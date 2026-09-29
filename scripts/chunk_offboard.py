#!/usr/bin/env python
"""
Velocity-chunk student offboard (method `agile_student_chunk`).

The student (ONNX, sidecar `arch: chunk_v1`) sees exactly what agile_student
sees -- the 224x224 depth of the `agile` sim camera and the 22-dim state --
but emits a 1.5 s heading-frame velocity + yaw-rate chunk per head and a gate
(superfly.policies.chunk). The executor (gate argmax with hysteresis,
same-head temporal ensemble, a lead into each chunk) runs at 15 Hz, the
python sim's decision rate; the ensembled command is streamed to PX4's
velocity loop at 50 Hz as SET_POSITION_TARGET_LOCAL_NED velocity + YAW RATE
(position/accel/yaw masked), the way depthnav_vel streams velocity.

Plant matching (sim_episode's velocity-setpoint mode): a = 3 (v_cmd - v), so
MPC_XY_VEL_P_ACC / MPC_Z_VEL_P_ACC are set to --kv (3.0); speed setpoints
above --v-cap (sim_episode.V_CAP 3.5) are scaled down to it; the vertical
setpoint is limited so the vehicle takes >= 0.5 s to reach either bound of
the student's z band (sim_episode.velocity_setpoint); yaw rate clipped to
+-2 rad/s (YAW_RATE_MAX).

Phases as agile_offboard: CLIMB (position hold) -> YAW (face goal) -> POLICY
-> LANDING when within --goal-radius horizontally.
"""

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pymavlink import mavutil

from superfly.common.px4_offboard import (
    HEARTBEAT_HZ, DroneState, wait_for_heartbeat, request_stream_rates,
    set_offboard_mode, arm, retry_offboard_arm, send_position_target_ned,
    send_land_command, send_heartbeat, set_param_float, receive_loop, make_clock,
)
from superfly.common.sentinels import mark_policy_phase, mark_offboard_done
from superfly.policies.chunk import ChunkPolicy

CONTROL_HZ = 50.0
DECISION_HZ = 15.0          # sim_episode.DECISION_HZ
YAW_RATE_MAX = 2.0          # sim_episode.YAW_RATE_MAX
Z_REF = (0.5, 4.0)          # sim_episode.Z_REF
Z_HEADROOM = 2.0            # as agile core's student_z_band
Z_SOFT_T = 0.5              # sim_episode.Z_SOFT_T


def send_velocity_yawrate_ned(mav, vx_n, vy_e, vz_d, yaw_rate_ned):
    """Velocity + yaw-RATE setpoint: position, accel and yaw masked."""
    IGNORE_POS = 1 | 2 | 4
    IGNORE_ACC = 64 | 128 | 256
    IGNORE_YAW = 1024
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        IGNORE_POS | IGNORE_ACC | IGNORE_YAW,
        0.0, 0.0, 0.0,
        float(vx_n), float(vy_e), float(vz_d),
        0.0, 0.0, 0.0,
        0.0, float(yaw_rate_ned))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="chunk_v1 student .onnx")
    ap.add_argument("--connect", default="udp:localhost:14550")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--goal", type=float, nargs=2, required=True, metavar=("X", "Y"))
    ap.add_argument("--climb-alt", type=float, default=2.0)
    ap.add_argument("--arrive-tol", type=float, default=0.3)
    ap.add_argument("--settle-speed", type=float, default=0.2)
    ap.add_argument("--yaw-tol-deg", type=float, default=5.0)
    ap.add_argument("--goal-radius", type=float, default=1.0,
                    help="HORIZONTAL distance [m] that hands off to landing (as agile).")
    ap.add_argument("--goal-speed", type=float, default=0.0)
    ap.add_argument("--lead", type=float, default=0.5,
                    help="Executor reach into each chunk [s] (python sim --chunk-lead).")
    ap.add_argument("--hysteresis", type=float, default=0.15)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--mix-heads", action="store_true",
                    help="Ensemble chunks across head switches (sim default off=same-head here).")
    ap.add_argument("--v-cap", type=float, default=3.5)
    ap.add_argument("--kv", type=float, default=3.0,
                    help="PX4 velocity P gain (MPC_XY/Z_VEL_P_ACC) = sim KV_VEL.")
    ap.add_argument("--clock", choices=["wall", "px4"], default="px4",
                    help="px4 (default): decisions, chunk ages and the ensemble run on "
                         "PX4's clock = sim time under lockstep SITL (see "
                         "px4_offboard.Px4Clock); wall = the host clock.")
    ap.add_argument("--policy-timeout", type=float, default=None,
                    help="Land (not reached) after this many --clock seconds of POLICY.")
    ap.add_argument("--hover-thrust", type=float, default=None,
                    help="MPC_THR_HOVER for the airframe (PX4's estimator refines it).")
    args = ap.parse_args()

    goal = np.array([args.goal[0], args.goal[1], args.climb_alt], float)
    from superfly.common.transport import DepthSubscriber
    depth_sub = DepthSubscriber() if args.depth else None

    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)
    request_stream_rates(mav, CONTROL_HZ)
    state = DroneState()
    stop = threading.Event()
    threading.Thread(target=receive_loop, args=(mav, state, stop), daemon=True).start()
    clock = make_clock(args.clock, state)

    policy = ChunkPolicy(args.checkpoint, lead=args.lead, hysteresis=args.hysteresis,
                         ensemble=args.ensemble, same_head=not args.mix_heads,
                         goal_speed=args.goal_speed)
    print(f"[chunk] {Path(args.checkpoint).name}: heads {policy.heads}, "
          f"{policy.steps} x {policy.cdt:g} s, lead {policy.lead:g} s, "
          f"hysteresis {policy.hysteresis:g}, ensemble {policy.ensemble} "
          f"({'same-head' if policy.same_head else 'mixed'}), v_cap {args.v_cap:g}, "
          f"forward {policy.forward_ms:.1f} ms", flush=True)

    for name, val in (("MPC_XY_VEL_P_ACC", args.kv), ("MPC_Z_VEL_P_ACC", args.kv),
                      ("MPC_XY_VEL_MAX", max(4.0, args.v_cap + 0.5)),
                      ("MPC_Z_VEL_MAX_UP", 3.0), ("MPC_Z_VEL_MAX_DN", 3.0)):
        set_param_float(mav, name, float(val))
    if args.hover_thrust is not None:
        set_param_float(mav, "MPC_THR_HOVER", float(np.clip(args.hover_thrust, 0.1, 0.9)))
    time.sleep(0.3)

    pos0, _, _, yaw0 = state.get()
    xn, ye, zd = pos0[1], pos0[0], -args.climb_alt
    yaw_ground = math.atan2(math.sin(math.pi / 2 - yaw0), math.cos(math.pi / 2 - yaw0))
    yaw_goal = math.atan2(goal[0] - pos0[0], goal[1] - pos0[1])     # NED: atan2(E, N)
    for _ in range(30):
        send_position_target_ned(mav, xn, ye, zd, yaw_ground)
        send_heartbeat(mav)
        time.sleep(0.05)
    set_offboard_mode(mav)
    time.sleep(0.5)
    arm(mav)
    time.sleep(1.0)

    dec_log = None
    sl = os.environ.get("SUPERFLY_STATE_LOG")
    if sl:
        dec_log = open(Path(sl).with_name("chunk_decisions.csv"), "w")
        dec_log.write("t,x,y,z,vx,vy,vz,yaw,head,reason,p0,p1,p2,p3,p4,"
                      "cmd_vx,cmd_vy,cmd_vz,cmd_yr,n_ens\n")

    dt = 1.0 / CONTROL_HZ
    phase = "CLIMB"
    t0 = clock()
    next_step = t0
    last_hb = 0.0
    last_arm = time.time()
    last_dec = -1e9
    policy_t0 = None
    landing_sent = False
    lo, hi = Z_REF
    n_dec = 0
    try:
        while True:
            now = clock()
            wall = time.time()
            if wall - last_hb >= 1.0 / HEARTBEAT_HZ:
                send_heartbeat(mav)
                last_hb = wall
            if now < next_step:
                time.sleep(0.0005)
                continue
            next_step += dt
            if next_step < now:
                next_step = now
            pos, vel, R, om, yaw = state.get_full()
            verbose = int(now) != int(now - dt)
            if phase == "CLIMB":
                send_position_target_ned(mav, xn, ye, zd, yaw_ground)
                last_arm = retry_offboard_arm(mav, state, last_arm)
                if (abs(pos[2] - args.climb_alt) < args.arrive_tol
                        and np.linalg.norm(vel) < args.settle_speed and state.offboard):
                    phase = "YAW"
                    print(f">>> climbed to {pos[2]:.2f} m -- turning to the goal", flush=True)
            elif phase == "YAW":
                send_position_target_ned(mav, xn, ye, zd, yaw_goal)
                ycur = math.atan2(math.sin(math.pi / 2 - yaw), math.cos(math.pi / 2 - yaw))
                err = math.atan2(math.sin(yaw_goal - ycur), math.cos(yaw_goal - ycur))
                if abs(math.degrees(err)) < args.yaw_tol_deg and state.offboard:
                    phase = "POLICY"
                    policy_t0 = now
                    mark_policy_phase("start")
                    policy.reset()
                    hi = max(Z_REF[1], float(pos[2]) + Z_HEADROOM)
                    print(f">>> HANDOFF to the chunk policy at z={pos[2]:.2f} "
                          f"(z band {lo:.1f}-{hi:.1f} m)", flush=True)
            elif phase == "POLICY":
                if now - last_dec >= 1.0 / DECISION_HZ or not policy.ring:
                    depth = depth_sub.latest() if depth_sub else None
                    rec = policy.decide(now, pos, R, vel, om, goal, depth)
                    last_dec = now
                    n_dec += 1
                v, yr, n_used = policy.command(now)
                sp = float(np.linalg.norm(v))
                if sp > args.v_cap:
                    v = v * (args.v_cap / sp)
                # sim_episode.velocity_setpoint: >= Z_SOFT_T to either bound
                vz_lo = (lo - pos[2]) / Z_SOFT_T
                vz_hi = (hi - pos[2]) / Z_SOFT_T
                v[2] = min(max(v[2], min(vz_lo, 0.0)), max(vz_hi, 0.0))
                yr = float(np.clip(yr, -YAW_RATE_MAX, YAW_RATE_MAX))
                # ENU (x E, y N, z U) -> NED; ENU yaw rate (CCW about up) -> NED -yr
                send_velocity_yawrate_ned(mav, v[1], v[0], -v[2], -yr)
                if dec_log is not None and last_dec == now:
                    p = policy.last["probs"]
                    dec_log.write(",".join(f"{x:.4f}" for x in (
                        now - t0, *pos, *vel, yaw)) + f",{policy.last['head']},"
                        f"{policy.last['reason']}," + ",".join(f"{x:.3f}" for x in p)
                        + "," + ",".join(f"{x:.3f}" for x in (*v, yr)) + f",{n_used}\n")
                dist = float(np.linalg.norm((goal - pos)[:2]))
                if verbose:
                    print(f"[POLICY t={now - t0:.1f}] pos={pos.round(2)} |v|={np.linalg.norm(vel):.2f} "
                          f"cmd={np.round(v, 2)} yr={yr:+.2f} head={policy.heads[policy.head]} "
                          f"p={np.round(policy.last['probs'], 2)} n_ens={n_used} "
                          f"switches={policy.switches} dist={dist:.1f} decisions={n_dec}",
                          flush=True)
                if dist < args.goal_radius:
                    phase = "LANDING"
                    mark_policy_phase("end")
                    print(f">>> goal reached at {pos.round(2)}; landing", flush=True)
                elif args.policy_timeout is not None and now - policy_t0 > args.policy_timeout:
                    phase = "LANDING"                   # no "end" marker: not reached
                    print(f">>> POLICY TIMEOUT after {now - policy_t0:.1f} s ({args.clock} "
                          f"clock) at {pos.round(2)}; landing", flush=True)
            elif phase == "LANDING":
                if not landing_sent:
                    send_land_command(mav)
                    landing_sent = True
                if not state.armed:
                    print(">>> landed and disarmed", flush=True)
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if dec_log is not None:
            dec_log.close()
        stop.set()
        mav.mav.command_long_send(mav.target_system, mav.target_component,
                                  mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                  0, 0, 0, 0, 0, 0, 0, 0)
        mark_offboard_done()


if __name__ == "__main__":
    main()
