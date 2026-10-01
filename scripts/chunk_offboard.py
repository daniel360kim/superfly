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

--smooth (off by default; without it the above is unchanged): the command is
not streamed as a bare velocity step but drives SmoothRef -- the sim's own
plant (a' = (kv (v_cmd - v) - a) / 0.1) with a jerk and an accel limit -- and
PX4 gets that plant's velocity AND acceleration (feedforward) at 60 Hz, the
yaw-rate command low-passed and rate-limited. Position stays masked (a
position setpoint would fight the z band / altitude logic).

--shield (off by default; without it nothing changes): a short-lived
egocentric obstacle memory (each decision's depth frame subsampled and
back-projected to local points with the odometry pose, ~1.2 s kept, capped)
and a clearance shield on the commanded velocity at 60 Hz: the component
toward any remembered point that the next --shield-horizon seconds of motion
would bring within --shield-margin is removed (reversed inside the margin),
with a repulsive gain; the shield never adds speed
(superfly.policies.chunk.ObstacleMemory / ClearanceShield). With --smooth it
filters the plant's target AND its output (written back into the plant).
--yaw-to-vel K (off by default) adds K * (heading(velocity) - yaw) to the yaw
rate so the camera turns with the velocity during swerves.

Logs (when the runner sets SUPERFLY_STATE_LOG): chunk_decisions.csv (one row
per decision, unchanged) and, unless --no-net-log, chunk_outputs.npy + .json
beside it -- every head's raw chunk, the gate logits, the decision pose (pos,
vel, R, heading yaw) and the setpoint sent, ~1.4 kB per decision
(superfly.policies.chunk.DecisionLog; paths via chunk_paths). Logging only.

--rgb (RGB students, 2026-10-01; required when the ONNX has an `rgb` input,
refused otherwise): the sim's policy RGB camera (run_px4_sim with
SUPERFLY_POLICY_RGB=1: 640x480, 87 deg HFOV, the Starling nose-camera mount)
arrives through shared memory (superfly.common.transport.RgbSubscriber) and is
fed through superfly.policies.rgb_preproc (exact area resize to the graph's
input size + ImageNet norm per the sidecar); depth inputs of an RGB graph get
the blank frame. With SUPERFLY_STATE_LOG set it also writes policy_inputs.csv
(per decision: RGB frame sim stamp and age, frames received / incomplete,
forward ms) and rgb_first_decision.npy (the 640x480 frame of the first
decision); --save-rgb-every S adds rgb_frames.npz (net-input frames every S s).
Depth students: nothing changes.

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
from superfly.policies.chunk import (ChunkPolicy, SmoothRef, ObstacleMemory, ClearanceShield,
                                     DecisionLog, yaw_toward)

CONTROL_HZ = 60.0          # a multiple of DECISION_HZ: exactly every 4th tick decides
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


def send_velocity_accel_yawrate_ned(mav, v_ned, a_ned, yaw_rate_ned):
    """Velocity + acceleration feedforward + yaw-RATE setpoint (--smooth):
    position and yaw masked. PX4's velocity controller adds the acceleration
    to its own P/I output (PositionControl: acc_sp = ff + vel loop)."""
    IGNORE_POS = 1 | 2 | 4
    IGNORE_YAW = 1024
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        IGNORE_POS | IGNORE_YAW,
        0.0, 0.0, 0.0,
        float(v_ned[0]), float(v_ned[1]), float(v_ned[2]),
        float(a_ned[0]), float(a_ned[1]), float(a_ned[2]),
        0.0, float(yaw_rate_ned))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="chunk_v1 student .onnx")
    ap.add_argument("--connect", default="udp:localhost:14550")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--rgb", action="store_true",
                    help="Subscribe to the sim's policy RGB camera (SUPERFLY_POLICY_RGB=1) and feed "
                         "the ONNX's rgb input. Required for an RGB student.")
    ap.add_argument("--save-rgb-every", type=float, default=0.0, metavar="S",
                    help="--rgb + SUPERFLY_STATE_LOG: keep the network-input RGB every S s of policy "
                         "time in rgb_frames.npz (0 = off; the first decision's frame is always kept).")
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
    ap.add_argument("--dwell", type=float, default=0.0,
                    help="Minimum time on a head [s]: inside it a switch needs --dwell-margin "
                         "(python sim --chunk-dwell; 0 = off).")
    ap.add_argument("--dwell-margin", type=float, default=0.3)
    # Plan B track 3 (2026-09-30): odometry-only executor guards, off by default = unchanged;
    # python sim twins --chunk-side-dwell/--chunk-side-margin/--chunk-flip-margin/--chunk-stuck*
    ap.add_argument("--side-dwell", type=float, default=0.0,
                    help="Once the executed head is left/right, leaving it within this many s "
                         "needs --side-margin (0 = off). Isaac walls 2026-09-29: 11-17 executed "
                         "L/R flips in front of the wall, most after 1-2 decisions.")
    ap.add_argument("--side-margin", type=float, default=1.0,
                    help="Margin inside the side dwell (1.0 = hard: no switch).")
    ap.add_argument("--flip-margin", type=float, default=0.0,
                    help="Margin for a switch into the side head opposite the last executed side "
                         "head, at any time (0 = off; a stronger hysteresis for side changes).")
    ap.add_argument("--stuck", type=float, default=0.0, metavar="WINDOW",
                    help="Stuck watchdog window [s] (0 = off): goal distance down by < "
                         "--stuck-progress over it while the mean commanded speed >= --stuck-v and "
                         "the goal > --stuck-min-dist away -> the best non-straight head by gate "
                         "(not the one in use, not one already tried) is forced for --stuck-hold s. "
                         "EnglishCollege 2026-09-29: pinned ~150 s commanding 1.2 m/s.")
    ap.add_argument("--stuck-progress", type=float, default=0.5)
    ap.add_argument("--stuck-v", type=float, default=0.5)
    ap.add_argument("--stuck-hold", type=float, default=2.5)
    ap.add_argument("--stuck-min-dist", type=float, default=1.5)
    ap.add_argument("--stuck-disp", type=float, default=1.0,
                    help="--stuck: only if the vehicle moved less than this [m] over the window "
                         "(a detour along a wall is not stuck).")
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--mix-heads", action="store_true",
                    help="Ensemble chunks across head switches (sim default off=same-head here).")
    ap.add_argument("--v-cap", type=float, default=3.5)
    ap.add_argument("--v-cap-xy", action="store_true",
                    help="--v-cap limits the HORIZONTAL speed only; the vertical setpoint passes "
                         "unscaled (the z band still applies). Off (default) = the whole 3-D "
                         "setpoint is scaled, so a chunk asking (vx 2.75, vz 0.5) flies "
                         "(1.47, 0.27) under --v-cap 1.5 -- the climb rate falls with the "
                         "forward speed (python sim twin: scratch_t6/ou_bench/sim_run_ou.py "
                         "OU_XYCAP=1).")
    ap.add_argument("--z-min", type=float, default=Z_REF[0],
                    help="Floor of the executor's z band [m] (python sim --chunk-z-min; default "
                         "0.5 = sim_episode.Z_REF[0]). The vertical setpoint is limited so the "
                         "vehicle takes >= 0.5 s to reach it. Isaac 2026-09-29: the recipe-v3 chunk "
                         "students sink from the 1.9 m handoff to the 0.5 m floor within 2-4 s "
                         "(python-sim flights: 1.3-2.3 m); all six of their diffphys contacts were "
                         "at 0.4-0.7 m (bars, box tops/undersides, low spheres). Off = 0.5.")
    ap.add_argument("--smooth", action="store_true",
                    help="Smooth executor (off = the velocity-step path, unchanged): the ensembled "
                         "command drives the python sim's plant (a' = (kv (v_cmd - v) - a) / 0.1) "
                         "with a jerk and an accel limit, and PX4 gets that plant's velocity AND "
                         "acceleration (feedforward); the yaw-rate command is low-passed and "
                         "rate-limited (superfly.policies.chunk.SmoothRef; sim --chunk-smooth).")
    ap.add_argument("--smooth-jerk", type=float, default=8.0, help="--smooth jerk limit [m/s^3].")
    ap.add_argument("--smooth-acc", type=float, default=4.0, help="--smooth accel limit [m/s^2].")
    ap.add_argument("--smooth-yaw-tau", type=float, default=0.25,
                    help="--smooth yaw-rate low-pass time constant [s].")
    ap.add_argument("--smooth-yaw-acc", type=float, default=4.0,
                    help="--smooth yaw acceleration limit [rad/s^2].")
    ap.add_argument("--smooth-ff", action="store_true",
                    help="--smooth: add the chunk's own slope to the plant's accel target "
                         "(sim --chunk-feedforward; off as there).")
    ap.add_argument("--shield", action="store_true",
                    help="Clearance shield (off = unchanged): remember ~--mem-horizon s of "
                         "back-projected depth points and remove/reverse the commanded velocity "
                         "component toward any point the next --shield-horizon s of motion brings "
                         "within --shield-margin. Never adds speed. Isaac 2026-09-29: 16 of 20 "
                         "passes under 0.25 m were side passes whose obstacle had left the 91 deg "
                         "view 0.06-0.8 s earlier.")
    ap.add_argument("--shield-margin", type=float, default=0.5,
                    help="--shield: centre-to-surface margin [m] (clearance ~ margin - 0.2 m).")
    ap.add_argument("--shield-horizon", type=float, default=0.5,
                    help="--shield: predicted motion checked against the memory [s].")
    ap.add_argument("--shield-gain", type=float, default=2.0,
                    help="--shield: allowed approach speed = gain * (distance - margin) [1/s].")
    ap.add_argument("--shield-vrep", type=float, default=0.5,
                    help="--shield: max retreat speed inside the margin [m/s].")
    ap.add_argument("--mem-horizon", type=float, default=1.2,
                    help="--shield: obstacle memory length [s].")
    ap.add_argument("--mem-stride", type=int, default=8,
                    help="--shield: depth subsample stride [px] (224/8 = 28x28 rays per frame).")
    ap.add_argument("--mem-range", type=float, default=5.0,
                    help="--shield: farthest depth point remembered [m].")
    ap.add_argument("--mem-cap", type=int, default=3000,
                    help="--shield: max points (the nearest are kept).")
    ap.add_argument("--mem-cam-pitch", type=float, default=0.0,
                    help="--shield: effective camera pitch [deg, + = down] for the back-"
                         "projection (Isaac with SUPERFLY_CAM_PITCH_DEG=-13: level = 0).")
    ap.add_argument("--yaw-lead", type=float, default=0.0,
                    help="Read each chunk's yaw rate on the step containing tau + YAW_LEAD [s] "
                         "(0 = at tau, the old behaviour; the turn test used YAW_LEAD = --lead).")
    ap.add_argument("--yaw-to-vel", type=float, default=0.0,
                    help="Add K * (heading(velocity) - yaw) to the yaw rate [1/s] above 0.5 m/s "
                         "(0 = off): turns the camera with the velocity during swerves.")
    ap.add_argument("--no-net-log", action="store_true",
                    help="Do not write chunk_outputs.npy/.json (default: written next to "
                         "chunk_decisions.csv whenever SUPERFLY_STATE_LOG is set -- every head's "
                         "raw chunk, the gate logits and the decision pose, ~1.4 kB per decision; "
                         "read by scratch_t6/isaac_eval/render_net_overlay.py). Logging only: "
                         "control is identical either way.")
    ap.add_argument("--kv", type=float, default=3.0,
                    help="PX4 velocity P gain (MPC_XY/Z_VEL_P_ACC) = sim KV_VEL.")
    ap.add_argument("--clock", choices=["wall", "px4"], default="px4",
                    help="px4 (default): decisions, chunk ages and the ensemble run on "
                         "PX4's clock = sim time under lockstep SITL (see "
                         "px4_offboard.Px4Clock); wall = the host clock.")
    ap.add_argument("--policy-timeout", type=float, default=None,
                    help="Land (not reached) after this many --clock seconds of POLICY.")
    ap.add_argument("--post-goal-hold", type=float, default=None, metavar="S",
                    help="After the goal is REACHED: send the land command as usual, keep "
                         "streaming for S --clock seconds, then exit without waiting for "
                         "touchdown (PX4 is in AUTO.LAND by then; no disarm is sent). The "
                         "harness scores clearance/speed up to the first 3D goal-sphere entry, "
                         "~0.2 s after the handoff, so S >= 1 leaves every metric unchanged "
                         "and saves the ~8 s descent. A policy TIMEOUT still lands fully (its "
                         "clearance window runs to the log end). Default: land fully.")
    ap.add_argument("--hover-thrust", type=float, default=None,
                    help="MPC_THR_HOVER for the airframe (PX4's estimator refines it).")
    args = ap.parse_args()

    goal = np.array([args.goal[0], args.goal[1], args.climb_alt], float)
    from superfly.common.transport import DepthSubscriber
    depth_sub = DepthSubscriber() if args.depth else None
    rgb_sub = None
    if args.rgb:
        from superfly.common.transport import RgbSubscriber
        rgb_sub = RgbSubscriber()

    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)
    request_stream_rates(mav, CONTROL_HZ)
    state = DroneState()
    stop = threading.Event()
    threading.Thread(target=receive_loop, args=(mav, state, stop), daemon=True).start()
    clock = make_clock(args.clock, state)

    policy = ChunkPolicy(args.checkpoint, lead=args.lead, hysteresis=args.hysteresis,
                         ensemble=args.ensemble, same_head=not args.mix_heads,
                         goal_speed=args.goal_speed, dwell=args.dwell,
                         dwell_margin=args.dwell_margin, side_dwell=args.side_dwell,
                         side_margin=args.side_margin, flip_margin=args.flip_margin,
                         stuck_window=args.stuck, stuck_progress=args.stuck_progress,
                         stuck_v=args.stuck_v, stuck_hold=args.stuck_hold,
                         stuck_min_dist=args.stuck_min_dist, stuck_disp=args.stuck_disp,
                         v_cap=args.v_cap, yaw_lead=args.yaw_lead)
    if policy.modality == "rgb" and rgb_sub is None:
        raise SystemExit(f"[chunk] {args.checkpoint} is an RGB student (input {policy.rgb['name']!r}) "
                         "-- run with --rgb (and the sim with SUPERFLY_POLICY_RGB=1)")
    if policy.modality != "rgb" and rgb_sub is not None:
        raise SystemExit(f"[chunk] --rgb given but {args.checkpoint} has no rgb input "
                         f"(inputs {list(policy.inputs)})")
    if policy.rgb is not None:
        print(f"[chunk] RGB student: input {policy.rgb['name']!r} {policy.inputs[policy.rgb['name']]} "
              f"layout {policy.rgb['layout']}, size {policy.rgb['size']}, norm {policy.rgb['norm']}"
              f"{'' if policy.rgb['norm_from_sidecar'] else ' (DEFAULT -- the sidecar has no rgb_input.norm)'}"
              f", area resize (superfly.policies.rgb_preproc); depth inputs get the blank frame", flush=True)
    print(f"[chunk] {Path(args.checkpoint).name}: heads {policy.heads}, "
          f"{policy.steps} x {policy.cdt:g} s, lead {policy.lead:g} s, "
          f"hysteresis {policy.hysteresis:g}, dwell {policy.dwell:g}/{policy.dwell_margin:g}, ensemble {policy.ensemble} "
          f"({'same-head' if policy.same_head else 'mixed'}), v_cap {args.v_cap:g}, "
          f"z floor {args.z_min:g}, forward {policy.forward_ms:.1f} ms", flush=True)
    if args.side_dwell > 0 or args.flip_margin > 0 or args.stuck > 0:
        print(f"[chunk] GUARDS: side dwell {args.side_dwell:g} s (margin {args.side_margin:g}), "
              f"flip margin {args.flip_margin:g}, stuck watchdog "
              + (f"{args.stuck:g} s / {args.stuck_progress:g} m / {args.stuck_v:g} m/s, hold "
                 f"{args.stuck_hold:g} s, beyond {args.stuck_min_dist:g} m, moved < {args.stuck_disp:g} m"
                 if args.stuck > 0 else "off"),
              flush=True)
    smooth = None
    if args.smooth:
        smooth = SmoothRef(kv=args.kv, jerk=args.smooth_jerk, acc_max=args.smooth_acc,
                           yaw_tau=args.smooth_yaw_tau, yaw_acc=args.smooth_yaw_acc)
        print(f"[chunk] SMOOTH executor: jerk {args.smooth_jerk:g} m/s^3, acc {args.smooth_acc:g} "
              f"m/s^2, yaw tau {args.smooth_yaw_tau:g} s, yaw acc {args.smooth_yaw_acc:g} rad/s^2, "
              f"chunk slope ff {'on' if args.smooth_ff else 'off'}; PX4 gets velocity + accel", flush=True)

    memory = shield = None
    if args.shield:
        memory = ObstacleMemory(horizon=args.mem_horizon, stride=args.mem_stride,
                                r_max=args.mem_range, cap=args.mem_cap,
                                cam_pitch_deg=args.mem_cam_pitch)
        shield = ClearanceShield(margin=args.shield_margin, horizon=args.shield_horizon,
                                 gain=args.shield_gain, v_rep=args.shield_vrep)
        print(f"[chunk] SHIELD: margin {args.shield_margin:g} m, horizon {args.shield_horizon:g} s, "
              f"gain {args.shield_gain:g}/s, v_rep {args.shield_vrep:g} m/s; memory "
              f"{args.mem_horizon:g} s, stride {args.mem_stride}, range {args.mem_range:g} m, cap "
              f"{args.mem_cap}, cam pitch {args.mem_cam_pitch:g} deg", flush=True)
        if not args.depth:
            print("[chunk] WARNING: --shield without --depth: the memory stays empty", flush=True)
    if args.yaw_lead > 0:
        print(f"[chunk] YAW-LEAD: chunk yaw rate read at tau + {args.yaw_lead:g} s", flush=True)
    if args.yaw_to_vel > 0:
        print(f"[chunk] YAW-TO-VEL: + {args.yaw_to_vel:g} * (heading(v) - yaw) rad/s", flush=True)
    sh_stat = dict(ticks=0, dv=0.0, dmin=float("inf"), n=0)   # since the last csv row
    sh_tot = dict(ticks=0, pol_ticks=0, dv=0.0, dmin=float("inf"))

    def shield_v(v_in, pos, vel, now):
        v_out, info = shield.apply(v_in, pos, memory.points(now, pos), vel)
        sh_stat["dmin"] = min(sh_stat["dmin"], info["dmin"])
        sh_tot["dmin"] = min(sh_tot["dmin"], info["dmin"])
        if info["dv"] > 1e-3:
            sh_stat["dv"] = max(sh_stat["dv"], info["dv"])
            sh_stat["n"] = max(sh_stat["n"], info["n"])
            sh_tot["dv"] = max(sh_tot["dv"], info["dv"])
        return v_out, info["dv"]

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
                      "cmd_vx,cmd_vy,cmd_vz,cmd_yr,n_ens"
                      + (",tgt_vx,tgt_vy,tgt_vz,tgt_yr,cmd_ax,cmd_ay,cmd_az" if smooth else "")
                      + (",mem_n,sh_ticks,sh_dv,sh_n,sh_dmin" if shield else "")
                      + (",yaw_vel_yr" if args.yaw_to_vel > 0 else "")
                      + "\n")
    net_log = None
    if sl and not args.no_net_log:
        try:
            net_log = DecisionLog(
                Path(sl).with_name("chunk_outputs.npy"), policy.heads, policy.steps,
                meta=dict(dt=policy.cdt, checkpoint=Path(args.checkpoint).name,
                          goal_local_enu=[float(x) for x in goal], lead=policy.lead,
                          hysteresis=policy.hysteresis, dwell=policy.dwell,
                          ensemble=policy.ensemble, same_head=policy.same_head,
                          v_cap=args.v_cap, v_cap_xy=bool(args.v_cap_xy), z_min=args.z_min, smooth=bool(args.smooth),
                          shield=bool(args.shield), yaw_to_vel=args.yaw_to_vel, yaw_lead=args.yaw_lead,
                          side_dwell=args.side_dwell, side_margin=args.side_margin,
                          flip_margin=args.flip_margin, stuck=args.stuck,
                          stuck_progress=args.stuck_progress, stuck_v=args.stuck_v,
                          stuck_hold=args.stuck_hold, stuck_min_dist=args.stuck_min_dist,
                          stuck_disp=args.stuck_disp,
                          decision_hz=DECISION_HZ, clock=args.clock, modality=policy.modality,
                          rgb_input=({k: v for k, v in policy.rgb.items() if k != "dtype"}
                                     if policy.rgb is not None else None)))
        except Exception as e:      # logging must never stop a flight
            print(f"[chunk] net log off: {type(e).__name__}: {e}", flush=True)
            net_log = None

    in_log = None
    rgb_keep, rgb_keep_t, rgb_next_keep = [], [], 0.0
    if sl and rgb_sub is not None:
        in_log = open(Path(sl).with_name("policy_inputs.csv"), "w")
        in_log.write("t,rgb_stamp,rgb_age_wall,rgb_frames,rgb_retries,rgb_mean,forward_ms\n")

    dt = 1.0 / CONTROL_HZ
    phase = "CLIMB"
    t0 = clock()
    next_step = t0
    last_hb = 0.0
    last_arm = time.time()
    last_dec = -1e9
    next_dec = -1e9
    policy_t0 = None
    landing_sent = False
    reached_t = None            # --clock time of the goal handoff (--post-goal-hold)
    skip_disarm = False
    lo, hi = float(args.z_min), Z_REF[1]
    n_dec = 0
    t_prev = None
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
                    next_dec = now
                    mark_policy_phase("start")
                    policy.reset()
                    if memory is not None:
                        memory.reset()
                    if smooth is not None:
                        smooth.reset(vel)
                    t_prev = now
                    hi = max(Z_REF[1], float(pos[2]) + Z_HEADROOM)
                    print(f">>> HANDOFF to the chunk policy at z={pos[2]:.2f} "
                          f"(z band {lo:.1f}-{hi:.1f} m)", flush=True)
            elif phase == "POLICY":
                # on a fixed 15 Hz schedule (not "15 Hz since the last one",
                # which quantizes to every 5th 60 Hz tick = 12 Hz)
                if now >= next_dec - 0.002 or not policy.ring:   # PX4 time is in ms
                    next_dec = max(next_dec + 1.0 / DECISION_HZ, now)
                    depth = depth_sub.latest() if depth_sub else None
                    rgb, rgb_st, rgb_wall = (rgb_sub.latest_stamped() if rgb_sub is not None
                                             else (None, None, None))
                    if rgb_sub is not None and rgb is None and n_dec == 0:
                        print("[chunk] WARNING: no RGB frame yet at the first decision "
                              "(the net sees mid-grey) -- is the sim running with SUPERFLY_POLICY_RGB=1?",
                              flush=True)
                    t_fw = time.perf_counter()
                    rec = policy.decide(now, pos, R, vel, om, goal, depth, rgb=rgb)
                    fw_ms = (time.perf_counter() - t_fw) * 1e3
                    if in_log is not None:
                        try:
                            in_log.write(f"{now - t0:.4f},{-1.0 if rgb_st is None else rgb_st:.4f},"
                                         f"{-1.0 if rgb_wall is None else time.time() - rgb_wall:.4f},"
                                         f"{rgb_sub.frames},{rgb_sub.incomplete},"
                                         f"{-1.0 if rgb is None else float(rgb.mean()):.2f},{fw_ms:.1f}\n")
                            if rgb is not None and n_dec == 0:
                                np.save(Path(sl).with_name("rgb_first_decision.npy"), rgb)
                            if (rgb is not None and args.save_rgb_every > 0
                                    and now - policy_t0 >= rgb_next_keep):
                                from superfly.policies.rgb_preproc import area_resize
                                rgb_keep.append(np.clip(np.rint(area_resize(
                                    rgb, tuple(policy.rgb["size"]))), 0, 255).astype(np.uint8))
                                rgb_keep_t.append(now - t0)
                                rgb_next_keep = now - policy_t0 + args.save_rgb_every
                        except Exception as e:      # logging never stops a flight
                            print(f"[chunk] input log off: {type(e).__name__}: {e}", flush=True)
                            in_log = None
                    if memory is not None:
                        memory.add(now, depth, pos, R)
                    last_dec = now
                    n_dec += 1
                v, yr, n_used = policy.command(now)
                if args.v_cap_xy:
                    sp = float(np.hypot(v[0], v[1]))
                    if sp > args.v_cap:
                        v = np.array([v[0] * args.v_cap / sp, v[1] * args.v_cap / sp, v[2]])
                else:
                    sp = float(np.linalg.norm(v))
                    if sp > args.v_cap:
                        v = v * (args.v_cap / sp)
                # sim_episode.velocity_setpoint: >= Z_SOFT_T to either bound
                vz_lo = (lo - pos[2]) / Z_SOFT_T
                vz_hi = (hi - pos[2]) / Z_SOFT_T
                v[2] = min(max(v[2], min(vz_lo, 0.0)), max(vz_hi, 0.0))
                sh_dv = 0.0
                if shield is not None:
                    v, sh_dv = shield_v(v, pos, vel, now)
                    v[2] = min(max(v[2], min(vz_lo, 0.0)), max(vz_hi, 0.0))
                yv_yr = 0.0
                if args.yaw_to_vel > 0:
                    yv_yr = yaw_toward(yaw, vel, args.yaw_to_vel)
                    yr = yr + yv_yr
                yr = float(np.clip(yr, -YAW_RATE_MAX, YAW_RATE_MAX))
                if smooth is None:
                    # ENU (x E, y N, z U) -> NED; ENU yaw rate (CCW about up) -> NED -yr
                    send_velocity_yawrate_ned(mav, v[1], v[0], -v[2], -yr)
                else:
                    v_tgt, yr_tgt = v.copy(), yr
                    a_ff = policy.command_acc(now) if args.smooth_ff else None
                    v, a, yr = smooth.step(v_tgt, yr_tgt, min(max(now - t_prev, 0.0), 0.1), a_ff)
                    if shield is not None:
                        # the plant lags its (already shielded) target: filter
                        # its output too, write it back, and drop any accel
                        # still pointing against the correction
                        v_s, dv2 = shield_v(v, pos, vel, now)
                        if dv2 > 1e-3:
                            u = (v_s - v) / dv2
                            au = float(a @ u)
                            if au < 0:
                                a = a - au * u
                                smooth.a = a.copy()
                            v = v_s
                            smooth.v = v.copy()
                            sh_dv = max(sh_dv, dv2)
                    # the z band again on the smoothed setpoint; no vertical
                    # feedforward pushing further out of it
                    vz = min(max(v[2], min(vz_lo, 0.0)), max(vz_hi, 0.0))
                    if vz != v[2]:
                        floor_bit = v[2] < vz
                        v[2] = vz
                        smooth.v[2] = vz
                        if (a[2] < 0) if floor_bit else (a[2] > 0):
                            a[2] = 0.0
                            smooth.a[2] = 0.0
                    send_velocity_accel_yawrate_ned(mav, (v[1], v[0], -v[2]),
                                                    (a[1], a[0], -a[2]), -yr)
                t_prev = now
                if shield is not None:
                    sh_tot["pol_ticks"] += 1
                    if sh_dv > 1e-3:
                        sh_stat["ticks"] += 1
                        sh_tot["ticks"] += 1
                if dec_log is not None and last_dec == now:
                    p = policy.last["probs"]
                    dec_log.write(",".join(f"{x:.4f}" for x in (
                        now - t0, *pos, *vel, yaw)) + f",{policy.last['head']},"
                        f"{policy.last['reason']}," + ",".join(f"{x:.3f}" for x in p)
                        + "," + ",".join(f"{x:.3f}" for x in (*v, yr)) + f",{n_used}"
                        + ("," + ",".join(f"{x:.3f}" for x in (*v_tgt, yr_tgt, *a))
                           if smooth is not None else "")
                        + (f",{len(memory.pts)},{sh_stat['ticks']},{sh_stat['dv']:.3f},"
                           f"{sh_stat['n']},{min(sh_stat['dmin'], 99.0):.3f}" if shield is not None else "")
                        + (f",{yv_yr:.3f}" if args.yaw_to_vel > 0 else "") + "\n")
                    sh_stat.update(ticks=0, dv=0.0, dmin=float("inf"), n=0)
                if net_log is not None and last_dec == now:
                    net_log.add(now - t0, pos, vel, R, om, policy.last, v, yr, n_used)
                dist = float(np.linalg.norm((goal - pos)[:2]))
                if verbose:
                    print(f"[POLICY t={now - t0:.1f}] pos={pos.round(2)} |v|={np.linalg.norm(vel):.2f} "
                          f"cmd={np.round(v, 2)} yr={yr:+.2f} head={policy.heads[policy.head]} "
                          f"p={np.round(policy.last['probs'], 2)} n_ens={n_used} "
                          f"switches={policy.switches} dist={dist:.1f} decisions={n_dec}"
                          + (f" mem={len(memory.pts)} shield_ticks={sh_tot['ticks']}"
                             if shield is not None else "")
                          + (f" rgb_frames={rgb_sub.frames}/{rgb_sub.written}"
                             f" fwd_ms={fw_ms:.0f}" if rgb_sub is not None and n_dec else ""),
                          flush=True)
                if dist < args.goal_radius:
                    phase = "LANDING"
                    reached_t = now
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
                if (args.post_goal_hold is not None and reached_t is not None
                        and now - reached_t >= args.post_goal_hold):
                    print(f">>> post-goal hold {args.post_goal_hold:g} s done at {pos.round(2)} "
                          f"-- exiting without waiting for touchdown", flush=True)
                    skip_disarm = True
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if policy.stuck is not None:
            print(f"[chunk] STUCK summary: {len(policy.stuck.fires)} fire(s) "
                  f"{policy.stuck.fires}", flush=True)
        if shield is not None:
            print(f"[chunk] SHIELD summary: active {sh_tot['ticks']}/{sh_tot['pol_ticks']} policy ticks "
                  f"({sh_tot['ticks'] / CONTROL_HZ:.1f} s), max |dv| {sh_tot['dv']:.2f} m/s, nearest "
                  f"remembered point {sh_tot['dmin']:.2f} m", flush=True)
        if dec_log is not None:
            dec_log.close()
        if net_log is not None:
            net_log.close()
        if in_log is not None:
            in_log.close()
        if rgb_keep and sl:
            try:
                np.savez_compressed(Path(sl).with_name("rgb_frames.npz"),
                                    rgb=np.stack(rgb_keep), t=np.asarray(rgb_keep_t))
            except Exception as e:
                print(f"[chunk] rgb_frames.npz not written: {e}", flush=True)
        if rgb_sub is not None:
            print(f"[chunk] RGB summary: {rgb_sub.frames} complete frames, "
                  f"{rgb_sub.incomplete} read retries, {rgb_sub.written} written by the sim", flush=True)
            rgb_sub.close()
        stop.set()
        if not skip_disarm:     # an in-air disarm would drop the vehicle
            mav.mav.command_long_send(mav.target_system, mav.target_component,
                                      mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                      0, 0, 0, 0, 0, 0, 0, 0)
        mark_offboard_done()


if __name__ == "__main__":
    main()
