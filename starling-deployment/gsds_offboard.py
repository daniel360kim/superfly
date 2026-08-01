#!/usr/bin/env python
"""
gs_drone_sim (gsds) offboard controller for PX4. Run under the gs_drone_sim
venv (/home/danielkim/gs_drone_sim/.venv: torch cu128 + pymavlink + scipy).

Connects to PX4 via MAVLink, arms, climbs to --climb-alt, yaws to face the
goal, then runs the gs_drone_sim multi-hypothesis trajectory student + its
flatness tracker (wrapper/gsds_core.py) at --control-hz sending
SET_ATTITUDE_TARGET. Mirrors agile_offboard.py's structure and CLI (--goal
takes X Y; goal altitude is --climb-alt), including the mandatory 50 Hz
LOCAL_POSITION_NED / ATTITUDE_QUATERNION stream request (the GCS link's 1 Hz
default position stream starves any tracker re-solving faster than that).

The obs modality follows the checkpoint: an RGB student subscribes to the
JPEG RGB stream (port 15002), a depth student to the depth stream (15001);
run_px4_sim.py --policy gsds/gsds_depth publishes both.

Usage (via compare/run_comparison.py, which manages PX4 + the sim):
    python gsds_offboard.py --checkpoint ../checkpoints/GSDroneSim/il_b8_tube8_4k.pt \
        --goal 40 30 --climb-alt 2.0 --max-vel 2.0
"""

import argparse
import math
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrapper.gsds_core import GsdsPolicy, GsdsObs
from depth_transport import DepthSubscriber, RgbSubscriber
from agile_debug_transport import AgileDebugPublisher

CONTROL_HZ = 50.0
NET_HZ = 15.0
HEARTBEAT_HZ = 2.0
G = 9.80665
MAX_ACCEL = 20.0          # thrust-accel at full throttle; hover = g/MAX_ACCEL
STREAM_HZ = 50.0

STREAMED_MSGS = {
    mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: "LOCAL_POSITION_NED",
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION: "ATTITUDE_QUATERNION",
}

OFFBOARD_DONE_FILE = "/tmp/superfly_offboard_done"
POLICY_PHASE_FILE = "/tmp/superfly_policy_phase"

PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def _mark_policy_phase(event: str):
    try:
        with open(POLICY_PHASE_FILE, "a") as f:
            f.write(f"{event} {time.time()}\n")
    except Exception:
        pass


class DroneState:
    def __init__(self):
        self._lock = threading.Lock()
        self.position_enu = np.zeros(3)
        self.velocity_enu = np.zeros(3)
        self.R_enu = np.eye(3)
        self.yaw = 0.0
        self.armed = False
        self.offboard = False
        self.msg_counts = {name: 0 for name in STREAMED_MSGS.values()}

    def update_from_attitude(self, msg):
        q_ned_frd = Rotation.from_quat([msg.q2, msg.q3, msg.q4, msg.q1])
        rot_enu_flu = _rot_ENU_to_NED.inv() * q_ned_frd * _rot_FLU_to_FRD.inv()
        with self._lock:
            self.R_enu = rot_enu_flu.as_matrix()
            fwd_enu = self.R_enu[:, 0]
            self.yaw = math.atan2(fwd_enu[1], fwd_enu[0])
            self.msg_counts["ATTITUDE_QUATERNION"] += 1

    def update_from_local_position(self, msg):
        with self._lock:
            self.position_enu = np.array([msg.y, msg.x, -msg.z])
            self.velocity_enu = np.array([msg.vy, msg.vx, -msg.vz])
            self.msg_counts["LOCAL_POSITION_NED"] += 1

    def update_from_heartbeat(self, msg):
        with self._lock:
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            custom_main = (msg.custom_mode >> 16) & 0xFF
            self.offboard = (custom_main == PX4_CUSTOM_MAIN_MODE_OFFBOARD)

    def get(self):
        with self._lock:
            return (self.position_enu.copy(), self.velocity_enu.copy(),
                    self.R_enu.copy(), self.yaw)

    def take_msg_rates(self, dt):
        with self._lock:
            rates = {k: v / max(dt, 1e-6) for k, v in self.msg_counts.items()}
            for k in self.msg_counts:
                self.msg_counts[k] = 0
        return rates


def wait_for_heartbeat(mav, timeout=120):
    print("Waiting for heartbeat...")
    mav.wait_heartbeat(timeout=timeout)
    print(f"Heartbeat received from system {mav.target_system} "
          f"component {mav.target_component}")


def request_stream_rates(mav, stream_hz=STREAM_HZ):
    interval_us = int(1e6 / stream_hz)
    for msg_id, name in STREAMED_MSGS.items():
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            float(msg_id), float(interval_us), 0, 0, 0, 0, 0,
        )
        print(f"Requested {name} at {stream_hz:.0f} Hz")


def set_offboard_mode(mav):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_CUSTOM_MAIN_MODE_OFFBOARD, 0, 0, 0, 0, 0,
    )


def arm(mav):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0,
    )


def retry_offboard_arm(mav, state, last_try_t, interval=2.0):
    now = time.time()
    if (state.armed and state.offboard) or now - last_try_t < interval:
        return last_try_t
    if not state.offboard:
        set_offboard_mode(mav)
    if not state.armed:
        arm(mav)
    print(f"[CLIMB] re-requesting OFFBOARD/arm "
          f"(offboard={state.offboard} armed={state.armed}) ...")
    return now


def send_attitude_target(mav, q_wxyz: np.ndarray, thrust: float):
    mav.mav.set_attitude_target_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        7,  # type_mask: ignore roll/pitch/yaw rate
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,
        float(thrust),
    )


def send_position_target_ned(mav, x_n, y_e, z_d, yaw=0.0):
    IGNORE_VEL = 8 | 16 | 32
    IGNORE_ACC = 64 | 128 | 256
    IGNORE_YAW_RATE = 2048
    type_mask = IGNORE_VEL | IGNORE_ACC | IGNORE_YAW_RATE
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        float(x_n), float(y_e), float(z_d),
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        float(yaw), 0.0,
    )


def send_land_command(mav):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, float("nan"),
        0.0, 0.0, 0.0,
    )


def send_heartbeat(mav):
    mav.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0,
    )


def set_param_float(mav, param_id: str, value: float):
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        param_id.encode("utf-8"), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    print(f"Set param {param_id} = {value}")


def receive_loop(mav, state: DroneState, stop_event: threading.Event):
    while not stop_event.is_set():
        msg = mav.recv_match(blocking=True, timeout=0.1)
        if msg is None:
            continue
        msg_type = msg.get_type()
        if msg_type == "ATTITUDE_QUATERNION":
            state.update_from_attitude(msg)
        elif msg_type == "LOCAL_POSITION_NED":
            state.update_from_local_position(msg)
        elif msg_type == "HEARTBEAT" and msg.get_srcSystem() != 255:
            state.update_from_heartbeat(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="gs_drone_sim student.pt checkpoint file")
    parser.add_argument("--connect", default="udp:localhost:14550")
    parser.add_argument("--depth", action="store_true",
                        help="Ignored (accepted for harness compatibility; obs "
                             "subscriptions follow the checkpoint's config).")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Goal XY (ENU). Goal altitude is --climb-alt. "
                             "If omitted, the drone climbs and hovers.")
    parser.add_argument("--climb-alt", type=float, default=2.0)
    parser.add_argument("--arrive-tol", type=float, default=0.3)
    parser.add_argument("--settle-speed", type=float, default=0.2)
    parser.add_argument("--yaw-tol-deg", type=float, default=5.0)
    parser.add_argument("--max-vel", "--max-speed", type=float, default=2.0,
                        dest="max_vel",
                        help="Reference-velocity cap [m/s]. The student's "
                             "labels fly ~1.5-2.0 m/s; values far above that "
                             "are out of its training distribution.")
    parser.add_argument("--goal-radius", type=float, default=1.0,
                        help="HORIZONTAL goal distance [m] that hands off to "
                             "landing (XY on purpose, see agile_offboard).")
    parser.add_argument("--max-tilt-deg", type=float, default=45.0)
    parser.add_argument("--alt-mode", choices=["plan", "hold"], default="plan",
                        help="plan: track the student's vertical waypoints "
                             "(gs_drone_sim deploy semantics). hold: agile-style "
                             "altitude hold at the climb altitude.")
    parser.add_argument("--yaw-mode", choices=["vel", "goal"], default="vel",
                        help="vel: yaw toward tracked velocity (gs_drone_sim "
                             "track() semantics). goal: yaw toward the goal.")
    parser.add_argument("--lookahead", type=int, default=3,
                        help="Waypoint lookahead index (controller.py default 3)")
    parser.add_argument("--depth-guard", action="store_true",
                        default=bool(os.environ.get("GSDS_DEPTH_GUARD")),
                        help="Veto hypotheses whose prop-disc path enters the "
                             "live depth frame's geometry; argmin-cost among "
                             "survivors (deploy-side fix for the geometry-blind "
                             "cost head; works for the RGB student too).")
    parser.add_argument("--depth-repulsion", type=float, metavar="GAIN",
                        default=float(os.environ.get("GSDS_DEPTH_REPULSION",
                                                     "0") or "0"),
                        help="Keepout-lite margin injection: signed lateral "
                             "bias on the selected plan's waypoints, pushing "
                             "away from geometry nearer than "
                             "$GSDS_REPULSION_THRESH (4 m) in the live depth "
                             "frame (plan/input space; composable with the "
                             "guard). GAIN = metres of bias at full ramp, "
                             "hard-capped at $GSDS_REPULSION_CAP (1 m). "
                             "0 disables. Also settable via "
                             "$GSDS_DEPTH_REPULSION.")
    parser.add_argument("--goal-clip", type=float, default=30.0,
                        help="Clip the body-frame goal DISTANCE fed to the net "
                             "[m] (training legs were 20-38 m; direction is "
                             "preserved). 0 disables.")
    parser.add_argument("--kp", type=float, default=6.0)
    parser.add_argument("--kv", type=float, default=4.0)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--stream-hz", type=float, default=STREAM_HZ)
    parser.add_argument("--device", default=None,
                        help="torch device override (default: cuda if available)")
    parser.add_argument("--dump-obs", default=os.environ.get("GSDS_DUMP_OBS"),
                        metavar="DIR",
                        help="Save every net tick's (obs, vec, hypotheses, "
                             "costs) as npz under DIR for offline analysis. "
                             "Also settable via $GSDS_DUMP_OBS (the comparison "
                             "harness passes no extra offboard flags).")
    parser.add_argument("--no-debug-viz", action="store_true")
    args = parser.parse_args()

    goal_xy = np.array(args.goal) if args.goal is not None else None

    control_hz = float(args.control_hz)
    stream_hz = max(float(args.stream_hz), control_hz)

    hover_thrust = float(np.clip(G / MAX_ACCEL, 0.0, 1.0))
    net_every = max(1, round(control_hz / NET_HZ))
    policy = GsdsPolicy(
        checkpoint_path=args.checkpoint,
        max_vel=args.max_vel,
        hover_thrust=hover_thrust,
        control_hz=control_hz,
        net_every=net_every,
        lookahead=args.lookahead,
        kp=args.kp, kv=args.kv,
        max_tilt_deg=args.max_tilt_deg,
        alt_mode=args.alt_mode,
        yaw_mode=args.yaw_mode,
        goal_clip=args.goal_clip,
        dump_obs_dir=args.dump_obs,
        depth_guard=args.depth_guard,
        depth_repulsion=args.depth_repulsion,
        device=args.device,
    )

    # depth-guard / depth-repulsion need the depth stream even for the RGB
    # student (the sim publishes both for gsds mode; run_px4_sim
    # _publish_depth_gsds)
    depth_sub = DepthSubscriber() if (policy.use_depth or args.depth_guard
                                      or args.depth_repulsion > 0.0) else None
    rgb_sub = RgbSubscriber() if policy.use_rgb else None
    print(f"Obs subscriptions: depth={'on' if depth_sub else 'off'} "
          f"rgb={'on' if rgb_sub else 'off'}")

    print(f"Connecting to {args.connect} ...")
    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)
    request_stream_rates(mav, stream_hz)

    state = DroneState()
    stop_event = threading.Event()
    recv_thread = threading.Thread(target=receive_loop,
                                   args=(mav, state, stop_event), daemon=True)
    recv_thread.start()

    debug_pub = None if args.no_debug_viz else AgileDebugPublisher()

    print(f"Control loop @ {control_hz:.0f} Hz, state stream @ {stream_hz:.0f} Hz, "
          f"net every {net_every} ticks (~{control_hz / net_every:.0f} Hz).")
    print(f"Setting PX4 MPC_THR_HOVER = {hover_thrust:.3f} ...")
    set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
    time.sleep(0.2)

    pos0, _, _, yaw0 = state.get()
    hold_x_n = pos0[1]
    hold_y_e = pos0[0]
    hold_z_d = -args.climb_alt

    yaw_ground = math.atan2(math.sin(math.pi / 2 - yaw0),
                            math.cos(math.pi / 2 - yaw0))
    if goal_xy is not None:
        d_north = goal_xy[1] - hold_x_n
        d_east = goal_xy[0] - hold_y_e
        yaw_goal = math.atan2(d_east, d_north)
    else:
        yaw_goal = yaw_ground

    print("Pre-arming: streaming position setpoints to satisfy PX4 OFFBOARD "
          "pre-condition...")
    for _ in range(30):
        send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_ground)
        send_heartbeat(mav)
        time.sleep(0.05)

    print("Setting OFFBOARD mode...")
    set_offboard_mode(mav)
    time.sleep(0.5)

    print("Arming...")
    arm(mav)
    time.sleep(1.0)

    policy.reset()

    control_dt = 1.0 / control_hz
    heartbeat_dt = 1.0 / HEARTBEAT_HZ
    last_heartbeat = time.time()
    start_time = time.time()
    next_step = time.time()
    last_arm_try = time.time()
    last_rate_t = time.time()
    step_count = 0

    phase = "CLIMB"
    landing_sent = False
    print(f"CLIMB: position-holding to {args.climb_alt:.1f} m ...")

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            if now - last_heartbeat >= heartbeat_dt:
                send_heartbeat(mav)
                last_heartbeat = now

            if now >= next_step:
                pos, vel, R_enu, yaw = state.get()
                verbose = elapsed < 5.0 or (int(now) != int(now - control_dt))

                depth = depth_sub.latest() if depth_sub else None
                rgb = rgb_sub.latest() if rgb_sub else None

                if goal_xy is not None:
                    goal_enu = np.array([goal_xy[0], goal_xy[1], args.climb_alt])
                else:
                    goal_enu = pos  # hover

                if phase == "CLIMB":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d,
                                             yaw_ground)
                    last_arm_try = retry_offboard_arm(mav, state, last_arm_try)
                    alt = pos[2]
                    speed = np.linalg.norm(vel)
                    if (abs(alt - args.climb_alt) < args.arrive_tol
                            and speed < args.settle_speed and state.offboard):
                        phase = "YAW"
                        print(f"\n>>> Climbed to alt={alt:.2f} m, "
                              f"speed={speed:.2f} m/s -- turning to face goal <<<\n")
                    if verbose:
                        print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/"
                              f"{args.climb_alt:.1f} speed={speed:.2f}  "
                              f"offboard={state.offboard} armed={state.armed}")
                elif phase == "YAW":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d,
                                             yaw_goal)
                    yaw_cur_ned = math.atan2(math.sin(math.pi / 2 - yaw),
                                             math.cos(math.pi / 2 - yaw))
                    yaw_err = math.atan2(math.sin(yaw_goal - yaw_cur_ned),
                                         math.cos(yaw_goal - yaw_cur_ned))
                    if abs(math.degrees(yaw_err)) < args.yaw_tol_deg and state.offboard:
                        phase = "POLICY"
                        _mark_policy_phase("start")
                        state.take_msg_rates(now - last_rate_t)
                        last_rate_t = now
                        print(f"\n>>> HANDOFF to policy facing goal, "
                              f"yaw={math.degrees(yaw_cur_ned):.1f} deg <<<\n")
                    if verbose:
                        print(f"[YAW t={elapsed:.2f}s] "
                              f"yaw={math.degrees(yaw_cur_ned):.1f} "
                              f"target={math.degrees(yaw_goal):.1f} "
                              f"err={math.degrees(yaw_err):.1f}")
                elif phase == "POLICY":
                    obs = GsdsObs(
                        position_enu=pos,
                        velocity_enu=vel,
                        R_enu=R_enu,
                        goal_enu=goal_enu,
                        depth=depth,
                        rgb=rgb,
                    )
                    cmd = policy.compute(obs)
                    send_attitude_target(mav, cmd.attitude_ned_frd_wxyz,
                                         cmd.thrust_norm)
                    if debug_pub is not None:
                        dbg = policy.debug_frame(pos, R_enu, cmd.tracker)
                        if dbg is not None:
                            from agile_debug_transport import AgileDebugFrame
                            debug_pub.send(AgileDebugFrame(
                                seq=step_count,
                                pos_local=dbg["pos_local"],
                                yaw=dbg["yaw"],
                                alphas=dbg["alphas"],
                                trajectories_local=dbg["trajectories_local"],
                                mode_idx=dbg["mode_idx"],
                                tracker=dbg["tracker"],
                            ))
                    if np.linalg.norm((goal_enu - pos)[:2]) < args.goal_radius:
                        phase = "LANDING"
                        _mark_policy_phase("end")
                        print(f"\n>>> HANDOFF to landing at pos={pos.round(2)} <<<\n")
                    if verbose:
                        rates = state.take_msg_rates(now - last_rate_t)
                        last_rate_t = now
                        meas_tilt = math.degrees(math.acos(
                            float(np.clip(R_enu[2, 2], -1, 1))))
                        rate_str = " ".join(f"{k.split('_')[0]}={v:.0f}Hz"
                                            for k, v in rates.items())
                        frame_str = (f"depth={'ok' if depth is not None else 'NONE'} "
                                     f"rgb={'ok' if rgb is not None else 'NONE'}")
                        print(
                            f"[POLICY t={elapsed:.2f}s step={step_count} "
                            f"trk={cmd.tracker}]\n"
                            f"  pos(ENU)={pos.round(2)} vel(ENU)={vel.round(2)} "
                            f"goal={np.round(goal_enu, 2)}\n"
                            f"  tilt cmd/meas={cmd.tilt_cmd_deg:.1f}/"
                            f"{meas_tilt:.1f} deg thrust={cmd.thrust_norm:.3f} "
                            f"mode={cmd.mode_idx} costs={np.round(cmd.costs, 3)} "
                            f"spread={cmd.spread_m:.2f}m "
                            # repl=... only when the knob is on, so flag-off
                            # logs stay byte-identical to the pre-knob stack
                            + (f"repl={cmd.repl_m:+.2f}m "
                               if args.depth_repulsion > 0.0 else "") +
                            f"guard={getattr(policy, '_guard_overrides', 0)}\n"
                            f"  obs: {frame_str}  msg rates: {rate_str}\n---"
                        )
                elif phase == "LANDING":
                    if not landing_sent:
                        send_land_command(mav)
                        landing_sent = True
                    if not state.armed:
                        print("\n>>> Landed and disarmed. Exiting.")
                        break
                    if verbose:
                        print(f"[LANDING t={elapsed:.2f}s] alt={pos[2]:.2f} m  "
                              f"armed={state.armed}")

                step_count += 1
                next_step += control_dt
                if next_step < time.time():
                    next_step = time.time()
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_event.set()
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        try:
            Path(OFFBOARD_DONE_FILE).write_text(str(time.time()))
        except Exception:
            pass


if __name__ == "__main__":
    main()
