#!/usr/bin/env python
"""
Agile Autonomy (Loquercio et al., uzh-rpg/agile_autonomy) offboard controller
for PX4. Run under agile_python.sh (the agile venv at
starling-deployment/.venv: TF + acados).

Connects to PX4 via MAVLink, arms, climbs to --climb-alt, yaws to face the
goal, then runs the PlaNet trajectory network + acados tracking MPC
(wrapper/agile_core.py) at 30 Hz sending SET_ATTITUDE_TARGET.

Mirrors diffaero_offboard.py's structure and CLI (--goal takes X Y; the goal
altitude is --climb-alt) with one agile-specific addition that is NOT
optional: right after the heartbeat it requests LOCAL_POSITION_NED and
ATTITUDE_QUATERNION at 50 Hz via MAV_CMD_SET_MESSAGE_INTERVAL. udp:14550 is
PX4's GCS link, which by default streams position at 1 Hz and attitude at
10 Hz -- an MPC re-solving at 30 Hz on second-stale state produces a 0.5 Hz
+-30 deg pitch/roll limit cycle that also tilts the camera off the obstacles.
Learned reactive policies tolerate that staleness; a stiff model-based
tracker does not.

Usage (after running run_px4_sim.py --policy agile):
    ./agile_python.sh agile_offboard.py \
        --checkpoint ../checkpoints/AgileAutonomy/ckpt-50 --depth \
        --goal 40 30 --climb-alt 2.0 --max-vel 7.0
"""

import argparse
import math
import os
import time
import threading
from pathlib import Path

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from wrapper.agile_core import AgilePolicy, AgileObs
from agile_debug_transport import AgileDebugPublisher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 30.0
# Target rate for the (expensive) net forward pass. The MPC/attitude loop can run
# faster than this; net_every = round(control_hz / NET_HZ) holds the net near here.
NET_HZ = 15.0
HEARTBEAT_HZ = 2.0
G = 9.80665

# Thrust-acceleration that maps to full throttle; hover throttle = g/MAX_ACCEL.
# Matches the other offboards (diffaero/diffphys) so PX4 sees the same vehicle.
MAX_ACCEL = 20.0

# PX4 message stream rates requested at startup (see module docstring).
STREAM_HZ = 50.0
STREAMED_MSGS = {
    mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: "LOCAL_POSITION_NED",
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION: "ATTITUDE_QUATERNION",
}

# Sentinel files shared with run_px4_sim.py --auto-stop / compare/run_comparison.py
# (must match OFFBOARD_DONE_FILE / POLICY_PHASE_FILE there).
OFFBOARD_DONE_FILE = "/tmp/superfly_offboard_done"
POLICY_PHASE_FILE = "/tmp/superfly_policy_phase"

PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def _mark_policy_phase(event: str):
    """Append '<event> <ts>' to POLICY_PHASE_FILE; never let it kill the loop."""
    try:
        with open(POLICY_PHASE_FILE, "a") as f:
            f.write(f"{event} {time.time()}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# State container (updated by MAVLink receive thread)
# ---------------------------------------------------------------------------

class DroneState:
    def __init__(self):
        self._lock = threading.Lock()
        self.position_enu = np.zeros(3)
        self.velocity_enu = np.zeros(3)
        self.R_enu = np.eye(3)
        self.angular_rate_body = np.zeros(3)
        self.yaw = 0.0
        self.armed = False
        self.offboard = False
        # message-rate counters (verbose diagnostics: catching a silently
        # ignored SET_MESSAGE_INTERVAL is the first step of any oscillation
        # debugging, so the rates are always measured)
        self.msg_counts = {name: 0 for name in STREAMED_MSGS.values()}

    def update_from_attitude(self, msg):
        q_ned_frd = Rotation.from_quat([msg.q2, msg.q3, msg.q4, msg.q1])
        rot_enu_flu = _rot_ENU_to_NED.inv() * q_ned_frd * _rot_FLU_to_FRD.inv()
        with self._lock:
            self.R_enu = rot_enu_flu.as_matrix()
            fwd_enu = self.R_enu[:, 0]
            self.yaw = math.atan2(fwd_enu[1], fwd_enu[0])
            w_frd = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed], dtype=np.float64)
            self.angular_rate_body = np.array([w_frd[0], -w_frd[1], -w_frd[2]])
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
                    self.R_enu.copy(), self.angular_rate_body.copy(), self.yaw)

    def take_msg_rates(self, dt):
        """Return {msg: Hz} since the last call and reset the counters."""
        with self._lock:
            rates = {k: v / max(dt, 1e-6) for k, v in self.msg_counts.items()}
            for k in self.msg_counts:
                self.msg_counts[k] = 0
        return rates


# ---------------------------------------------------------------------------
# MAVLink helpers (same wire protocol as diffaero_offboard.py)
# ---------------------------------------------------------------------------

def wait_for_heartbeat(mav, timeout=120):
    print("Waiting for heartbeat...")
    mav.wait_heartbeat(timeout=timeout)
    print(f"Heartbeat received from system {mav.target_system} component {mav.target_component}")


def request_stream_rates(mav, stream_hz=STREAM_HZ):
    """Ask PX4 to stream the state messages at stream_hz on THIS link.
    Without this the GCS link's defaults apply (position 1 Hz, attitude
    10 Hz) and the MPC runs on second-stale state -> attitude limit cycle.
    Should be >= the control rate so each MPC solve sees fresh state."""
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
    """Re-request OFFBOARD mode + arming until PX4 accepts both (the one-shot
    commands are silently rejected while the EKF is still converging; see
    diffaero_offboard.py)."""
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


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="TF2 checkpoint PREFIX (e.g. .../ckpt-50, with "
                             "ckpt-50.index alongside) or a directory containing one")
    parser.add_argument("--connect", default="udp:localhost:14550",
                        help="MAVLink connection string")
    parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Goal XY (ENU). Goal altitude is --climb-alt (horizontal "
                             "cruise). If omitted, the drone climbs and hovers.")
    parser.add_argument("--depth", action="store_true",
                        help="Subscribe to live depth frames over UDP (from the sim)")
    parser.add_argument("--climb-alt", type=float, default=2.0,
                        help="Climb to this altitude [m] before the policy takes over; "
                             "also the cruise/goal altitude.")
    parser.add_argument("--arrive-tol", type=float, default=0.3,
                        help="Altitude tolerance [m] for the climb target")
    parser.add_argument("--settle-speed", type=float, default=0.2,
                        help="Speed [m/s] below which the drone counts as settled")
    parser.add_argument("--yaw-tol-deg", type=float, default=5.0,
                        help="Yaw tolerance [deg] for the goal-facing turn")
    parser.add_argument("--max-vel", "--max-speed", type=float, default=7.0,
                        dest="max_vel",
                        help="Cruise speed cap [m/s] for the MPC reference and net "
                             "plan scaling (upstream agile_autonomy "
                             "test_time_velocity=7). --max-speed is an alias.")
    parser.add_argument("--goal-radius", type=float, default=1.0,
                        help="HORIZONTAL distance [m] to the goal that hands off to "
                             "landing. XY on purpose: altitude is held near the cruise "
                             "alt with some steady-state sag, and a 3D check against "
                             "that sag can never fire -- the drone then orbits the goal "
                             "until the harness timeout (observed live).")
    parser.add_argument("--max-tilt-deg", type=float, default=90.0,
                        help="Attitude-setpoint tilt clamp [deg]. Default 90 disables "
                             "clamping (upstream agile_autonomy has no tilt clamp).")
    parser.add_argument("--att-lp", type=float, default=1.0,
                        help="Attitude low-pass alpha (1.0 = disabled, upstream default).")
    parser.add_argument("--ref-lookahead-s", type=float, default=5.0,
                        help="Goal-direction lookahead [s] on the mission reference "
                             "(upstream test_settings.yaml future_time).")
    parser.add_argument("--keepout", action="store_true",
                        help="Enable depth-derived obstacle-memory keep-out constraints "
                             "in the MPC (not used upstream).")
    parser.add_argument("--att-lookahead-s", type=float, default=None,
                        help="How far ahead [s] to sample the MPC attitude setpoint "
                             "streamed to PX4. Default = control period (1/control_hz); "
                             "the legacy behaviour is 0.1 (the MPC stage-1 node), which "
                             "over-anticipates and drives the attitude limit cycle.")
    parser.add_argument("--q-att", type=float, default=None,
                        help="MPC attitude tracking weight (Q_attitude). Upstream "
                             "mpc_params.yaml uses 200; the port defaults to 50. Sets "
                             "AGILE_MPC_Q_ATT before the acados solver is built.")
    parser.add_argument("--r", type=float, default=None,
                        help="MPC input weight R = diag([thrust, wx, wy, wz]). Port "
                             "defaults to 0.1; upstream mpc_params.yaml uses 1.0. Higher "
                             "R relative to Q makes control smoother/less aggressive. Sets "
                             "AGILE_MPC_R before the acados solver is built.")
    parser.add_argument("--t-min", type=float, default=None,
                        help="MPC min collective thrust [m/s^2] (mass-normalized). "
                             "Upstream mpc_params.yaml = 5.0 (~0.5 g), now the port "
                             "default; the legacy port used 1.0. Sets AGILE_MPC_T_MIN.")
    parser.add_argument("--t-max", type=float, default=None,
                        help="MPC max collective thrust [m/s^2] (mass-normalized). "
                             "Upstream mpc_params.yaml = 20.0 (~2 g), now the port "
                             "default; the legacy port used 40.0. Wider bands let the MPC "
                             "plan more extreme attitudes. Sets AGILE_MPC_T_MAX.")
    parser.add_argument("--max-bodyrate-xy", type=float, default=None,
                        help="MPC max roll/pitch rate [rad/s]. Upstream = 6.0 (default). "
                             "Sets AGILE_MPC_MAX_BODYRATE_XY.")
    parser.add_argument("--max-bodyrate-z", type=float, default=None,
                        help="MPC max yaw rate [rad/s]. Upstream = 2.0 (default). "
                             "Sets AGILE_MPC_MAX_BODYRATE_Z.")
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ,
                        help=f"Rate [Hz] of the MPC solve + SET_ATTITUDE_TARGET loop "
                             f"(default {CONTROL_HZ:.0f}). Higher rates make the attitude "
                             f"setpoint more achievable and reduce the limit cycle; the net "
                             f"still runs near {NET_HZ:.0f} Hz (net_every scales with it).")
    parser.add_argument("--stream-hz", type=float, default=STREAM_HZ,
                        help=f"Rate [Hz] to request PX4 stream state (default {STREAM_HZ:.0f}). "
                             f"Automatically raised to at least --control-hz so no MPC solve "
                             f"runs on stale state.")
    parser.add_argument("--no-debug-viz", action="store_true",
                        help="Disable UDP debug frames for sim overhead trajectory viz.")
    args = parser.parse_args()

    # Must be set BEFORE AgilePolicy builds the acados MPC (make_solver reads it).
    if args.q_att is not None:
        os.environ["AGILE_MPC_Q_ATT"] = str(args.q_att)
    if args.r is not None:
        os.environ["AGILE_MPC_R"] = str(args.r)
    if args.t_min is not None:
        os.environ["AGILE_MPC_T_MIN"] = str(args.t_min)
    if args.t_max is not None:
        os.environ["AGILE_MPC_T_MAX"] = str(args.t_max)
    if args.max_bodyrate_xy is not None:
        os.environ["AGILE_MPC_MAX_BODYRATE_XY"] = str(args.max_bodyrate_xy)
    if args.max_bodyrate_z is not None:
        os.environ["AGILE_MPC_MAX_BODYRATE_Z"] = str(args.max_bodyrate_z)

    goal_xy = np.array(args.goal) if args.goal is not None else None

    depth_sub = None
    if args.depth:
        from depth_transport import DepthSubscriber
        depth_sub = DepthSubscriber()
        print("Depth subscriber listening for frames over UDP.")

    print(f"Connecting to {args.connect} ...")
    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)

    # Control rate: the loop resolves the MPC and streams SET_ATTITUDE_TARGET at
    # this rate. Upstream agile_autonomy closes the low level loop far faster than
    # the 30 Hz stage-1 node, so a higher rate here (with att_lookahead_s = control
    # period) keeps the attitude setpoint achievable and kills the limit cycle. The
    # net still runs at ~NET_HZ (net_every scales with the control rate).
    control_hz = float(args.control_hz)
    # Stream state at least as fast as we control so no MPC solve sees stale state.
    stream_hz = max(float(args.stream_hz), control_hz)
    request_stream_rates(mav, stream_hz)

    state = DroneState()
    stop_event = threading.Event()
    recv_thread = threading.Thread(target=receive_loop, args=(mav, state, stop_event),
                                   daemon=True)
    recv_thread.start()

    hover_thrust = float(np.clip(G / MAX_ACCEL, 0.0, 1.0))
    # Keep the expensive net forward pass near NET_HZ regardless of control rate.
    net_every = max(1, round(control_hz / NET_HZ))
    print(f"Control loop @ {control_hz:.0f} Hz, state stream @ {stream_hz:.0f} Hz, "
          f"net every {net_every} ticks (~{control_hz / net_every:.0f} Hz).")
    policy = AgilePolicy(
        checkpoint_path=args.checkpoint,
        max_vel=args.max_vel,
        hover_thrust=hover_thrust,
        control_hz=control_hz,
        net_every=net_every,
        max_tilt_deg=args.max_tilt_deg,
        att_lp=args.att_lp,
        ref_lookahead_s=args.ref_lookahead_s,
        use_keepout=args.keepout,
        att_lookahead_s=args.att_lookahead_s,
    )
    debug_pub = None if args.no_debug_viz else AgileDebugPublisher()
    if debug_pub is not None:
        print("Agile debug viz publisher active (sim writes agile_overhead_debug.png).")

    print(f"Setting PX4 MPC_THR_HOVER = {hover_thrust:.3f} ...")
    set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
    time.sleep(0.2)

    # Pre-arm: stream POSITION setpoints (hold + climb) so PX4 accepts OFFBOARD.
    pos0, _, _, _, yaw0 = state.get()
    hold_x_n = pos0[1]   # North = ENU.y
    hold_y_e = pos0[0]   # East  = ENU.x
    hold_z_d = -args.climb_alt

    # Hold the current heading on the ground/climb; turn to the goal at altitude.
    yaw_ground = math.atan2(math.sin(math.pi / 2 - yaw0), math.cos(math.pi / 2 - yaw0))
    if goal_xy is not None:
        d_north = goal_xy[1] - hold_x_n
        d_east = goal_xy[0] - hold_y_e
        yaw_goal = math.atan2(d_east, d_north)
    else:
        yaw_goal = yaw_ground

    print("Pre-arming: streaming position setpoints to satisfy PX4 OFFBOARD pre-condition...")
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
                pos, vel, R_enu, omega_body, yaw = state.get()
                verbose = elapsed < 5.0 or (int(now) != int(now - control_dt))

                depth = depth_sub.latest() if depth_sub else None

                if goal_xy is not None:
                    goal_enu = np.array([goal_xy[0], goal_xy[1], args.climb_alt])
                else:
                    goal_enu = pos  # hover

                if phase == "CLIMB":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_ground)
                    last_arm_try = retry_offboard_arm(mav, state, last_arm_try)
                    alt = pos[2]
                    speed = np.linalg.norm(vel)
                    if (abs(alt - args.climb_alt) < args.arrive_tol
                            and speed < args.settle_speed and state.offboard):
                        phase = "YAW"
                        print(f"\n>>> Climbed to alt={alt:.2f} m, speed={speed:.2f} m/s -- "
                              f"turning to face goal <<<\n")
                    if verbose:
                        print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/{args.climb_alt:.1f} "
                              f"speed={speed:.2f}  offboard={state.offboard} armed={state.armed}")
                elif phase == "YAW":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_goal)
                    yaw_cur_ned = math.atan2(math.sin(math.pi / 2 - yaw), math.cos(math.pi / 2 - yaw))
                    yaw_err = math.atan2(math.sin(yaw_goal - yaw_cur_ned), math.cos(yaw_goal - yaw_cur_ned))
                    if abs(math.degrees(yaw_err)) < args.yaw_tol_deg and state.offboard:
                        phase = "POLICY"
                        _mark_policy_phase("start")
                        state.take_msg_rates(now - last_rate_t)  # reset counters
                        last_rate_t = now
                        print(f"\n>>> HANDOFF to policy facing goal, "
                              f"yaw={math.degrees(yaw_cur_ned):.1f} deg <<<\n")
                    if verbose:
                        print(f"[YAW t={elapsed:.2f}s] yaw={math.degrees(yaw_cur_ned):.1f} "
                              f"target={math.degrees(yaw_goal):.1f} err={math.degrees(yaw_err):.1f}")
                elif phase == "POLICY":
                    obs = AgileObs(
                        position_enu=pos,
                        velocity_enu=vel,
                        R_enu=R_enu,
                        angular_rate_body=omega_body,
                        goal_enu=goal_enu,
                        depth=depth,
                    )
                    cmd = policy.compute(obs)
                    send_attitude_target(mav, cmd.attitude_ned_frd_wxyz, cmd.thrust_norm)
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
                        cur_rpy = Rotation.from_matrix(R_enu).as_euler("xyz", degrees=True)
                        meas_tilt = math.degrees(math.acos(float(np.clip(R_enu[2, 2], -1, 1))))
                        rate_str = " ".join(f"{k.split('_')[0]}={v:.0f}Hz"
                                            for k, v in rates.items())
                        print(
                            f"[POLICY t={elapsed:.2f}s step={step_count} trk={cmd.tracker}]\n"
                            f"  pos(ENU)={pos.round(2)} vel(ENU)={vel.round(2)} "
                            f"goal={np.round(goal_enu, 2)}\n"
                            f"  tilt cmd/meas={cmd.tilt_cmd_deg:.1f}/{meas_tilt:.1f} deg "
                            f"thrust={cmd.thrust_norm:.3f} mode={cmd.mode_idx} "
                            f"alphas={np.round(cmd.alphas, 3)} keepout={cmd.n_keepout}\n"
                            f"  msg rates: {rate_str} "
                            f"(want {stream_hz:.0f}; ~1 Hz position = stale-state limit cycle)\n"
                            f"  cur RPY(ENU)={np.round(cur_rpy, 1)} armed={state.armed} "
                            f"offboard={state.offboard}\n---"
                        )
                elif phase == "LANDING":
                    if not landing_sent:
                        send_land_command(mav)
                        landing_sent = True
                    if not state.armed:
                        print("\n>>> Landed and disarmed. Exiting.")
                        break
                    if verbose:
                        print(f"[LANDING t={elapsed:.2f}s] alt={pos[2]:.2f} m  armed={state.armed}")

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
