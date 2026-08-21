#!/usr/bin/env python
"""
DiffAero velocity-command offboard controller for PX4.

Connects to PX4 via MAVLink, arms the drone, enters OFFBOARD mode, then runs a
velocity-command DiffAero policy at 30 Hz (training dt=0.0333) sending
SET_POSITION_TARGET_LOCAL_NED with velocity + yaw.

Unlike diffaero_offboard.py (accel → SET_ATTITUDE_TARGET), this script feeds
the policy's world-frame velocity setpoint straight to PX4's velocity loop.

  * Observation (obs_frame=local, velocity point-mass): state =
    [target_vel_local(3), v_local(3)]. The vel_nodepth checkpoint (ne sha2c_vel_cmd) was trained
    with env=pc (no depth); env=oa checkpoints also consume 9x16 perception.
  * Action (action_frame=local): world-frame velocity setpoint
    vel_cmd = Rz @ scaled_action, sent to PX4 as NED (vx, vy, vz).
  * Yaw: rate-limited slew toward velocity EMA (matches training); held below
    yaw_hold_speed. Pre-policy YAW phase still faces the goal once.
  * Altitude: planar policies output horizontal velocity only; a light altitude
    PID supplies vz while holding --climb-alt.

Usage:
    # Against PX4 SITL (after running run_px4_sim.py --policy diffaero):
    python run_px4_sim.py ... --no-debug-frames   # skip camera_debug.png writes in sim
    python scripts/diffaero_vel_offboard.py \\
        --checkpoint checkpoints/DiffAero/vel_depth \\
        --goal 15 0 --climb-alt 2.0 --quiet

    # Against real VOXL2 over UDP:
    python diffaero_vel_offboard.py --checkpoint <dir> --connect udp:192.168.1.x:14550
"""

import argparse
import math
import sys
import time
import threading
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pymavlink import mavutil

from superfly.common.frames import enu_vel_to_ned
from superfly.common.px4_offboard import (
    HEARTBEAT_HZ,
    DroneState, wait_for_heartbeat, wait_for_position, set_offboard_mode, arm,
    send_position_target_ned, send_velocity_target_ned,
    send_land_command, send_heartbeat, set_param_float, receive_loop,
)
from superfly.common.sentinels import mark_policy_phase, mark_offboard_done
from superfly.policies.diffaero import DiffAeroObs, DA_INTRINSICS

CONTROL_HZ = 30.0


def stream_setpoints_loop(mav, send_fn, send_args, stop_event: threading.Event):
    """Background setpoint stream; keeps PX4 OFFBOARD preconditions satisfied."""
    last_hb = time.time()
    while not stop_event.is_set():
        send_fn(mav, *send_args)
        now = time.time()
        if now - last_hb >= 0.5:
            send_heartbeat(mav)
            last_hb = now
        time.sleep(1.0 / CONTROL_HZ)


def stream_setpoints(mav, send_fn, send_args, duration: float):
    """Keep OFFBOARD alive by streaming setpoints (no gaps after arming)."""
    end = time.time() + duration
    last_hb = time.time()
    while time.time() < end:
        send_fn(mav, *send_args)
        now = time.time()
        if now - last_hb >= 0.5:
            send_heartbeat(mav)
            last_hb = now
        time.sleep(1.0 / CONTROL_HZ)


def altitude_vz_ned(
    cruise_alt: float,
    pos_enu: np.ndarray,
    vel_enu: np.ndarray,
    kp: float = 2.0,
    kd: float = 1.0,
    max_vz: float = 1.5,
) -> float:
    """NED vz command to hold cruise_alt (planar policies do not command z)."""
    alt_err = cruise_alt - float(pos_enu[2])
    vz_meas_ned = -float(vel_enu[2])
    vz_cmd = -kp * alt_err - kd * vz_meas_ned
    return float(np.clip(vz_cmd, -max_vz, max_vz))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to velocity-command checkpoint dir or .pt2 file")
    parser.add_argument("--connect", default="udp:localhost:14550",
                        help="MAVLink connection string")
    parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Goal XY (ENU). Z = --climb-alt. Omit to hover.")
    parser.add_argument("--depth", action="store_true",
                        help="Subscribe to depth UDP (only used for env=oa checkpoints)")
    parser.add_argument("--climb-alt", type=float, default=10.0,
                        help="Climb altitude [m] before policy handoff")
    parser.add_argument("--climb-rate", type=float, default=1.0,
                        help="Vertical climb rate during CLIMB phase [m/s]")
    parser.add_argument("--arrive-tol", type=float, default=0.3,
                        help="Altitude tolerance [m] for climb handoff")
    parser.add_argument("--settle-speed", type=float, default=0.2,
                        help="Speed [m/s] below which climb is considered settled")
    parser.add_argument("--yaw-tol-deg", type=float, default=5.0,
                        help="Yaw tolerance [deg] for goal-facing turn")
    parser.add_argument("--max-vel", type=float, default=None,
                        help="Target cruise speed [m/s] for goal heuristic "
                             "(default: from checkpoint config)")
    parser.add_argument("--max-vel-xy", type=float, default=None,
                        help="PX4 MPC_XY_VEL_MAX and action XY limit override [m/s]")
    parser.add_argument("--max-vel-z", type=float, default=None,
                        help="PX4 MPC_Z_VEL_MAX_UP/DN and action Z limit override [m/s]")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-phase status prints (phase handoffs still print)")
    args = parser.parse_args()

    goal_xy = np.array(args.goal) if args.goal is not None else None

    depth_sub = None
    if args.depth:
        from superfly.common.transport import DepthSubscriber
        depth_sub = DepthSubscriber()
        print("Depth subscriber listening for frames over UDP.")

    print(f"Connecting to {args.connect} ...")
    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)

    state = DroneState()
    stop_event = threading.Event()
    recv_thread = threading.Thread(
        target=receive_loop, args=(mav, state, stop_event), daemon=True
    )
    recv_thread.start()
    wait_for_position(state, raise_on_timeout=True)

    policy_kwargs = {"intrinsics": DA_INTRINSICS, "checkpoint_path": args.checkpoint}
    if args.max_vel is not None:
        policy_kwargs["max_vel"] = args.max_vel
    if args.max_vel_xy is not None:
        policy_kwargs["max_vel_xy"] = args.max_vel_xy
    if args.max_vel_z is not None:
        policy_kwargs["max_vel_z"] = args.max_vel_z

    pos0, _, _, yaw0 = state.get()
    hold_x_n = pos0[1]
    hold_y_e = pos0[0]
    hold_z_d = -args.climb_alt

    yaw_ground = math.atan2(math.sin(math.pi / 2 - yaw0), math.cos(math.pi / 2 - yaw0))
    if goal_xy is not None:
        d_north = goal_xy[1] - hold_x_n
        d_east = goal_xy[0] - hold_y_e
        yaw_goal = math.atan2(d_east, d_north)
    else:
        yaw_goal = yaw_ground
    yaw_ned = yaw_ground

    stream_stop = threading.Event()
    stream_thread = threading.Thread(
        target=stream_setpoints_loop,
        args=(mav, send_position_target_ned, (hold_x_n, hold_y_e, hold_z_d, yaw_ned), stream_stop),
        daemon=True,
    )
    stream_thread.start()

    print("Loading policy (setpoints streaming in background) ...")
    from superfly.policies.diffaero_vel import DiffAeroVelPolicy
    policy = DiffAeroVelPolicy(**policy_kwargs)
    max_vel_xy = policy.max_vel_xy
    max_vel_z = policy.max_vel_z

    print("Setting OFFBOARD mode...")
    set_offboard_mode(mav)
    time.sleep(0.5)

    print("Arming...")
    arm(mav)
    time.sleep(0.5)

    print(f"Setting PX4 MPC_XY_VEL_MAX = {max_vel_xy:.1f} ...")
    set_param_float(mav, "MPC_XY_VEL_MAX", max_vel_xy)
    print(f"Setting PX4 MPC_Z_VEL_MAX_UP = {max_vel_z:.1f} ...")
    set_param_float(mav, "MPC_Z_VEL_MAX_UP", max_vel_z)
    print(f"Setting PX4 MPC_Z_VEL_MAX_DN = {max_vel_z:.1f} ...")
    set_param_float(mav, "MPC_Z_VEL_MAX_DN", max_vel_z)

    stream_stop.set()
    stream_thread.join(timeout=1.0)

    policy.reset()

    control_dt = 1.0 / CONTROL_HZ
    heartbeat_dt = 1.0 / HEARTBEAT_HZ
    last_heartbeat = time.time()
    start_time = time.time()
    next_step = time.time()
    step_count = 0

    phase = "CLIMB"
    landing_sent = False
    yaw_ned_cmd = yaw_ned
    last_arm_attempt = time.time()
    grounded_since = None
    recoveries = 0
    print(f"CLIMB: velocity climb to {args.climb_alt:.1f} m at {args.climb_rate:.1f} m/s ...")
    if policy.planar:
        print(
            f"Planar policy: horizontal velocity from actor, altitude PID + "
            f"rate-limited yaw (max {policy.max_yaw_rate_deg:.0f} deg/s).",
            flush=True,
        )

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            if now - last_heartbeat >= heartbeat_dt:
                send_heartbeat(mav)
                last_heartbeat = now

            if now >= next_step:
                pos, vel, R_enu, yaw = state.get()
                verbose = (
                    not args.quiet
                    and (elapsed < 5.0 or int(now) != int(now - control_dt))
                )
                depth_range = depth_sub.latest() if depth_sub else None

                if goal_xy is not None:
                    goal_enu = np.array([goal_xy[0], goal_xy[1], args.climb_alt])
                else:
                    goal_enu = pos

                if phase == "CLIMB":
                    alt = pos[2]
                    speed = np.linalg.norm(vel)
                    if not state.offboard:
                        set_offboard_mode(mav)
                    if not state.armed and now - last_arm_attempt >= 2.0:
                        # PX4 rejects arming while the EKF is still settling
                        # (in sim, the first trial's shader-compile stall;
                        # on hardware, a slow GPS/vision lock). The single
                        # pre-loop arm() then leaves the drone parked until
                        # the pre-policy timeout -- retry until it sticks.
                        arm(mav)
                        last_arm_attempt = now
                    if alt < args.climb_alt - args.arrive_tol:
                        # NED vz < 0 commands upward motion.
                        send_velocity_target_ned(
                            mav, 0.0, 0.0, -args.climb_rate, yaw_ground,
                        )
                    else:
                        send_velocity_target_ned(mav, 0.0, 0.0, 0.0, yaw_ground)
                    arrived = alt >= args.climb_alt - args.arrive_tol
                    settled = speed < args.settle_speed
                    if arrived and settled and state.offboard:
                        phase = "YAW"
                        print(f"\n>>> Climbed to alt={alt:.2f} m, speed={speed:.2f} m/s -- "
                              f"turning to face goal <<<\n")
                    if verbose:
                        print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/{args.climb_alt:.1f} "
                              f"speed={speed:.2f}  offboard={state.offboard} "
                              f"armed={state.armed}")
                elif phase == "YAW":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_goal)
                    yaw_cur_ned = math.atan2(
                        math.sin(math.pi / 2 - yaw), math.cos(math.pi / 2 - yaw)
                    )
                    yaw_err = math.atan2(
                        math.sin(yaw_goal - yaw_cur_ned), math.cos(yaw_goal - yaw_cur_ned)
                    )
                    if abs(math.degrees(yaw_err)) < args.yaw_tol_deg and state.offboard:
                        phase = "POLICY"
                        yaw_ned_cmd = yaw_cur_ned
                        mark_policy_phase("start")
                        print(f"\n>>> HANDOFF to velocity policy, "
                              f"yaw={math.degrees(yaw_cur_ned):.1f} deg <<<\n")
                    if verbose:
                        print(f"[YAW t={elapsed:.2f}s] yaw={math.degrees(yaw_cur_ned):.1f} "
                              f"target={math.degrees(yaw_goal):.1f} err={math.degrees(yaw_err):.1f}")
                elif phase == "POLICY":
                    obs = DiffAeroObs(
                        position_enu=pos,
                        velocity_enu=vel,
                        R_enu=R_enu,
                        goal_enu=goal_enu,
                        depth_planar=depth_range,
                    )
                    cmd = policy.compute(obs)
                    vx_n, vy_e, _ = enu_vel_to_ned(cmd.vel_cmd_enu)
                    if policy.planar:
                        vz_d = altitude_vz_ned(
                            args.climb_alt, pos, vel, max_vz=max_vel_z,
                        )
                    else:
                        _, _, vz_d = enu_vel_to_ned(cmd.vel_cmd_enu)
                    yaw_ned_cmd = policy.slew_yaw_ned_cmd(yaw_ned_cmd, control_dt)
                    yaw_out = yaw_ned_cmd
                    send_velocity_target_ned(mav, vx_n, vy_e, vz_d, yaw_out)
                    # Grounded-recovery: after an upset (obstacle graze, rough
                    # tracking) PX4's land detector can latch with the drone
                    # parked on the ground, ignoring climb setpoints forever.
                    # If we sit grounded and motionless mid-policy, re-run the
                    # CLIMB/YAW sequence from the current spot instead of
                    # burning the rest of the flight budget.
                    if pos[2] < 0.5 and np.linalg.norm(vel) < 0.3:
                        if grounded_since is None:
                            grounded_since = now
                        elif now - grounded_since > 3.0 and recoveries < 3:
                            recoveries += 1
                            grounded_since = None
                            hold_x_n, hold_y_e = pos[1], pos[0]
                            yaw_goal = math.atan2(goal_enu[0] - pos[0],
                                                  goal_enu[1] - pos[1])
                            policy.reset()
                            last_arm_attempt = 0.0
                            phase = "CLIMB"
                            print(f"\n>>> GROUNDED mid-policy at pos={pos.round(2)} -- "
                                  f"recovery {recoveries}/3: re-arm + climb <<<\n")
                            continue
                    else:
                        grounded_since = None
                    if np.linalg.norm(goal_enu - pos) < 0.5:
                        phase = "LANDING"
                        mark_policy_phase("end")
                        print(f"\n>>> HANDOFF to landing at pos={pos.round(2)} <<<\n")
                    if verbose:
                        print(
                            f"[POLICY t={elapsed:.2f}s step={step_count}]\n"
                            f"  pos(ENU)     = {pos.round(2)}\n"
                            f"  vel(ENU)     = {vel.round(2)}\n"
                            f"  goal(ENU)    = {np.round(goal_enu, 2)}\n"
                            f"  vel_cmd(ENU) = {np.round(cmd.vel_cmd_enu, 2)}  "
                            f"|v|={cmd.vel_norm:.2f}\n"
                            f"  vz_ned       = {vz_d:.2f}\n"
                            f"  yaw_ned(deg) = {math.degrees(yaw_out):.1f}\n"
                            f"  offboard={state.offboard}  armed={state.armed}\n"
                            "---"
                        )
                elif phase == "LANDING":
                    if not landing_sent:
                        send_land_command(mav)
                        landing_sent = True
                    if not state.armed:
                        print("\n>>> Landed and disarmed. Exiting.")
                        break
                    if verbose:
                        print(f"[LANDING t={elapsed:.2f}s] alt={pos[2]:.2f} m")

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
        mark_offboard_done()


if __name__ == "__main__":
    main()
