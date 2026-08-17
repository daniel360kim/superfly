#!/usr/bin/env python
"""
DepthNav VELOCITY-command offboard controller for PX4.

Companion to depthnav_offboard.py. That script converts the policy's thrust
vector into SET_ATTITUDE_TARGET, bypassing PX4's position and velocity loops
entirely. This one drives a policy trained with action_type=VELOCITY_YAW and
feeds its world-frame velocity setpoint straight to PX4's velocity loop via
SET_POSITION_TARGET_LOCAL_NED -- the same interface diffaero_vel_offboard.py
uses, so the two can be compared on equal terms.

  * Observation: identical to the thrust policy (state 7, target 4, depth
    1x72x128). Only the output head's meaning changes.
  * Action: action[:3] is a velocity setpoint in m/s in the START frame,
    rotated to world ENU by DepthNavPolicy and converted to NED here.
    action[3] is an absolute yaw in the START frame.
  * Yaw: absolute (yawspeed is masked off), with a rate limiter as a safety
    clamp on top of the policy's command.
  * NO client-side velocity lag is applied. wrapper/diffaero_vel_core.py adds a
    first-order lag to emulate its training plant on top of PX4's own loop;
    depthnav's training sim models PX4's velocity loop explicitly (a PI
    controller in PointMassDynamics), so re-applying a lag here would
    double-count it.

Differences from depthnav_offboard.py worth knowing about:
  * There is a YAW phase before handoff. Training spawns the agent with its
    START frame x-axis pointing at the target (navigation_env.reset_agents
    builds start_rot from the target direction, with only ~0.2 rad of noise).
    depthnav_offboard.py captures START at whatever heading the climb ended on,
    which is off-distribution. Here we turn to face the goal first.
  * Goal-reach transitions to LANDING and writes the "end" phase marker and the
    done-file. depthnav_offboard.py writes only "start", which makes
    compare/run_comparison.py's `policy_reported_reached` permanently False for
    depthnav while it is True for the other methods.

Run with the depthnav venv (torch + depthnav + pymavlink):
  python depthnav_vel_offboard.py --checkpoint <vel .pth> \\
      --depth --climb-alt 2 --goal -3 40 1 --target-speed 1.1
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

from superfly.common.frames import enu_vel_to_ned, wrap_pi, yaw_enu_to_ned
from superfly.common.px4_offboard import (
    HEARTBEAT_HZ,
    DroneState, wait_for_heartbeat, wait_for_position, set_offboard_mode, arm,
    send_position_target_ned, send_velocity_target_ned, send_land_command,
    send_heartbeat, set_param_float, receive_loop, retry_offboard_arm,
)
from superfly.common.sentinels import mark_policy_phase, mark_offboard_done
from superfly.policies.depthnav import DepthNavPolicy, ACTION_VELOCITY

CONTROL_HZ = 50.0   # depthnav ctrl_dt = 0.02

# Defaults mirror policy_cfg/small_yaw_vel.yaml and the training dynamics.
DEFAULT_MAX_VEL_XY = 2.5    # VelocityBoundedYaw max_vel_xy
DEFAULT_MAX_VEL_Z = 1.5     # VelocityBoundedYaw max_vel_z
DEFAULT_TARGET_SPEED = 1.1  # centre of the trained [0.7, 1.5] band
DEFAULT_ACC_HOR = 3.0       # must match dynamics_kwargs.vel_setpoint_slew
DEFAULT_MAX_YAW_RATE_DEG = 60.0


def slew_yaw(yaw_cmd_ned: float, yaw_target_ned: float,
             max_rate_deg: float, dt: float) -> float:
    """Rate-limit the commanded yaw toward the policy's target.

    The policy's yaw is already trained smooth (lambda_yaw plus the body-rate
    penalty), so this is a safety clamp against a step in the command rather
    than the primary shaping, which is why it is not in the training loop.
    """
    err = wrap_pi(yaw_target_ned - yaw_cmd_ned)
    max_step = math.radians(max_rate_deg) * dt
    return wrap_pi(yaw_cmd_ned + max(-max_step, min(max_step, err)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="depthnav velocity-command .pth")
    parser.add_argument("--policy-cfg", default=None,
                        help="policy yaml (default: policy_cfg/small_yaw_vel.yaml)")
    parser.add_argument("--connect", default="udp:localhost:14550")
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Goal position (ENU). target = goal - pos each step.")
    parser.add_argument("--target-speed", type=float, default=DEFAULT_TARGET_SPEED,
                        help=f"Cruise speed [m/s] (trained 0.7-1.5, default {DEFAULT_TARGET_SPEED})")
    parser.add_argument("--max-vel-xy", type=float, default=DEFAULT_MAX_VEL_XY,
                        help="PX4 MPC_XY_VEL_MAX; match VelocityBoundedYaw max_vel_xy")
    parser.add_argument("--max-vel-z", type=float, default=DEFAULT_MAX_VEL_Z,
                        help="PX4 MPC_Z_VEL_MAX_UP/DN; match VelocityBoundedYaw max_vel_z")
    parser.add_argument("--acc-hor", type=float, default=DEFAULT_ACC_HOR,
                        help="PX4 MPC_ACC_HOR; match dynamics vel_setpoint_slew")
    parser.add_argument("--max-yaw-rate-deg", type=float, default=DEFAULT_MAX_YAW_RATE_DEG)
    parser.add_argument("--depth", action="store_true",
                        help="Subscribe to live 72x128 depth frames over UDP.")
    parser.add_argument("--climb-alt", type=float, default=2.0)
    parser.add_argument("--climb-rate", type=float, default=1.0)
    parser.add_argument("--arrive-tol", type=float, default=0.3)
    parser.add_argument("--settle-speed", type=float, default=0.2)
    parser.add_argument("--yaw-tol-deg", type=float, default=5.0)
    parser.add_argument("--goal-tol", type=float, default=0.5,
                        help="Distance [m] to goal that ends the policy phase")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.goal is None:
        raise SystemExit("--goal X Y Z is required (depthnav is goal-seeking).")
    goal_enu = np.array(args.goal, dtype=float)

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
    wait_for_position(state)

    print("Loading depthnav velocity policy ...")
    policy_kwargs = {
        "target_speed": args.target_speed,
        "action_mode": ACTION_VELOCITY,
    }
    if args.checkpoint:
        policy_kwargs["checkpoint_path"] = args.checkpoint
    if args.policy_cfg:
        policy_kwargs["cfg_path"] = args.policy_cfg
    policy = DepthNavPolicy(**policy_kwargs)

    # Make PX4 saturate the same way training did.
    print(f"Setting PX4 MPC_XY_VEL_MAX   = {args.max_vel_xy:.2f}")
    set_param_float(mav, "MPC_XY_VEL_MAX", args.max_vel_xy)
    print(f"Setting PX4 MPC_Z_VEL_MAX_UP = {args.max_vel_z:.2f}")
    set_param_float(mav, "MPC_Z_VEL_MAX_UP", args.max_vel_z)
    print(f"Setting PX4 MPC_Z_VEL_MAX_DN = {args.max_vel_z:.2f}")
    set_param_float(mav, "MPC_Z_VEL_MAX_DN", args.max_vel_z)
    # Left at PX4's default by every other script in this harness, which lets
    # the autopilot's acceleration limit differ from the one trained against.
    print(f"Setting PX4 MPC_ACC_HOR      = {args.acc_hor:.2f}")
    set_param_float(mav, "MPC_ACC_HOR", args.acc_hor)
    time.sleep(0.2)

    pos0, _, _, yaw0 = state.get()
    hold_x_n = pos0[1]
    hold_y_e = pos0[0]
    hold_z_d = -args.climb_alt

    yaw_ground = yaw_enu_to_ned(yaw0)
    d_east = goal_enu[0] - pos0[0]
    d_north = goal_enu[1] - pos0[1]
    yaw_goal = math.atan2(d_east, d_north)

    print("Pre-arming: streaming position setpoints ...")
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

    control_dt = 1.0 / CONTROL_HZ
    heartbeat_dt = 1.0 / HEARTBEAT_HZ
    last_heartbeat = time.time()
    start_time = time.time()
    next_step = time.time()
    last_arm_try = time.time()
    step_count = 0

    phase = "CLIMB"
    landing_sent = False
    yaw_ned_cmd = yaw_ground
    print(f"CLIMB: velocity climb to {args.climb_alt:.1f} m at {args.climb_rate:.1f} m/s ...")

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            if now - last_heartbeat >= heartbeat_dt:
                send_heartbeat(mav)
                last_heartbeat = now

            if now < next_step:
                time.sleep(0.0005)
                continue

            pos, vel, R_enu, yaw = state.get()
            verbose = (not args.quiet
                       and (elapsed < 5.0 or int(now) != int(now - control_dt)))
            depth_m = depth_sub.latest() if depth_sub else None
            alt = pos[2]
            speed = float(np.linalg.norm(vel))

            if phase == "CLIMB":
                if alt < args.climb_alt - args.arrive_tol:
                    # NED vz < 0 commands upward motion.
                    send_velocity_target_ned(mav, 0.0, 0.0, -args.climb_rate, yaw_ground)
                else:
                    send_velocity_target_ned(mav, 0.0, 0.0, 0.0, yaw_ground)
                last_arm_try = retry_offboard_arm(mav, state, last_arm_try)
                if (alt >= args.climb_alt - args.arrive_tol
                        and speed < args.settle_speed and state.offboard):
                    phase = "YAW"
                    print(f"\n>>> Climbed to alt={alt:.2f} m, speed={speed:.2f} m/s -- "
                          f"turning to face goal <<<\n")
                if verbose:
                    print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/{args.climb_alt:.1f} "
                          f"speed={speed:.2f} offboard={state.offboard} armed={state.armed}")

            elif phase == "YAW":
                # Face the goal before capturing START. Training builds the
                # START frame from the target direction, so handing off at an
                # arbitrary heading puts the policy off-distribution.
                send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_goal)
                yaw_cur_ned = yaw_enu_to_ned(yaw)
                yaw_err = wrap_pi(yaw_goal - yaw_cur_ned)
                if abs(math.degrees(yaw_err)) < args.yaw_tol_deg and state.offboard:
                    policy.reset(R_enu)   # capture the START frame, now goal-facing
                    yaw_ned_cmd = yaw_cur_ned
                    phase = "POLICY"
                    mark_policy_phase("start")
                    print(f"\n>>> HANDOFF to velocity policy at "
                          f"yaw={math.degrees(yaw_cur_ned):.1f} deg; START frame captured <<<\n")
                if verbose:
                    print(f"[YAW t={elapsed:.2f}s] yaw={math.degrees(yaw_cur_ned):.1f} "
                          f"target={math.degrees(yaw_goal):.1f} err={math.degrees(yaw_err):.1f}")

            elif phase == "POLICY":
                vel_cmd_enu, yaw_world_enu = policy.step(
                    pos, vel, R_enu, goal_enu, depth_m=depth_m)
                vx_n, vy_e, vz_d = enu_vel_to_ned(vel_cmd_enu)
                yaw_ned_cmd = slew_yaw(
                    yaw_ned_cmd, yaw_enu_to_ned(yaw_world_enu),
                    args.max_yaw_rate_deg, control_dt,
                )
                send_velocity_target_ned(mav, vx_n, vy_e, vz_d, yaw_ned_cmd)

                dist = float(np.linalg.norm(goal_enu - pos))
                if dist < args.goal_tol:
                    phase = "LANDING"
                    mark_policy_phase("end")
                    print(f"\n>>> Goal reached at pos={pos.round(2)} "
                          f"(dist={dist:.2f} m); landing <<<\n")
                if verbose:
                    print(
                        f"[POLICY t={elapsed:.2f}s step={step_count}]\n"
                        f"  pos(ENU)     = {pos.round(2)}  dist_to_goal={dist:.1f}\n"
                        f"  vel(ENU)     = {vel.round(2)}  |v|={speed:.2f}\n"
                        f"  vel_cmd(ENU) = {np.round(vel_cmd_enu, 2)}  "
                        f"|v_cmd|={np.linalg.norm(vel_cmd_enu):.2f}\n"
                        f"  vz_ned       = {vz_d:.2f}\n"
                        f"  yaw_ned(deg) = {math.degrees(yaw_ned_cmd):.1f}\n"
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
                    print(f"[LANDING t={elapsed:.2f}s] alt={alt:.2f} m")

            step_count += 1
            next_step += control_dt
            if next_step < time.time():
                next_step = time.time()

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_event.set()
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        # run_comparison.py polls this to know the run finished.
        mark_offboard_done()


if __name__ == "__main__":
    main()
