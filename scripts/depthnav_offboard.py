#!/usr/bin/env python
"""
DepthNav offboard controller for PX4.

Same PX4 climb->policy state machine as diffdrone_offboard.py, but driving the
trained depthnav MultiInputPolicy. Reuses diffdrone_offboard's MAVLink I/O,
DroneState (ENU<->NED), and the PX4 setup; swaps the policy + action decode.

Key differences from DiffPhysDrone:
  - 50 Hz control (depthnav ctrl_dt 0.02).
  - Policy output thrust is GRAVITY-INCLUSIVE (no +g add-back).
  - START-frame everything (captured at handoff; see depthnav_policy.py).
  - Explicit yaw command from the policy (APPLIED, not yaw-hold).
  - Depth: raw 72x128 metric metres (subscriber resizes/forwards).

Run with the depthnav venv (torch + depthnav + pymavlink):
  methods/depthnav/.venv/bin/python scripts/depthnav_offboard.py \
      --depth --climb-alt 2 --goal -3 40 1 --target-speed 3
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
from scipy.spatial.transform import Rotation

from superfly.common.frames import quat_ENU_FLU_to_NED_FRD
from superfly.common.px4_offboard import (
    MASS_KG, MAX_ACCEL, HEARTBEAT_HZ,
    DroneState, wait_for_heartbeat, set_offboard_mode, arm,
    send_attitude_target, send_position_target_ned, send_heartbeat,
    set_param_float, receive_loop, retry_offboard_arm,
)
from superfly.common.sentinels import mark_policy_phase
from superfly.policies.depthnav import DepthNavPolicy

CONTROL_HZ = 50.0   # depthnav ctrl_dt = 0.02


def thrust_world_to_attitude_target(thrust_accel_world: np.ndarray,
                                    yaw_world: float, verbose: bool = False):
    """Convert a GRAVITY-INCLUSIVE world-frame thrust acceleration + desired yaw
    to (q_des NED/FRD [w,x,y,z], thrust_norm [0-1]) for SET_ATTITUDE_TARGET.

    Unlike DiffPhysDrone's version, NO +g is added (depthnav thrust already
    includes gravity), and yaw is the policy's explicit command (not yaw-hold).
    """
    F_des = MASS_KG * np.asarray(thrust_accel_world, float)
    F_norm = np.linalg.norm(F_des)
    if F_norm < 1e-3:
        F_des = np.array([0.0, 0.0, MASS_KG * 9.80665])
        F_norm = np.linalg.norm(F_des)

    Z_b_des = F_des / F_norm
    X_c = np.array([math.cos(yaw_world), math.sin(yaw_world), 0.0])
    Z_cross_X = np.cross(Z_b_des, X_c)
    n = np.linalg.norm(Z_cross_X)
    if n < 1e-6:
        X_c = np.array([math.cos(yaw_world + 0.1), math.sin(yaw_world + 0.1), 0.0])
        Z_cross_X = np.cross(Z_b_des, X_c)
        n = np.linalg.norm(Z_cross_X)
    Y_b_des = Z_cross_X / n
    X_b_des = np.cross(Y_b_des, Z_b_des)
    R_des_enu = np.column_stack([X_b_des, Y_b_des, Z_b_des])
    q_des = quat_ENU_FLU_to_NED_FRD(R_des_enu)
    thrust_norm = float(np.clip(F_norm / (MASS_KG * MAX_ACCEL), 0.0, 1.0))

    if verbose:
        des_rpy = Rotation.from_matrix(R_des_enu).as_euler("xyz", degrees=True)
        print(
            f"  thrust(ENU)     = {np.round(thrust_accel_world, 3)}  |F|={F_norm:.2f} N\n"
            f"  Z_b_des(ENU)    = {np.round(Z_b_des, 3)}\n"
            f"  R_des RPY       = roll={des_rpy[0]:.1f}° pitch={des_rpy[1]:.1f}° yaw={des_rpy[2]:.1f}°\n"
            f"  yaw_cmd         = {math.degrees(yaw_world):.1f}°\n"
            f"  thrust_norm     = {thrust_norm:.3f}"
        )
    return q_des, thrust_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="depthnav .pth (default: level1_4_iteration_13500.pth)")
    parser.add_argument("--connect", default="udp:localhost:14550")
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Goal position (ENU). target = goal - pos each step.")
    parser.add_argument("--target-speed", type=float, default=3.0,
                        help="Desired cruise speed [m/s] (depthnav trained 1-5).")
    parser.add_argument("--depth", action="store_true",
                        help="Subscribe to live 72x128 depth frames over UDP.")
    parser.add_argument("--climb-alt", type=float, default=2.0)
    parser.add_argument("--arrive-tol", type=float, default=0.3)
    parser.add_argument("--settle-speed", type=float, default=0.2)
    args = parser.parse_args()

    if args.goal is None:
        raise SystemExit("--goal X Y Z is required (depthnav is goal-seeking).")
    goal_enu = np.array(args.goal)

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
    recv_thread = threading.Thread(target=receive_loop, args=(mav, state, stop_event), daemon=True)
    recv_thread.start()

    print("Loading depthnav policy ...")
    ckpt = args.checkpoint
    policy = (DepthNavPolicy(ckpt, target_speed=args.target_speed) if ckpt
              else DepthNavPolicy(target_speed=args.target_speed))

    hover_thrust = float(np.clip(MASS_KG * 9.80665 / (MASS_KG * MAX_ACCEL), 0.0, 1.0))
    print(f"Setting PX4 MPC_THR_HOVER = {hover_thrust:.3f} ...")
    set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
    time.sleep(0.2)

    # Pre-arm: stream position setpoints so PX4 holds OFFBOARD with no gaps.
    pos0, _, _, _ = state.get()
    hold_x_n = pos0[1]
    hold_y_e = pos0[0]
    hold_z_d = -args.climb_alt
    yaw_ned = 0.0

    print("Pre-arming: streaming position setpoints ...")
    for _ in range(30):
        send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_ned)
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
                depth_m = depth_sub.latest() if depth_sub else None

                if phase == "CLIMB":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_ned)
                    # Re-request OFFBOARD/arm until PX4 accepts: on big USD stages
                    # Isaac loads past the warmup and the EKF settles after the
                    # one-shot arm, which PX4 silently rejects (drone sits disarmed).
                    last_arm_try = retry_offboard_arm(mav, state, last_arm_try)
                    alt = pos[2]
                    speed = np.linalg.norm(vel)
                    arrived = abs(alt - args.climb_alt) < args.arrive_tol
                    settled = speed < args.settle_speed
                    if arrived and settled and state.offboard:
                        # Capture the START frame at handoff (the drone's current
                        # world attitude defines depthnav's fixed START frame).
                        policy.reset(R_enu)
                        phase = "POLICY"
                        mark_policy_phase("start")
                        print(f"\n>>> HANDOFF at alt={alt:.2f} m, speed={speed:.2f} m/s; "
                              f"START frame captured <<<\n")
                    if verbose:
                        print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/{args.climb_alt:.1f} "
                              f"speed={speed:.2f}  offboard={state.offboard} armed={state.armed}")

                else:  # POLICY
                    thrust_world, yaw_world = policy.step(pos, vel, R_enu, goal_enu, depth_m=depth_m)
                    q_des, thrust_norm = thrust_world_to_attitude_target(
                        thrust_world, yaw_world, verbose=verbose)
                    send_attitude_target(mav, q_des, thrust_norm)
                    if verbose:
                        dist = float(np.linalg.norm(goal_enu - pos))
                        print(
                            f"[POLICY t={elapsed:.2f}s step={step_count}]\n"
                            f"  pos(ENU)      = {pos.round(2)}  dist_to_goal={dist:.1f}\n"
                            f"  vel(ENU)      = {vel.round(2)}\n"
                            f"  thrust(ENU)   = {np.round(thrust_world, 2)} (gravity-incl)\n"
                            f"  armed={state.armed}  offboard={state.offboard}\n"
                            "---"
                        )

                step_count += 1
                next_step += control_dt
                if next_step < time.time():
                    next_step = time.time()
            else:
                time.sleep(0.0005)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_event.set()
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )


if __name__ == "__main__":
    main()
