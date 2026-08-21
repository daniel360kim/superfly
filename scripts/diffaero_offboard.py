#!/usr/bin/env python
"""
DiffAero (SHA2C, continuous point-mass) offboard controller for PX4.

Connects to PX4 via MAVLink, arms the drone, enters OFFBOARD mode, then runs the
DiffAero policy at 30 Hz (matching its training dt=0.0333) sending
SET_ATTITUDE_TARGET.

This mirrors diffdrone_offboard.py but adapts every interface to the DiffAero
policy, whose inputs and outputs differ from DiffPhysDrone:

  * Observation (obs_frame=local, point-mass): state = [target_vel_local(3),
    uz(3), v_local(3)] plus a 9x16 depth "perception" image. target/velocity are
    expressed in the yaw-only (local) frame; uz is the body up-axis in world.
  * Perception: Euclidean range, 16(w) x 9(h), hfov 86 deg, max_dist 5 m, forward
    camera (no downward pitch). The network consumes depth = 1 - clamp(r,0,5)/5.
  * Action (action_frame=local): the policy emits a world-frame THRUST
    acceleration command acc_cmd = Rz @ scaled_action (gravity is handled
    separately by the point-mass model), so unlike DiffPhysDrone we do NOT add
    gravity when forming the attitude/thrust setpoint.
  * Yaw aligns with the velocity EMA (align_yaw_with_vel_ema), which the exported
    actor bakes into the returned attitude quaternion.

Inference uses the self-contained TorchScript actor
(checkpoints/exported_actor.pt2), which bakes in tanh -> rescale -> Rz@action ->
point_mass_quat and returns (acc_cmd, quat_xyzw_cmd, acc_norm).

Usage:
    # Against PX4 SITL (after running run_px4_sim.py --policy diffaero):
    python diffaero_offboard.py --checkpoint checkpoints/DiffAero/thrust_pmc --depth \
        --goal <px> <py>

    # Against real VOXL2 over UDP:
    python diffaero_offboard.py --checkpoint <dir-or-pt2> --connect udp:192.168.1.x:14550
"""

import argparse
import math
import sys
import time
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Run-from-checkout convenience: make `superfly` importable without an
# installed package (method venvs should still `pip install -e . --no-deps`).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pymavlink import mavutil

from superfly.common.px4_offboard import (
    HEARTBEAT_HZ, G,
    DroneState, wait_for_heartbeat, set_offboard_mode, arm,
    retry_offboard_arm, send_attitude_target, send_position_target_ned,
    send_land_command, send_heartbeat, set_param_float, receive_loop,
)
from superfly.common.sentinels import mark_policy_phase, mark_offboard_done
from superfly.policies.diffaero import DiffAeroPolicy, DiffAeroObs, DA_INTRINSICS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 30.0          # DiffAero trained at dt=0.0333 s

# DiffAero perception (camera) shape: height x width.
DEPTH_H, DEPTH_W = 9, 16
# DiffAero camera max range [m] (sensor.max_dist); depth = 1 - clamp(r,0,5)/5.
CAM_MAX_DIST = 5.0

# DiffAero point-mass action limits (max_acc.xy / max_acc.z defaults).
MAX_ACC_XY = 20.0
MAX_ACC_Z = 40.0


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the DiffAero checkpoint dir (containing "
                             "checkpoints/exported_actor.pt2) or a .pt2 file directly")
    parser.add_argument("--connect", default="udp:localhost:14550",
                        help="MAVLink connection string")
    parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Goal XY (ENU). The goal altitude is fixed to --climb-alt "
                             "for a horizontal cruise toward the field's XY goal. If "
                             "omitted, the drone hovers (zero target velocity).")
    parser.add_argument("--depth", action="store_true",
                        help="Subscribe to live depth frames over UDP (from the sim/camera)")
    parser.add_argument("--climb-alt", type=float, default=10.0,
                        help="Climb to this altitude [m] via position control before "
                             "the policy takes over; also the goal altitude.")
    parser.add_argument("--arrive-tol", type=float, default=0.3,
                        help="Altitude tolerance [m] to consider the climb target reached")
    parser.add_argument("--settle-speed", type=float, default=0.2,
                        help="Speed [m/s] below which the drone is considered settled")
    parser.add_argument("--yaw-tol-deg", type=float, default=5.0,
                        help="Yaw tolerance [deg] to consider the goal-facing turn complete")
    parser.add_argument("--max-vel", type=float, default=5.0,
                        help="Target cruise speed [m/s]; target_vel = (goal-pos) "
                             "normalized to this. Training sampled [3, 6].")
    parser.add_argument("--max-accel", type=float, default=20.0,
                        help="Thrust-acceleration that maps to full throttle [m/s^2]. "
                             "Hover throttle is set to g/max_accel. Default matches "
                             "diffphys/depthnav's MAX_ACCEL (diffdrone_offboard.py).")
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
    recv_thread = threading.Thread(target=receive_loop, args=(mav, state, stop_event), daemon=True)
    recv_thread.start()

    policy = DiffAeroPolicy(
        intrinsics=DA_INTRINSICS, 
        checkpoint_path=args.checkpoint,
        max_vel=args.max_vel,
        max_accel=args.max_accel,
    )

    # Thrust normalization: hover (~g) -> MPC_THR_HOVER = g / max_accel.
    hover_thrust = float(np.clip(G / args.max_accel, 0.0, 1.0))
    print(f"Setting PX4 MPC_THR_HOVER = {hover_thrust:.3f} ...")
    set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
    time.sleep(0.2)

    # Pre-arm: stream POSITION setpoints (hold + climb) so PX4 accepts OFFBOARD.
    pos0, _, _, yaw0 = state.get()
    hold_x_n = pos0[1]   # North = ENU.y
    hold_y_e = pos0[0]   # East  = ENU.x
    hold_z_d = -args.climb_alt

    # Goal-facing yaw, computed up front but NOT commanded until the drone is
    # off the ground (rotating while still on the ground can dig a skid/leg in
    # and trip a sim collision). On the ground and during the climb we hold the
    # current heading; the turn-to-goal happens at climb_alt, before handoff.
    # `yaw0` is ENU math-convention (atan2(North, East)); convert to the NED
    # compass heading used by send_position_target_ned: ned = pi/2 - enu.
    yaw_ground = math.atan2(math.sin(math.pi / 2 - yaw0), math.cos(math.pi / 2 - yaw0))
    if goal_xy is not None:
        d_north = goal_xy[1] - hold_x_n
        d_east = goal_xy[0] - hold_y_e
        yaw_goal = math.atan2(d_east, d_north)
    else:
        yaw_goal = yaw0
    yaw_ned = yaw_ground

    print("Pre-arming: streaming position setpoints to satisfy PX4 OFFBOARD pre-condition...")
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

    policy.reset()

    control_dt = 1.0 / CONTROL_HZ
    heartbeat_dt = 1.0 / HEARTBEAT_HZ
    last_heartbeat = time.time()
    start_time = time.time()
    next_step = time.time()
    last_arm_try = time.time()
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
                cur_rpy = Rotation.from_matrix(R_enu).as_euler("xyz", degrees=True)
                verbose = elapsed < 5.0 or (int(now) != int(now - control_dt))

                depth_range = depth_sub.latest() if depth_sub else None

                # Horizontal-cruise goal: XY from field goal, Z = flight altitude.
                if goal_xy is not None:
                    goal_enu = np.array([goal_xy[0], goal_xy[1], args.climb_alt])
                else:
                    goal_enu = pos  # zero target velocity -> hover

                if phase == "CLIMB":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_ground)
                    # Re-request OFFBOARD/arm until PX4 accepts: on big USD stages
                    # Isaac loads past the warmup and the EKF settles after the
                    # one-shot arm, which PX4 silently rejects (drone sits disarmed).
                    last_arm_try = retry_offboard_arm(mav, state, last_arm_try)
                    alt = pos[2]
                    speed = np.linalg.norm(vel)
                    arrived = abs(alt - args.climb_alt) < args.arrive_tol
                    settled = speed < args.settle_speed
                    if arrived and settled and state.offboard:
                        phase = "YAW"
                        print(f"\n>>> Climbed to alt={alt:.2f} m, speed={speed:.2f} m/s -- "
                              f"turning to face goal <<<\n")
                    if verbose:
                        print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/{args.climb_alt:.1f} "
                              f"speed={speed:.2f}  offboard={state.offboard} armed={state.armed}")
                elif phase == "YAW":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_goal)
                    # `yaw` (from state) is the ENU math-convention heading
                    # (atan2(North, East)); yaw_goal is a NED compass heading
                    # (atan2(East, North)) = pi/2 - yaw_enu. Convert before diffing.
                    yaw_cur_ned = math.atan2(math.sin(math.pi / 2 - yaw), math.cos(math.pi / 2 - yaw))
                    yaw_err = math.atan2(math.sin(yaw_goal - yaw_cur_ned), math.cos(yaw_goal - yaw_cur_ned))
                    if abs(math.degrees(yaw_err)) < args.yaw_tol_deg and state.offboard:
                        phase = "POLICY"
                        mark_policy_phase("start")
                        print(f"\n>>> HANDOFF to policy facing goal, "
                              f"yaw={math.degrees(yaw_cur_ned):.1f} deg <<<\n")
                    if verbose:
                        print(f"[YAW t={elapsed:.2f}s] yaw={math.degrees(yaw_cur_ned):.1f} "
                              f"target={math.degrees(yaw_goal):.1f} err={math.degrees(yaw_err):.1f}")
                elif phase == "POLICY":
                    pos, vel, R_enu, yaw = state.get()
                    obs = DiffAeroObs(
                        position_enu=pos,
                        velocity_enu=vel,
                        R_enu=R_enu,
                        goal_enu=goal_enu,
                        depth_planar=depth_range,
                        
                    )
                    cmd = policy.compute(obs)
                    send_attitude_target(mav, cmd.attitude_ned_frd_wxyz, cmd.thrust_norm)
                    if np.linalg.norm(goal_enu - pos) < 0.5:
                        phase = "LANDING"
                        mark_policy_phase("end")
                        print(f"\n>>> HANDOFF to landing at pos={pos.round(2)} <<<\n")
                    if verbose:
                        print(
                            f"[POLICY t={elapsed:.2f}s step={step_count}]\n"
                            f"  pos(ENU)        = {pos.round(2)}\n"
                            f"  vel(ENU)        = {vel.round(2)}\n"
                            f"  goal(ENU)       = {np.round(goal_enu, 2)}\n"
                            f"  acc_cmd(ENU)    = {np.round(cmd.acc_cmd_enu, 2)}  |acc|={cmd.acc_norm:.2f}\n"
                            f"  cur_att RPY(ENU)= roll={cur_rpy[0]:.1f} pitch={cur_rpy[1]:.1f} yaw={cur_rpy[2]:.1f}\n"
                            f"  armed={state.armed}  offboard={state.offboard}\n"
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
        # Tell run_px4_sim.py (--auto-stop) this offboard is done -- however
        # it ended (landed, Ctrl-C, crash).
        mark_offboard_done()


if __name__ == "__main__":
    main()
