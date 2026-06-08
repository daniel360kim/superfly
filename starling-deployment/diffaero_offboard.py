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
    python diffaero_offboard.py --checkpoint checkpoints/DiffAero/sha2c_pmc --depth \
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
import torch
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 30.0          # DiffAero trained at dt=0.0333 s
HEARTBEAT_HZ = 2.0
G = 9.80665

# PX4 custom mode for OFFBOARD
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# DiffAero perception (camera) shape: height x width.
DEPTH_H, DEPTH_W = 9, 16
# DiffAero camera max range [m] (sensor.max_dist); depth = 1 - clamp(r,0,5)/5.
CAM_MAX_DIST = 5.0

# DiffAero point-mass action limits (max_acc.xy / max_acc.z defaults).
MAX_ACC_XY = 20.0
MAX_ACC_Z = 40.0


# ---------------------------------------------------------------------------
# Frame conversion helpers (identical convention to diffdrone_offboard.py)
# ---------------------------------------------------------------------------

# ENU inertial -> NED inertial: same rotation Pegasus uses
_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
# FLU body -> FRD body: +PI around X
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def quat_ENU_FLU_to_NED_FRD(R_enu_flu: np.ndarray) -> np.ndarray:
    """Return [w, x, y, z] quaternion in NED/FRD for a given ENU/FLU rotation matrix."""
    rot = _rot_ENU_to_NED * Rotation.from_matrix(R_enu_flu) * _rot_FLU_to_FRD
    q = rot.as_quat()  # [x, y, z, w] scipy convention
    return np.array([q[3], q[0], q[1], q[2]])  # -> [w, x, y, z] MAVLink convention


def diffaero_quat_to_attitude_target(quat_xyzw_enu_flu: np.ndarray, acc_norm: float,
                                     max_accel: float, verbose: bool = False):
    """
    Convert the DiffAero attitude command (ENU/FLU quaternion, xyzw) and thrust
    acceleration magnitude into (q_des_ned_frd [w,x,y,z], thrust_norm [0-1]) for
    SET_ATTITUDE_TARGET.

    The DiffAero world frame is ENU/FLU (z up, body x forward, body z along the
    thrust direction), matching the converter above. acc_norm is |acc_cmd|, the
    thrust-acceleration magnitude the rotors must produce; thrust is normalized
    by `max_accel` so that hover (~g) maps to MPC_THR_HOVER = g / max_accel.
    """
    R_des_enu = Rotation.from_quat(
        [quat_xyzw_enu_flu[0], quat_xyzw_enu_flu[1],
         quat_xyzw_enu_flu[2], quat_xyzw_enu_flu[3]]
    ).as_matrix()
    q_des = quat_ENU_FLU_to_NED_FRD(R_des_enu)
    thrust_norm = float(np.clip(acc_norm / max_accel, 0.0, 1.0))

    if verbose:
        des_rpy = Rotation.from_matrix(R_des_enu).as_euler("xyz", degrees=True)
        print(
            f"  acc_norm        = {acc_norm:.3f} m/s^2\n"
            f"  R_des_enu RPY   = roll={des_rpy[0]:.1f} pitch={des_rpy[1]:.1f} yaw={des_rpy[2]:.1f}\n"
            f"  q_des(NED/FRD)  = w={q_des[0]:.3f} x={q_des[1]:.3f} y={q_des[2]:.3f} z={q_des[3]:.3f}\n"
            f"  thrust_norm     = {thrust_norm:.3f}"
        )
    return q_des, thrust_norm


# ---------------------------------------------------------------------------
# State container (updated by MAVLink receive thread)
# ---------------------------------------------------------------------------

class DroneState:
    def __init__(self):
        self._lock = threading.Lock()
        self.position_enu = np.zeros(3)
        self.velocity_enu = np.zeros(3)
        self.R_enu = np.eye(3)
        self.yaw = 0.0
        self.armed = False
        self.offboard = False
        self.last_update = 0.0

    def update_from_attitude(self, msg):
        """Update from ATTITUDE_QUATERNION message (NED/FRD quaternion)."""
        q_ned_frd = Rotation.from_quat([msg.q2, msg.q3, msg.q4, msg.q1])  # -> scipy [x,y,z,w]
        rot_enu_flu = _rot_ENU_to_NED.inv() * q_ned_frd * _rot_FLU_to_FRD.inv()
        with self._lock:
            self.R_enu = rot_enu_flu.as_matrix()
            fwd_enu = self.R_enu[:, 0]
            self.yaw = math.atan2(fwd_enu[1], fwd_enu[0])
        self.last_update = time.time()

    def update_from_local_position(self, msg):
        """Update from LOCAL_POSITION_NED message (NED -> ENU)."""
        with self._lock:
            self.position_enu = np.array([msg.y, msg.x, -msg.z])
            self.velocity_enu = np.array([msg.vy, msg.vx, -msg.vz])

    def update_from_heartbeat(self, msg):
        with self._lock:
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            custom_main = (msg.custom_mode >> 16) & 0xFF
            self.offboard = (custom_main == PX4_CUSTOM_MAIN_MODE_OFFBOARD)

    def get(self):
        with self._lock:
            return (
                self.position_enu.copy(),
                self.velocity_enu.copy(),
                self.R_enu.copy(),
                self.yaw,
            )


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

class DiffAeroPolicy:
    """Loads and runs the exported DiffAero actor (TorchScript) for inference.

    The exported actor forward signature (MLP + perception):
        ((state[1,9], perception[1,H,W]), orientation[1,3], Rz[1,3,3],
         min_action[1,3], max_action[1,3])
        -> (acc_cmd[1,3], quat_xyzw_cmd[1,4], acc_norm[1])
    where acc_cmd is the world-frame thrust acceleration and quat_xyzw_cmd is the
    ENU/FLU attitude (xyzw) with yaw aligned to the velocity EMA.
    """

    def __init__(self, checkpoint_path: str, vel_ema_factor: float = 0.1,
                 max_acc_xy: float = MAX_ACC_XY, max_acc_z: float = MAX_ACC_Z):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt2_path = self._resolve_pt2(checkpoint_path)
        print(f"Loading DiffAero TorchScript actor from {pt2_path} ...")
        self.module = torch.jit.load(str(pt2_path), map_location=self.device)
        self.module.eval()

        self.min_action = torch.tensor(
            [[-max_acc_xy, -max_acc_xy, 0.0]], dtype=torch.float32, device=self.device)
        self.max_action = torch.tensor(
            [[max_acc_xy, max_acc_xy, max_acc_z]], dtype=torch.float32, device=self.device)

        self.vel_ema_factor = float(vel_ema_factor)
        self.vel_ema = None  # world-frame velocity EMA (drives commanded yaw)
        self._up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

    @staticmethod
    def _resolve_pt2(checkpoint_path: str) -> Path:
        p = Path(checkpoint_path)
        if p.is_file():
            return p
        candidates = [
            p / "checkpoints" / "exported_actor.pt2",
            p / "exported_actor.pt2",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"Could not find exported_actor.pt2 under {checkpoint_path}. "
            f"Tried: {[str(c) for c in candidates]}")

    def reset(self):
        self.vel_ema = None

    def normalize_depth(self, range_m: np.ndarray) -> torch.Tensor:
        """Convert metric Euclidean range (H,W) to the trained perception encoding:
            depth = 1 - clamp(range, 0, max_dist) / max_dist
        Far / no-return -> 0; very close -> ~1. Output (1, H, W).
        """
        d = torch.as_tensor(range_m, dtype=torch.float32, device=self.device)
        d = d.clamp(0.0, CAM_MAX_DIST)
        perception = 1.0 - d / CAM_MAX_DIST
        return perception.reshape(1, DEPTH_H, DEPTH_W)

    @torch.no_grad()
    def step(self, position_enu: np.ndarray, velocity_enu: np.ndarray,
             R_enu: np.ndarray, target_pos_enu: np.ndarray,
             max_vel: float = 5.0, depth_range: np.ndarray = None):
        """Run one policy step.

        Returns (acc_cmd_world[3], quat_xyzw_enu_flu[4], acc_norm) where acc_cmd
        is the world-frame thrust acceleration (debug) and quat/acc_norm form the
        attitude+thrust setpoint.
        """
        R = torch.tensor(R_enu, dtype=torch.float32, device=self.device)  # (3,3)
        v_world = torch.tensor(velocity_enu, dtype=torch.float32, device=self.device)

        # Yaw-only (local) frame: columns are [fwd, left, up] in world, matching
        # diffaero axis_rotmat("Z", yaw).
        fwd = R[:, 0].clone()
        fwd[2] = 0.0
        fwd = torch.nn.functional.normalize(fwd, dim=0)
        left = torch.cross(self._up, fwd, dim=0)
        Rz = torch.stack([fwd, left, self._up], dim=1)  # (3,3)
        uz = R[:, 2]  # body up axis in world

        # target_vel_world = (target_pos - p) / max(dist/max_vel, 1)
        relpos = (torch.tensor(target_pos_enu, dtype=torch.float32, device=self.device)
                  - torch.tensor(position_enu, dtype=torch.float32, device=self.device))
        dist = relpos.norm()
        mv = torch.tensor(float(max_vel), device=self.device)
        denom = torch.maximum(dist / mv, torch.ones((), device=self.device))
        target_vel_world = relpos / denom

        target_vel_local = Rz.t() @ target_vel_world
        v_local = Rz.t() @ v_world
        state9 = torch.cat([target_vel_local, uz, v_local]).unsqueeze(0)  # (1,9)

        # Velocity EMA -> commanded yaw (matches align_yaw_with_vel_ema).
        if self.vel_ema is None:
            self.vel_ema = v_world.clone()
        else:
            self.vel_ema = torch.lerp(self.vel_ema, v_world, self.vel_ema_factor)
        orientation = self.vel_ema.unsqueeze(0)
        if orientation.norm() < 1e-3:
            # Yaw undefined when near-stationary; fall back to current heading.
            orientation = fwd.unsqueeze(0)

        if depth_range is not None:
            perception = self.normalize_depth(depth_range)
        else:
            perception = torch.zeros(1, DEPTH_H, DEPTH_W, device=self.device)

        acc_cmd, quat_cmd, acc_norm = self.module(
            (state9, perception), orientation, Rz.unsqueeze(0),
            self.min_action, self.max_action)

        return (acc_cmd.squeeze(0).cpu().numpy(),
                quat_cmd.squeeze(0).cpu().numpy(),
                float(acc_norm.reshape(-1)[0].cpu()))


# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def wait_for_heartbeat(mav, timeout=30):
    print("Waiting for heartbeat...")
    mav.wait_heartbeat(timeout=timeout)
    print(f"Heartbeat received from system {mav.target_system} component {mav.target_component}")


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


def send_attitude_target(mav, q_wxyz: np.ndarray, thrust: float):
    """Send SET_ATTITUDE_TARGET (msg id 82), ignoring body rates."""
    mav.mav.set_attitude_target_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        7,  # type_mask: ignore roll/pitch/yaw rate
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,
        float(thrust),
    )


def send_position_target_ned(mav, x_n: float, y_e: float, z_d: float, yaw: float = 0.0):
    """Send SET_POSITION_TARGET_LOCAL_NED (msg id 84), position + yaw only."""
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


# ---------------------------------------------------------------------------
# MAVLink receive thread
# ---------------------------------------------------------------------------

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
    parser.add_argument("--max-vel", type=float, default=5.0,
                        help="Target cruise speed [m/s]; target_vel = (goal-pos) "
                             "normalized to this. Training sampled [3, 6].")
    parser.add_argument("--max-accel", type=float, default=30.0,
                        help="Thrust-acceleration that maps to full throttle [m/s^2]. "
                             "Hover throttle is set to g/max_accel.")
    args = parser.parse_args()

    goal_xy = np.array(args.goal) if args.goal is not None else None

    depth_sub = None
    if args.depth:
        from depth_transport import DepthSubscriber
        depth_sub = DepthSubscriber()
        print("Depth subscriber listening for frames over UDP.")

    print(f"Connecting to {args.connect} ...")
    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)

    state = DroneState()
    stop_event = threading.Event()
    recv_thread = threading.Thread(target=receive_loop, args=(mav, state, stop_event), daemon=True)
    recv_thread.start()

    policy = DiffAeroPolicy(args.checkpoint)

    # Thrust normalization: hover (~g) -> MPC_THR_HOVER = g / max_accel.
    hover_thrust = float(np.clip(G / args.max_accel, 0.0, 1.0))
    print(f"Setting PX4 MPC_THR_HOVER = {hover_thrust:.3f} ...")
    set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
    time.sleep(0.2)

    # Pre-arm: stream POSITION setpoints (hold + climb) so PX4 accepts OFFBOARD.
    pos0, _, _, _ = state.get()
    hold_x_n = pos0[1]   # North = ENU.y
    hold_y_e = pos0[0]   # East  = ENU.x
    hold_z_d = -args.climb_alt
    yaw_ned = 0.0

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
                cur_rpy = Rotation.from_matrix(R_enu).as_euler("xyz", degrees=True)
                verbose = elapsed < 5.0 or (int(now) != int(now - control_dt))

                depth_range = depth_sub.latest() if depth_sub else None

                # Horizontal-cruise goal: XY from field goal, Z = flight altitude.
                if goal_xy is not None:
                    goal_enu = np.array([goal_xy[0], goal_xy[1], args.climb_alt])
                else:
                    goal_enu = pos  # zero target velocity -> hover

                if phase == "CLIMB":
                    send_position_target_ned(mav, hold_x_n, hold_y_e, hold_z_d, yaw_ned)
                    alt = pos[2]
                    speed = np.linalg.norm(vel)
                    arrived = abs(alt - args.climb_alt) < args.arrive_tol
                    settled = speed < args.settle_speed
                    if arrived and settled and state.offboard:
                        phase = "POLICY"
                        print(f"\n>>> HANDOFF to policy at alt={alt:.2f} m, "
                              f"speed={speed:.2f} m/s <<<\n")
                    if verbose:
                        print(f"[CLIMB t={elapsed:.2f}s] alt={alt:.2f}/{args.climb_alt:.1f} "
                              f"speed={speed:.2f}  offboard={state.offboard} armed={state.armed}")
                else:  # POLICY
                    acc_cmd, quat_cmd, acc_norm = policy.step(
                        pos, vel, R_enu, goal_enu,
                        max_vel=args.max_vel, depth_range=depth_range)
                    q_des, thrust_norm = diffaero_quat_to_attitude_target(
                        quat_cmd, acc_norm, args.max_accel, verbose=verbose)
                    send_attitude_target(mav, q_des, thrust_norm)
                    if verbose:
                        print(
                            f"[POLICY t={elapsed:.2f}s step={step_count}]\n"
                            f"  pos(ENU)        = {pos.round(2)}\n"
                            f"  vel(ENU)        = {vel.round(2)}\n"
                            f"  goal(ENU)       = {np.round(goal_enu, 2)}\n"
                            f"  acc_cmd(ENU)    = {np.round(acc_cmd, 2)}  |acc|={acc_norm:.2f}\n"
                            f"  cur_att RPY(ENU)= roll={cur_rpy[0]:.1f} pitch={cur_rpy[1]:.1f} yaw={cur_rpy[2]:.1f}\n"
                            f"  armed={state.armed}  offboard={state.offboard}\n"
                            "---"
                        )

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


if __name__ == "__main__":
    main()
