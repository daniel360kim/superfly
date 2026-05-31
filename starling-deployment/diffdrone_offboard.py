#!/usr/bin/env python
"""
DiffPhysDrone offboard controller for PX4.

Connects to PX4 via MAVLink, arms the drone, enters OFFBOARD mode,
then runs the DiffPhysDrone policy at 15 Hz sending SET_ATTITUDE_TARGET.

Usage:
    # Against PX4 SITL (after running run_px4_sim.py):
    python diffdrone_offboard.py --checkpoint path/to/model.pt

    # Against real VOXL2 over UDP:
    python diffdrone_offboard.py --checkpoint path/to/model.pt --connect udp:192.168.1.x:14550

    # Against real VOXL2 over USB serial:
    python diffdrone_offboard.py --checkpoint path/to/model.pt --connect /dev/ttyUSB0
"""

import argparse
import math
import sys
import time
import threading

import numpy as np
import torch
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

sys.path.insert(0, "DiffPhysDrone")
from model import Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_HZ = 15.0
HEARTBEAT_HZ = 2.0
MASS_KG = 1.5
MAX_ACCEL = 20.0  # m/s², used to normalize thrust to [0, 1]
G = np.array([0.0, 0.0, -9.80665])

# PX4 custom mode for OFFBOARD
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# Depth image shape expected by the policy (after 4x max-pool of 48x64)
DEPTH_H, DEPTH_W = 12, 16


# ---------------------------------------------------------------------------
# Frame conversion helpers
# ---------------------------------------------------------------------------

# ENU inertial → NED inertial: same rotation Pegasus uses
_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
# FLU body → FRD body: +PI around X
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def rotation_matrix_ENU_FLU_to_NED_FRD(R_enu_flu: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix from ENU/FLU convention to NED/FRD."""
    rot = _rot_ENU_to_NED * Rotation.from_matrix(R_enu_flu) * _rot_FLU_to_FRD
    return rot.as_matrix()


def quat_ENU_FLU_to_NED_FRD(R_enu_flu: np.ndarray) -> np.ndarray:
    """Return [w, x, y, z] quaternion in NED/FRD for a given ENU/FLU rotation matrix."""
    rot = _rot_ENU_to_NED * Rotation.from_matrix(R_enu_flu) * _rot_FLU_to_FRD
    q = rot.as_quat()  # [x, y, z, w] scipy convention
    return np.array([q[3], q[0], q[1], q[2]])  # → [w, x, y, z] MAVLink convention


# ---------------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------------

def act_world_to_attitude_target(act_world: np.ndarray, current_yaw: float, verbose: bool = False):
    """
    Convert a DiffPhysDrone world-frame thrust-acceleration vector to
    (q_des_ned_frd [w,x,y,z], thrust_norm [0-1]) for SET_ATTITUDE_TARGET.

    act_world: [m/s²] in ENU world frame (output of policy decode formula)
    current_yaw: current drone yaw in radians (ENU convention, CCW from East)
    """
    # Desired force in world frame [N]
    F_des = MASS_KG * act_world

    F_norm = np.linalg.norm(F_des)
    if F_norm < 1e-3:
        # Degenerate: just hover
        F_des = np.array([0.0, 0.0, MASS_KG * 9.80665])
        F_norm = np.linalg.norm(F_des)

    # Desired body-z axis (thrust direction) in ENU
    Z_b_des = F_des / F_norm

    # Yaw-hold: desired body-x points in current heading direction
    X_c = np.array([math.cos(current_yaw), math.sin(current_yaw), 0.0])

    Z_cross_X = np.cross(Z_b_des, X_c)
    z_cross_x_norm = np.linalg.norm(Z_cross_X)
    if z_cross_x_norm < 1e-6:
        # Near-singular (thrust pointing along yaw axis) — use fallback X_c
        X_c = np.array([math.cos(current_yaw + 0.1), math.sin(current_yaw + 0.1), 0.0])
        Z_cross_X = np.cross(Z_b_des, X_c)
        z_cross_x_norm = np.linalg.norm(Z_cross_X)

    Y_b_des = Z_cross_X / z_cross_x_norm
    X_b_des = np.cross(Y_b_des, Z_b_des)

    # ENU/FLU rotation matrix: columns are [X_b, Y_b, Z_b] in world frame
    R_des_enu = np.column_stack([X_b_des, Y_b_des, Z_b_des])

    # Convert to NED/FRD quaternion for PX4
    q_des = quat_ENU_FLU_to_NED_FRD(R_des_enu)

    # Normalized thrust [0, 1]
    thrust_norm = float(np.clip(F_norm / (MASS_KG * MAX_ACCEL), 0.0, 1.0))

    if verbose:
        des_rpy = Rotation.from_matrix(R_des_enu).as_euler("xyz", degrees=True)
        print(
            f"  act_world(ENU)  = {np.round(act_world, 3)}\n"
            f"  F_des(ENU)      = {np.round(F_des, 3)}  |F|={F_norm:.2f} N\n"
            f"  Z_b_des(ENU)    = {np.round(Z_b_des, 3)}\n"
            f"  R_des_enu RPY   = roll={des_rpy[0]:.1f}° pitch={des_rpy[1]:.1f}° yaw={des_rpy[2]:.1f}°\n"
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
        # ENU position [m]
        self.position_enu = np.zeros(3)
        # ENU linear velocity [m/s]
        self.velocity_enu = np.zeros(3)
        # ENU/FLU rotation matrix
        self.R_enu = np.eye(3)
        # Yaw angle [rad], ENU convention
        self.yaw = 0.0
        # Armed and in offboard
        self.armed = False
        self.offboard = False
        self.last_update = 0.0

    def update_from_attitude(self, msg):
        """Update from ATTITUDE_QUATERNION message (NED/FRD quaternion)."""
        # MAVLink quaternion is [w, x, y, z]
        q_ned_frd = Rotation.from_quat([msg.q2, msg.q3, msg.q4, msg.q1])  # → scipy [x,y,z,w]
        # Convert NED/FRD → ENU/FLU
        rot_enu_flu = _rot_ENU_to_NED.inv() * q_ned_frd * _rot_FLU_to_FRD.inv()
        with self._lock:
            self.R_enu = rot_enu_flu.as_matrix()
            # Yaw: angle of body-x projection onto ENU XY plane
            fwd_enu = self.R_enu[:, 0]
            self.yaw = math.atan2(fwd_enu[1], fwd_enu[0])
        self.last_update = time.time()

    def update_from_local_position(self, msg):
        """Update from LOCAL_POSITION_NED message."""
        # NED → ENU: x↔y, negate z
        with self._lock:
            self.position_enu = np.array([msg.y, msg.x, -msg.z])
            self.velocity_enu = np.array([msg.vy, msg.vx, -msg.vz])

    def update_from_heartbeat(self, msg):
        with self._lock:
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            # PX4 custom mode: offboard = main mode 6
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

class DiffDronePolicy:
    def __init__(self, checkpoint_path: str, no_odom: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = 7 if no_odom else 10
        self.model = Model(obs_dim, 6).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.no_odom = no_odom
        self.h = None  # GRU hidden state
        self.g = torch.tensor([0.0, 0.0, -9.80665], device=self.device)
        # Blank depth image (no obstacles) — 12x16 after 4x pool of 48x64
        self._blank_depth = torch.full(
            (1, 1, DEPTH_H, DEPTH_W), 3.0 / 24.0 - 0.6, device=self.device
        )  # 24m depth normalized: 3/24 - 0.6

    def reset(self):
        self.h = None

    @torch.no_grad()
    def step(self, position_enu: np.ndarray, velocity_enu: np.ndarray,
             R_enu: np.ndarray, target_velocity_enu: np.ndarray,
             margin: float = 10.0) -> np.ndarray:
        """
        Run one policy step. Returns act_world [m/s²] in ENU frame.

        position_enu, velocity_enu, R_enu: current drone state (ENU/FLU)
        target_velocity_enu: desired velocity vector in ENU [m/s]
        margin: distance to nearest obstacle [m] (use large value if unknown)
        """
        R = torch.tensor(R_enu, dtype=torch.float32, device=self.device)  # (3,3)

        # Build yaw-aligned frame (strip pitch/roll, keep only yaw)
        fwd = R[:, 0].clone()
        fwd[2] = 0.0
        fwd = torch.nn.functional.normalize(fwd, dim=0)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        R_yaw = torch.stack([fwd, torch.cross(up, fwd), up], dim=1)  # (3,3)

        # Target velocity in yaw-aligned body frame, clamped to max_speed
        tv = torch.tensor(target_velocity_enu, dtype=torch.float32, device=self.device)
        tv_norm = tv.norm()
        max_speed = torch.tensor(1.0, device=self.device)
        if tv_norm > 1e-4:
            tv_clamped = (tv / tv_norm) * torch.minimum(tv_norm, max_speed)
        else:
            tv_clamped = torch.zeros(3, device=self.device)
        target_v_body = R_yaw.T @ tv_clamped  # (3,)

        # Body-up axis in world frame (third column of R)
        body_up = R[:, 2]  # (3,)

        # Local velocity in yaw-aligned body frame
        v_world = torch.tensor(velocity_enu, dtype=torch.float32, device=self.device)
        local_v = R_yaw.T @ v_world  # (3,)

        margin_t = torch.tensor([[margin]], dtype=torch.float32, device=self.device)

        if self.no_odom:
            state = torch.cat([target_v_body, body_up, margin_t.squeeze(0)]).unsqueeze(0)  # (1,7)
        else:
            state = torch.cat([local_v, target_v_body, body_up, margin_t.squeeze(0)]).unsqueeze(0)  # (1,10)

        # Depth image: (1, 1, H, W) — blank = no obstacles
        depth = self._blank_depth

        act_raw, _, self.h = self.model(depth, state, self.h)  # (1, 6)

        # Decode: rotate 6D output by yaw frame, split into a_pred and v_pred
        act_6d = act_raw.squeeze(0)  # (6,)
        R_yaw_batch = R_yaw.unsqueeze(0)  # (1,3,3)
        decoded = (R_yaw_batch @ act_6d.reshape(1, 3, 2))  # (1,3,2)
        a_pred, v_pred = decoded.squeeze(0).unbind(-1)  # each (3,)

        # Final thrust acceleration command (no thrust estimation noise at inference)
        act_world = (a_pred - v_pred - self.g) + self.g  # simplifies to a_pred - v_pred
        # Equivalently: act_world = a_pred - v_pred (thr_est_error=1 at inference)

        return act_world.cpu().numpy()


# ---------------------------------------------------------------------------
# MAVLink helpers
# ---------------------------------------------------------------------------

def wait_for_heartbeat(mav, timeout=30):
    print("Waiting for heartbeat...")
    mav.wait_heartbeat(timeout=timeout)
    print(f"Heartbeat received from system {mav.target_system} component {mav.target_component}")


def set_offboard_mode(mav):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        PX4_CUSTOM_MAIN_MODE_OFFBOARD,
        0, 0, 0, 0, 0,
    )


def arm(mav):
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0,
    )


def send_attitude_target(mav, q_wxyz: np.ndarray, thrust: float):
    """Send SET_ATTITUDE_TARGET (msg id 82)."""
    # type_mask: ignore body rates (0b00000111 = 7)
    mav.mav.set_attitude_target_send(
        int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
        mav.target_system,
        mav.target_component,
        7,  # type_mask: ignore roll/pitch/yaw rate
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,  # body rates (ignored)
        float(thrust),
    )


def send_heartbeat(mav):
    mav.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0,
    )


def set_param_float(mav, param_id: str, value: float):
    """Set a PX4 float parameter via MAVLink PARAM_SET."""
    mav.mav.param_set_send(
        mav.target_system,
        mav.target_component,
        param_id.encode("utf-8"),
        value,
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
    parser.add_argument("--checkpoint", required=True, help="Path to DiffPhysDrone model checkpoint (.pt)")
    parser.add_argument("--connect", default="udp:localhost:14550", help="MAVLink connection string")
    parser.add_argument("--no-odom", action="store_true", help="Use 7D obs (no velocity in state)")
    parser.add_argument("--target-vx", type=float, default=0.0, help="Target velocity x (ENU) m/s")
    parser.add_argument("--target-vy", type=float, default=0.0, help="Target velocity y (ENU) m/s")
    parser.add_argument("--target-vz", type=float, default=0.0, help="Target velocity z (ENU) m/s")
    args = parser.parse_args()

    target_vel = np.array([args.target_vx, args.target_vy, args.target_vz])

    # Connect
    print(f"Connecting to {args.connect} ...")
    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)

    state = DroneState()
    stop_event = threading.Event()

    recv_thread = threading.Thread(target=receive_loop, args=(mav, state, stop_event), daemon=True)
    recv_thread.start()

    # Load policy
    print(f"Loading policy from {args.checkpoint} ...")
    policy = DiffDronePolicy(args.checkpoint, no_odom=args.no_odom)

    hover_q = np.array([1.0, 0.0, 0.0, 0.0])  # level attitude in NED/FRD
    hover_thrust = float(np.clip(MASS_KG * 9.80665 / (MASS_KG * MAX_ACCEL), 0.0, 1.0))

    # Tell PX4 what hover throttle to expect so its internal throttle curve
    # maps our normalized thrust correctly.
    print(f"Setting PX4 MPC_THR_HOVER = {hover_thrust:.3f} ...")
    set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
    time.sleep(0.2)

    # Send a few attitude targets before arming so PX4 accepts OFFBOARD mode
    print("Pre-arming: sending attitude setpoints to satisfy PX4 OFFBOARD pre-condition...")
    for _ in range(30):
        send_attitude_target(mav, hover_q, hover_thrust)
        send_heartbeat(mav)
        time.sleep(0.05)

    print("Setting OFFBOARD mode...")
    set_offboard_mode(mav)
    time.sleep(0.5)

    print("Arming...")
    arm(mav)
    time.sleep(1.0)

    # Warm up the GRU by running the policy for 1 second while holding a safe
    # hover command. This prevents the cold-start garbage output from reaching
    # the motors immediately after arming.
    WARMUP_SECS = 1.0
    print(f"Warming up GRU for {WARMUP_SECS}s (holding hover, not sending policy output)...")
    policy.reset()
    pos, vel, R_enu, yaw = state.get()
    for _ in range(int(WARMUP_SECS * CONTROL_HZ)):
        policy.step(pos, vel, R_enu, target_vel)  # run but discard output
        time.sleep(1.0 / CONTROL_HZ)

    print("Starting policy loop at 15 Hz. Ctrl+C to stop.")

    control_dt = 1.0 / CONTROL_HZ
    heartbeat_dt = 1.0 / HEARTBEAT_HZ
    last_heartbeat = time.time()
    start_time = time.time()
    next_step = time.time()
    step_count = 0

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            # Heartbeat keepalive
            if now - last_heartbeat >= heartbeat_dt:
                send_heartbeat(mav)
                last_heartbeat = now

            # Control step
            if now >= next_step:
                pos, vel, R_enu, yaw = state.get()

                # Current attitude as RPY for logging
                cur_rpy = Rotation.from_matrix(R_enu).as_euler("xyz", degrees=True)

                act_world = policy.step(pos, vel, R_enu, target_vel)

                # Verbose every step for first 5 seconds, then at 1 Hz
                verbose = elapsed < 5.0 or (int(now) != int(now - control_dt))
                q_des, thrust_norm = act_world_to_attitude_target(act_world, yaw, verbose=verbose)
                send_attitude_target(mav, q_des, thrust_norm)

                if verbose:
                    print(
                        f"[t={elapsed:.2f}s step={step_count}] "
                        f"pos={pos.round(2)}  vel={vel.round(2)}\n"
                        f"  cur_att RPY(ENU) = roll={cur_rpy[0]:.1f}° pitch={cur_rpy[1]:.1f}° yaw={cur_rpy[2]:.1f}°\n"
                        f"  state.armed={state.armed}  state.offboard={state.offboard}\n"
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
        # Disarm
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )


if __name__ == "__main__":
    main()
