"""Shared PX4 MAVLink offboard layer: state estimation mirror, mode/arming,
setpoint senders, and the receive thread.

Extracted from diffdrone_offboard.py (the canonical copy the other offboards
either imported or re-declared verbatim). Depends only on numpy + scipy +
pymavlink, so it imports cleanly inside every method venv.

The per-method phase logic (CLIMB/YAW/POLICY/LANDING loops) deliberately
stays in each scripts/*_offboard.py: the loops differ in documented,
load-bearing ways (depthnav has no YAW/LANDING phase; agile requests stream
rates; the vel offboards climb on velocity setpoints) and a shared
state-machine abstraction cannot be validated without hardware. Only code
that was literally identical across offboards lives here.
"""

import math
import threading
import time

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from superfly.common.frames import ROT_ENU_TO_NED, ROT_FLU_TO_FRD

# --- vehicle constants shared by every offboard (PX4 sees the same drone) ---
MASS_KG = 1.5
MAX_ACCEL = 20.0   # thrust-acceleration [m/s^2] that maps to full throttle
G = 9.80665
HEARTBEAT_HZ = 2.0

# PX4 custom mode for OFFBOARD.
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# PX4 message stream rates requested by request_stream_rates (agile needs
# this; see that function's docstring).
STREAM_HZ = 50.0
STREAMED_MSGS = {
    mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: "LOCAL_POSITION_NED",
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION: "ATTITUDE_QUATERNION",
}


class DroneState:
    """Thread-safe mirror of the drone state, updated by receive_loop.

    Superset of the per-offboard variants: position_valid (the vel offboards
    gate on it), angular_rate_body + msg-rate counters (agile's MPC needs
    them; the counters are cheap enough to keep for everyone -- catching a
    silently ignored SET_MESSAGE_INTERVAL is the first step of any
    oscillation debugging)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.position_enu = np.zeros(3)      # ENU position [m]
        self.velocity_enu = np.zeros(3)      # ENU linear velocity [m/s]
        self.R_enu = np.eye(3)               # ENU/FLU rotation matrix
        self.angular_rate_body = np.zeros(3)  # FLU body rates [rad/s]
        self.yaw = 0.0                       # ENU math-convention yaw [rad]
        self.armed = False
        self.offboard = False
        self.position_valid = False
        self.last_update = 0.0
        self.msg_counts = {name: 0 for name in STREAMED_MSGS.values()}

    def update_from_attitude(self, msg):
        """Update from ATTITUDE_QUATERNION message (NED/FRD quaternion)."""
        # MAVLink quaternion is [w, x, y, z] -> scipy [x, y, z, w].
        q_ned_frd = Rotation.from_quat([msg.q2, msg.q3, msg.q4, msg.q1])
        rot_enu_flu = ROT_ENU_TO_NED.inv() * q_ned_frd * ROT_FLU_TO_FRD.inv()
        with self._lock:
            self.R_enu = rot_enu_flu.as_matrix()
            # Yaw: angle of body-x projection onto the ENU XY plane.
            fwd_enu = self.R_enu[:, 0]
            self.yaw = math.atan2(fwd_enu[1], fwd_enu[0])
            w_frd = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed],
                             dtype=np.float64)
            self.angular_rate_body = np.array([w_frd[0], -w_frd[1], -w_frd[2]])
            self.msg_counts["ATTITUDE_QUATERNION"] += 1
        self.last_update = time.time()

    def update_from_local_position(self, msg):
        """Update from LOCAL_POSITION_NED message (NED -> ENU)."""
        with self._lock:
            self.position_enu = np.array([msg.y, msg.x, -msg.z])
            self.velocity_enu = np.array([msg.vy, msg.vx, -msg.vz])
            self.position_valid = True
            self.msg_counts["LOCAL_POSITION_NED"] += 1
        self.last_update = time.time()

    def update_from_heartbeat(self, msg):
        with self._lock:
            self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            # PX4 custom mode: offboard = main mode 6.
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

    def get_full(self):
        """Like get(), plus the FLU body angular rate (agile's MPC state)."""
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
# MAVLink helpers
# ---------------------------------------------------------------------------

def wait_for_heartbeat(mav, timeout=120):
    print("Waiting for heartbeat...")
    mav.wait_heartbeat(timeout=timeout)
    print(f"Heartbeat received from system {mav.target_system} "
          f"component {mav.target_component}")


def wait_for_position(state: DroneState, timeout: float = 30.0,
                      raise_on_timeout: bool = False):
    """Block until the EKF has produced a local position estimate."""
    print("Waiting for LOCAL_POSITION_NED...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if state.position_valid:
            pos, _, _, _ = state.get()
            print(f"Position estimate ready: ENU pos={pos.round(2)}")
            return True
        time.sleep(0.05)
    if raise_on_timeout:
        raise TimeoutError("Timed out waiting for LOCAL_POSITION_NED")
    print("WARNING: no position estimate before timeout; continuing anyway.")
    return False


def request_stream_rates(mav, stream_hz=STREAM_HZ):
    """Ask PX4 to stream the state messages at stream_hz on THIS link.

    udp:14550 is PX4's GCS link, whose defaults stream position at 1 Hz and
    attitude at 10 Hz. Learned reactive policies tolerate that staleness; a
    stiff model-based tracker (agile's MPC) re-solving at 30 Hz on
    second-stale state produces a 0.5 Hz +-30 deg attitude limit cycle.
    Should be >= the control rate so each solve sees fresh state."""
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


def disarm(mav):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0,
    )


def retry_offboard_arm(mav, state, last_try_t, interval=2.0):
    """Re-request OFFBOARD mode + arming until PX4 accepts both.

    The one-shot mode/arm commands at startup are silently rejected if the EKF
    hasn't converged yet (PX4 denies the command and only prints 'Ready for
    takeoff!' seconds later) -- observed on big USD stages, where Isaac loads
    long past the harness warmup and PX4's EKF settles well after the first
    heartbeat, leaving the offboard streaming CLIMB setpoints disarmed forever.
    Call every control tick during CLIMB: while not armed+offboard, the
    commands are re-sent every `interval` seconds. Returns updated last_try_t."""
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
    """Send SET_ATTITUDE_TARGET (msg id 82), ignoring body rates."""
    mav.mav.set_attitude_target_send(
        int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
        mav.target_system, mav.target_component,
        7,  # type_mask: ignore roll/pitch/yaw rate
        [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        0.0, 0.0, 0.0,  # body rates (ignored)
        float(thrust),
    )


def send_position_target_ned(mav, x_n: float, y_e: float, z_d: float, yaw: float = 0.0):
    """Send SET_POSITION_TARGET_LOCAL_NED (msg id 84), position + yaw only.

    Coordinates are NED (z DOWN, so 10 m altitude => z_d = -10). type_mask
    ignores velocity, accel, and yaw_rate -- we command position + yaw."""
    # type_mask bits (SET => IGNORE that field):
    #   x=1,y=2,z=4, vx=8,vy=16,vz=32, ax=64,ay=128,az=256, force=512,
    #   yaw=1024, yaw_rate=2048.
    IGNORE_VEL = 8 | 16 | 32
    IGNORE_ACC = 64 | 128 | 256
    IGNORE_YAW_RATE = 2048
    type_mask = IGNORE_VEL | IGNORE_ACC | IGNORE_YAW_RATE  # = 2552
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        float(x_n), float(y_e), float(z_d),
        0.0, 0.0, 0.0,   # velocity (ignored)
        0.0, 0.0, 0.0,   # accel (ignored)
        float(yaw), 0.0,  # yaw, yaw_rate (yaw_rate ignored)
    )


def send_velocity_target_ned(mav, vx_n: float, vy_e: float, vz_d: float, yaw: float):
    """Velocity + yaw setpoint (SET_POSITION_TARGET_LOCAL_NED, msg id 84).

    Position, acceleration and yaw-rate fields are masked off, so PX4 runs its
    velocity controller on (vx, vy, vz) and its yaw controller on `yaw`."""
    IGNORE_POS = 1 | 2 | 4
    IGNORE_ACC = 64 | 128 | 256
    IGNORE_YAW_RATE = 2048
    type_mask = IGNORE_POS | IGNORE_ACC | IGNORE_YAW_RATE
    mav.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0.0, 0.0, 0.0,
        float(vx_n), float(vy_e), float(vz_d),
        0.0, 0.0, 0.0,
        float(yaw), 0.0,
    )


def send_land_command(mav):
    """Command PX4 to land at the current XY position via MAV_CMD_NAV_LAND."""
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, float("nan"),  # abort_alt, precision_mode, empty, yaw (nan=keep)
        0.0, 0.0, 0.0,           # lat, lon, alt (0 = current position)
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
        mav.target_system, mav.target_component,
        param_id.encode("utf-8"), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    print(f"Set param {param_id} = {value}")


def receive_loop(mav, state: DroneState, stop_event: threading.Event):
    """MAVLink receive thread body: mirrors PX4 state into DroneState."""
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
