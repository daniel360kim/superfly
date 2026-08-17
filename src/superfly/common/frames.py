"""Frame conversions shared by every offboard controller and the sim.

Conventions (identical across all methods -- extracted verbatim from the
per-offboard copies, which had drifted into 4 identical declarations):
  * World: ENU (x East, y North, z Up); PX4 speaks NED (x North, y East,
    z Down).
  * Body: FLU (x Forward, y Left, z Up); PX4 speaks FRD.
  * ENU yaw is math-convention (CCW from East, atan2(North-ish fwd_y, fwd_x));
    NED yaw is a compass heading (CW from North) = pi/2 - yaw_enu.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

# ENU inertial -> NED inertial: same rotation Pegasus uses.
ROT_ENU_TO_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
# FLU body -> FRD body: +PI around X.
ROT_FLU_TO_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def rotation_matrix_ENU_FLU_to_NED_FRD(R_enu_flu: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix from ENU/FLU convention to NED/FRD."""
    rot = ROT_ENU_TO_NED * Rotation.from_matrix(R_enu_flu) * ROT_FLU_TO_FRD
    return rot.as_matrix()


def quat_ENU_FLU_to_NED_FRD(R_enu_flu: np.ndarray) -> np.ndarray:
    """Return [w, x, y, z] quaternion in NED/FRD for a given ENU/FLU rotation matrix."""
    rot = ROT_ENU_TO_NED * Rotation.from_matrix(R_enu_flu) * ROT_FLU_TO_FRD
    q = rot.as_quat()  # [x, y, z, w] scipy convention
    return np.array([q[3], q[0], q[1], q[2]])  # -> [w, x, y, z] MAVLink convention


def rotation_NED_FRD_to_ENU_FLU(q_wxyz_msg) -> Rotation:
    """MAVLink ATTITUDE_QUATERNION ([w,x,y,z] NED/FRD) -> scipy Rotation in ENU/FLU."""
    w, x, y, z = q_wxyz_msg
    q_ned_frd = Rotation.from_quat([x, y, z, w])  # scipy [x,y,z,w]
    return ROT_ENU_TO_NED.inv() * q_ned_frd * ROT_FLU_TO_FRD.inv()


def enu_vel_to_ned(vel_enu: np.ndarray):
    """ENU velocity -> NED components (vx=North, vy=East, vz=Down)."""
    return float(vel_enu[1]), float(vel_enu[0]), float(-vel_enu[2])


def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_enu_to_ned(yaw_enu: float) -> float:
    """ENU heading (CCW from East) -> NED compass heading (CW from North)."""
    return wrap_pi(math.pi / 2.0 - yaw_enu)
