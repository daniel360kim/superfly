"""
Dataclasses that define the setpoint commands sent to the PX4. Policy outputs are converted to these dataclasses and sent to the PX4.

PX4 uses NED/FRD convention for velocities and quaternions.
"""

from dataclasses import dataclass
import numpy as np
from typing import Union

@dataclass
class AttitudeSetpoint:
    q_wxyz_ned_frd: np.ndarray #(4,) NED/FRD quaternion
    thrust_norm: float # [0, 1]
    
@dataclass 
class VelocitySetpoint:
    vel_ned: np.ndarray # (3, ) NED velocity
    yaw_ned: float # NED yaw in radians

@dataclass
class PositionSetpoint:
    pos_ned: np.ndarray # (3, ) NED position
    yaw_ned: float # NED yaw in radians
    
PX4Setpoint = Union[AttitudeSetpoint, VelocitySetpoint, PositionSetpoint]
