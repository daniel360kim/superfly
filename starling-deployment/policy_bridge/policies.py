from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

from diffaero_offboard import DroneState
from policy_bridge.px4_setpoint import PX4Setpoint, AttitudeSetpoint


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PolicyBase(ABC):
    """Abstract base class for all PX4 offboard policies.

    Every concrete policy must implement three members:

    * ``setpoint_type`` — class-level declaration of which ``PX4Setpoint``
      subclass ``step()`` returns.  The bridge reads this at init to verify
      it knows how to send that setpoint type before any flight code runs.

    * ``hover_thrust`` — normalized [0, 1] throttle that produces 1-g of
      thrust on this airframe.  The bridge writes it to ``MPC_THR_HOVER``
      before arming so PX4's mixer is calibrated to the policy's thrust scale.

    * ``reset()`` — clears any per-episode state (RNN hidden states, EMA
      accumulators, etc.).  Called by the bridge before handing control to
      the policy after the CLIMB phase.

    * ``step()`` — runs one inference step and returns a typed ``PX4Setpoint``
      that the bridge dispatches to the appropriate MAVLink sender.
    """

    #: Subclasses must override with the concrete PX4Setpoint type they return.
    setpoint_type: type = NotImplemented

    @property
    @abstractmethod
    def hover_thrust(self) -> float:
        """Normalized throttle [0, 1] at which this airframe hovers (≈ g / max_accel).

        The bridge sets PX4's ``MPC_THR_HOVER`` to this value before arming so
        that the normalized thrust sent in ``SET_ATTITUDE_TARGET`` is correctly
        scaled to physical thrust on the real motor/ESC stack.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-episode internal state.

        Called once by the bridge immediately before the POLICY phase begins
        (i.e. after the CLIMB phase has settled).  Implementations must reset
        any stateful accumulators — EMA buffers, RNN hidden states, integrators
        — so the first ``step()`` call starts from a clean slate.
        """
        ...

    @abstractmethod
    def step(
        self,
        state: DroneState,
        goal_enu: np.ndarray,
        perception: np.ndarray | None,
    ) -> PX4Setpoint:
        """Run one policy inference step and return a PX4 setpoint.

        Args:
            state:      Current drone state read from the MAVLink receive thread.
                        Contains position, velocity, rotation matrix, and yaw,
                        all expressed in ENU / FLU convention.
            goal_enu:   Desired goal position in the ENU world frame, shape (3,).
            perception: Raw sensor frame, or ``None`` if unavailable.
                        Interpretation is policy-specific (e.g. Euclidean range
                        image for DiffAero, RGB for an end-to-end vision policy).

        Returns:
            A ``PX4Setpoint`` instance whose concrete type matches
            ``self.setpoint_type``.  The bridge dispatches it to the
            appropriate MAVLink sender without inspecting its internals.
        """
        ...


class DiffAeroPolicy(PolicyBase):
    """PX4 policy wrapper for the exported DiffAero SHA2C point-mass actor.

    Loads a TorchScript ``.pt2`` checkpoint and translates between the
    DroneState / PX4Setpoint interface expected by ``PolicyBridge`` and the
    internal observation / action conventions used during DiffAero training:

    Observation (9-D state + depth image):
        * ``target_vel_local`` (3): desired velocity toward goal, projected into
          the yaw-only (Rz) frame.
        * ``uz`` (3): body up-axis expressed in the world frame.
        * ``v_local`` (3): current velocity projected into the Rz frame.
        * ``perception`` (H×W): inverse-depth image,
          ``1 - clamp(range, 0, max_dist) / max_dist``.

    Action:
        The actor returns ``(acc_cmd, quat_xyzw_enu_flu, acc_norm)``.
        ``acc_cmd`` (3-D world-frame thrust acceleration) is debug-only.
        ``quat_xyzw_enu_flu`` and ``acc_norm`` are converted to an
        ``AttitudeSetpoint`` in NED/FRD with normalized thrust.

    Yaw:
        Commanded yaw aligns with a low-pass EMA of the current velocity
        (``align_yaw_with_vel_ema``), matching the training convention baked
        into the exported actor.
    """

    #: This policy always produces an attitude + thrust command.
    setpoint_type: type = AttitudeSetpoint

    # Fixed frame-conversion rotations (constructed once at class definition).
    # ENU inertial → NED inertial (same convention as Pegasus Simulator).
    _rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
    # FLU body → FRD body (+π around X).
    _rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])

    def __init__(
        self,
        checkpoint_path: str,
        vel_ema_factor: float = 0.1,
        max_acc_xy: float = 20.0,
        max_acc_z: float = 40.0,
        max_vel: float = 5.0,
        max_accel: float = 30.0,
        cam_max_dist: float = 5.0,
        depth_h: int = 9,
        depth_w: int = 16,
    ):
        """Load the TorchScript actor and configure inference hyperparameters.

        Args:
            checkpoint_path: Path to either the ``.pt2`` file directly or a
                checkpoint directory containing
                ``checkpoints/exported_actor.pt2`` or ``exported_actor.pt2``.
            vel_ema_factor:  EMA smoothing factor α for the velocity-derived
                yaw command: ``ema = lerp(ema, v_world, α)``.  Smaller values
                give a slower, smoother yaw response.  Default 0.1.
            max_acc_xy:      Action-space upper bound for horizontal thrust
                acceleration [m/s²].  Must match the training config
                ``env.max_acc.xy``.  Default 20.0.
            max_acc_z:       Action-space upper bound for vertical thrust
                acceleration [m/s²].  Must match ``env.max_acc.z``.
                Default 40.0.
            max_vel:         Speed [m/s] at which the target-velocity heuristic
                saturates.  The drone is commanded to approach the goal at this
                speed and slow down proportionally inside one second of travel
                time.  Default 5.0.
            max_accel:       Thrust acceleration [m/s²] that corresponds to
                normalized throttle = 1.0.  Sets the hover throttle via
                ``g / max_accel`` and clips ``acc_norm`` before sending.
                Default 30.0.
            cam_max_dist:    Sensor maximum range [m] used to invert depth:
                ``perception = 1 - clamp(range, 0, cam_max_dist) / cam_max_dist``.
                Must match ``sensor.max_dist`` in the training config.
                Default 5.0.
            depth_h:         Depth image height in pixels.  Default 9.
            depth_w:         Depth image width in pixels.  Default 16.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pt2_path = self._resolve_pt2(checkpoint_path)
        print(f"Loading DiffAero TorchScript actor from {pt2_path} ...")
        self.module = torch.jit.load(str(pt2_path), map_location=self.device)
        self.module.eval()

        self.min_action = torch.tensor(
            [[-max_acc_xy, -max_acc_xy, 0.0]], dtype=torch.float32, device=self.device
        )
        self.max_action = torch.tensor(
            [[max_acc_xy, max_acc_xy, max_acc_z]], dtype=torch.float32, device=self.device
        )

        self.vel_ema_factor = vel_ema_factor
        self.vel_ema: torch.Tensor | None = None
        self.max_vel = max_vel
        self.max_accel = max_accel

        self.cam_max_dist = cam_max_dist
        self.depth_h = depth_h
        self.depth_w = depth_w

        self._up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

    @property
    def hover_thrust(self) -> float:
        """Normalized throttle at which this airframe hovers.

        Computed as ``g / max_accel`` so that when the policy commands
        ``acc_norm ≈ g``, the normalized thrust sent to PX4 matches
        ``MPC_THR_HOVER`` and the drone holds altitude without integrator
        windup.
        """
        return float(np.clip(9.80665 / self.max_accel, 0.0, 1.0))

    def reset(self) -> None:
        """Clear the velocity EMA accumulated from the previous episode.

        Must be called before the first ``step()`` of each flight segment so
        the yaw command does not carry over stale velocity history from a
        previous run or the CLIMB phase.
        """
        self.vel_ema = None

    @torch.no_grad()
    def step(
        self,
        state: DroneState,
        goal_enu: np.ndarray,
        perception: np.ndarray | None,
    ) -> AttitudeSetpoint:
        """Run one DiffAero inference step at 30 Hz.

        Builds the 9-D state observation and depth perception tensor from the
        current drone state, runs the TorchScript actor, and converts the
        output attitude quaternion and thrust acceleration into an
        ``AttitudeSetpoint`` in PX4's NED/FRD convention.

        Args:
            state:      Current drone state (ENU/FLU).
            goal_enu:   Goal position in ENU, shape (3,).
            perception: Euclidean range image [m], shape (H, W), or ``None``
                        to substitute a zero (all-clear) depth frame.

        Returns:
            ``AttitudeSetpoint`` containing the desired NED/FRD quaternion
            ``[w, x, y, z]`` and normalized thrust in ``[0, 1]``.
        """
        pos, vel, R_enu, _ = state.get()

        R = torch.tensor(R_enu, dtype=torch.float32, device=self.device)   # (3, 3)
        v_world = torch.tensor(vel, dtype=torch.float32, device=self.device)

        Rz = self._build_yaw_frame(R)           # (3, 3) yaw-only rotation
        uz = R[:, 2]                            # body up-axis in world frame

        target_vel_world = self._compute_target_vel(goal_enu, pos)

        # Project velocities into the yaw-only (local) frame to match the
        # obs_frame=local convention used during training.
        target_vel_local = Rz.t() @ target_vel_world   # (3,)
        v_local = Rz.t() @ v_world                     # (3,)
        state9 = torch.cat([target_vel_local, uz, v_local]).unsqueeze(0)  # (1, 9)

        # Update velocity EMA; used by the actor to derive commanded yaw.
        if self.vel_ema is None:
            self.vel_ema = v_world.clone()
        else:
            self.vel_ema = torch.lerp(self.vel_ema, v_world, self.vel_ema_factor)

        # Fall back to current heading when nearly stationary (yaw undefined).
        orientation = self.vel_ema.unsqueeze(0)     # (1, 3)
        if orientation.norm() < 1e-3:
            orientation = Rz[:, 0].unsqueeze(0)    # (1, 3)

        perception_t = (
            self._normalize_depth(perception)
            if perception is not None
            else torch.zeros(1, self.depth_h, self.depth_w, device=self.device)
        )

        _, quat_cmd, acc_norm = self.module(
            (state9, perception_t),
            orientation,
            Rz.unsqueeze(0),
            self.min_action,
            self.max_action,
        )

        q_des, thrust_norm = self._to_attitude_setpoint(
            quat_cmd.squeeze(0).cpu().numpy(),
            float(acc_norm.reshape(-1)[0].cpu()),
        )
        return AttitudeSetpoint(q_des, thrust_norm)


    def _build_yaw_frame(self, R: torch.Tensor) -> torch.Tensor:
        """Build the yaw-only rotation matrix Rz from the full attitude R.

        Strips pitch and roll from the body forward axis, then constructs an
        orthonormal frame whose columns are [forward, left, up] in the world
        frame — matching DiffAero's ``axis_rotmat('Z', yaw)`` convention.

        Args:
            R: Full body-to-world rotation matrix, shape (3, 3).

        Returns:
            Rz: Yaw-only rotation matrix, shape (3, 3).
        """
        fwd = R[:, 0].clone()
        fwd[2] = 0.0
        fwd = torch.nn.functional.normalize(fwd, dim=0)
        left = torch.cross(self._up, fwd, dim=0)
        return torch.stack([fwd, left, self._up], dim=1)  # (3, 3)

    def _compute_target_vel(
        self, goal_enu: np.ndarray, position: np.ndarray
    ) -> torch.Tensor:
        """Compute the desired world-frame velocity toward the goal.

        Uses the heuristic: ``target_vel = (goal - pos) / max(dist / max_vel, 1)``.
        This saturates approach speed to ``max_vel`` far from the goal and
        smoothly decelerates to zero as the drone arrives, without requiring a
        separate trajectory planner.

        Args:
            goal_enu:  Goal position in ENU, shape (3,).
            position:  Current drone position in ENU, shape (3,).

        Returns:
            target_vel_world: Desired velocity vector in the ENU world frame,
            shape (3,).  Magnitude is at most ``max_vel`` [m/s].
        """
        rel = (
            torch.tensor(goal_enu, dtype=torch.float32, device=self.device)
            - torch.tensor(position, dtype=torch.float32, device=self.device)
        )
        dist = rel.norm()
        mv = torch.tensor(float(self.max_vel), device=self.device)
        denom = torch.maximum(dist / mv, torch.ones((), device=self.device))
        return rel / denom

    def _normalize_depth(self, range_m: np.ndarray) -> torch.Tensor:
        """Convert a metric Euclidean range image to the DiffAero perception encoding.

        DiffAero trains with inverted, normalized depth:
        ``perception = 1 - clamp(range, 0, max_dist) / max_dist``.
        This maps close obstacles to values near 1 and free space (or
        no-return pixels) to 0, matching the ``sensor.encoding = inverse``
        convention in the training config.

        Args:
            range_m: Euclidean range image [m], shape (H, W).

        Returns:
            Normalized perception tensor, shape (1, H, W), on ``self.device``.
        """
        d = torch.as_tensor(range_m, dtype=torch.float32, device=self.device)
        d = d.clamp(0.0, self.cam_max_dist)
        return (1.0 - d / self.cam_max_dist).reshape(1, self.depth_h, self.depth_w)

    def _to_attitude_setpoint(
        self, quat_xyzw_enu_flu: np.ndarray, acc_norm: float
    ) -> tuple[np.ndarray, float]:
        """Convert DiffAero actor output to PX4 attitude + thrust.

        Performs two conversions:
        1. **Frame**: ENU/FLU quaternion (xyzw, scipy convention) →
           NED/FRD quaternion (wxyz, MAVLink convention) via the fixed
           ENU→NED and FLU→FRD rotations.
        2. **Thrust**: ``acc_norm`` [m/s²] → normalized throttle [0, 1]
           by dividing by ``max_accel``, so hover (≈ g) maps to
           ``MPC_THR_HOVER = g / max_accel``.

        Args:
            quat_xyzw_enu_flu: Desired attitude quaternion in ENU/FLU,
                               xyzw order (scipy convention), shape (4,).
            acc_norm:          Thrust-acceleration magnitude [m/s²] the
                               rotors must produce (gravity not included;
                               point-mass model handles gravity separately).

        Returns:
            q_des:       Desired quaternion in NED/FRD, wxyz order, shape (4,).
            thrust_norm: Normalized thrust in [0, 1].
        """
        R_des_enu = Rotation.from_quat(quat_xyzw_enu_flu).as_matrix()
        q_des = self._quat_ENU_FLU_to_NED_FRD(R_des_enu)
        thrust_norm = float(np.clip(acc_norm / self.max_accel, 0.0, 1.0))
        return q_des, thrust_norm

    @staticmethod
    def _quat_ENU_FLU_to_NED_FRD(R_enu_flu: np.ndarray) -> np.ndarray:
        """Convert an ENU/FLU rotation matrix to a NED/FRD quaternion.

        Applies the fixed frame change:
        ``R_ned_frd = R_ENU_to_NED @ R_enu_flu @ R_FLU_to_FRD``
        and returns the result in MAVLink wxyz order.

        Args:
            R_enu_flu: Rotation matrix in ENU/FLU convention, shape (3, 3).

        Returns:
            Quaternion [w, x, y, z] in NED/FRD convention, shape (4,).
        """
        rot = (
            DiffAeroPolicy._rot_ENU_to_NED
            * Rotation.from_matrix(R_enu_flu)
            * DiffAeroPolicy._rot_FLU_to_FRD
        )
        q = rot.as_quat()   # scipy: [x, y, z, w]
        return np.array([q[3], q[0], q[1], q[2]])  # MAVLink: [w, x, y, z]

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
            f"Tried: {[str(c) for c in candidates]}"
        )