from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import torch

from wrapper.perception_builder import PerceptionBuilder, Intrinsics

@dataclass
class DiffAeroObs:
    position_enu:  np.ndarray   # (3,) ENU world
    velocity_enu:  np.ndarray   # (3,) ENU world — MEASURED (never finite-differenced)
    R_enu:         np.ndarray   # (3,3) FLU-body -> ENU world rotation matrix
    goal_enu:      np.ndarray   # (3,) ENU world
    depth_planar:  np.ndarray | None   # (H,W) metric planar (image-plane) depth raw from sim (PerceptionBuilder handles conversion to DiffAero perception encoding)

@dataclass
class DiffAeroCmd:
    attitude_ned_frd_wxyz: np.ndarray  # PX4-ready
    attitude_enu_flu_xyzw: np.ndarray  # debug / sim
    thrust_norm: float                 # [0,1], anchored to hover_thrust
    acc_cmd_enu: np.ndarray            # debug
    acc_norm: float
   

class DiffAeroPolicy:
    # Fixed frame-conversion rotations (constructed once at class definition).
    # ENU inertial → NED inertial (same convention as Pegasus Simulator).
    _rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
    # FLU body → FRD body (+π around X).
    _rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])

    def __init__(
        self,
        intrinsics: Intrinsics,
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
        self.perception_builder = PerceptionBuilder(intrinsics, out_h=depth_h, out_w=depth_w, target_fov_deg=86.0, max_dist=cam_max_dist)
        self._up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        """Clear the velocity EMA accumulated from the previous episode."""
        self.vel_ema = None

    @torch.no_grad()
    def compute(
        self,
        obs: DiffAeroObs,
    ) -> DiffAeroCmd:

        R = torch.tensor(obs.R_enu, dtype=torch.float32, device=self.device)   # (3, 3)
        v_world = torch.tensor(obs.velocity_enu, dtype=torch.float32, device=self.device)

        Rz = self._build_yaw_frame(R)           # (3, 3) yaw-only rotation
        uz = R[:, 2]                            # body up-axis in world frame

        target_vel_world = self._compute_target_vel(obs.goal_enu, obs.position_enu)

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

        if obs.depth_planar is not None:
            perception = self.perception_builder(obs.depth_planar)
            perception_t = torch.tensor(perception, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            perception_t = torch.zeros(
                1, self.perception_builder.out_h, self.perception_builder.out_w,
                dtype=torch.float32, device=self.device,
            )

        acc_cmd, quat_cmd, acc_norm = self.module(
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
        
        return DiffAeroCmd(
            attitude_ned_frd_wxyz=q_des,
            attitude_enu_flu_xyzw=quat_cmd.squeeze(0).cpu().numpy(),
            thrust_norm=thrust_norm,
            acc_cmd_enu=acc_cmd.squeeze(0).cpu().numpy(),
            acc_norm=float(acc_norm.reshape(-1)[0].cpu()),
        )
            
        


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
        left = torch.nn.functional.normalize(left, dim=0)
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