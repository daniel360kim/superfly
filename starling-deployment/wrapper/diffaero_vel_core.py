"""DiffAero velocity-command policy wrapper for PX4 offboard control.

Loads a TorchScript actor exported with ``action_is_velocity=true`` (see
``diffaero/utils/exporter.py``). The actor returns a single world-frame
velocity setpoint ``[vx, vy, vz]`` in ENU; PX4 receives NED via the offboard
script.

Observation layout (``obs_frame=local``): ``[target_vel_local(3), v_local(3)]``.
Perception is only used when the checkpoint was trained with ``env=oa``; the
``sha2c_vel_cmd`` run used ``env=pc`` (6-D state only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from wrapper.diffaero_core import DiffAeroObs
from wrapper.perception_builder import Intrinsics, PerceptionBuilder, PerceptionGrid


@dataclass
class DiffAeroVelCmd:
    vel_cmd_enu: np.ndarray  # (3,) world-frame velocity setpoint [m/s]
    vel_norm: float
    yaw_ned: float           # compass heading for PX4 SET_POSITION_TARGET_LOCAL_NED


class DiffAeroVelPolicy:
    def __init__(
        self,
        intrinsics: Intrinsics,
        checkpoint_path: str,
        grid: PerceptionGrid = PerceptionGrid(),
        vel_ema_factor: float | None = None,
        max_vel_xy: float | None = None,
        max_vel_z: float | None = None,
        max_vel: float | None = None,
        flip_lr: bool = False,
        flip_ud: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = self._resolve_checkpoint_dir(checkpoint_path)
        cfg = self._load_hydra_config(ckpt_dir)
        self._validate_config(cfg)

        self.uses_perception = cfg["env"]["name"] == "obstacle_avoidance"
        dyn = cfg["dynamics"]
        self.vel_ema_factor = (
            vel_ema_factor
            if vel_ema_factor is not None
            else float(dyn["vel_ema_factor"]["default"])
        )
        max_vel_xy = max_vel_xy if max_vel_xy is not None else float(dyn["max_vel"]["xy"]["default"])
        max_vel_z = max_vel_z if max_vel_z is not None else float(dyn["max_vel"]["z"]["default"])
        if max_vel is None:
            env_cfg = cfg["env"]
            max_vel = float(env_cfg.get("max_target_vel", max_vel_xy))

        pt2_path = self._resolve_pt2(checkpoint_path)
        print(f"Loading DiffAero velocity TorchScript actor from {pt2_path} ...")
        self.module = torch.jit.load(str(pt2_path), map_location=self.device)
        self.module.eval()

        self.min_action = torch.tensor(
            [[-max_vel_xy, -max_vel_xy, -max_vel_z]], dtype=torch.float32, device=self.device
        )
        self.max_action = torch.tensor(
            [[max_vel_xy, max_vel_xy, max_vel_z]], dtype=torch.float32, device=self.device
        )
        self.max_vel_t = torch.tensor(max_vel, dtype=torch.float32, device=self.device)
        self.vel_ema: torch.Tensor | None = None

        self.perception_builder = PerceptionBuilder(
            intrinsics, grid=grid, flip_lr=flip_lr, flip_ud=flip_ud
        )
        self._up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

        if self.uses_perception:
            print("Checkpoint uses obstacle-avoidance perception (state + depth).")
        else:
            print("Checkpoint uses state-only observations (no depth in the actor).")

    def reset(self) -> None:
        self.vel_ema = None

    @torch.no_grad()
    def compute(self, obs: DiffAeroObs) -> DiffAeroVelCmd:
        R = torch.tensor(obs.R_enu, dtype=torch.float32, device=self.device)
        v_world = torch.tensor(obs.velocity_enu, dtype=torch.float32, device=self.device)
        Rz = self._build_yaw_frame(R)

        target_vel_world = self._compute_target_vel(obs.goal_enu, obs.position_enu)
        target_vel_local = Rz.t() @ target_vel_world
        v_local = Rz.t() @ v_world
        state6 = torch.cat([target_vel_local, v_local]).unsqueeze(0)

        if self.vel_ema is None:
            self.vel_ema = v_world.clone()
        else:
            self.vel_ema = torch.lerp(self.vel_ema, v_world, self.vel_ema_factor)

        # orientation/Rz are required by the exported module signature but unused in velocity mode
        orientation = self.vel_ema.unsqueeze(0)
        if orientation.norm() < 1e-3:
            orientation = Rz[:, 0].unsqueeze(0)

        if self.uses_perception:
            if obs.depth_planar is None:
                perception_t = torch.zeros(
                    1, self.perception_builder.grid.H, self.perception_builder.grid.W,
                    dtype=torch.float32, device=self.device,
                )
            else:
                perception = self.perception_builder(obs.depth_planar)
                perception_t = torch.tensor(
                    perception, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            actor_state = (state6, perception_t)
        else:
            actor_state = state6

        vel_cmd = self.module(
            actor_state,
            orientation,
            Rz.unsqueeze(0),
            self.min_action,
            self.max_action,
        )
        vel_cmd_enu = vel_cmd.squeeze(0).cpu().numpy()
        vel_norm = float(np.linalg.norm(vel_cmd_enu))
        yaw_ned = self._yaw_ned_from_vel_ema(Rz)

        return DiffAeroVelCmd(
            vel_cmd_enu=vel_cmd_enu,
            vel_norm=vel_norm,
            yaw_ned=yaw_ned,
        )

    def _yaw_ned_from_vel_ema(self, Rz: torch.Tensor) -> float:
        """NED compass heading from the velocity EMA (matches training yaw alignment)."""
        if self.vel_ema is not None and self.vel_ema.norm() >= 1e-3:
            # ENU: x=East, y=North → NED yaw = atan2(East, North)
            return float(np.arctan2(self.vel_ema[0].item(), self.vel_ema[1].item()))
        fwd = Rz[:, 0]
        return float(np.arctan2(fwd[0].item(), fwd[1].item()))

    def _build_yaw_frame(self, R: torch.Tensor) -> torch.Tensor:
        fwd = R[:, 0].clone()
        fwd[2] = 0.0
        fwd = torch.nn.functional.normalize(fwd, dim=0)
        left = torch.cross(self._up, fwd, dim=0)
        left = torch.nn.functional.normalize(left, dim=0)
        return torch.stack([fwd, left, self._up], dim=1)

    def _compute_target_vel(
        self, goal_enu: np.ndarray, position: np.ndarray
    ) -> torch.Tensor:
        rel = (
            torch.tensor(goal_enu, dtype=torch.float32, device=self.device)
            - torch.tensor(position, dtype=torch.float32, device=self.device)
        )
        dist = rel.norm()
        denom = torch.maximum(dist / self.max_vel_t, torch.ones((), device=self.device))
        return rel / denom

    @staticmethod
    def _resolve_checkpoint_dir(checkpoint_path: str) -> Path:
        p = Path(checkpoint_path)
        if p.is_file():
            p = p.parent
        if (p / ".hydra" / "config.yaml").exists():
            return p
        if (p.parent / ".hydra" / "config.yaml").exists():
            return p.parent
        raise FileNotFoundError(
            f"Could not find .hydra/config.yaml near checkpoint path {checkpoint_path}"
        )

    @staticmethod
    def _resolve_pt2(checkpoint_path: str) -> Path:
        p = Path(checkpoint_path)
        if p.is_file():
            return p
        for c in (p / "checkpoints" / "exported_actor.pt2", p / "exported_actor.pt2"):
            if c.exists():
                return c
        raise FileNotFoundError(
            f"Could not find exported_actor.pt2 under {checkpoint_path}"
        )

    @staticmethod
    def _load_hydra_config(ckpt_dir: Path) -> dict:
        with open(ckpt_dir / ".hydra" / "config.yaml") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate_config(cfg: dict) -> None:
        dyn = cfg.get("dynamics", {})
        if dyn.get("name") != "velocity_pointmass":
            raise ValueError(
                f"Expected dynamics.name=velocity_pointmass, got {dyn.get('name')!r}"
            )
        if not dyn.get("action_is_velocity", False):
            raise ValueError("Checkpoint dynamics.action_is_velocity is not true")
