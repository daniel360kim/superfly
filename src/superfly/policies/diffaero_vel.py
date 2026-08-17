"""DiffAero velocity-command policy wrapper for PX4 offboard control.

Loads a TorchScript actor exported with ``action_is_velocity=true`` (see
``diffaero/utils/exporter.py``). The actor returns a world-frame velocity
setpoint ``[vx, vy, vz]`` in ENU; before sending to PX4 the output is
per-axis clamped to the deployed cruise limits and passed through the same
first-order velocity lag used by ``VelocityPointMassModel`` in training
(``lmbda`` from the checkpoint hydra config).

Supports both full 3-D velocity policies (``planar: false``) and planar
policies (``planar: true``) that output horizontal ``[vx, vy]`` only; the
exporter pads ``vz=0`` and the deploy bridge holds altitude externally.

Observation layout (``obs_frame=local``): ``[target_vel_local(3), v_local(3)]``.
Perception is used when the checkpoint was trained with ``env=obstacle_avoidance``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from superfly.policies.diffaero import DiffAeroObs
from superfly.perception.builder import Intrinsics, PerceptionBuilder, PerceptionGrid


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
        lmbda: float | None = None,
        control_hz: float = 30.0,
        flip_lr: bool = False,
        flip_ud: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = self._resolve_checkpoint_dir(checkpoint_path)
        cfg = self._load_hydra_config(ckpt_dir)
        self._validate_config(cfg)

        self.uses_perception = cfg["env"]["name"] == "obstacle_avoidance"
        dyn = cfg["dynamics"]
        net_cfg = cfg.get("network", {})
        self.network_name = str(net_cfg.get("name", "mlp"))
        self.planar = bool(dyn.get("planar", False))
        self.vel_ema_factor = (
            vel_ema_factor
            if vel_ema_factor is not None
            else float(dyn["vel_ema_factor"]["default"])
        )
        self.lmbda = (
            lmbda if lmbda is not None else float(dyn["lmbda"]["default"])
        )
        self.control_dt = 1.0 / float(control_hz)
        self._vel_lag_alpha = 1.0 - math.exp(-self.lmbda * self.control_dt)
        self.max_yaw_rate_deg = float(dyn.get("max_yaw_rate", {}).get("default", 60.0))
        self.yaw_hold_speed = float(dyn.get("yaw_hold_speed", 0.3))

        if max_vel_xy is None:
            max_vel_xy = (
                float(max_vel)
                if max_vel is not None
                else float(dyn["max_vel"]["xy"]["default"])
            )
        if max_vel_z is None:
            max_vel_z = float(dyn["max_vel"]["z"]["default"])
        if max_vel is None:
            env_cfg = cfg["env"]
            max_vel = float(env_cfg.get("max_target_vel", max_vel_xy))
        self.max_vel_xy = float(max_vel_xy)
        self.max_vel_z = float(max_vel_z)

        pt2_path = self._resolve_pt2(checkpoint_path)
        print(f"Loading DiffAero velocity TorchScript actor from {pt2_path} ...")
        self.module = torch.jit.load(str(pt2_path), map_location=self.device)
        self.module.eval()

        if self.planar:
            self.min_action = torch.tensor(
                [[-self.max_vel_xy, -self.max_vel_xy]],
                dtype=torch.float32, device=self.device,
            )
            self.max_action = torch.tensor(
                [[self.max_vel_xy, self.max_vel_xy]],
                dtype=torch.float32, device=self.device,
            )
        else:
            self.min_action = torch.tensor(
                [[-self.max_vel_xy, -self.max_vel_xy, -self.max_vel_z]],
                dtype=torch.float32, device=self.device,
            )
            self.max_action = torch.tensor(
                [[self.max_vel_xy, self.max_vel_xy, self.max_vel_z]],
                dtype=torch.float32, device=self.device,
            )
        self.max_vel_t = torch.tensor(max_vel, dtype=torch.float32, device=self.device)
        self.vel_ema: torch.Tensor | None = None
        self._vel_setpoint: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        if self.network_name == "rcnn":
            self._hidden_shape = (
                int(net_cfg.get("rnn_n_layers", 1)),
                1,
                int(net_cfg.get("rnn_hidden_dim", 512)),
            )

        self.perception_builder = PerceptionBuilder(
            intrinsics, grid=grid, flip_lr=flip_lr, flip_ud=flip_ud
        )
        self._up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

        mode = "planar horizontal" if self.planar else "full 3-D"
        if self.uses_perception:
            print(f"Checkpoint uses obstacle-avoidance perception ({self.network_name}, {mode}).")
        else:
            print(f"Checkpoint uses state-only observations ({self.network_name}, {mode}).")
        print(
            f"Velocity lag: lmbda={self.lmbda:.2f}, alpha={self._vel_lag_alpha:.4f} "
            f"@ {control_hz:.1f} Hz; action clamp xy=±{self.max_vel_xy:.1f} "
            f"z=±{self.max_vel_z:.1f} m/s.",
            flush=True,
        )
        if self.planar:
            print(
                f"Planar yaw bridge: max_yaw_rate={self.max_yaw_rate_deg:.0f} deg/s, "
                f"yaw_hold_speed={self.yaw_hold_speed:.2f} m/s.",
                flush=True,
            )

    def reset(self) -> None:
        self.vel_ema = None
        self._vel_setpoint = None
        self._hidden = None

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

        orientation = self.vel_ema.unsqueeze(0)
        if orientation.norm() < 1e-3:
            orientation = Rz[:, 0].unsqueeze(0)

        perception_t = self._build_perception(obs)
        vel_cmd_raw = self._run_actor(state6, perception_t, orientation, Rz).squeeze(0)
        vel_cmd_raw = self._clamp_vel_cmd(vel_cmd_raw)
        vel_cmd_enu_t = self._apply_velocity_lag(vel_cmd_raw, v_world)
        if self.planar:
            vel_cmd_enu_t[2] = 0.0
        vel_cmd_enu = vel_cmd_enu_t.cpu().numpy()
        vel_norm = float(np.linalg.norm(vel_cmd_enu))
        yaw_ned = self._yaw_ned_from_vel_ema(Rz)

        return DiffAeroVelCmd(
            vel_cmd_enu=vel_cmd_enu,
            vel_norm=vel_norm,
            yaw_ned=yaw_ned,
        )

    def _build_perception(self, obs: DiffAeroObs) -> torch.Tensor | None:
        if not self.uses_perception:
            return None
        if obs.depth_planar is None:
            return torch.zeros(
                1, self.perception_builder.grid.H, self.perception_builder.grid.W,
                dtype=torch.float32, device=self.device,
            )
        perception = self.perception_builder(obs.depth_planar)
        return torch.tensor(
            perception, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def _run_actor(
        self,
        state6: torch.Tensor,
        perception_t: torch.Tensor | None,
        orientation: torch.Tensor,
        Rz: torch.Tensor,
    ) -> torch.Tensor:
        rz = Rz.unsqueeze(0)
        if self.network_name == "cnn":
            if perception_t is None:
                raise ValueError("CNN checkpoint requires depth perception")
            return self.module(
                state6, perception_t, orientation, rz,
                self.min_action, self.max_action,
            )
        if self.network_name == "rcnn":
            if perception_t is None:
                raise ValueError("RCNN checkpoint requires depth perception")
            if self._hidden is None:
                self._hidden = torch.zeros(
                    self._hidden_shape, dtype=torch.float32, device=self.device
                )
            out, self._hidden = self.module(
                state6, perception_t, orientation, rz,
                self.min_action, self.max_action, self._hidden,
            )
            return out

        # MLP (default): state may be a tuple with perception.
        if self.uses_perception:
            actor_state = (state6, perception_t)
        else:
            actor_state = state6
        return self.module(
            actor_state, orientation, rz, self.min_action, self.max_action,
        )

    def _clamp_vel_cmd(self, vel_cmd: torch.Tensor) -> torch.Tensor:
        if self.planar:
            xy = torch.clamp(
                vel_cmd[:2],
                self.min_action.squeeze(0),
                self.max_action.squeeze(0),
            )
            return torch.cat([xy, torch.zeros(1, device=vel_cmd.device, dtype=vel_cmd.dtype)])
        lo = self.min_action.squeeze(0)
        hi = self.max_action.squeeze(0)
        return torch.clamp(vel_cmd, lo, hi)

    def _apply_velocity_lag(
        self, vel_cmd_raw: torch.Tensor, v_measured: torch.Tensor
    ) -> torch.Tensor:
        """First-order lag matching VelocityPointMassModel training dynamics."""
        if self._vel_setpoint is None:
            self._vel_setpoint = v_measured.clone()
        self._vel_setpoint = torch.lerp(
            self._vel_setpoint, vel_cmd_raw, self._vel_lag_alpha
        )
        return self._vel_setpoint

    def _yaw_ned_from_vel_ema(self, Rz: torch.Tensor) -> float:
        """NED compass heading aligned with the velocity EMA (instantaneous)."""
        return self._desired_yaw_ned_vel_ema(Rz)

    def _desired_yaw_ned_vel_ema(self, Rz: torch.Tensor) -> float:
        """PX4 NED yaw from velocity EMA: atan2(East, North)."""
        if self.vel_ema is not None and float(self.vel_ema[:2].norm()) >= 1e-3:
            return float(torch.atan2(self.vel_ema[0], self.vel_ema[1]).item())
        fwd = Rz[:, 0]
        return float(torch.atan2(fwd[0], fwd[1]).item())

    def slew_yaw_ned_cmd(self, yaw_ned: float, control_dt: float) -> float:
        """Rate-limited yaw toward velocity EMA, matching training dynamics.

        Below ``yaw_hold_speed`` the commanded yaw is held fixed so the drone
        does not spin when nearly stationary (e.g. at the goal).
        """
        if self.vel_ema is None:
            return yaw_ned
        speed_xy = float(self.vel_ema[:2].norm().item())
        if speed_xy < self.yaw_hold_speed:
            return yaw_ned
        desired = float(torch.atan2(self.vel_ema[0], self.vel_ema[1]).item())
        err = math.atan2(
            math.sin(desired - yaw_ned),
            math.cos(desired - yaw_ned),
        )
        max_step = math.radians(self.max_yaw_rate_deg) * control_dt
        return yaw_ned + max(-max_step, min(max_step, err))

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
