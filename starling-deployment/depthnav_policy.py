#!/usr/bin/env python
"""
DepthNav policy wrapper for PX4 offboard deployment.

Loads a trained depthnav MultiInputPolicy WITHOUT spinning up the habitat env
(the feature extractor builds purely from an observation_space — extractors.py
reads shapes only), then exposes a `.step()` that mirrors how navigation_env
builds the obs and decodes the action.

Ground-truth config (saved logs/level1/level1.yaml for ckpt level1_4_iteration_13500):
  action_type=THRUST_YAW, inertial_frame=START, target_type=TARGET_VELOCITY_TARGET_DISTANCE
  output_activation=acceleration_bounded_yaw  -> action[:3]=thrust accel (gravity-INCLUSIVE),
                                                 action[3]=yaw in [-pi,pi]
  depth 72x128, near 0.25, far 20.0 ; GRU hidden 192 ; 50 Hz ; target_speed in [1,5]

Frames: depthnav uses ENU (std frame). START = world-aligned at spawn, fixed thereafter.
We capture R_ws (world<-start) once at policy handoff; state/target/thrust are START-frame.
"""

import os
import sys
import yaml
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation

# depthnav package
_DEPTHNAV = "/home/ubuntu/superfly/depthnav"
sys.path.insert(0, _DEPTHNAV)
from gymnasium import spaces
from depthnav.policies.multi_input_policy import MultiInputPolicy

# --- ground-truth artefacts ---
DEFAULT_CKPT = os.path.join(
    _DEPTHNAV, "examples/navigation/logs/level1/level1_4_iteration_13500.pth")

# The training run's saved logs/level1/level1.yaml no longer exists on disk.
# policy_cfg/small_yaw.yaml is the matching policy config: its layer shapes
# (state/target Linear(*, 192)+LN, 3-layer depth CNN, single policy_net.0
# 192->4) line up with this checkpoint's state_dict, and its
# update_env_kwargs (THRUST_YAW / START / TARGET_VELOCITY_TARGET_DISTANCE)
# matches the docstring above. DepthNavPolicy only reads the `policy:` key.
DEFAULT_CFG = os.path.join(
    _DEPTHNAV, "examples/navigation/policy_cfg/small_yaw.yaml")

DEPTH_H, DEPTH_W = 72, 128
DEPTH_NEAR, DEPTH_FAR = 0.25, 20.0
STATE_DIM, TARGET_DIM = 7, 4
GRAVITY = 9.81


def _build_obs_space():
    """Hand-build the spaces.Dict the extractor reads shapes from."""
    return spaces.Dict({
        "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32),
        "target": spaces.Box(-np.inf, np.inf, (TARGET_DIM,), dtype=np.float32),
        "depth": spaces.Box(0.0, np.inf, (1, DEPTH_H, DEPTH_W), dtype=np.float32),
    })


def _quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> [w,x,y,z], with w>=0 (matches navigation_env.py:148-149)."""
    q = Rotation.from_matrix(R).as_quat()          # scipy: [x,y,z,w]
    q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    if q[0] < 0:
        q = -q
    return q


class DepthNavPolicy:
    def __init__(self, checkpoint_path: str = DEFAULT_CKPT,
                 cfg_path: str = DEFAULT_CFG, target_speed: float = 3.0,
                 device: str = None):
        self.device = th.device(device or ("cuda" if th.cuda.is_available() else "cpu"))
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        policy_kwargs = dict(cfg["policy"])
        policy_kwargs["device"] = str(self.device)

        self.model = MultiInputPolicy(_build_obs_space(), **policy_kwargs)
        self.model.load(checkpoint_path)   # load_state_dict + to(device)
        self.model.eval()

        self.target_speed = float(target_speed)
        self.latent = None      # GRU hidden state (init lazily to zeros)
        self.R_ws = None        # world<-start rotation, captured at handoff

    def reset(self, R_ws_enu: np.ndarray):
        """Call at CLIMB->POLICY handoff. R_ws_enu = drone world rotation (ENU/FLU)
        at that instant; defines the fixed START frame."""
        self.R_ws = np.asarray(R_ws_enu, dtype=np.float64)
        self.latent = th.zeros((1, self.model.latent_dim), device=self.device)

    def normalize_depth(self, depth_m: np.ndarray) -> th.Tensor:
        """Clamp to [near, far], NaN/inf -> far (matches base_env.py:387-395).
        The network itself inverts (1/(d+1e-6)) and maxpools — we do NOT."""
        d = np.asarray(depth_m, dtype=np.float32)
        d = np.nan_to_num(d, nan=DEPTH_FAR, posinf=DEPTH_FAR, neginf=DEPTH_FAR)
        d = np.clip(d, DEPTH_NEAR, DEPTH_FAR)
        return th.as_tensor(d, dtype=th.float32, device=self.device).view(1, 1, DEPTH_H, DEPTH_W)

    @th.no_grad()
    def step(self, position_enu, velocity_enu, R_enu, goal_enu, depth_m=None):
        """One control step. Returns (thrust_world(3) [m/s^2, gravity-incl], yaw_world).

        position_enu, velocity_enu, R_enu: current drone state (ENU/FLU world frame).
        goal_enu: goal position (ENU). depth_m: (72,128) metric metres or None.
        """
        assert self.R_ws is not None, "call reset(R_ws) at handoff first"
        R_ws = self.R_ws
        R_sw = R_ws.T                      # world->start
        pos = np.asarray(position_enu, float)
        vel = np.asarray(velocity_enu, float)
        R_wb = np.asarray(R_enu, float)    # world<-body

        # --- state(7): [quat(START<-body) wxyz, vel_start] ---
        R_sb = R_sw @ R_wb
        quat_sb = _quat_wxyz_from_R(R_sb)
        vel_start = R_sw @ vel
        state = np.concatenate([quat_sb, vel_start]).astype(np.float32)

        # --- target(4): [target_vel_start, 1/clamp(dist,0.5)] ---
        target_vec = np.asarray(goal_enu, float) - pos
        dist = np.linalg.norm(target_vec)
        # PD desired velocity (Kp=1.5, Kd=0), clamped to target_speed (navigation_env.py:505-518)
        des_v = 1.5 * target_vec
        des_v_norm = np.linalg.norm(des_v)
        if des_v_norm > 1e-9:
            des_v = des_v / des_v_norm * min(des_v_norm, self.target_speed)
        else:
            des_v = np.zeros(3)
        target_vel_start = R_sw @ des_v
        inv_dist = 1.0 / max(dist, 0.5)
        target = np.concatenate([target_vel_start, [inv_dist]]).astype(np.float32)

        # --- depth ---
        if depth_m is not None:
            depth = self.normalize_depth(depth_m)
        else:
            depth = th.full((1, 1, DEPTH_H, DEPTH_W), DEPTH_FAR,
                            dtype=th.float32, device=self.device)

        obs = {
            "state": th.as_tensor(state, device=self.device).unsqueeze(0),
            "target": th.as_tensor(target, device=self.device).unsqueeze(0),
            "depth": depth,
        }
        action, self.latent = self.model(obs, self.latent)
        action = action.squeeze(0).cpu().numpy()

        thrust_start = action[:3]               # gravity-inclusive thrust accel, START frame
        yaw_start = float(action[3])            # desired yaw in START frame [-pi,pi]
        thrust_world = R_ws @ thrust_start

        # world yaw = start_yaw + policy yaw; start_yaw = heading of START frame x-axis
        start_fwd = R_ws[:, 0]
        start_yaw = float(np.arctan2(start_fwd[1], start_fwd[0]))
        yaw_world = start_yaw + yaw_start

        return thrust_world, yaw_world
