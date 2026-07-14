"""Agile Autonomy policy core: visual observation + MAVLink state -> attitude/thrust.

Pipeline per control tick (100 Hz in the comparison harness):
  1. (decimated, ~15 Hz) PlaNet inference (wrapper/agile_model.py): either the
     original 224x224 depth input or 224x224 RGB encoded by frozen CL4Nav ONNX,
     plus the 21-dim state -> `modes` candidate body-frame trajectories + costs.
  2. Mode selection: fly mode 0 (lowest alpha), matching upstream
     agile_autonomy trajectory_decision.
  3. (every tick) acados MPC (wrapper/agile_mpc.py) tracks the selected
     trajectory; stage-1 attitude + altitude-hold thrust become SET_ATTITUDE_TARGET.

State encoding matches upstream PlannerBase.update_input_queues:
  full body rotation matrix, body-frame velocity, body-frame angular rates, and
  body-frame goal direction (reference point future_time seconds ahead on the
  mission line).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from wrapper.agile_model import LoquercioModelConfig, TensorFlowLoquercioBackend
from wrapper.agile_mpc import (
    MPC, state_x0, clamp_attitude_tilt, flatness_attitude, G,
)

# Sim depth camera (run_px4_sim.py --policy agile): 640x480 render bilinear-downsampled
# to 224x224, hfov 91 deg (flightmare.yaml), forward-facing, planar Z-depth in metres.
AGILE_IMG_SIZE = 224
AGILE_FOV_DEG = 91.0
AGILE_FAR = 20.0

# Upstream test_settings.yaml future_time [s]; reference is discretized at 50 Hz in
# PlannerBase (ref_idx = progress + int(future_time * 50)). On a straight start->goal
# line at cruise speed that is ~future_time * max_vel metres ahead.
REF_LOOKAHEAD_S = 5.0

# The net predicts out_seq_len waypoints at a FIXED 0.1 s spacing (a 1 s horizon).
WAYPOINT_DT = 0.1
# Upstream default.yaml test_time_velocity; body-frame plans are scaled down when
# max_vel is below this so the spatial horizon matches the commanded cruise speed.
NATIVE_PLAN_SPEED = 7.0

# ENU inertial -> NED inertial and FLU body -> FRD body (Pegasus convention).
_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def quat_enu_flu_to_ned_frd_wxyz(R_enu_flu: np.ndarray) -> np.ndarray:
    rot = _rot_ENU_to_NED * Rotation.from_matrix(R_enu_flu) * _rot_FLU_to_FRD
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        return v / n
    return np.asarray(fallback, dtype=np.float64)


@dataclass
class AgileObs:
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    R_enu: np.ndarray            # body FLU -> world ENU
    angular_rate_body: np.ndarray  # body FLU [rad/s]
    goal_enu: np.ndarray
    depth: np.ndarray | None = None  # (224, 224) planar Z-depth [m], row 0 = up
    rgb: np.ndarray | None = None    # (224, 224, 3) uint8 RGB, row 0 = up


@dataclass
class AgileCmd:
    attitude_ned_frd_wxyz: np.ndarray
    thrust_norm: float
    # diagnostics for the offboard's verbose log
    tracker: str = "mpc"         # "mpc" | "pd"
    mode_idx: int = 0
    n_keepout: int = 0
    tilt_cmd_deg: float = 0.0
    alphas: np.ndarray = field(default_factory=lambda: np.zeros(3))


class AgilePolicy:
    """Owns the net, the MPC, the obstacle memory, and all frame plumbing.

    compute(obs) is the only entry point; it must be called at control_hz. The
    expensive net forward pass runs every `net_every` ticks (about 14-15 Hz
    in deployment); the MPC re-solves EVERY control tick against fresh state,
    which is what keeps the
    attitude stream continuous for PX4."""

    # -- mode selection ----------------------------------------------------
    SELECT_MARGIN = 0.5       # [m] required free depth beyond a candidate waypoint
    # -- obstacle memory ---------------------------------------------------
    OBS_CELL = 0.5            # [m] world XY bin size
    OBS_TTL = 3.0             # [s] cell lifetime after last sighting
    OBS_RANGE_MAX = 4.5       # [m] only trust depth returns nearer than this
    OBS_Z_HALF_BAND = 2.5     # [m] accept cells within cruise_alt +- this
    OBS_Z_FLOOR = 0.8         # [m] never accept cells below this (ground returns)
    OBS_POOL = 8              # min-pool factor before backprojection (224 -> 28)
    OBS_R = 0.45              # [m] keep-out radius per cell (sized for ~0.5 m
                              # depth-lag misregistration, on top of OBS_MARGIN)

    def __init__(self, checkpoint_path: str, max_vel: float = 7.0,
                 hover_thrust: float = G / 20.0, control_hz: float = 30.0,
                 net_every: int = 2, max_tilt_deg: float = 90.0,
                 att_lp: float = 1.0, ref_lookahead_s: float = REF_LOOKAHEAD_S,
                 use_keepout: bool = False, att_lookahead_s: float | None = None,
                 visual_input: str = "depth", cl4nav_onnx_path: str | None = None,
                 cl4nav_provider: str = "CUDAExecutionProvider"):
        if visual_input not in ("depth", "cl4nav_rgb"):
            raise ValueError(f"Unsupported Agile visual input: {visual_input}")
        self.visual_input = visual_input
        self.config = (
            LoquercioModelConfig(
                use_rgb=True, use_depth=False, visual_input="cl4nav_frozen")
            if visual_input == "cl4nav_rgb" else LoquercioModelConfig()
        )
        if use_keepout and visual_input == "cl4nav_rgb":
            raise ValueError("--keepout requires depth and is unavailable for agile_rgb")
        print(f"[agile] loading {visual_input} PlaNet checkpoint from "
              f"{checkpoint_path} ...", flush=True)
        self.net = TensorFlowLoquercioBackend(
            checkpoint_path, self.config, cl4nav_onnx_path, cl4nav_provider)
        print(f"[agile] {self.net.loaded_weight_count} weights loaded from "
              f"{self.net.checkpoint_prefix}; building acados MPC ...", flush=True)
        self.mpc = MPC()
        print("[agile] MPC ready.", flush=True)

        self.max_vel = float(max_vel)
        self.hover_thrust = float(hover_thrust)
        self.control_dt = 1.0 / float(control_hz)
        self.net_every = max(1, int(net_every))
        self.max_tilt_deg = float(max_tilt_deg)
        self.att_lp = float(att_lp)
        self.ref_lookahead_s = float(ref_lookahead_s)
        self.use_keepout = bool(use_keepout)
        # Sample the MPC attitude at the control period by default (not the 0.1 s
        # stage-1 node) so the setpoint doesn't over-anticipate at control_hz.
        self.att_lookahead_s = (self.control_dt if att_lookahead_s is None
                                else float(att_lookahead_s))

        # altitude-hold thrust PD + slow integrator. The I-term matters: the
        # nominal hover_thrust (g/20) is below the Iris's true hover point, and
        # a pure PD equilibrates ~0.5 m BELOW cruise_alt (observed live) --
        # enough to keep a 3D goal check from ever firing.
        self.kp_alt, self.kd_alt, self.ki_alt = 4.0, 4.0, 0.4
        # PD fallback tracker gains (only used when an MPC solve fails)
        self.kp_pos, self.kd_vel = 6.0, 4.0
        self.pd_lookahead = 5     # waypoint index for the PD fallback

        # depth-camera geometry (square image, fx == fy)
        n = AGILE_IMG_SIZE
        fx = 0.5 * n / math.tan(0.5 * math.radians(AGILE_FOV_DEG))
        self._fx = fx
        self._cx = 0.5 * n
        cols = (self._cx - (np.arange(n) + 0.5)) / fx       # +left
        rows = (self._cx - (np.arange(n) + 0.5)) / fx       # +up (square image)
        yy, zz = np.meshgrid(cols, rows)                     # (row, col) grids
        # Unit rays in the camera/body frame (x fwd, y left, z up) per pixel.
        d = np.stack([np.ones_like(yy), yy, zz], axis=-1)
        self._rays = (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)

        self.reset()

    def reset(self):
        self._tick = -1
        self._world_points = None            # cached selected-mode trajectory (T,3)
        self._world_points_per_mode = None   # all candidate trajectories (modes, T, 3)
        self._alphas = np.zeros(self.config.modes, dtype=np.float32)
        self._mode_idx = 0
        self._prev_q = None                  # low-pass state for the sent attitude
        self._ref_start = None               # mission reference line (set on first tick)
        self._ref_goal = None
        self._cruise_alt = None
        self._obs_cells: dict[tuple[int, int], float] = {}   # (ix,iy) -> last-seen ts
        self._alt_i = 0.0                    # altitude integrator [m/s^2]
        self.mpc._warmed = False             # re-converge the first solve

    # ------------------------------------------------------------------ #
    # Net input encoding
    # ------------------------------------------------------------------ #
    def _depth_to_model_input(self, depth_hw) -> np.ndarray:
        if depth_hw is None:
            depth_m = np.full((AGILE_IMG_SIZE, AGILE_IMG_SIZE), AGILE_FAR, dtype=np.float32)
        else:
            depth_m = np.asarray(depth_hw, dtype=np.float32)
            depth_m = np.nan_to_num(depth_m, nan=AGILE_FAR, posinf=AGILE_FAR, neginf=0.0)
        depth_mm = np.clip(depth_m * 1000.0, 0.0, AGILE_FAR * 1000.0)
        # The sim already bilinear-downsamples to the net's 224x224 input (matching
        # Loquercio's training loader), so no resize here in the common case. Keep a
        # nearest-neighbour fallback only for an unexpected off-size frame.
        size = self.config.img_height
        if depth_mm.shape != (size, size):
            idx = np.linspace(0, depth_mm.shape[0] - 1, size).astype(np.int64)
            jdx = np.linspace(0, depth_mm.shape[1] - 1, size).astype(np.int64)
            depth_mm = depth_mm[idx[:, None], jdx[None, :]]
        normalized = depth_mm / 80.0
        model_depth = np.repeat(normalized[..., None], 3, axis=-1)
        return model_depth.reshape((1, 1, size, size, 3)).astype(np.float32)

    def _rgb_to_model_input(self, rgb_hwc) -> np.ndarray:
        if rgb_hwc is None:
            raise RuntimeError(
                "No RGB frame received. Start run_px4_sim.py with --policy agile_rgb "
                "and launch agile_offboard.py with --rgb.")
        rgb = np.asarray(rgb_hwc)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Agile RGB frame must be HxWx3, got {rgb.shape}")
        if not np.all(np.isfinite(rgb)):
            raise ValueError("Agile RGB frame contains NaN or Inf")
        return rgb.reshape((1, 1, *rgb.shape))

    def _state_to_model_input(self, pos_enu, R_enu, vel_enu, angular_body,
                              goal_dir_world) -> np.ndarray:
        # Upstream PlannerBase: full camera/body rotation, body-frame velocity and
        # rates, body-frame goal direction (ref_frame=bf, velocity_frame=bf).
        local_velocity = R_enu.T @ vel_enu
        local_goal = R_enu.T @ goal_dir_world
        # De-yaw the rotation-matrix input. The checkpoint was trained on flights
        # along world +x (near-zero yaw), and the net reads absolute yaw in R as
        # an error to correct: feeding the raw matrix at yaw=90 deg curls an
        # empty-scene straight-ahead plan ~50 deg LEFT (end wp y=+8.7 vs +0.8 at
        # yaw 0, ckpt-50 ablation 2026-07-08) and drowns out obstacle avoidance.
        # Body-frame velocity/rates/goal are yaw-invariant already; the plan is
        # mapped back to world with the FULL R_enu in compute(), so only the net
        # input is de-yawed (tilt is preserved).
        yaw = math.atan2(R_enu[1, 0], R_enu[0, 0])
        cz, sz = math.cos(-yaw), math.sin(-yaw)
        R_deyaw = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]) @ R_enu
        state = np.concatenate([
            np.asarray(pos_enu, np.float32),
            np.asarray(R_deyaw, np.float32).reshape(-1),
            local_velocity, np.asarray(angular_body, np.float32),
            local_goal,
        ]).astype(np.float32)
        return state.reshape((1, 1, self.config.raw_state_dim))

    def _goal_dir(self, pos_enu, goal_enu):
        """Direction to a point future_time [s] ahead on the straight start->goal
        reference line (clamped to the goal), matching upstream PlannerBase's
        ref_idx = progress + int(future_time * 50) on the 50 Hz reference."""
        if self._ref_start is None:
            self._ref_start = np.asarray(pos_enu, np.float64).copy()
            self._ref_goal = np.asarray(goal_enu, np.float64).copy()
        line = self._ref_goal - self._ref_start
        line_len = float(np.linalg.norm(line))
        if self.ref_lookahead_s <= 0.0 or line_len < 1e-3:
            return _unit(goal_enu - pos_enu)
        line_dir = line / line_len
        progress = float(np.clip(np.dot(pos_enu - self._ref_start, line_dir), 0.0, line_len))
        lookahead_m = self.ref_lookahead_s * self.max_vel
        target_s = min(progress + lookahead_m, line_len)
        return _unit(self._ref_start + target_s * line_dir - pos_enu, fallback=line_dir)

    def _scale_body_plan(self, local_xyz: np.ndarray) -> np.ndarray:
        """Shrink the net's body-frame plan when flying below training speed."""
        if self.max_vel <= 0.0 or self.max_vel >= NATIVE_PLAN_SPEED:
            return local_xyz
        return local_xyz * (self.max_vel / NATIVE_PLAN_SPEED)

    # ------------------------------------------------------------------ #
    def _select_mode(self, local_xyz_per_mode, depth_hw) -> int:
        """Upstream agile_autonomy always tracks mode 0 (lowest alpha)."""
        return 0

    # ------------------------------------------------------------------ #
    # Obstacle memory: depth -> world XY keep-out cells with a TTL
    # ------------------------------------------------------------------ #
    def _update_obstacles(self, depth_hw, R_enu, pos_enu, now):
        if depth_hw is None or self._cruise_alt is None:
            return
        d = np.asarray(depth_hw, dtype=np.float32)
        d = np.nan_to_num(d, nan=AGILE_FAR, posinf=AGILE_FAR, neginf=0.0)
        p = self.OBS_POOL
        n = AGILE_IMG_SIZE // p
        d_pool = d[:n * p, :n * p].reshape(n, p, n, p).min(axis=(1, 3))
        rays_pool = self._rays[p // 2::p, p // 2::p][:n, :n]      # central ray per cell
        mask = d_pool < self.OBS_RANGE_MAX
        if not mask.any():
            self._expire_obstacles(now)
            return
        # planar Z-depth -> point along the ray: scale by 1/x-component of the ray
        rays = rays_pool[mask]                                     # (M, 3) body frame
        rng = d_pool[mask][:, None] / np.maximum(rays[:, :1], 1e-3)
        pts_world = pos_enu[None, :] + (R_enu @ (rays * rng).T).T  # (M, 3) ENU
        z_lo = max(self._cruise_alt - self.OBS_Z_HALF_BAND, self.OBS_Z_FLOOR)
        z_hi = self._cruise_alt + self.OBS_Z_HALF_BAND
        keep = (pts_world[:, 2] >= z_lo) & (pts_world[:, 2] <= z_hi)
        for x, y in pts_world[keep, :2]:
            cell = (int(math.floor(x / self.OBS_CELL)), int(math.floor(y / self.OBS_CELL)))
            self._obs_cells[cell] = now
        self._expire_obstacles(now)

    def _expire_obstacles(self, now):
        dead = [c for c, ts in self._obs_cells.items() if now - ts > self.OBS_TTL]
        for c in dead:
            del self._obs_cells[c]

    def _keepout_list(self):
        h = 0.5 * self.OBS_CELL
        return [(ix * self.OBS_CELL + h, iy * self.OBS_CELL + h, self.OBS_R)
                for ix, iy in self._obs_cells]

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def compute(self, obs: AgileObs) -> AgileCmd:
        self._tick += 1
        now = time.time()
        pos = np.asarray(obs.position_enu, np.float64)
        vel = np.asarray(obs.velocity_enu, np.float64)
        R_enu = np.asarray(obs.R_enu, np.float64)
        goal = np.asarray(obs.goal_enu, np.float64)

        if self._cruise_alt is None:
            # POLICY handoff happens at climb altitude; hold that for the cruise.
            self._cruise_alt = float(pos[2])

        goal_dir = self._goal_dir(pos, goal)
        yaw_des = math.atan2(float(goal_dir[1]), float(goal_dir[0]))

        if self.use_keepout:
            self._update_obstacles(obs.depth, R_enu, pos, now)

        # --- net inference (decimated) ---
        if self._world_points is None or (self._tick % self.net_every) == 0:
            visual_in = (self._rgb_to_model_input(obs.rgb)
                         if self.visual_input == "cl4nav_rgb"
                         else self._depth_to_model_input(obs.depth))
            state_in = self._state_to_model_input(
                pos, R_enu, vel, obs.angular_rate_body, goal_dir)
            alphas, trajectories = self.net.infer(visual_in, state_in)
            local_per_mode = [t.reshape(self.config.state_dim, self.config.out_seq_len)
                              for t in trajectories]
            self._mode_idx = self._select_mode(local_per_mode, obs.depth)
            per_mode = []
            for local_xyz in local_per_mode:
                local_xyz = self._scale_body_plan(local_xyz)
                per_mode.append(pos[None, :] + (R_enu @ local_xyz).T)
            self._world_points_per_mode = np.stack(per_mode, axis=0)  # (modes, T, 3)
            self._world_points = self._world_points_per_mode[self._mode_idx]
            self._alphas = alphas

        world_points = self._world_points

        # --- MPC tracking (every tick) ---
        attitude_q = None
        tracker = "pd"
        keepout = self._keepout_list() if self.use_keepout else None
        x0 = state_x0(pos, R_enu, vel)
        try:
            _u0, status, minfo = self.mpc.compute(
                x0, world_points.astype(np.float64), self._cruise_alt, yaw_des,
                dt_wp=WAYPOINT_DT, max_vel=self.max_vel,
                obstacles_xy_r=keepout, alt_hold=True,
                att_lookahead_s=self.att_lookahead_s)
            if status in (0, 2):
                attitude_q = np.asarray(minfo["q_pred"], dtype=np.float64)
                if self.max_tilt_deg < 89.0:
                    attitude_q = np.asarray(
                        clamp_attitude_tilt(attitude_q, self.max_tilt_deg, yaw_des),
                        dtype=np.float64)
                tracker = "mpc"
        except Exception as exc:
            if self._tick < 3 or self._tick % 300 == 0:
                print(f"[agile] MPC solve raised ({exc}); PD fallback this tick.",
                      flush=True)

        if attitude_q is None:
            # PD fallback: track a lookahead waypoint with a velocity feedforward,
            # then flatness -> attitude, tilt-clamped like the MPC path.
            k = min(self.pd_lookahead, world_points.shape[0] - 1)
            p_ref = world_points[k]
            nxt, prv = min(k + 1, world_points.shape[0] - 1), max(k - 1, 0)
            v_ref = (world_points[nxt] - world_points[prv]) / (max(nxt - prv, 1) * WAYPOINT_DT)
            s = float(np.linalg.norm(v_ref))
            if s > self.max_vel > 0.0:
                v_ref *= self.max_vel / s
            accel = self.kp_pos * (p_ref - pos) + self.kd_vel * (v_ref - vel)
            accel[2] = 0.0                    # altitude is the thrust PD's job
            q_flat, _ = flatness_attitude(accel + np.array([0.0, 0.0, G]), yaw_des)
            attitude_q = np.asarray(q_flat, dtype=np.float64)
            if self.max_tilt_deg < 89.0:
                attitude_q = np.asarray(
                    clamp_attitude_tilt(attitude_q, self.max_tilt_deg, yaw_des),
                    dtype=np.float64)

        # Optional low-pass on the attitude sent to PX4 (disabled when att_lp>=1).
        if self.att_lp < 1.0 and self._prev_q is not None:
            if float(np.dot(attitude_q, self._prev_q)) < 0.0:
                attitude_q = -attitude_q
            attitude_q = (1.0 - self.att_lp) * self._prev_q + self.att_lp * attitude_q
            attitude_q /= np.linalg.norm(attitude_q) + 1e-12
        self._prev_q = attitude_q.copy()

        # Thrust: altitude-hold PD on cruise_alt, tilt-compensated, normalized by
        # the hover throttle (the MPC only steers laterally; z never follows the
        # net's unreliable vertical plan).
        R_cmd = Rotation.from_quat(
            [attitude_q[1], attitude_q[2], attitude_q[3], attitude_q[0]]).as_matrix()
        alt_err = self._cruise_alt - float(pos[2])
        self._alt_i = float(np.clip(self._alt_i + self.ki_alt * alt_err * self.control_dt,
                                    -2.0, 2.0))
        az = float(np.clip(
            self.kp_alt * alt_err - self.kd_alt * float(vel[2]) + self._alt_i,
            -4.0, 8.0))
        cos_tilt = max(0.5, float(R_cmd[2, 2]))
        thrust = float(np.clip((az + G) / cos_tilt / G * self.hover_thrust, 0.05, 0.9))
        tilt_cmd = math.degrees(math.acos(float(np.clip(R_cmd[2, 2], -1.0, 1.0))))

        return AgileCmd(
            attitude_ned_frd_wxyz=quat_enu_flu_to_ned_frd_wxyz(R_cmd),
            thrust_norm=thrust,
            tracker=tracker,
            mode_idx=self._mode_idx,
            n_keepout=len(self._obs_cells),
            tilt_cmd_deg=tilt_cmd,
            alphas=self._alphas,
        )

    def debug_frame(self, pos_enu, R_enu, tracker: str) -> dict | None:
        """Latest net trajectories in local ENU for the overhead debug view."""
        if self._world_points_per_mode is None:
            return None
        yaw = math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0]))
        return dict(
            pos_local=np.asarray(pos_enu, dtype=np.float64).reshape(3),
            yaw=yaw,
            alphas=np.asarray(self._alphas, dtype=np.float64).reshape(3),
            trajectories_local=np.asarray(self._world_points_per_mode, dtype=np.float64),
            mode_idx=int(self._mode_idx),
            tracker=tracker,
        )
