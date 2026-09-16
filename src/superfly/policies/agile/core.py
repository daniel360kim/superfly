"""Agile Autonomy (Loquercio) policy core: depth + MAVLink state -> attitude/thrust.

Pipeline per control tick (agile_offboard.py runs this at 30 Hz):
  1. (decimated, ~15 Hz) PlaNet inference (wrapper/agile_model.py): 224x224 depth
     (the sim renders 640x480 and bilinear-downsamples to 224, matching
     Loquercio's training loader) + 21-dim state -> `modes` candidate body-frame
     trajectories + alpha costs.
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
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation

from superfly.policies.agile.model import (
    LoquercioModelConfig, TensorFlowLoquercioBackend,
    OnnxStudentBackend, is_onnx_checkpoint,
)
from superfly.policies.agile.mpc import (
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

# ---------------------------------------------------------------------------
# anyanything student (ONNX) constants
# ---------------------------------------------------------------------------
# The student's waypoints are ABSOLUTE body-frame metres at a 0.5 s spacing
# (agile_student/INPUTS.md; waypoint j is at t_j = 0.5 j s, waypoint 0 is the
# state itself and is NOT emitted). Two consequences, both handled explicitly:
#   * no _scale_body_plan rescaling -- the metric plan already encodes the speed
#     the labeller chose, and shrinking it would slow the student by max_vel/7.
#   * the MPC's dt_wp is 0.5, not 0.1. build_reference() only ever uses dt_wp to
#     build the time base `t = arange(nwp) * dt_wp` for its cubic fit, and then
#     samples that cubic at the solver's own nodes (N=10 x DT=0.1 s = a 1.0 s
#     horizon), so passing the true spacing is exact and NO resampling of the
#     waypoints themselves is needed or wanted.
STUDENT_WAYPOINT_DT = 0.5
# ...but a cubic least-squares fitted over the student's whole 5 s plan is a bad
# local fit for the MPC's first 1.0 s. Feed build_reference the current position
# (t=0) plus the first three waypoints (t = 0.5, 1.0, 1.5 s): four points, one
# exactly-determined cubic, covering the 1.0 s horizon with 0.5 s of margin --
# the same window the test-5 evaluation harness fits
# (sim_episode.fit_cubic: a cubic through the state and waypoints 1-3).
STUDENT_MPC_WAYPOINTS = 4
# Depth-veto defaults = sim_episode.DepthVetoPolicy's (the deployed rule).
STUDENT_VETO_LOOK_M = 3.5
STUDENT_VETO_STEP_M = 0.25
STUDENT_VETO_MARGIN_M = 0.15
STUDENT_VETO_RADIUS_M = 0.35
# Upstream agile_autonomy accept_thresh, used by the veto's tie rule.
STUDENT_ACCEPT = 0.9
# The render pinhole the students were trained against (render_depth.Camera):
# 640x480 at 91 deg hfov, principal point at the centre of the PIXEL GRID
# ((w-1)/2, not w/2), bilinear-resized to 224x224. The resize is anisotropic --
# 640 -> 224 horizontally, 480 -> 224 vertically -- so there are two scales.
STUDENT_CAM_W, STUDENT_CAM_H, STUDENT_CAM_HFOV_DEG = 640, 480, 91.0

# ENU inertial -> NED inertial and FLU body -> FRD body (Pegasus convention).
_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def quat_enu_flu_to_ned_frd_wxyz(R_enu_flu: np.ndarray) -> np.ndarray:
    rot = _rot_ENU_to_NED * Rotation.from_matrix(R_enu_flu) * _rot_FLU_to_FRD
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def depth_veto_blocked(depth: np.ndarray, modes_body: np.ndarray,
                       look_m: float = STUDENT_VETO_LOOK_M,
                       step_m: float = STUDENT_VETO_STEP_M,
                       margin: float = STUDENT_VETO_MARGIN_M,
                       radius_m: float = STUDENT_VETO_RADIUS_M):
    """Port of sim_episode.DepthVetoPolicy.blocked -- the test-5 deployment's
    mode gate, using nothing the vehicle does not already have.

    `modes_body` is (M, N, 3) body-frame waypoints (FLU: x forward, y left,
    z up), WITHOUT the implicit waypoint 0 at the origin. Each mode's path is
    walked from the body out to `look_m` metres in `step_m` steps and every
    sample is projected into the 224x224 depth frame with the render pinhole;
    a mode whose sample sits behind the depth surface on its own ray by more
    than `margin` metres (taking the nearest return in a `radius_m` window
    around the pixel) is vetoed. Only the NEAR path is walked: a far point can
    be occluded by an obstacle the path flies over or around, which is not a
    collision. Samples outside the frame never veto.

    Returns (blocked (M,) bool, scores (M,) float = worst signed clearance)."""
    depth = np.asarray(depth, dtype=np.float64)
    modes_body = np.asarray(modes_body, dtype=np.float64)
    H, W = depth.shape
    f = 0.5 * STUDENT_CAM_W / math.tan(math.radians(STUDENT_CAM_HFOV_DEG) / 2)
    cx, cy = (STUDENT_CAM_W - 1) / 2.0, (STUDENT_CAM_H - 1) / 2.0
    sx, sy = W / float(STUDENT_CAM_W), H / float(STUDENT_CAM_H)
    out = np.zeros(modes_body.shape[0], bool)
    scores = np.full(modes_body.shape[0], np.inf)
    for k, wps in enumerate(modes_body):
        pts = np.vstack([np.zeros((1, 3)), wps])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s = np.arange(step_m, min(look_m, cum[-1]) + 1e-9, step_m)
        worst = np.inf
        for si in s:
            j = int(np.searchsorted(cum, si, side="right") - 1)
            j = min(j, len(seg) - 1)
            frac = (si - cum[j]) / seg[j] if seg[j] > 1e-9 else 0.0
            pb = pts[j] + frac * (pts[j + 1] - pts[j])
            zc = pb[0]                      # camera z = body x (forward)
            if zc < 0.2:
                continue
            xc, yc = -pb[1], -pb[2]         # camera x = -body y, camera y = -body z
            u = (f * xc / zc + cx) * sx
            v = (f * yc / zc + cy) * sy
            if not (0 <= u < W and 0 <= v < H):
                continue
            r = int(np.ceil(f * radius_m / zc * sx))
            u0, u1 = max(0, int(u) - r), min(W, int(u) + r + 1)
            v0, v1 = max(0, int(v) - r), min(H, int(v) + r + 1)
            worst = min(worst, float(np.min(depth[v0:v1, u0:u1])) - zc)
        scores[k] = worst
        out[k] = worst < -margin
    return out, scores


def _pad3(a) -> np.ndarray:
    """The debug transport carries exactly three alphas; students have 1-3."""
    out = np.zeros(3, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).ravel()[:3]
    out[:a.size] = a
    return out


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
    depth: np.ndarray | None     # (224, 224) planar Z-depth [m], row 0 = up


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
    # depth-veto worst signed clearance per mode (None unless --mode-select veto)
    veto_scores: np.ndarray | None = None


class AgilePolicy:
    """Owns the net, the MPC, the obstacle memory, and all frame plumbing.

    compute(obs) is the only entry point; it must be called at control_hz. The
    expensive net forward pass runs every `net_every` ticks (15 Hz at the 30 Hz
    control rate -- the rate the MPC reference pipeline was designed for); the
    MPC re-solves EVERY tick against fresh state, which is what keeps the
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
                 depth_inflate_px: int = 0, alt_follow: bool = False,
                 net_thread: bool = False, goal_speed: float = 0.0,
                 mode_select: str = "auto",
                 veto_look_m: float = STUDENT_VETO_LOOK_M,
                 veto_step_m: float = STUDENT_VETO_STEP_M,
                 veto_margin_m: float = STUDENT_VETO_MARGIN_M,
                 veto_radius_m: float = STUDENT_VETO_RADIUS_M):
        # Which net: a .onnx artifact is an anyanything student (22-dim state,
        # metric 0.5 s waypoints, any mode/waypoint count); anything else is
        # the legacy TF2 PlaNet checkpoint prefix (21-dim state, 0.1 s plan).
        self.is_student = is_onnx_checkpoint(checkpoint_path)
        if self.is_student:
            print(f"[agile] loading ONNX student from {checkpoint_path} ...", flush=True)
            self.net = OnnxStudentBackend(checkpoint_path)
            self.config = self.net.config
            self.waypoint_dt = STUDENT_WAYPOINT_DT
            print(f"[agile] student: {self.config.modes} modes x "
                  f"{self.config.out_seq_len} waypoints @ {self.waypoint_dt:g} s "
                  f"({self.config.out_seq_len * self.waypoint_dt:g} s horizon), "
                  f"state dim {self.config.raw_state_dim}; building acados MPC ...",
                  flush=True)
        else:
            self.config = LoquercioModelConfig()
            self.waypoint_dt = WAYPOINT_DT
            print(f"[agile] loading PlaNet checkpoint from {checkpoint_path} ...", flush=True)
            self.net = TensorFlowLoquercioBackend(checkpoint_path, self.config)
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
        # Student knobs. goal_speed is the ARRIVAL speed v_goal the student was
        # conditioned on (agile_student/INPUTS.md); mode_select "auto" means
        # veto for the student, upstream cost-only for the TF checkpoint.
        self.goal_speed = float(goal_speed)
        if mode_select == "auto":
            mode_select = "veto" if self.is_student else "cost"
        if mode_select not in ("cost", "veto"):
            raise ValueError(f"mode_select must be cost|veto|auto, got {mode_select!r}")
        self.mode_select = mode_select
        self.veto_look_m = float(veto_look_m)
        self.veto_step_m = float(veto_step_m)
        self.veto_margin_m = float(veto_margin_m)
        self.veto_radius_m = float(veto_radius_m)
        # How many plan points the MPC's reference cubic is fitted through.
        # Legacy: all 10 of the net's 0.1 s waypoints (a 0.9 s span). Student:
        # the current position plus the first three 0.5 s waypoints.
        self.mpc_waypoints = STUDENT_MPC_WAYPOINTS if self.is_student else None
        # 2026-07-30 margin-tuning campaign knobs (see notes/robust_2026-07/
        # agile_diagnosis.md in gs_drone_sim):
        #  - depth_inflate_px: odd minimum-filter kernel on the 224x224 depth fed
        #    to the NET ONLY (obstacle memory keeps the raw frame). Makes every
        #    obstacle look wider/closer so the thin-margin net dodges wider.
        #  - alt_follow: follow the net's z through the MPC reference (upstream
        #    behaviour) instead of locking z to cruise_alt; the altitude-hold
        #    thrust PD is slaved to the stage-1 reference z.
        #  - net_thread: run the ~61 ms CPU net forward pass in a worker thread
        #    so it no longer blocks the control/MPC/attitude loop.
        self.depth_inflate_px = int(depth_inflate_px)
        self.alt_follow = bool(alt_follow)
        self.net_thread = bool(net_thread)
        # Keep-out cell radius, env-overridable for the campaign sweep.
        self.OBS_R = float(os.environ.get("AGILE_OBS_R", self.OBS_R))
        # Sample the MPC attitude at the control period by default (not the 0.1 s
        # stage-1 node) so the setpoint doesn't over-anticipate at control_hz.
        self.att_lookahead_s = (self.control_dt if att_lookahead_s is None
                                else float(att_lookahead_s))
        # net-thread machinery (started lazily on the first compute() call)
        self._net_lock = threading.Lock()
        self._net_req = None          # latest pending (depth_in, state_in, pos, R)
        self._net_res = None          # latest finished (alphas, trajs, pos, R)
        self._net_event = threading.Event()
        self._net_worker = None

        # altitude-hold thrust PD + slow integrator. The I-term matters: the
        # nominal hover_thrust (g/20) is below the Iris's true hover point, and
        # a pure PD equilibrates ~0.5 m BELOW cruise_alt (observed live) --
        # enough to keep a 3D goal check from ever firing.
        self.kp_alt, self.kd_alt, self.ki_alt = 4.0, 4.0, 0.4
        # PD fallback tracker gains (only used when an MPC solve fails)
        self.kp_pos, self.kd_vel = 6.0, 4.0
        # PD-fallback lookahead index: 0.5 s ahead in both plans (legacy index
        # 5 x 0.1 s; student index 1 on the origin-prefixed 0.5 s plan).
        self.pd_lookahead = 1 if self.is_student else 5

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
        self._veto_scores = None             # last depth-veto clearances (modes,)
        self._prev_mode_world = None         # last tracked mode's world waypoints
        self._prev_q = None                  # low-pass state for the sent attitude
        self._ref_start = None               # mission reference line (set on first tick)
        self._ref_goal = None
        self._cruise_alt = None
        self._obs_cells: dict[tuple[int, int], float] = {}   # (ix,iy) -> last-seen ts
        self._alt_i = 0.0                    # altitude integrator [m/s^2]
        self.mpc._warmed = False             # re-converge the first solve
        if getattr(self, "_net_lock", None) is not None:     # drop stale net i/o
            with self._net_lock:
                self._net_req = None
                self._net_res = None
                self._net_event.clear()

    # ------------------------------------------------------------------ #
    # Net input encoding
    # ------------------------------------------------------------------ #
    def _depth_to_model_input(self, depth_hw) -> np.ndarray:
        if depth_hw is None:
            depth_m = np.full((AGILE_IMG_SIZE, AGILE_IMG_SIZE), AGILE_FAR, dtype=np.float32)
        else:
            depth_m = np.asarray(depth_hw, dtype=np.float32)
            depth_m = np.nan_to_num(depth_m, nan=AGILE_FAR, posinf=AGILE_FAR, neginf=0.0)
        if self.depth_inflate_px > 1:
            # Obstacle inflation in INPUT space: grey-morphology erosion keeps
            # the nearest return within a KxK window, widening every obstacle by
            # ~(K//2) px per side (~d*(K//2)/110 m at distance d, fx=110 px).
            # Net input only -- the keep-out obstacle memory sees the raw frame.
            depth_m = minimum_filter(depth_m, size=self.depth_inflate_px,
                                     mode="nearest")
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

    def _student_state_to_model_input(self, pos_enu, R_enu, vel_enu, angular_body,
                                      goal_enu) -> np.ndarray:
        """The 22-dim student state of agile_student/INPUTS.md:
        [pos(3), R(9) row-major, v_body(3), omega_body(3), goal_body(3), v_goal].

        Three deliberate differences from the legacy 21-dim encoding:
          * R is the RAW body->world matrix, not de-yawed. The de-yaw is a
            workaround for the ckpt-50 checkpoint, which was trained only on
            near-zero-yaw flights; the students are trained on drawn states at
            every yaw, and de-yawing here would rotate the goal out of the
            frame the net learned.
          * the goal is the METRIC body-frame vector to the real goal, clamped
            to 10 m (Obs.goal_body), not a unit direction to a look-ahead point
            on the mission line.
          * a trailing v_goal = the arrival speed the student is conditioned on.
        Byte-for-byte the same vector as
        superfly_expert_sampler.sim_episode.OnnxPolicy.encode_state -- asserted
        in tests/test_agile_student.py."""
        R_enu = np.asarray(R_enu, np.float64)
        g = R_enu.T @ (np.asarray(goal_enu, np.float64) - np.asarray(pos_enu, np.float64))
        n = float(np.linalg.norm(g))
        if n > 1e-9:
            g = g * (min(n, 10.0) / n)
        state = np.concatenate([
            np.asarray(pos_enu, np.float64).reshape(3),
            R_enu.reshape(-1),
            R_enu.T @ np.asarray(vel_enu, np.float64).reshape(3),
            np.asarray(angular_body, np.float64).reshape(3),
            g,
            [self.goal_speed],
        ]).astype(np.float32)
        return state.reshape((1, 1, self.config.raw_state_dim))

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
        # The student's waypoints are absolute metres at a known 0.5 s spacing:
        # the geometry IS the schedule, so rescaling would both slow it down and
        # move the waypoints off the times the MPC reference assumes. Identity.
        if self.is_student:
            return local_xyz
        if self.max_vel <= 0.0 or self.max_vel >= NATIVE_PLAN_SPEED:
            return local_xyz
        return local_xyz * (self.max_vel / NATIVE_PLAN_SPEED)

    # ------------------------------------------------------------------ #
    def _select_mode(self, local_xyz_per_mode, depth_hw, alphas=None,
                     world_per_mode=None) -> int:
        """`cost`: upstream agile_autonomy always tracks mode 0 (lowest alpha).

        `veto`: the test-5 deployment rule (sim_episode.DepthVetoPolicy +
        select_mode). The cost head is not reliably learnable from these labels,
        so each mode's near path is projected into the depth frame the policy
        was just given and any mode that goes behind the surface is given an
        infinite cost; the choice is then argmin cost over the survivors, with
        upstream's ACCEPT=0.9 "sent set" and the nearest-to-previously-tracked
        tie break. A veto that would reject EVERY mode is ignored (there is no
        better option to fall back to), exactly as the reference does."""
        if self.mode_select == "cost" or alphas is None:
            return 0
        n_modes = len(local_xyz_per_mode)
        if n_modes == 1:
            return 0
        costs = np.abs(np.asarray(alphas, np.float64)).reshape(n_modes).copy()
        self._veto_scores = None
        if depth_hw is not None:
            # (3, N) per mode -> (M, N, 3) body-frame waypoints for the veto.
            modes_body = np.stack([np.asarray(m, np.float64).T
                                   for m in local_xyz_per_mode], axis=0)
            bad, scores = depth_veto_blocked(
                depth_hw, modes_body, self.veto_look_m, self.veto_step_m,
                self.veto_margin_m, self.veto_radius_m)
            self._veto_scores = scores
            if bad.any() and not bad.all():
                costs = np.where(bad, np.inf, costs)
        order = np.argsort(costs, kind="stable")
        best = int(order[0])
        sent = [best]
        for k in order[1:]:
            if not np.isfinite(costs[k]):
                continue
            if (costs[best] + 1e-6) / (costs[k] + 1e-6) > STUDENT_ACCEPT:
                sent.append(int(k))
        if self._prev_mode_world is None or len(sent) == 1 or world_per_mode is None:
            return best
        prev = self._prev_mode_world
        n = min(len(prev), world_per_mode.shape[1])
        d = [float(np.sum(np.linalg.norm(world_per_mode[k][:n] - prev[:n], axis=1)))
             for k in sent]
        return int(sent[int(np.argmin(d))])

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
    # Net plan adoption + optional worker thread (--net-thread)
    # ------------------------------------------------------------------ #
    def _adopt_plan(self, alphas, trajectories, pos, R_enu, depth_hw):
        """Turn a net output into the cached world-frame plan. pos/R_enu must be
        the SNAPSHOT the net input was built from (matters in threaded mode)."""
        local_per_mode = [np.asarray(t, np.float64).reshape(
            self.config.state_dim, self.config.out_seq_len) for t in trajectories]
        # World-frame waypoints first: the veto's tie rule needs them, and the
        # selection must not change the plans it is choosing between.
        wp_per_mode = np.stack(
            [pos[None, :] + (R_enu @ self._scale_body_plan(m)).T
             for m in local_per_mode], axis=0)                     # (modes, N, 3)
        self._mode_idx = self._select_mode(local_per_mode, depth_hw, alphas,
                                           wp_per_mode)
        if self.is_student:
            # The student emits waypoints 1..N at t = 0.5 j s; waypoint 0 is the
            # state itself, which it does not emit. Prepend it so the MPC's time
            # base `t = arange(nwp) * dt_wp` lines up with the real schedule.
            origin = np.repeat(pos[None, None, :], wp_per_mode.shape[0], axis=0)
            self._world_points_per_mode = np.concatenate([origin, wp_per_mode], axis=1)
        else:
            self._world_points_per_mode = wp_per_mode             # (modes, T, 3)
        self._world_points = self._world_points_per_mode[self._mode_idx]
        self._prev_mode_world = wp_per_mode[self._mode_idx].copy()
        self._alphas = alphas

    def _net_worker_loop(self):
        """Latest-only inference worker: always consumes the freshest snapshot,
        so the effective net rate saturates at 1/forward-time (~16 Hz on this
        CPU) instead of being gated AND blocked by the control loop."""
        while True:
            self._net_event.wait()
            with self._net_lock:
                req = self._net_req
                self._net_req = None
                self._net_event.clear()
            if req is None:
                continue
            depth_in, state_in, pos, R_enu, depth_hw = req
            try:
                alphas, trajs = self.net.infer(depth_in, state_in)
            except Exception as exc:      # never kill the worker
                print(f"[agile] net worker infer failed: {exc}", flush=True)
                continue
            with self._net_lock:
                self._net_res = (alphas, trajs, pos, R_enu, depth_hw)

    def _encode_state(self, pos, R_enu, vel, angular_body, goal_enu, goal_dir):
        """Student -> the 22-dim metric encoding; TF checkpoint -> the legacy
        21-dim de-yawed/look-ahead-direction one."""
        if self.is_student:
            return self._student_state_to_model_input(
                pos, R_enu, vel, angular_body, goal_enu)
        return self._state_to_model_input(pos, R_enu, vel, angular_body, goal_dir)

    def _net_tick_threaded(self, obs, pos, R_enu, vel, goal_enu, goal_dir):
        """Submit the freshest observation, adopt the latest finished plan.
        Blocks only on the very first call (no plan exists yet)."""
        if self._net_worker is None:
            self._net_worker = threading.Thread(target=self._net_worker_loop,
                                                daemon=True)
            self._net_worker.start()
        depth_in = self._depth_to_model_input(obs.depth)
        state_in = self._encode_state(pos, R_enu, vel, obs.angular_rate_body,
                                      goal_enu, goal_dir)
        with self._net_lock:
            self._net_req = (depth_in, state_in, pos.copy(), R_enu.copy(), obs.depth)
            self._net_event.set()
            res, self._net_res = self._net_res, None
        if res is not None:
            self._adopt_plan(*res)
        elif self._world_points is None:
            t0 = time.time()
            while self._world_points is None and time.time() - t0 < 3.0:
                time.sleep(0.005)
                with self._net_lock:
                    res, self._net_res = self._net_res, None
                if res is not None:
                    self._adopt_plan(*res)
            if self._world_points is None:
                raise RuntimeError("[agile] net worker produced no plan within 3 s")

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

        # --- net inference (decimated; optionally off-loop in a worker thread) ---
        if self.net_thread:
            self._net_tick_threaded(obs, pos, R_enu, vel, goal, goal_dir)
        elif self._world_points is None or (self._tick % self.net_every) == 0:
            depth_in = self._depth_to_model_input(obs.depth)
            state_in = self._encode_state(pos, R_enu, vel, obs.angular_rate_body,
                                          goal, goal_dir)
            alphas, trajectories = self.net.infer(depth_in, state_in)
            self._adopt_plan(alphas, trajectories, pos, R_enu, obs.depth)

        world_points = self._world_points
        # Points the MPC's reference cubic is fitted through (see
        # STUDENT_MPC_WAYPOINTS): the whole plan for the legacy net, the first
        # four (t = 0, 0.5, 1.0, 1.5 s) for the student.
        mpc_points = (world_points if self.mpc_waypoints is None
                      else world_points[:self.mpc_waypoints])

        # --- MPC tracking (every tick) ---
        attitude_q = None
        tracker = "pd"
        alt_target = self._cruise_alt
        keepout = self._keepout_list() if self.use_keepout else None
        x0 = state_x0(pos, R_enu, vel)
        try:
            _u0, status, minfo = self.mpc.compute(
                x0, mpc_points.astype(np.float64), self._cruise_alt, yaw_des,
                dt_wp=self.waypoint_dt, max_vel=self.max_vel,
                obstacles_xy_r=keepout, alt_hold=not self.alt_follow,
                att_lookahead_s=self.att_lookahead_s)
            if status in (0, 2):
                attitude_q = np.asarray(minfo["q_pred"], dtype=np.float64)
                if self.max_tilt_deg < 89.0:
                    attitude_q = np.asarray(
                        clamp_attitude_tilt(attitude_q, self.max_tilt_deg, yaw_des),
                        dtype=np.float64)
                tracker = "mpc"
                if self.alt_follow:
                    # Slave the altitude-hold thrust PD to the stage-1 reference
                    # z (the net's clamped vertical plan) so cyl_h obstacles can
                    # be over/under-flown; band-limited around the cruise alt.
                    alt_target = float(np.clip(minfo["p_ref1"][2], 1.0,
                                               self._cruise_alt + 3.0))
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
            v_ref = (world_points[nxt] - world_points[prv]) / (max(nxt - prv, 1) * self.waypoint_dt)
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
        alt_err = alt_target - float(pos[2])
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
            veto_scores=self._veto_scores,
        )

    def debug_frame(self, pos_enu, R_enu, tracker: str) -> dict | None:
        """Latest net trajectories in local ENU for the overhead debug view."""
        if self._world_points_per_mode is None:
            return None
        yaw = math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0]))
        return dict(
            pos_local=np.asarray(pos_enu, dtype=np.float64).reshape(3),
            yaw=yaw,
            alphas=_pad3(self._alphas),
            trajectories_local=np.asarray(self._world_points_per_mode, dtype=np.float64),
            mode_idx=int(self._mode_idx),
            tracker=tracker,
        )
