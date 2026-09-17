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
    MPC, state_x0, clamp_attitude_tilt, clamp_speed_command, scale_tilt_to_thrust,
    flatness_attitude, G, fit_cubic_state,
    eval_cubic,
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
# sim_episode.Z_REF: the evaluation harness clips every mode's WORLD z into this
# band before it selects between them, and then flies the clipped plan. Same
# frame here -- PX4's local z is height above the arming point, i.e. above the
# ground, which is what the reference's world z is.
#
# The band is only meaningful if it CONTAINS the vehicle. Several shipped
# scenarios climb to 5 m (and some to 3): clipping a level plan to 4.0 m while
# the prepended origin sits at 5.0 makes the fitted cubic command a ~2 m/s
# descent from the first decision, and caps alt_target at the ceiling -- the
# student dives and then flies pinned to it. So the ceiling is raised to keep
# 2 m of headroom above wherever the policy actually takes over
# (student_z_band). The right answer is still to hand over INSIDE the band --
# see AGILE_STUDENT_CLIMB_ALT in compare/registry.py, which defaults the method
# to a 2.0 m handover.
STUDENT_Z_REF = (0.5, 4.0)
#: How far the altitude setpoint may lead the vehicle [m]. The student's
#: vertical plan is RELATIVE (body-frame z), so integrating its slope into an
#: absolute setpoint ratchets for as long as the vehicle has not caught up
#: (measured 2026-09-17: 1.6 -> 4.0 m in 3 s, integrator at its limit, the
#: vehicle then climbing at 2.30 m/s against a setpoint moving 0.60 m/s).
#: Larger than the 0.43 m/s sink 16d6251 cured, so that fix is untouched.
STUDENT_ALT_LEAD = 0.75
#: Attitude-loop response time used by the command speed clamp to decide how
#: much along-track acceleration can still be stopped before V_CAP. Measured
#: only indirectly (commanded vs measured tilt in the offboard logs track
#: within a few degrees at 1 Hz sampling); the bound degrades gracefully --
#: too small and the clamp reverts to a plain delete, too large and the
#: approach to the cap is gentler than it needs to be.
STUDENT_CMD_LAG_S = 0.15
#: Headroom kept above the handover altitude when it is outside STUDENT_Z_REF.
STUDENT_Z_HEADROOM = 2.0

# ---------------------------------------------------------------------------
# Hover throttle
# ---------------------------------------------------------------------------
# The offboard's default hover_thrust is G / MAX_ACCEL = 0.490, but the Iris in
# the 2026-09-16 airstation03 trial hovers at ~0.577 (read off that log: level
# segments at t = 20.8-23.8 s sit at thrust 0.580-0.582 with vz ~ +0.05). An 18 %
# throttle-model error is invisible to the evaluation simulator, which applies
# f = a_cmd + g exactly (sim_episode.clamp_command), but on PX4 it has to be
# balanced by something. With --alt-follow and an alt_target re-pinned to the
# vehicle every replan, the only term that can supply it is -kd_alt * vz, i.e.
# a PERMANENT sink of -G (h_true/h_assumed - 1) / kd_alt = -0.43 m/s -- the
# -0.40/-0.49 m/s observed, and the reason the student flew the whole field at
# 0.1-0.6 m instead of 2 m. Both halves of the fix live here:
#   (1) _alt_sp: an ABSOLUTE altitude setpoint advanced by the reference's own
#       vertical velocity, so a sag produces a real error (see compute()).
#   (2) this parameter, a bigger ki_alt, and an online estimate that replaces
#       the parameter once the vehicle has held level flight.
STUDENT_HOVER_THRUST = 0.577
#: Integral gain while following a vertical plan. 0.4 (the legacy value) takes
#: ~5 s to absorb an 18 % throttle error; 1.5 takes ~1 s.
STUDENT_KI_ALT = 1.5
#: Online hover-throttle estimator: sample the commanded vertical throttle only
#: in near-equilibrium (at equilibrium thrust*cos_tilt IS the true hover
#: throttle, whatever parameter was assumed -- the integrator supplies the
#: difference), EMA it, and adopt it after enough samples.
# The gate has to mean SETTLED, not merely "passing through level": sampling
# during the integrator's wind-up averages in throttles that are still wrong
# (measured: an 0.577 airframe estimated at 0.564, and the bumpless rebase then
# left a 0.25 m overshoot). Hence a tight error window as well as a tight
# velocity one, and enough consecutive settled samples to outlast a transient.
HOVER_EST_VZ_TOL = 0.05        # [m/s]
HOVER_EST_ERR_TOL = 0.05       # [m]
HOVER_EST_ALPHA = 0.02         # EMA weight per sample
HOVER_EST_MIN_SAMPLES = 200    # 2 s of settled flight at 100 Hz per adoption
HOVER_EST_ADOPT_STEP = 0.005   # don't re-adopt for less than this
HOVER_EST_BOUNDS = (0.20, 0.85)


def student_z_band(cruise_alt: float | None) -> tuple[float, float]:
    """sim_episode.Z_REF, widened upward so it always contains the handover
    altitude with STUDENT_Z_HEADROOM to spare."""
    lo, hi = STUDENT_Z_REF
    if cruise_alt is not None:
        hi = max(hi, float(cruise_alt) + STUDENT_Z_HEADROOM)
    return lo, hi
# Decision rate of the evaluation harness (sim_episode.DECISION_HZ). The student
# submits inference no faster than this, so a fast box cannot out-run the rate
# the policy was scored at; a slow one is reported in the log.
STUDENT_DECISION_HZ = 15.0
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
    depth_probe: dict | None = None      # metres at a few pixels of the frame the net saw (debug)
    net_hz: float = 0.0          # achieved decision rate (student only)
    alt_sp: float = 0.0          # absolute altitude setpoint being tracked [m]
    hover_thrust: float = 0.0    # hover throttle in use (measured once converged)


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
                 hover_thrust: float | None = None, control_hz: float = 30.0,
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
        # hover_thrust None -> the student's measured value, else the legacy
        # G/MAX_ACCEL assumption. AGILE_HOVER_THRUST overrides both.
        if hover_thrust is None:
            hover_thrust = (STUDENT_HOVER_THRUST if self.is_student else G / 20.0)
        self.hover_thrust = float(os.environ.get("AGILE_HOVER_THRUST", hover_thrust))
        self.hover_thrust_param = self.hover_thrust     # what was configured
        self.hover_estimate_enabled = True
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
        # Cap on the threaded inference submission rate (0 = uncapped).
        self.net_decision_hz = STUDENT_DECISION_HZ if self.is_student else 0.0
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
        self._net_stop = threading.Event()
        self._net_worker = None

        # altitude-hold thrust PD + slow integrator. The I-term matters: the
        # nominal hover_thrust (g/20) is below the Iris's true hover point, and
        # a pure PD equilibrates ~0.5 m BELOW cruise_alt (observed live) --
        # enough to keep a 3D goal check from ever firing.
        self.kp_alt, self.kd_alt = 4.0, 4.0
        # A vertical plan needs an integrator fast enough to absorb a throttle
        # model error within a decision or two, not within the whole flight.
        self.ki_alt = STUDENT_KI_ALT if self.is_student else 0.4
        # ...and enough integrator authority to actually hold it. Covering an
        # h_true/h_assumed ratio r costs az_ss = G(r - 1): the legacy +-2 m/s^2
        # clamp saturates at r = 1.20, and the trial's own airframe is already
        # at 1.18. +-4 covers r = 1.41.
        self.alt_i_limit = 4.0 if self.is_student else 2.0
        self.alt_lead = STUDENT_ALT_LEAD if self.is_student else 0.0
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

        if self.is_student and not self.alt_follow:
            # Without --alt-follow the MPC reference forces vz = 0 / z = cruise
            # alt and thrust becomes an altitude PD: the student's vertical plan
            # is deleted while the depth veto still judges modes on their
            # vertical geometry, so it can pick a climb-over and fly flat into
            # the obstacle. The agile_student registry entry always passes it.
            print("[agile] WARNING: student without --alt-follow -- its vertical "
                  "plan is discarded and altitude is a PD hold on the climb "
                  "altitude. This is NOT how the student was evaluated.",
                  flush=True)

        self.reset()

    def reset(self):
        self._tick = -1
        self._world_points = None            # cached selected-mode trajectory (T,3)
        self._world_points_per_mode = None   # all candidate trajectories (modes, T, 3)
        self._alphas = np.zeros(self.config.modes, dtype=np.float32)
        self._mode_idx = 0
        self._veto_scores = None             # last depth-veto clearances (modes,)
        self._cubic = None                   # state-pinned cubic of the live plan
        self._plan_z = None                  # (times, world z) of the live plan
        self._plan_time = None               # wall time the live plan was adopted
        self._net_submit_t = 0.0             # last threaded inference submission
        self._net_stamps = []                # recent adoption times, for net_hz
        self._prev_mode_world = None         # last tracked mode's world waypoints
        self._prev_q = None                  # low-pass state for the sent attitude
        self._ref_start = None               # mission reference line (set on first tick)
        self._ref_goal = None
        self._cruise_alt = None
        self._obs_cells: dict[tuple[int, int], float] = {}   # (ix,iy) -> last-seen ts
        self._alt_i = 0.0                    # altitude integrator [m/s^2]
        self._alt_sp = None                  # ABSOLUTE altitude setpoint [m]
        self._hover_est = None               # EMA of the observed hover throttle
        self._hover_n = 0                    # samples in that EMA
        self._hover_adopted = False
        self.mpc._warmed = False             # re-converge the first solve
        if getattr(self, "_net_stop", None) is not None:
            self._net_stop.clear()           # a policy reused after shutdown()
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
        if depth_m.shape[0] >= 224 and depth_m.shape[1] >= 224:
            h, w = depth_m.shape[:2]
            self._depth_probe = {"top": float(depth_m[h // 6, w // 2]), "centre": float(depth_m[h // 2, w // 2]),
                                 "low": float(depth_m[3 * h // 4, w // 2]), "bottom": float(depth_m[h - 12, w // 2]),
                                 "min": float(depth_m.min()), "median": float(np.median(depth_m)), "shape": [int(h), int(w)]}
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
            # R^T omega, NOT omega. The evaluation harness's Obs.omega is
            # ALREADY a body rate (sim_episode.run_episode integrates it from
            # dR = R^T R_new) and encode_state then applies R^T again, so what
            # the student was scored with is R^T omega_body. Physically odd, but
            # it is also harmless: the TRAINING data's omega column is drawn
            # noise, not a measured rate (draw_states.synthesize_attitude:
            # omega = rng.normal(0, 0.3, 3), independent of R) and the loader
            # passes it through unrotated (data_loader rotates velocity only),
            # so the net learned nothing frame-dependent from this channel.
            # Matching test-5 is therefore the only tie-break, and this is it.
            R_enu.T @ np.asarray(angular_body, np.float64).reshape(3),
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
    def _adopt_plan(self, alphas, trajectories, pos, R_enu, depth_hw, vel=None,
                    t_stamp=None):
        """Turn a net output into the cached world-frame plan. pos/R_enu/vel must
        be the SNAPSHOT the net input was built from (matters in threaded mode)."""
        if not np.isfinite(np.asarray(trajectories)).all() or not np.isfinite(
                np.asarray(alphas)).all():
            # A non-finite plan would propagate NaN through build_reference into
            # the attitude quaternion streamed to PX4, and the veto would pass it
            # (min of a NaN window is NaN, NaN < -margin is False). Keep flying
            # the previous plan instead; it is at most one decision period old.
            print("[agile] net returned a non-finite plan; keeping the previous one.",
                  flush=True)
            if self._world_points is not None:
                return
            raise RuntimeError("[agile] first net output is non-finite")
        local_per_mode = [np.asarray(t, np.float64).reshape(
            self.config.state_dim, self.config.out_seq_len) for t in trajectories]
        # World-frame waypoints first: the veto's tie rule needs them, and the
        # selection must not change the plans it is choosing between.
        wp_per_mode = np.stack(
            [pos[None, :] + (R_enu @ self._scale_body_plan(m)).T
             for m in local_per_mode], axis=0)                     # (modes, N, 3)
        if self.is_student:
            # sim_episode.run_episode clips the WORLD z of every mode into Z_REF
            # before select_mode and flies the clipped plan. Do it here, in the
            # same order and the same frame. The veto still runs on the RAW body
            # waypoints, exactly as DepthVetoPolicy does.
            wp_per_mode[..., 2] = np.clip(wp_per_mode[..., 2], *self.z_band())
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
        if self.is_student:
            # sim_episode.fit_cubic(p, v, modes_w[sel]) -- pinned to the state
            # the plan was made from, tracked from there at tau = t - t_decision.
            v0 = np.zeros(3) if vel is None else np.asarray(vel, np.float64)
            self._cubic = fit_cubic_state(pos, v0, wp_per_mode[self._mode_idx],
                                          self.waypoint_dt)
            # The plan's vertical profile, origin-prefixed: times 0, dt, 2dt...
            # against the CLIPPED world z. _reference_vz differentiates this.
            z_prof = self._world_points_per_mode[self._mode_idx][:, 2]
            self._plan_z = (self.waypoint_dt * np.arange(len(z_prof)),
                            np.asarray(z_prof, np.float64).copy())
            # The cubic is pinned to the state the net INPUT was built from, so
            # the plan's clock starts there -- not at adoption. In threaded mode
            # those differ by one forward pass (~140 ms), which at 3 m/s would
            # park the reference's t=0 point ~0.4 m behind the vehicle for the
            # plan's whole life: a constant backwards position error into the
            # MPC, i.e. chronic braking.
            now = time.time() if t_stamp is None else float(t_stamp)
            self._plan_time = now
            self._net_stamps.append(now)
            if len(self._net_stamps) > 16:
                del self._net_stamps[:-16]

    def _net_worker_loop(self):
        """Latest-only inference worker: always consumes the freshest snapshot,
        so the effective net rate saturates at 1/forward-time (~16 Hz on this
        CPU) instead of being gated AND blocked by the control loop."""
        while not self._net_stop.is_set():
            self._net_event.wait()
            if self._net_stop.is_set():
                return
            with self._net_lock:
                req = self._net_req
                self._net_req = None
                self._net_event.clear()
            if req is None:
                continue
            depth_in, state_in, pos, R_enu, depth_hw, vel, t_submit = req
            try:
                alphas, trajs = self.net.infer(depth_in, state_in)
            except Exception as exc:      # never kill the worker
                print(f"[agile] net worker infer failed: {exc}", flush=True)
                continue
            with self._net_lock:
                self._net_res = (alphas, trajs, pos, R_enu, depth_hw, vel, t_submit)

    def _encode_state(self, pos, R_enu, vel, angular_body, goal_enu, goal_dir):
        """Student -> the 22-dim metric encoding; TF checkpoint -> the legacy
        21-dim de-yawed/look-ahead-direction one."""
        if self.is_student:
            return self._student_state_to_model_input(
                pos, R_enu, vel, angular_body, goal_enu)
        return self._state_to_model_input(pos, R_enu, vel, angular_body, goal_dir)

    def _reference_vz(self, minfo) -> float:
        """The vertical velocity of the reference being tracked this tick.

        Student: the cubic's own vz at the time the tracker is at
        (tau = plan age + one control period, where the setpoint lands). Legacy
        --alt-follow: the MPC's stage-1 reference velocity if it reports one,
        else a finite difference against the current setpoint."""
        if self.is_student and self._plan_z is not None:
            # NOT the cubic's vz: the cubic is pinned to p'(0) = v_current, so
            # for the first half-second of every plan its vertical velocity IS
            # the vehicle's own. Advancing the setpoint with that makes it chase
            # the vehicle again -- slower than p_ref1 did, but the same disease
            # (measured: a level plan ratcheted the setpoint up 0.28 m during
            # the recovery climb and stayed there).
            #
            # The plan's vertical INTENT is the shape of its waypoint z profile,
            # which is a difference and so carries none of the vehicle's sag or
            # velocity. Level plan -> exactly 0; a climb-over -> its climb rate.
            t, z = self._plan_z
            tau = self.control_dt
            if self._plan_time is not None:
                tau += max(0.0, time.time() - self._plan_time)
            return float((np.interp(tau + self.control_dt, t, z)
                          - np.interp(tau, t, z)) / self.control_dt)
        v_ref1 = minfo.get("v_ref1") if isinstance(minfo, dict) else None
        if v_ref1 is not None:
            return float(np.asarray(v_ref1, np.float64)[2])
        if self._alt_sp is not None:
            return (float(minfo["p_ref1"][2]) - self._alt_sp) / max(self.control_dt, 1e-6)
        return 0.0

    def _advance_alt_setpoint(self, minfo, lo: float, hi: float,
                              z_now: float | None = None) -> float:
        if self._alt_sp is None:
            # Start where the policy took over, not at the plan's first node:
            # the handover altitude is the only absolute the vehicle agrees on.
            self._alt_sp = float(np.clip(self._cruise_alt, lo, hi))
        vz_ref = self._reference_vz(minfo)
        sp = self._alt_sp + vz_ref * self.control_dt
        if z_now is not None and self.alt_lead > 0.0:
            # The plan's vertical intent is RELATIVE ("be 0.6 m higher in 0.5
            # s"), and the net keeps asking for it as long as the vehicle has
            # not got there -- so integrating its slope into an absolute
            # setpoint is a ratchet: measured 1.6 -> 4.0 m in three seconds,
            # against a vehicle that had not moved anything like that far. Cap
            # how far the setpoint may run ahead of the vehicle. This does NOT
            # reintroduce 16d6251's disease (a target pinned to the vehicle, so
            # alt_err == 0 by construction and a throttle-model error can only
            # appear as a permanent sink): every error smaller than the lead is
            # untouched, and the sink it cured was 0.43 m/s against a lead of
            # 0.75 m.
            sp = float(np.clip(sp, z_now - self.alt_lead, z_now + self.alt_lead))
        self._alt_sp = float(np.clip(sp, lo, hi))
        return self._alt_sp

    def _update_hover_estimate(self, thrust: float, cos_tilt: float,
                               vz: float, alt_err: float) -> None:
        """At equilibrium the commanded vertical throttle IS the true hover
        throttle -- whatever parameter was assumed, the integrator has supplied
        the difference. Sample only there, EMA, and adopt once converged. The
        integrator is rebased on adoption so the swap is bumpless."""
        if not self.hover_estimate_enabled:
            return
        if abs(vz) > HOVER_EST_VZ_TOL or abs(alt_err) > HOVER_EST_ERR_TOL:
            self._hover_n = 0          # consecutive settled samples only
            return
        sample = float(thrust) * float(cos_tilt)
        if not (HOVER_EST_BOUNDS[0] <= sample <= HOVER_EST_BOUNDS[1]):
            self._hover_n = 0
            return
        self._hover_est = (sample if self._hover_est is None else
                           (1 - HOVER_EST_ALPHA) * self._hover_est
                           + HOVER_EST_ALPHA * sample)
        self._hover_n += 1
        if self._hover_n < HOVER_EST_MIN_SAMPLES:
            return
        old, new = self.hover_thrust, float(self._hover_est)
        if abs(new - old) < HOVER_EST_ADOPT_STEP:
            return
        # Each adoption costs the loop a transient, so make the next one earn
        # another full settled window rather than re-rebasing every tick.
        n_settled, self._hover_n = self._hover_n, 0
        # Rebase the integrator: it was holding az_ss = G (h_true/h_old - 1) to
        # cover the old parameter's error; with the new parameter that demand is
        # gone, so remove it rather than letting it drive a climb.
        self._alt_i = float(np.clip(self._alt_i - G * (new / max(old, 1e-6) - 1.0),
                                    -self.alt_i_limit, self.alt_i_limit))
        self.hover_thrust = new
        if not self._hover_adopted:
            self._hover_adopted = True
            print(f"[agile] hover throttle: configured {self.hover_thrust_param:.3f}, "
                  f"measured {new:.3f} over {n_settled} settled samples -- using "
                  f"the measurement. (Too low a hover throttle with --alt-follow "
                  f"shows up as a steady sink, not as an offset.)", flush=True)

    def z_band(self) -> tuple[float, float]:
        """The student's vertical band for this flight (see student_z_band)."""
        return student_z_band(self._cruise_alt)

    def shutdown(self):
        """Stop the inference worker and join it. onnxruntime tears its thread
        pool down from a daemon thread mid-run with `terminate called without an
        active exception` -- a SIGABRT that the comparison harness would score as
        a trial error. The offboard calls this from its finally block."""
        self._net_stop.set()
        self._net_event.set()
        w = self._net_worker
        if w is not None and w.is_alive():
            w.join(timeout=2.0)
        self._net_worker = None

    def net_hz(self) -> float:
        """Achieved decision rate over the last few plans (0 until there are two).
        The offboard logs this: one forward pass costs 120-220 ms of CPU, so the
        student can fall short of sim_episode's 15 Hz on a loaded box and any
        result has to be read knowing which it was."""
        if len(self._net_stamps) < 2:
            return 0.0
        span = self._net_stamps[-1] - self._net_stamps[0]
        return (len(self._net_stamps) - 1) / span if span > 1e-6 else 0.0

    def _net_tick_threaded(self, obs, pos, R_enu, vel, goal_enu, goal_dir):
        """Submit the freshest observation, adopt the latest finished plan.
        Blocks only on the very first call (no plan exists yet)."""
        if self._net_worker is None:
            self._net_worker = threading.Thread(target=self._net_worker_loop,
                                                daemon=True)
            self._net_worker.start()
        # Submit no faster than the rate the student was scored at: the worker
        # is latest-only, so without this gate a fast box would decide at
        # 1/forward_time instead of sim_episode's DECISION_HZ.
        now = time.time()
        submit = (now - self._net_submit_t) >= (1.0 / self.net_decision_hz) \
            if self.net_decision_hz > 0 else True
        res = None
        if submit:
            depth_in = self._depth_to_model_input(obs.depth)
            state_in = self._encode_state(pos, R_enu, vel, obs.angular_rate_body,
                                          goal_enu, goal_dir)
            self._net_submit_t = now
        with self._net_lock:
            if submit:
                self._net_req = (depth_in, state_in, pos.copy(), R_enu.copy(),
                                 obs.depth, vel.copy(), now)
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
            if self.is_student:
                lo, hi = self.z_band()
                print(f"[agile] student vertical band {lo:.2f}-{hi:.2f} m "
                      f"(handover at {self._cruise_alt:.2f} m; the labels flew "
                      f"{STUDENT_Z_REF[0]:.1f}-{STUDENT_Z_REF[1]:.1f} m"
                      + ("" if hi <= STUDENT_Z_REF[1] + 1e-9 else
                         " -- HANDOVER IS ABOVE THE TRAINING BAND, the ceiling "
                         "was raised to contain it; prefer a lower --climb-alt")
                      + ").", flush=True)
                # Named at launch so a log says, without inference, that the cap
                # reached the paths that matter (both were silently inert once).
                print(f"[agile] student speed cap {self.max_vel:.2f} m/s: "
                      f"MPC reference velocity + along-track command clamp "
                      f"(3-D, lag {STUDENT_CMD_LAG_S:.2f} s); tilt rescaled to "
                      f"the MPC's own thrust; altitude setpoint leads by at "
                      f"most {self.alt_lead:.2f} m "
                      f"(sim_episode.V_CAP = 3.50).", flush=True)

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
            t_submit = time.time()
            alphas, trajectories = self.net.infer(depth_in, state_in)
            self._adopt_plan(alphas, trajectories, pos, R_enu, obs.depth, vel,
                             t_submit)

        world_points = self._world_points
        # Points the MPC's reference cubic is fitted through (see
        # STUDENT_MPC_WAYPOINTS): the whole plan for the legacy net, the first
        # four (t = 0, 0.5, 1.0, 1.5 s) for the student.
        mpc_points = (world_points if self.mpc_waypoints is None
                      else world_points[:self.mpc_waypoints])

        # --- MPC tracking (every tick) ---
        attitude_q = None
        tracker = "pd"
        t_mpc = None            # the MPC's own collective thrust [m/s^2]
        alt_target = self._cruise_alt
        keepout = self._keepout_list() if self.use_keepout else None
        x0 = state_x0(pos, R_enu, vel)
        try:
            _u0, status, minfo = self.mpc.compute(
                x0, mpc_points.astype(np.float64), self._cruise_alt, yaw_des,
                dt_wp=self.waypoint_dt, max_vel=self.max_vel,
                obstacles_xy_r=keepout, alt_hold=not self.alt_follow,
                att_lookahead_s=self.att_lookahead_s,
                # Student: track the state-pinned cubic itself, offset by the
                # plan's age so the manoeuvre is consumed between net updates
                # rather than dragged along with the vehicle (sim_episode
                # evaluates at tau = t - t_decision).
                cubic=self._cubic if self.is_student else None,
                t_offset=(0.0 if self._plan_time is None
                          else max(0.0, now - self._plan_time)),
                min_alt=self.z_band()[0] if self.is_student else 0.15,
                max_alt=self.z_band()[1] if self.is_student else None)
            if status in (0, 2):
                attitude_q = np.asarray(minfo["q_pred"], dtype=np.float64)
                if self.max_tilt_deg < 89.0:
                    attitude_q = np.asarray(
                        clamp_attitude_tilt(attitude_q, self.max_tilt_deg, yaw_des),
                        dtype=np.float64)
                tracker = "mpc"
                # .get: a stub/older MPC may not report it; the rescale is
                # then simply not applied, never an exception on the hot path.
                u0 = minfo.get("u0") if isinstance(minfo, dict) else None
                t_mpc = None if u0 is None else float(np.asarray(u0, np.float64)[0])
                if self.alt_follow:
                    # Follow the plan's vertical profile WITHOUT losing the
                    # absolute reference. p_ref1[2] is node 1 of a cubic pinned
                    # to the vehicle's own position at every replan, so using it
                    # directly makes alt_err ~ 0 by construction no matter how
                    # far the vehicle has sagged, and a steady throttle-model
                    # error can then only be balanced by a permanent sink.
                    # Instead: integrate an ABSOLUTE setpoint at the reference's
                    # own vertical velocity. Climbs and dives in the plan are
                    # still tracked; a sag is now a real error kp/ki can remove.
                    lo, hi = (self.z_band() if self.is_student
                              else (1.0, self._cruise_alt + 3.0))
                    alt_target = self._advance_alt_setpoint(minfo, lo, hi,
                                                            z_now=float(pos[2]))
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

        # --- altitude loop. Computed BEFORE the horizontal corrections below,
        # because az does not depend on the attitude and they need the specific
        # force it implies. alt_target is the cruise altitude without
        # --alt-follow, and the absolute _alt_sp (advanced by the plan's own vz,
        # and kept within STUDENT_ALT_LEAD of the vehicle) with it -- never a
        # target re-pinned to the vehicle, which would make the error
        # identically zero and leave a throttle-model error to be balanced by a
        # permanent sink.
        alt_err = alt_target - float(pos[2])
        # Conditional integration: the integrator exists to remove a STEADY
        # error at equilibrium (the hover-throttle mismatch of 16d6251). While
        # the setpoint is ramping away faster than the vehicle can follow, the
        # error is not an equilibrium error, and integrating it is how a 2026-
        # 09-17 climb reached the +4 m/s^2 integrator limit and then overshot
        # its own setpoint 4x (setpoint +0.60 m/s, vehicle +2.30 m/s). Inside
        # the lead band nothing changes.
        if (not self.is_student) or abs(alt_err) < self.alt_lead:
            self._alt_i = float(np.clip(
                self._alt_i + self.ki_alt * alt_err * self.control_dt,
                -self.alt_i_limit, self.alt_i_limit))
        az = float(np.clip(
            self.kp_alt * alt_err - self.kd_alt * float(vel[2]) + self._alt_i,
            -4.0, 8.0))

        if self.is_student:
            # The specific force the vehicle is about to be given [m/s^2], from
            # the attitude as it stands. One pass: both corrections below only
            # ever REDUCE tilt, so the final cos(tilt) is larger and the real
            # force slightly smaller -- this estimate is conservative in the
            # direction that matters.
            R0 = Rotation.from_quat([attitude_q[1], attitude_q[2], attitude_q[3],
                                     attitude_q[0]]).as_matrix()
            f_cmd = (az + G) / max(0.5, float(R0[2, 2]))
            # (1) the MPC solved its tilt against its OWN collective thrust; we
            # stream the altitude loop's. Rescale so the horizontal
            # acceleration is the one it planned.
            if t_mpc is not None:
                attitude_q = np.asarray(
                    scale_tilt_to_thrust(attitude_q, t_mpc, f_cmd, yaw_des),
                    dtype=np.float64)
            # (2) sim_episode.clamp_command, lag-compensated: above V_CAP the
            # tracker may turn and brake but may not add speed along its own
            # direction of travel. Neither the capped reference (the MPC trades
            # velocity error against a position reference a whole horizon
            # ahead, 100 vs 10) nor the MPC's input bounds enforce this.
            attitude_q = np.asarray(
                clamp_speed_command(attitude_q, vel, self.max_vel, yaw_des,
                                    thrust_cmd=f_cmd, lag_s=STUDENT_CMD_LAG_S),
                dtype=np.float64)
            # The clamp BRAKES above the cap, and a large excess asks for a
            # large deceleration -- which is still an attitude PX4 has to fly.
            # Re-apply the same tilt limit the tracker's own command got.
            if self.max_tilt_deg < 89.0:
                attitude_q = np.asarray(
                    clamp_attitude_tilt(attitude_q, self.max_tilt_deg, yaw_des),
                    dtype=np.float64)
        self._prev_q = attitude_q.copy()

        # Thrust: the altitude loop's az, tilt-compensated through the FINAL
        # attitude and normalized by the hover throttle.
        R_cmd = Rotation.from_quat(
            [attitude_q[1], attitude_q[2], attitude_q[3], attitude_q[0]]).as_matrix()
        cos_tilt = max(0.5, float(R_cmd[2, 2]))
        thrust = float(np.clip((az + G) / cos_tilt / G * self.hover_thrust, 0.05, 0.9))
        self._update_hover_estimate(thrust, cos_tilt, float(vel[2]), alt_err)
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
            depth_probe=getattr(self, '_depth_probe', None),
            net_hz=self.net_hz(),
            alt_sp=float(alt_target),
            hover_thrust=self.hover_thrust,
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
