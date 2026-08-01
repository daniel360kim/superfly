"""gs_drone_sim multi-hypothesis student policy core: obs + MAVLink state -> attitude/thrust.

Loads a MultiHypothesisTrajectoryNet student checkpoint (student.pt) from the
gs_drone_sim repo and reproduces its closed-loop deploy semantics
(scripts/eval_il_closedloop.py + policy/controller.py:track) against PX4:

  net (decimated, ~15 Hz): 224x224 obs (RGB in [0,1] and/or log-encoded metric
    depth, per the checkpoint's use_rgb/use_depth config) + 6-dim state vec
    [R^T v / 5, R^T (goal - pos) / 10]  ->  M=3 body-frame waypoint
    trajectories (T=10 @ label_dt=0.1 s) + per-mode costs; fly argmin cost.
  tracker (every tick): gs_drone_sim's geometric flatness tracker re-expressed
    for PX4 SET_ATTITUDE_TARGET: f_des = kp (p_des - p) + kv (v_des - v) + g e3
    -> desired attitude (yaw toward v_des, matching track()); collective from
    f_des . b3 -> normalized throttle. PX4's inner attitude loop replaces the
    katt body-rate law (we stream attitude setpoints, not body rates).

Frame conventions: gs_drone_sim's dynamics frame is z-up metric with an FLU
body (x fwd, y left, z up) — identical to the ENU world / FLU body used by the
other offboards here, so no extra remapping: vel_body = R_enu^T vel_enu,
world waypoints = pos + R_enu @ wp_body.

X15 test-time augmentation: env GSDS_TTA_N=N (N>1) averages the policy over
N-1 color-jittered copies of the RGB frame per net tick (clean frame always
included; per-sample mode commitment before averaging). See the GSDS_TTA_*
constants below. Off by default; unset => byte-identical behavior.

Depth far-cap remap: the u16-mm wire codec saturates at 65.535 m but the
student's encode_depth treats far/sky as DEPTH_HI = 100 m (gsplat renders mask
sky to "no return" -> 100). Pixels arriving >= FAR_CAP are remapped to 100 m
before encoding so Isaac sky pixels land on the value the net saw in training.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

GSDS_REPO = os.environ.get("GSDS_REPO", "/home/danielkim/gs_drone_sim")
if os.path.join(GSDS_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(GSDS_REPO, "src"))

import torch  # noqa: E402

from gs_drone_sim.policy.model import MultiHypothesisTrajectoryNet  # noqa: E402
from gs_drone_sim.policy.data import (  # noqa: E402
    DEPTH_HI, GOAL_SCALE, VEL_SCALE, encode_depth,
)

GSDS_IMG_SIZE = 224
GSDS_FOV_DEG = 90.0
# Depth-guard veto radius: honest Iris prop-tip radius by default; widen
# (e.g. 0.45-0.55) to veto side-swipe-range hypotheses earlier.
GSDS_GUARD_RADIUS = float(os.environ.get("GSDS_GUARD_RADIUS", "0.354"))
# Keepout-lite depth repulsion (see _repulsion_bias_y): tunables. THRESH =
# planar depth [m] below which pixels repel; CAP = hard bound on the lateral
# plan bias [m]; COVER = fraction of the central window that must be
# proximity-weighted for full authority (ramp; stray pixels ~ no bias).
GSDS_REPULSION_THRESH = float(os.environ.get("GSDS_REPULSION_THRESH", "4.0"))
GSDS_REPULSION_CAP = float(os.environ.get("GSDS_REPULSION_CAP", "1.0"))
GSDS_REPULSION_COVER = float(os.environ.get("GSDS_REPULSION_COVER", "0.15"))
# X15 deploy-side test-time augmentation (TTA), env-gated. GSDS_TTA_N=N (N>1)
# runs the student, per net tick, on the clean frame PLUS N-1 lightly
# color-jittered copies (one batched forward), then flies the AVERAGE of the
# per-sample selected trajectories — see tta_select_average for why the
# average must happen AFTER per-sample mode commitment, never per-mode.
# Jitter ranges (uniform draws, gs_drone_sim.domain_rand.color_jitter):
# hue ±GSDS_TTA_HUE of the hue circle, saturation/contrast scale 1±delta.
# T4 purity rule: this is still RGB-only at test — the jitter perturbs the
# RGB input tensor only; no depth or other modality is consulted by the
# augmentation, so TTA'd headline rows remain legal "RGB-only at deploy"
# (label them +TTA). Unset or N<=1 => byte-identical pre-X15 behavior.
GSDS_TTA_HUE = float(os.environ.get("GSDS_TTA_HUE", "0.03"))
GSDS_TTA_SAT = float(os.environ.get("GSDS_TTA_SAT", "0.15"))       # 0.85-1.15x
GSDS_TTA_CONTRAST = float(os.environ.get("GSDS_TTA_CONTRAST", "0.15"))
GSDS_TTA_SEED = int(os.environ.get("GSDS_TTA_SEED", "0"))
FAR_CAP = 65.0          # wire-codec saturation threshold -> remap to DEPTH_HI
G = 9.80665

# ENU inertial -> NED inertial and FLU body -> FRD body (same as agile_core).
_rot_ENU_to_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
_rot_FLU_to_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])


def quat_enu_flu_to_ned_frd_wxyz(R_enu_flu: np.ndarray) -> np.ndarray:
    rot = _rot_ENU_to_NED * Rotation.from_matrix(R_enu_flu) * _rot_FLU_to_FRD
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _tta_n_from_env() -> int:
    """Parse GSDS_TTA_N. Unset/empty -> 1 (TTA off). N<=1 -> 1. A set-but-
    unparsable value raises: a typo'd screen env must fail loudly, not
    silently fly non-TTA and pollute the comparison."""
    raw = os.environ.get("GSDS_TTA_N", "").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(f"GSDS_TTA_N must be an integer, got {raw!r}") from e
    return n if n > 1 else 1


def tta_select_average(traj: np.ndarray, cost: np.ndarray,
                       modes: np.ndarray | None = None):
    """X15 TTA aggregation: per-sample mode commitment, then average.

    traj (S,M,T,3), cost (S,M) — S TTA samples (sample 0 = clean frame) of an
    M-hypothesis head. `modes` (S,) optionally overrides the per-sample
    selection (deploy passes the depth-guarded selection); default is
    per-sample argmin cost.

    Why not average traj/cost over samples per mode index: the M hypothesis
    slots have no canonical identity across different inputs — under jitter
    the head may emit the same hypothesis set permuted, and per-index
    averaging would then blend DIFFERENT maneuvers (e.g. dodge-left averaged
    with dodge-right = fly straight into the obstacle). Instead each sample
    first commits to ONE trajectory (its selected mode); the committed
    trajectories are averaged. Result is invariant to any per-sample
    permutation of the (traj, cost) mode axis.

    Returns (avg_traj (T,3), modes (S,), disagree) where disagree is the
    fraction of samples 1..S-1 whose selected mode INDEX differs from the
    clean sample's — the selection-instability diagnostic (high = tiny
    appearance jitter flips which hypothesis wins).
    """
    traj = np.asarray(traj)
    cost = np.asarray(cost)
    S = traj.shape[0]
    if modes is None:
        modes = cost.argmin(axis=1)
    modes = np.asarray(modes, dtype=np.int64)
    picked = traj[np.arange(S), modes]                     # (S,T,3)
    avg = picked.mean(axis=0)
    disagree = float(np.mean(modes[1:] != modes[0])) if S > 1 else 0.0
    return avg, modes, disagree


@dataclass
class GsdsObs:
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    R_enu: np.ndarray              # body FLU -> world ENU
    goal_enu: np.ndarray
    depth: np.ndarray | None       # (224,224) planar Z-depth [m], row 0 = up
    rgb: np.ndarray | None         # (224,224,3) uint8 RGB, row 0 = up


@dataclass
class GsdsCmd:
    attitude_ned_frd_wxyz: np.ndarray
    thrust_norm: float
    tracker: str = "pd"            # "pd" | "wait" (no obs frame yet)
    mode_idx: int = 0
    tilt_cmd_deg: float = 0.0
    costs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spread_m: float = 0.0          # inter-hypothesis disagreement (diagnostic)
    repl_m: float = 0.0            # keepout-lite lateral bias, signed body-y [m]


class GsdsPolicy:
    """Owns the student net and the flatness tracker; compute(obs) at control_hz."""

    def __init__(self, checkpoint_path: str, max_vel: float = 2.0,
                 hover_thrust: float = G / 20.0, control_hz: float = 50.0,
                 net_every: int = 3, lookahead: int = 3,
                 kp: float = 6.0, kv: float = 4.0,
                 max_tilt_deg: float = 45.0, alt_mode: str = "plan",
                 yaw_mode: str = "vel", goal_clip: float = 30.0,
                 dump_obs_dir: str | None = None,
                 depth_guard: bool = False,
                 depth_repulsion: float = 0.0,
                 device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"[gsds] loading student checkpoint {checkpoint_path} "
              f"on {self.device} ...", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt["config"]
        self.use_rgb = bool(cfg.get("use_rgb", True))
        self.use_depth = bool(cfg.get("use_depth", False))
        self.dt_wp = float(cfg.get("label_dt", 0.1))
        self.model = MultiHypothesisTrajectoryNet(
            modes=cfg["modes"], horizon=cfg["horizon"],
            use_rgb=self.use_rgb, use_depth=self.use_depth, pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[gsds] student ready: modes={cfg['modes']} horizon={cfg['horizon']} "
              f"use_rgb={self.use_rgb} use_depth={self.use_depth} "
              f"dt_wp={self.dt_wp} trained_scenes={cfg.get('scenes')}", flush=True)

        self.max_vel = float(max_vel)
        self.hover_thrust = float(hover_thrust)
        self.control_dt = 1.0 / float(control_hz)
        self.net_every = max(1, int(net_every))
        self.lookahead = int(lookahead)
        self.kp, self.kv = float(kp), float(kv)
        self.max_tilt_deg = float(max_tilt_deg)
        assert alt_mode in ("plan", "hold")
        assert yaw_mode in ("vel", "goal")
        self.alt_mode = alt_mode
        self.yaw_mode = yaw_mode
        # Training goal legs were 20-38 m; a 90 m+ scenario goal would push the
        # vec's goal channel ~2.5x out of the training range. Clip the DISTANCE
        # (keep the direction) so the net input stays in-distribution; 0 = off.
        self.goal_clip = float(goal_clip)
        # Depth-guarded hypothesis selection (see _depth_guarded_mode): veto
        # visibly blocked hypotheses using the live depth frame. Off by
        # default; enabled via gsds_offboard --depth-guard.
        self.depth_guard = bool(depth_guard)
        self._guard_overrides = 0
        self._guard_last_firstbad = None
        # Keepout-lite depth repulsion (gain in metres at full ramp; 0 = off):
        # input/plan-space margin injection — shifts the PLAN the tracker
        # follows, never vetoes/reselects hypotheses (composable with the
        # guard, which runs first at mode selection). See _repulsion_bias_y.
        self.depth_repulsion = float(depth_repulsion)
        # Optional flight-recorder: save (obs, vec, traj, cost) at every net
        # tick so the policy's real inputs/outputs can be probed offline.
        self.dump_obs_dir = dump_obs_dir
        self._dump_n = 0
        if dump_obs_dir:
            os.makedirs(dump_obs_dir, exist_ok=True)

        # X15 TTA (see module-level GSDS_TTA_* docs). Everything below is
        # inert when GSDS_TTA_N is unset or <=1 — the deploy path is then
        # byte-identical to pre-X15.
        self.tta_n = _tta_n_from_env()
        if self.tta_n > 1 and not self.use_rgb:
            print(f"[gsds] TTA: GSDS_TTA_N={self.tta_n} ignored -- checkpoint "
                  f"has use_rgb=False (color jitter has nothing to touch)",
                  flush=True)
            self.tta_n = 1
        self._tta_gen = None
        self._tta_dis_sum = 0.0
        self._tta_dis_ticks = 0
        self._tta_last_log = 0.0
        if self.tta_n > 1:
            # domain_rand lives in the same gs_drone_sim package this module
            # already hard-imports above (policy.model / policy.data), so in
            # any venv where GsdsPolicy loads at all this import cannot fail
            # independently; it is pure torch (no torchvision/extra deps).
            from gs_drone_sim.domain_rand import ObsDRConfig, color_jitter
            self._tta_jitter_fn = color_jitter
            self._tta_cfg = ObsDRConfig(
                rgb_hue_delta=GSDS_TTA_HUE,
                rgb_sat_delta=GSDS_TTA_SAT,
                rgb_contrast_delta=GSDS_TTA_CONTRAST)
            # color_jitter draws via an explicit generator on the tensor's
            # device; seeded so a given flight's jitter stream is reproducible.
            self._tta_gen = torch.Generator(device=self.device)
            self._tta_gen.manual_seed(GSDS_TTA_SEED)
            self._tta_latency_probe()

        # altitude-hold PD+I (alt_mode="hold", same constants as agile_core);
        # in alt_mode="plan" the I-term still trims hover-thrust mismatch.
        self.kp_alt, self.kd_alt, self.ki_alt = 4.0, 4.0, 0.4

        self.reset()

    def _tta_latency_probe(self) -> None:
        """Measure and print per-net-tick policy time for the chosen TTA N,
        once at startup. TTA runs as ONE batched forward of N samples, so the
        real cost is the batched time (<< naive serial N*F on GPU); both are
        printed. Budget: the control loop ticks at up to 28 Hz (~35.7 ms);
        warn if the measured TTA forward exceeds ~20 ms."""
        n = GSDS_IMG_SIZE
        probe = {"vec": torch.zeros(1, 6, device=self.device)}
        probe["rgb"] = torch.rand(1, 3, n, n, device=self.device)
        if self.use_depth:
            probe["depth"] = torch.zeros(1, 1, n, n, device=self.device)

        def _time_ms(item, reps: int = 10) -> float:
            with torch.no_grad():
                for _ in range(3):                       # warmup
                    self.model(item)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    self.model(item)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps * 1e3

        f_ms = _time_ms(probe)
        tta_ms = _time_ms(self._tta_expand(probe))
        # the probe consumed jitter draws; restore the seeded flight stream
        self._tta_gen.manual_seed(GSDS_TTA_SEED)
        print(f"[gsds] TTA enabled: N={self.tta_n} "
              f"(hue ±{GSDS_TTA_HUE}, sat/contrast ±{GSDS_TTA_SAT}/"
              f"±{GSDS_TTA_CONTRAST}, seed {GSDS_TTA_SEED}); "
              f"single forward {f_ms:.1f} ms, batched TTA forward "
              f"{tta_ms:.1f} ms (naive serial N*F ~ {self.tta_n * f_ms:.1f} ms); "
              f"tick budget 35.7 ms @ 28 Hz", flush=True)
        if tta_ms > 20.0:
            print(f"[gsds] WARNING: TTA per-tick policy time {tta_ms:.1f} ms "
                  f"exceeds the ~20 ms guard (28 Hz budget 35.7 ms) -- risk "
                  f"of missed control ticks; lower GSDS_TTA_N", flush=True)

    def _tta_expand(self, item: dict) -> dict:
        """Replicate a batch-1 net input to tta_n samples: sample 0 keeps the
        clean frame, samples 1..N-1 get independent per-image color jitter on
        the RGB channel ONLY (depth/vec are replicated untouched — RGB-only
        at test, see GSDS_TTA_* docs)."""
        S = self.tta_n
        out = {k: v.expand(S, *v.shape[1:]) for k, v in item.items()}
        jittered = self._tta_jitter_fn(
            item["rgb"].expand(S - 1, -1, -1, -1).contiguous(),
            self._tta_cfg, self._tta_gen)
        out["rgb"] = torch.cat([item["rgb"], jittered], dim=0)
        return out

    def _tta_log(self, modes: np.ndarray, disagree: float) -> None:
        """Once-per-second stderr line: across-sample mode-selection
        disagreement rate (mean over the second's net ticks). Diagnostic for
        appearance sensitivity — a high rate means imperceptible color jitter
        flips which hypothesis wins (the camouflage-failure signature)."""
        self._tta_dis_sum += disagree
        self._tta_dis_ticks += 1
        now = time.time()
        if now - self._tta_last_log >= 1.0:
            print(f"[gsds][tta] N={self.tta_n} sel-disagree="
                  f"{self._tta_dis_sum / max(self._tta_dis_ticks, 1):.2f} "
                  f"over {self._tta_dis_ticks} net ticks; "
                  f"last modes={modes.tolist()}",
                  file=sys.stderr, flush=True)
            self._tta_dis_sum = 0.0
            self._tta_dis_ticks = 0
            self._tta_last_log = now

    def reset(self):
        self._tick = -1
        self._world_points = None        # (T,3) selected trajectory, world ENU
        self._plan_t = 0.0               # wall time of the last net inference
        self._costs = np.zeros(3, dtype=np.float32)
        self._mode_idx = 0
        self._spread = 0.0
        self._repl_bias = 0.0            # last keepout-lite lateral bias [m]
        self._cruise_alt = None
        self._alt_i = 0.0

    # ------------------------------------------------------------------ #
    # Obs encoding (must match policy/data.py + eval_il_closedloop.py)
    # ------------------------------------------------------------------ #
    # Guard camera model — matches run_px4_sim._setup_camera_gsds: 224x224,
    # 90 deg FOV (fx=fy=112, cx=cy=112), forward-facing, 0.10 m ahead of body.
    _G_N = 224
    _G_F = 0.5 * 224 / math.tan(math.radians(45.0))
    _G_OFF = 0.10

    def _depth_guarded_mode(self, traj: np.ndarray, cost: np.ndarray,
                            obs: GsdsObs, mode: int) -> int:
        """Veto hypotheses whose prop-disc-swept path enters observed geometry
        (planar-Z depth frame); pick argmin cost among survivors. If all are
        vetoed, pick the one whose first violation is furthest along the plan.

        The learned cost head is geometry-blind (regressed to
        distance-from-expert); on OOD frames it collapses to a static ranking
        (ATTEMPTS 'net-responsiveness' + s1/s0 diagnoses). This is the
        deploy-side correction: never fly a visibly blocked hypothesis.
        """
        d = np.asarray(obs.depth, dtype=np.float32)
        if d.shape != (self._G_N, self._G_N):
            return mode
        M, T, _ = traj.shape
        first_bad = np.full(M, T + 1, dtype=np.int64)
        r_prop = GSDS_GUARD_RADIUS
        for m in range(M):
            for t in range(T):
                xb, yb, zb = traj[m, t]
                zc = xb - self._G_OFF          # camera fwd
                if zc < 0.4 or zc > 30.0:
                    continue
                u = self._G_N / 2 - self._G_F * (yb / zc)   # left -> smaller u? (x_c = -y_b)
                v = self._G_N / 2 - self._G_F * (zb / zc)   # up -> smaller v  (y_c = -z_b)
                if not (0 <= u < self._G_N and 0 <= v < self._G_N):
                    continue
                rp = int(np.clip(self._G_F * r_prop / zc, 2, 40))
                u0, u1 = max(0, int(u) - rp), min(self._G_N, int(u) + rp + 1)
                v0, v1 = max(0, int(v) - rp), min(self._G_N, int(v) + rp + 1)
                patch = d[v0:v1, u0:u1]
                if patch.size and float(patch.min()) < zc + 0.25:
                    # nearest surface inside the disc window is at or in front
                    # of the waypoint -> flying there sweeps the disc into it
                    first_bad[m] = t
                    break
        clear = np.where(first_bad > T)[0]
        if len(clear):
            sel = int(clear[np.argmin(cost[clear])])
        else:
            far = first_bad.max()
            cands = np.where(first_bad == far)[0]
            sel = int(cands[np.argmin(cost[cands])])
        if sel != mode:
            self._guard_overrides += 1
        self._guard_last_firstbad = first_bad
        return sel

    def _repulsion_bias_y(self, obs: GsdsObs) -> float:
        """Keepout-lite margin injection (transfer memo, agile keepout analog):
        a signed lateral (body-y) bias, computed per net tick from the live
        planar depth frame, that pushes the SELECTED PLAN away from near
        geometry. Acts in input/plan space — the tracker follows the shifted
        waypoints — so it ADDS dodge amplitude the hypothesis set lacks,
        unlike the veto-space guard which can only select among the 3 plans.

        Method: inside the central half FOV window (±26.6 deg at f=112, rows
        and cols; the row band keeps ground/sky out at cruise altitude),
        pixels with planar depth < GSDS_REPULSION_THRESH get proximity weight
        w = (thresh - d)/thresh. The push direction is opposite the weighted
        mean lateral bearing of those pixels (obstacle mass left -> push
        right). Magnitude = gain * (mean-bearing / max-bearing) * ramp, where
        ramp saturates once the weighted coverage reaches GSDS_REPULSION_COVER
        of the window; hard-capped at GSDS_REPULSION_CAP metres. Antisymmetric
        under left-right image mirroring by construction; exactly 0.0 when no
        window pixel is nearer than the threshold.
        """
        d = np.asarray(obs.depth, dtype=np.float32)
        n = self._G_N
        if d.shape != (n, n):
            return 0.0
        c, hw = n // 2, n // 4
        sub = d[c - hw:c + hw, c - hw:c + hw]
        w = np.where(np.isfinite(sub) & (sub > 0.05),
                     np.clip((GSDS_REPULSION_THRESH - sub)
                             / GSDS_REPULSION_THRESH, 0.0, 1.0), 0.0)
        wsum = float(w.sum())
        if wsum <= 0.0:
            return 0.0
        # per-column lateral bearing sine, body-y (left = +): u<c is left
        u = np.arange(c - hw, c + hw, dtype=np.float32) + 0.5
        s_y = -(u - c) / np.hypot(u - c, self._G_F)
        push = -float((w * s_y[None, :]).sum()) / wsum      # away from mass
        sin_max = float(hw / math.hypot(hw, self._G_F))
        ramp = min(1.0, wsum / (GSDS_REPULSION_COVER * w.size))
        bias = self.depth_repulsion * (push / sin_max) * ramp
        return float(np.clip(bias, -GSDS_REPULSION_CAP, GSDS_REPULSION_CAP))

    def _encode_image(self, obs: GsdsObs) -> dict | None:
        item = {}
        n = GSDS_IMG_SIZE
        if self.use_rgb:
            if obs.rgb is None:
                return None
            rgb = np.asarray(obs.rgb)
            if rgb.shape[:2] != (n, n):
                idx = np.linspace(0, rgb.shape[0] - 1, n).astype(np.int64)
                jdx = np.linspace(0, rgb.shape[1] - 1, n).astype(np.int64)
                rgb = rgb[idx[:, None], jdx[None, :]]
            t = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
            t = t.float().div_(255.0).clamp_(0.0, 1.0)
            item["rgb"] = t.permute(2, 0, 1).unsqueeze(0)          # (1,3,H,W)
        if self.use_depth:
            if obs.depth is None:
                return None
            d = np.asarray(obs.depth, dtype=np.float32)
            if d.shape != (n, n):
                idx = np.linspace(0, d.shape[0] - 1, n).astype(np.int64)
                jdx = np.linspace(0, d.shape[1] - 1, n).astype(np.int64)
                d = d[idx[:, None], jdx[None, :]]
            d = d.copy()
            d[~np.isfinite(d)] = DEPTH_HI
            d[d >= FAR_CAP] = DEPTH_HI     # wire-codec saturation -> training far
            t = torch.from_numpy(d).to(self.device)
            item["depth"] = encode_depth(t).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return item

    def _encode_vec(self, obs: GsdsObs) -> torch.Tensor:
        R_bw = obs.R_enu.T
        vel_body = R_bw @ obs.velocity_enu
        goal_body = R_bw @ (obs.goal_enu - obs.position_enu)
        if self.goal_clip > 0.0:
            n = float(np.linalg.norm(goal_body))
            if n > self.goal_clip:
                goal_body = goal_body * (self.goal_clip / n)
        vec = np.concatenate([vel_body / VEL_SCALE,
                              goal_body / GOAL_SCALE]).astype(np.float32)
        return torch.from_numpy(vec).to(self.device).unsqueeze(0)   # (1,6)

    # ------------------------------------------------------------------ #
    def _infer(self, obs: GsdsObs) -> bool:
        item = self._encode_image(obs)
        if item is None:
            return False
        item["vec"] = self._encode_vec(obs)
        tta = self.tta_n > 1
        if tta:
            item = self._tta_expand(item)          # batch of tta_n samples
        with torch.no_grad():
            traj, cost = self.model(item)          # (S,M,T,3), (S,M); S=1 off
        if tta:
            traj_s = traj.cpu().numpy()            # (S,M,T,3)
            cost_s = cost.cpu().numpy()            # (S,M)
            modes = cost_s.argmin(axis=1)
            if self.depth_guard and obs.depth is not None:
                # guard semantics preserved PER SAMPLE (never average in a
                # visibly blocked hypothesis); shares the one live depth frame
                modes = np.array(
                    [self._depth_guarded_mode(traj_s[s], cost_s[s], obs,
                                              int(modes[s]))
                     for s in range(self.tta_n)], dtype=np.int64)
            wp_body, modes, disagree = tta_select_average(traj_s, cost_s,
                                                          modes)
            self._tta_log(modes, disagree)
            # clean-frame sample keeps the diagnostic surface (costs/spread/
            # dump) comparable with non-TTA flights
            traj = traj_s[0]                       # (M,T,3) body frame
            cost = cost_s[0]
            mode = int(modes[0])
        else:
            traj = traj[0].cpu().numpy()           # (M,T,3) body frame
            cost = cost[0].cpu().numpy()
            mode = int(np.argmin(cost))
            if self.depth_guard and obs.depth is not None:
                mode = self._depth_guarded_mode(traj, cost, obs, mode)
            wp_body = traj[mode]                   # (T,3)
        # world-frame anchoring at the pose the obs was rendered from
        if self.depth_repulsion > 0.0:
            # keepout-lite: uniform lateral plan shift (p_des moves, v_des —
            # a finite difference — is untouched). Recomputed per net tick.
            self._repl_bias = (self._repulsion_bias_y(obs)
                               if obs.depth is not None else 0.0)
            if self._repl_bias != 0.0:
                wp_body = wp_body + np.array(
                    [0.0, self._repl_bias, 0.0], dtype=wp_body.dtype)
        self._world_points = (obs.position_enu[None, :]
                              + (obs.R_enu @ wp_body.T).T)
        self._plan_t = time.time()
        self._costs = cost
        self._mode_idx = mode
        centroid = traj.mean(axis=0, keepdims=True)
        self._spread = float(np.linalg.norm(traj - centroid, axis=-1).mean())
        if self.dump_obs_dir:
            try:
                extra = {}
                if tta:
                    # traj_bf/cost/mode above are the CLEAN sample's; the
                    # flown plan is the TTA average — record it + selections
                    extra = dict(tta_modes=modes, tta_wp_body=wp_body,
                                 tta_disagree=disagree)
                np.savez_compressed(
                    os.path.join(self.dump_obs_dir,
                                 f"net_{self._dump_n:05d}.npz"),
                    t=time.time(),
                    depth=(obs.depth if obs.depth is not None else np.zeros(0)),
                    rgb=(obs.rgb if obs.rgb is not None else np.zeros(0)),
                    vec=item["vec"][:1].cpu().numpy(),
                    pos=obs.position_enu, R=obs.R_enu, vel=obs.velocity_enu,
                    goal=obs.goal_enu, traj_bf=traj, cost=cost, mode=mode,
                    **extra)
                self._dump_n += 1
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------ #
    def compute(self, obs: GsdsObs) -> GsdsCmd:
        self._tick += 1
        pos = np.asarray(obs.position_enu, np.float64)
        vel = np.asarray(obs.velocity_enu, np.float64)
        R_enu = np.asarray(obs.R_enu, np.float64)

        if self._cruise_alt is None:
            self._cruise_alt = float(pos[2])

        if self._world_points is None or (self._tick % self.net_every) == 0:
            self._infer(obs)

        if self._world_points is None:
            # No obs frame yet: hold level attitude at hover-ish thrust.
            yaw = math.atan2(R_enu[1, 0], R_enu[0, 0])
            R_cmd = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0]).as_matrix()
            return GsdsCmd(
                attitude_ned_frd_wxyz=quat_enu_flu_to_ned_frd_wxyz(R_cmd),
                thrust_norm=self._hold_thrust(pos, vel, R_cmd),
                tracker="wait")

        wp = self._world_points
        T = wp.shape[0]
        # Receding-horizon index: advance with elapsed time since the plan was
        # made so a 15 Hz replan still tracks a moving reference at control_hz
        # (eval_il_closedloop re-infers every tick; this is the equivalent).
        adv = int((time.time() - self._plan_t) / self.dt_wp)
        li = min(self.lookahead + adv, T - 1)
        p_des = wp[li]
        lj = min(li + 1, T - 1)
        lk = max(li - 1, 0)
        v_des = (wp[lj] - wp[lk]) / ((lj - lk) * self.dt_wp + 1e-6)
        s = float(np.linalg.norm(v_des))
        if s > self.max_vel > 0.0:
            v_des = v_des * (self.max_vel / s)

        f_des = (self.kp * (p_des - pos) + self.kv * (v_des - vel)
                 + np.array([0.0, 0.0, G]))
        if self.alt_mode == "hold":
            # override vertical: PD+I on the cruise altitude (agile-style)
            f_des[2] = G + self._alt_accel(pos, vel)
        else:
            # trim steady hover-thrust mismatch with the slow integrator
            alt_err = float(p_des[2] - pos[2])
            self._alt_i = float(np.clip(
                self._alt_i + self.ki_alt * alt_err * self.control_dt, -2.0, 2.0))
            f_des[2] += self._alt_i

        # tilt clamp: bound the horizontal specific force so the commanded
        # attitude never exceeds max_tilt_deg (gs eval had no clamp; PX4 +
        # a real vehicle model reward one)
        f_z = max(float(f_des[2]), 1.0)
        h_norm = float(np.linalg.norm(f_des[:2]))
        h_max = f_z * math.tan(math.radians(self.max_tilt_deg))
        if h_norm > h_max:
            f_des[:2] *= h_max / h_norm

        # desired attitude: b3 along f_des, yaw toward v_des (track()) or goal
        b3 = f_des / (np.linalg.norm(f_des) + 1e-9)
        if self.yaw_mode == "goal":
            head = (obs.goal_enu - pos).copy()
        else:
            head = v_des.copy()
        head[2] = 0.0
        if np.linalg.norm(head) < 1e-3:
            head = R_enu[:, 0].copy()
            head[2] = 0.0
        xc = head / (np.linalg.norm(head) + 1e-9)
        b2 = np.cross(b3, xc)
        b2 /= (np.linalg.norm(b2) + 1e-9)
        b1 = np.cross(b2, b3)
        R_cmd = np.stack([b1, b2, b3], axis=-1)

        # collective thrust: projection of f_des on the CURRENT body z
        # (controller.py:100), normalized by the harness thrust convention
        # (MAX_ACCEL = G / hover_thrust, same vehicle map as the other offboards)
        b3_meas = R_enu[:, 2]
        c = float(np.clip(np.dot(f_des, b3_meas), 0.0, 40.0))
        thrust = float(np.clip(c / G * self.hover_thrust, 0.05, 0.9))
        tilt_cmd = math.degrees(math.acos(float(np.clip(R_cmd[2, 2], -1.0, 1.0))))

        return GsdsCmd(
            attitude_ned_frd_wxyz=quat_enu_flu_to_ned_frd_wxyz(R_cmd),
            thrust_norm=thrust,
            tracker="pd",
            mode_idx=self._mode_idx,
            tilt_cmd_deg=tilt_cmd,
            costs=self._costs,
            spread_m=self._spread,
            repl_m=self._repl_bias,
        )

    # ------------------------------------------------------------------ #
    def _alt_accel(self, pos, vel):
        alt_err = self._cruise_alt - float(pos[2])
        self._alt_i = float(np.clip(
            self._alt_i + self.ki_alt * alt_err * self.control_dt, -2.0, 2.0))
        return float(np.clip(
            self.kp_alt * alt_err - self.kd_alt * float(vel[2]) + self._alt_i,
            -4.0, 8.0))

    def _hold_thrust(self, pos, vel, R_cmd):
        az = self._alt_accel(pos, vel)
        cos_tilt = max(0.5, float(R_cmd[2, 2]))
        return float(np.clip((az + G) / cos_tilt / G * self.hover_thrust,
                             0.05, 0.9))

    def debug_frame(self, pos_enu, R_enu, tracker: str) -> dict | None:
        """Latest selected trajectory for the overhead debug view (agile-compatible)."""
        if self._world_points is None:
            return None
        yaw = math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0]))
        return dict(
            pos_local=np.asarray(pos_enu, dtype=np.float64).reshape(3),
            yaw=yaw,
            alphas=np.asarray(self._costs, dtype=np.float64).reshape(-1),
            trajectories_local=self._world_points[None, :, :],
            mode_idx=0,
            tracker=tracker,
        )
