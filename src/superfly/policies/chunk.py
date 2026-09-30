"""Velocity-chunk student (ONNX sidecar ``arch: chunk_v1``) for PX4.

Port of superfly_expert_sampler ``sim_episode.OnnxChunkPolicy`` (branch
v8-chunk, 35ebbd3) -- the executor the chunk students are scored with in the
python sim -- minus the point-mass plant: here the command goes to PX4's
velocity loop (chunk_offboard.py).

Contract (agile_student/INPUTS.md, "Velocity chunk"):
  in   depth (1,1,224,224,3) mm/80 clipped at 20 m, tiled   -- as the student
       imu   (1,1,22) [pos, R row-major, v_body, omega_body, goal_body
             (clamped 10 m), v_goal]                          -- as the student
       prev_chunk (1,60) the chunk executed from at the previous decision,
             re-expressed in the CURRENT heading frame, zeros if none
  out  chunk (1,H,60) per head [vx_1..15 | vy | vz | yr] heading frame, SI,
             t = 0.1 j s
       gate  (1,H) logits
       intent (1,H,15) (auxiliary, never executed)
Heading frame: gravity aligned, x along yaw = atan2(R[1,0], R[0,0]).

Executor (per decision, 15 Hz):
  * head = argmax softmax(gate), kept unless another head's probability beats
    it by `hysteresis` (0.15); with `dwell` > 0 (off by default; sim
    --chunk-dwell, v8-chunk 4393ab4) a head just switched to needs a lead of
    max(hysteresis, `dwell_margin` 0.3) for `dwell` seconds after the switch;
  * Plan B track 3 (2026-09-30; off by default = unchanged; odometry only, so
    they carry to an RGB student; python sim twins in v8-chunk sim_episode):
    `side_dwell` -- once the executed head is left/right, leaving it within
    side_dwell s needs `side_margin` (1.0 = hard); `flip_margin` -- a switch
    into the side head opposite the last executed side head needs this margin
    at any time; `stuck_window` -- StuckWatchdog: no goal progress over the
    window while commanding speed -> the best non-straight head is forced for
    `stuck_hold` s (reason "stuck");
  * the selected chunk goes to WORLD frame at its issue time into a ring of the
    last `ensemble` (4) chunks; with `same_head` (default here) the ring is
    emptied on a head switch, so two routes are never blended;
  * command(t) = 0.5^age-weighted mean over ring chunks whose horizon still
    covers t of each chunk's velocity at tau + lead (linear between steps;
    tau = t - issue time) and its yaw rate on the step containing tau.
  * optional smooth execution (SmoothRef, chunk_offboard --smooth, off by
    default): the command is integrated through the sim's plant with jerk /
    accel limits and sent as velocity + acceleration feedforward, yaw rate
    low-passed.
  * optional clearance shield (ObstacleMemory + ClearanceShield,
    chunk_offboard --shield, off by default): a ~1.2 s egocentric memory of
    back-projected depth points, and a velocity filter that removes (or
    reverses) the commanded component toward any remembered point the next
    ~0.5 s of motion would bring inside a margin. Never adds speed.
Dependency-light: numpy + onnxruntime.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

CHUNK_DT = 0.1
CHUNK_STEPS = 15
HEADS = ("straight", "left", "right", "over", "under")
AGILE_FAR = 20.0
GOAL_CLAMP_M = 10.0


def read_sidecar(path) -> dict:
    f = Path(str(path) + ".json")
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def is_chunk_checkpoint(path) -> bool:
    return read_sidecar(path).get("arch") == "chunk_v1"


def heading_yaw(R) -> float:
    R = np.asarray(R, float)
    return math.atan2(R[1, 0], R[0, 0])


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def softmax(x) -> np.ndarray:
    x = np.asarray(x, float)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def encode_depth(depth_m) -> np.ndarray:
    """mm / 80, clipped at 20 m, tiled to 3 channels: (1, 1, 224, 224, 3)."""
    if depth_m is None:
        d = np.full((224, 224), AGILE_FAR, np.float32)
    else:
        d = np.nan_to_num(np.asarray(depth_m, np.float32), nan=AGILE_FAR,
                          posinf=AGILE_FAR, neginf=0.0)
    mm = np.clip(d * 1000.0, 0.0, AGILE_FAR * 1000.0)
    x = (mm / 80.0).astype(np.float32)
    return np.tile(x[None, None, :, :, None], (1, 1, 1, 1, 3))


def encode_state(pos, R, vel, omega_body, goal, goal_speed) -> np.ndarray:
    """The student's 22-dim state, byte-for-byte AgilePolicy.
    _student_state_to_model_input: omega_body goes in unrotated, like training."""
    R = np.asarray(R, np.float64)
    g = R.T @ (np.asarray(goal, np.float64) - np.asarray(pos, np.float64))
    n = float(np.linalg.norm(g))
    if n > 1e-9:
        g = g * (min(n, GOAL_CLAMP_M) / n)
    v = np.concatenate([np.asarray(pos, np.float64).reshape(3), R.reshape(-1),
                        R.T @ np.asarray(vel, np.float64).reshape(3),
                        np.asarray(omega_body, np.float64).reshape(3),
                        g, [float(goal_speed)]])
    return v.astype(np.float32)[None, None]


class SmoothRef:
    """Smooth executor (chunk_offboard --smooth; python sim --chunk-smooth): the
    ensembled chunk command is not sent to PX4 as a bare velocity step every
    1/15 s but integrated through the python sim's own velocity-setpoint plant

        a' = (KV (v_cmd - v_ref) - a) / LAG     (sim_episode KV_VEL 3, LAG 0.1)
        v_ref' = a

    with a jerk limit on a' (|a'| <= jerk) and an acceleration limit
    (|a| <= acc_max); PX4 then gets v_ref AND a as velocity setpoint + accel
    feedforward, so it follows the trajectory the sim's point mass would fly
    instead of reacting to each decision's step. ``a_ff`` (optional) adds the
    chunk's own slope to the plant's acceleration target (the sim's
    --chunk-feedforward; off by default there too). The yaw-rate command goes
    through a first-order low-pass (``yaw_tau``) and a yaw-acceleration limit
    (``yaw_acc``): the network's yaw rate scatters 0.2-0.4 rad/s between
    decisions (Isaac 2026-09-29), which is what shakes the onboard video."""

    def __init__(self, kv: float = 3.0, lag: float = 0.1, jerk: float = 8.0,
                 acc_max: float = 4.0, yaw_tau: float = 0.25, yaw_acc: float = 4.0):
        self.kv, self.lag = float(kv), float(lag)
        self.jerk, self.acc_max = float(jerk), float(acc_max)
        self.yaw_tau, self.yaw_acc = float(yaw_tau), float(yaw_acc)
        self.reset(np.zeros(3))

    def reset(self, v0, a0=None, yr0: float = 0.0):
        self.v = np.asarray(v0, float).reshape(3).copy()
        self.a = np.zeros(3) if a0 is None else np.asarray(a0, float).reshape(3).copy()
        self.yr = float(yr0)

    @staticmethod
    def _clip_norm(x, m):
        n = float(np.linalg.norm(x))
        return x * (m / n) if (m > 0 and n > m) else x

    def step(self, v_cmd, yr_cmd: float, dt: float, a_ff=None):
        """Advance by dt toward (v_cmd, yr_cmd). Returns (v_ref, a_ref, yr_ref)."""
        dt = float(dt)
        if dt <= 0:
            return self.v.copy(), self.a.copy(), self.yr
        a_tgt = self.kv * (np.asarray(v_cmd, float) - self.v)
        if a_ff is not None:
            a_tgt = a_tgt + np.asarray(a_ff, float)
        a_tgt = self._clip_norm(a_tgt, self.acc_max)
        da = (a_tgt - self.a) * min(dt / self.lag, 1.0) if self.lag > 0 else a_tgt - self.a
        da = self._clip_norm(da, self.jerk * dt)
        self.a = self._clip_norm(self.a + da, self.acc_max)
        self.v = self.v + self.a * dt
        # yaw rate: low-pass, then a yaw-acceleration limit
        tgt = self.yr + (float(yr_cmd) - self.yr) * (1.0 - math.exp(-dt / self.yaw_tau)
                                                     if self.yaw_tau > 0 else 1.0)
        dyr = tgt - self.yr
        lim = self.yaw_acc * dt if self.yaw_acc > 0 else abs(dyr)
        self.yr = self.yr + max(-lim, min(lim, dyr))
        return self.v.copy(), self.a.copy(), self.yr


class ObstacleMemory:
    """Short-lived egocentric obstacle memory (chunk_offboard --shield).

    Why: the chunk student is purely reactive, and 16 of the 20 Isaac close
    passes under 0.25 m (2026-09-29, chunkv8L_s2 vc15/sm runs) were SIDE
    passes -- bearing ~90 deg from the velocity -- whose closest obstacle patch
    had left the 91 deg depth view 0.06-0.8 s before closest approach. So each
    depth frame is subsampled and back-projected to local-ENU points with the
    odometry pose at the decision, points older than ``horizon`` are dropped,
    and the set is thinned on a voxel grid (newest wins) and capped to the
    ``cap`` points nearest the vehicle. numpy only, O(1e3) points: cheap
    enough for the Starling's CPU.

    Camera (superfly.sim.px4_sim POLICY_CAMERAS['agile'] + render_depth.Camera,
    the training renderer): 640x480 render, 91 deg hfov, square pixels,
    bilinear to the shipped frame (224x224, so fy != fx there), planar z-depth,
    row 0 = up, col 0 = left, mounted ``cam_pos`` in the FLU body frame. The
    effective mount is level (``cam_pitch_deg`` 0, positive = nose-down):
    Isaac's SUPERFLY_CAM_PITCH_DEG=-13 cancels the 13 deg the raw mount looks
    down, which is what makes its frame match the training renderer.

    Silhouettes: resizing 640x480 -> 224 (cv2 bilinear in px4_sim) blends
    foreground and background into one-pixel-wide phantom depths in free
    space. Those pixels are detected (``_mixed``: strictly between two
    opposite neighbours, a big jump to each) and dropped, and each sampled
    pixel then takes the MIN depth of its (2 edge_px + 1)^2 neighbourhood
    (3x3), so thin bars and silhouettes are still caught, on the foreground
    surface, inflated by <= edge_px pixels (~1.4 cm at 1.5 m)."""

    def __init__(self, horizon: float = 1.2, stride: int = 8, r_min: float = 0.35,
                 r_max: float = 5.0, cap: int = 3000, voxel: float = 0.08,
                 z_floor: float = 0.2, cam_pos=(0.10, 0.0, 0.0), cam_pitch_deg: float = 0.0,
                 hfov_deg: float = 91.0, render_wh=(640, 480), edge_px: int = 1,
                 mix_rel: float = 0.05, mix_abs: float = 0.05):
        self.horizon, self.stride = float(horizon), max(1, int(stride))
        self.edge_px = max(0, int(edge_px))
        self.mix_rel, self.mix_abs = float(mix_rel), float(mix_abs)
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.cap, self.voxel, self.z_floor = int(cap), float(voxel), float(z_floor)
        self.cam_pos = np.asarray(cam_pos, float).reshape(3)
        th = math.radians(cam_pitch_deg)
        # camera (x right, y down, z forward) -> FLU body, then pitched
        # nose-down by th about body +y
        R_bc = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        Ry = np.array([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0],
                       [-math.sin(th), 0.0, math.cos(th)]])
        self.R_bc = Ry @ R_bc
        self.hfov = math.radians(hfov_deg)
        self.render_wh = (int(render_wh[0]), int(render_wh[1]))
        self._grid_key = None
        self.reset()

    def reset(self):
        self.frames: list[tuple[float, np.ndarray]] = []   # (t, (K,3) local ENU)
        self.pts = np.zeros((0, 3))
        self._last_depth = None

    def _grid(self, h: int, w: int):
        """Sampled pixel indices and their camera-frame rays (z_c = 1)."""
        if self._grid_key != (h, w):
            s = self.stride
            rows = np.arange(s // 2, h, s)
            cols = np.arange(s // 2, w, s)
            W0, H0 = self.render_wh
            fx_r = 0.5 * W0 / math.tan(0.5 * self.hfov)
            fx, fy = fx_r * w / W0, fx_r * h / H0
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            C, Rr = np.meshgrid(cols, rows)
            self._rows, self._cols = Rr.ravel(), C.ravel()
            self._rays = np.stack([(self._cols - cx) / fx, (self._rows - cy) / fy,
                                   np.ones(self._rows.size)], 1)
            self._grid_key = (h, w)
        return self._rows, self._cols, self._rays

    @staticmethod
    def _mixed(d, rel: float, tol: float) -> np.ndarray:
        """Pixels strictly between two opposite neighbours (horizontal,
        vertical or either diagonal) with a jump of more than rel * depth + tol
        to EACH: a resampler's blend of foreground and background, i.e. a
        phantom depth in free space. A continuous surface -- even a wall seen at
        grazing incidence, ~1 cm per pixel -- does not trip it; a steep limb
        may, which only drops real points."""
        m = np.zeros(d.shape, bool)
        c = d[1:-1, 1:-1]
        tau = rel * c + tol
        for a, b in ((d[1:-1, :-2], d[1:-1, 2:]), (d[:-2, 1:-1], d[2:, 1:-1]),
                     (d[:-2, :-2], d[2:, 2:]), (d[:-2, 2:], d[2:, :-2])):
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            m[1:-1, 1:-1] |= ((c - lo) > tau) & ((hi - c) > tau)
        return m

    def add(self, t: float, depth, pos, R) -> int:
        """Add one depth frame seen from (pos, R: local ENU <- FLU body).
        A frame object already added (the subscriber's last-seen frame, no new
        one arrived) is skipped. Returns the number of points added."""
        if depth is None or depth is self._last_depth:
            return 0
        self._last_depth = depth
        d = np.nan_to_num(np.asarray(depth, np.float32), nan=np.inf, posinf=np.inf,
                          neginf=0.0)
        h, w = d.shape
        rows, cols, rays = self._grid(h, w)
        d = np.where(self._mixed(d, self.mix_rel, self.mix_abs), np.inf, d)
        z = d[rows, cols]
        e = self.edge_px
        for dr in range(-e, e + 1):
            rr = np.clip(rows + dr, 0, h - 1)
            for dc in range(-e, e + 1):
                z = np.minimum(z, d[rr, np.clip(cols + dc, 0, w - 1)])
        z = np.nan_to_num(z.astype(float), nan=np.inf, posinf=np.inf, neginf=0.0)
        rng = z * np.sqrt((rays ** 2).sum(1))
        ok = (rng > self.r_min) & (rng < self.r_max)
        if not ok.any():
            self._push(t, np.zeros((0, 3)))
            return 0
        p_c = rays[ok] * z[ok, None]
        p_b = p_c @ self.R_bc.T + self.cam_pos
        p_w = p_b @ np.asarray(R, float).T + np.asarray(pos, float).reshape(3)
        p_w = p_w[p_w[:, 2] > self.z_floor]          # the ground is not an obstacle here
        self._push(t, p_w)
        return len(p_w)

    def _push(self, t: float, p_w: np.ndarray):
        self.frames.insert(0, (float(t), p_w))
        self._rebuild(t)

    def _rebuild(self, t: float):
        self.frames = [f for f in self.frames if t - f[0] <= self.horizon + 1e-9]
        pts = np.concatenate([f[1] for f in self.frames], 0) if self.frames else np.zeros((0, 3))
        if len(pts) and self.voxel > 0:                # newest first -> unique keeps the newest
            k = np.floor(pts / self.voxel).astype(np.int64) + (1 << 20)
            key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
            _, idx = np.unique(key, return_index=True)
            pts = pts[np.sort(idx)]
        self.pts = pts

    def points(self, t: float, pos=None) -> np.ndarray:
        """The remembered points at time t (frames older than horizon
        dropped); if more than ``cap``, the cap nearest ``pos``."""
        if self.frames and t - self.frames[-1][0] > self.horizon + 1e-9:
            self._rebuild(t)
        pts = self.pts
        if len(pts) > self.cap and pos is not None:
            d2 = ((pts - np.asarray(pos, float).reshape(3)) ** 2).sum(1)
            pts = pts[np.argpartition(d2, self.cap)[:self.cap]]
        return pts


class ClearanceShield:
    """Velocity filter over ObstacleMemory points (chunk_offboard --shield).

    For every remembered point p (vehicle centre x, c = |p - x|, n = (p - x)/c)
    the commanded velocity must satisfy the discrete barrier condition

        v . n  <=  gain * (c - margin)        (clipped below at -v_rep)

    i.e. the approach speed toward p shrinks linearly to zero at ``margin`` and
    turns into a retreat of up to ``v_rep`` inside it. A point is only
    constrained when the predicted next ``horizon`` seconds of straight-line
    motion (along the command and along the current velocity) pass within
    ``margin`` of it -- with a weight ramping 0 -> 1 as that predicted miss
    distance goes from margin to margin - ``soft`` (always 1 once c < margin)
    so a point entering the gate does not step the command. The worst
    violation is removed by projection, ``iters`` times; finally the result is
    rescaled to at most the input speed: the shield NEVER ADDS SPEED.
    ``margin`` is a centre-to-surface distance: clearance to the airframe is
    ~margin - 0.2 m (drone_radius)."""

    def __init__(self, margin: float = 0.5, horizon: float = 0.5, gain: float = 2.0,
                 v_rep: float = 0.5, soft: float = 0.2, iters: int = 3):
        self.margin, self.horizon, self.gain = float(margin), float(horizon), float(gain)
        self.v_rep, self.soft, self.iters = float(v_rep), float(soft), int(iters)

    @staticmethod
    def _seg_dist(d, u):
        """Distance from points d (relative to the segment start) to the segment 0 -> u."""
        uu = float(u @ u)
        if uu < 1e-12:
            return np.sqrt((d ** 2).sum(1))
        s = np.clip(d @ u / uu, 0.0, 1.0)
        return np.sqrt(((d - s[:, None] * u) ** 2).sum(1))

    def apply(self, v, pos, pts, vel=None):
        """Filter v (local ENU, (3,)). Returns (v_out, info) with info
        {n: points constrained, dmin: nearest remembered point [m], dv: |v_out - v|}."""
        v = np.asarray(v, float).reshape(3).copy()
        info = {"n": 0, "dmin": float("inf"), "dv": 0.0}
        if pts is None or len(pts) == 0:
            return v, info
        x = np.asarray(pos, float).reshape(3)
        d = pts - x
        c = np.sqrt((d ** 2).sum(1))
        info["dmin"] = float(c.min())
        sp = float(np.linalg.norm(v))
        vv = np.zeros(3) if vel is None else np.asarray(vel, float).reshape(3)
        reach = self.margin + self.horizon * max(sp, float(np.linalg.norm(vv))) + 1e-6
        near = c < reach
        if not near.any():
            return v, info
        d, c = d[near], c[near]
        dseg = np.minimum(self._seg_dist(d, v * self.horizon),
                          self._seg_dist(d, vv * self.horizon))
        w = np.clip((self.margin - dseg) / max(self.soft, 1e-6), 0.0, 1.0)
        w[c < self.margin] = 1.0
        act = w > 0
        if not act.any():
            return v, info
        n = d[act] / np.maximum(c[act], 1e-6)[:, None]
        allow = np.maximum(self.gain * (c[act] - self.margin), -self.v_rep)
        w = w[act]
        info["n"] = int(act.sum())
        v0 = v.copy()
        for _ in range(max(self.iters, 1)):
            viol = w * (n @ v - allow)
            k = int(np.argmax(viol))
            if viol[k] <= 1e-6:
                break
            v = v - viol[k] * n[k]
        s1 = float(np.linalg.norm(v))
        if s1 > sp:
            v = v * (sp / s1) if s1 > 1e-9 else v
        info["dv"] = float(np.linalg.norm(v - v0))
        return v, info


REASON_CODES = {"gate": 0, "switch": 1, "hysteresis": 2, "dwell": 3}


class DecisionLog:
    """Per-decision record of the FULL network output (chunk_offboard, on by
    default whenever the runner sets SUPERFLY_STATE_LOG; --no-net-log = off):
    ``chunk_outputs.npy`` + ``chunk_outputs.json`` next to chunk_decisions.csv.

    One structured row per decision (15 Hz), ~1.4 kB: all heads' chunks as the
    network emitted them (H x [vx_1..T | vy | vz | yr], heading frame, SI, step
    j at t = j dt), the gate logits, the selected head / reason, the pose the
    decision was taken at, and the setpoint actually sent on that tick. A
    150 s flight is ~3 MB; typical trials < 1 MB.

    The file is a valid .npy after EVERY row (the header's row count is
    rewritten in place, fixed width), so ``np.load`` reads a flight whose
    offboard was killed. Every write is guarded: a logging failure disables the
    log and never reaches the control loop. Nothing here feeds back into
    control. Reproduce a head's path in the local frame with ``chunk_paths``."""

    VERSION = 1

    def __init__(self, path, heads, steps: int, meta: dict | None = None):
        self.path = Path(path)
        H, T = len(heads), int(steps)
        self.dtype = np.dtype([
            ("t", "<f8"),                  # the chunk_decisions.csv clock (s since offboard start)
            ("pos", "<f4", (3,)),          # local ENU (= csv x, y, z)
            ("vel", "<f4", (3,)),          # local ENU
            ("R", "<f4", (3, 3)),          # local ENU <- FLU body
            ("omega", "<f4", (3,)),        # FLU body rate
            ("yaw", "<f4"),                # heading yaw = atan2(R[1,0], R[0,0]): the chunk frame
            ("head", "i1"),                # executed head (after hysteresis / dwell)
            ("reason", "i1"),              # REASON_CODES
            ("gate", "<f4", (H,)),         # gate LOGITS (softmax -> the csv p0..p4)
            ("chunk", "<f4", (H, 4 * T)),  # raw network chunk per head
            ("cmd", "<f4", (4,)),          # setpoint sent on the decision tick: v ENU, yaw rate
            ("n_ens", "i1"),               # chunks in the temporal ensemble on that tick
        ])
        self.n = 0
        self.ok = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(self._header(0))
        self._f.flush()
        info = {
            "format": f"superfly chunk_outputs v{self.VERSION}",
            "heads": list(heads), "steps": T,
            "chunk_layout": "chunk[h] = [vx_1..vx_T | vy_1..vy_T | vz_1..vz_T | yr_1..yr_T], "
                            "heading frame (x along yaw, gravity aligned), SI; step j is t = j*dt "
                            "after the decision",
            "frames": "pos/vel/goal local ENU of the offboard (origin PX4 home, the frame of "
                      "chunk_decisions.csv x,y,z); R = local ENU <- FLU body; heading frame = "
                      "rz(yaw), yaw = atan2(R[1,0], R[0,0])",
            "integration": "p_0 = pos; p_j = pos + rz(yaw) @ sum_{i<=j} v_i * dt (the executor's "
                           "vh @ rz(yaw).T and the student's ChunkClearanceLoss.positions)",
            "t": "same clock as chunk_decisions.csv t (--clock, seconds since offboard start)",
            "gate": "logits; probabilities = softmax(gate) (csv p0..p4)",
            "reason_codes": REASON_CODES,
            "cmd": "setpoint on the decision tick after the ensemble, lead, v-cap, z band, "
                   "shield, yaw clip (and the smooth plant, if on) = csv cmd_vx..cmd_yr",
        }
        info.update(meta or {})
        Path(str(self.path.with_suffix("")) + ".json").write_text(json.dumps(info, indent=1) + "\n")

    def _header(self, n: int) -> bytes:
        """npy v1.0 header, fixed width whatever n (the count is %12d)."""
        d = ("{'descr': %r, 'fortran_order': False, 'shape': (%12d,), }"
             % (np.lib.format.dtype_to_descr(self.dtype), int(n)))
        pre = 10                                          # magic (8) + uint16 length
        pad = (-(pre + len(d) + 1)) % 64
        d = d + " " * pad + "\n"
        return b"\x93NUMPY\x01\x00" + len(d).to_bytes(2, "little") + d.encode("latin1")

    def add(self, t, pos, vel, R, omega, last: dict, cmd_v, cmd_yr, n_ens) -> None:
        if not self.ok:
            return
        try:
            r = np.zeros((), self.dtype)
            r["t"], r["pos"], r["vel"] = float(t), pos, vel
            r["R"], r["omega"], r["yaw"] = R, omega, last["yaw"]
            r["head"] = int(last["head"])
            r["reason"] = REASON_CODES.get(last.get("reason"), -1)
            r["gate"], r["chunk"] = last["gate"], last["chunk"]
            r["cmd"] = (*np.asarray(cmd_v, float).reshape(3), float(cmd_yr))
            r["n_ens"] = int(n_ens)
            self._f.seek(0, 2)
            self._f.write(r.tobytes())
            self.n += 1
            self._f.seek(0)
            self._f.write(self._header(self.n))
            self._f.flush()
        except Exception as e:                       # never let the log touch control
            self.ok = False
            print(f"[chunk] net log disabled: {type(e).__name__}: {e}", flush=True)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def chunk_paths(chunk, pos, yaw: float, dt: float = CHUNK_DT) -> np.ndarray:
    """Positions of every head's chunk, integrated from the decision pose:
    (H, 4T) chunk + pos (3,) + heading yaw -> (H, T + 1, 3), p_0 = pos,
    p_j = pos + rz(yaw) @ sum_{i<=j} v_i dt -- the executor's heading-to-world
    rotation and the student's ChunkClearanceLoss.positions."""
    c = np.asarray(chunk, float)
    c = c.reshape(-1, c.shape[-1])
    T = c.shape[1] // 4
    v = np.stack([c[:, 0:T], c[:, T:2 * T], c[:, 2 * T:3 * T]], -1)       # (H, T, 3) heading
    p = np.cumsum(v, 1) * float(dt) @ rz(float(yaw)).T
    p = np.concatenate([np.zeros((len(c), 1, 3)), p], 1)
    return p + np.asarray(pos, float).reshape(1, 1, 3)


def yaw_toward(yaw: float, vel, gain: float, v_min: float = 0.5) -> float:
    """Extra yaw rate turning the camera toward the horizontal velocity
    (chunk_offboard --yaw-to-vel): gain * wrap(heading(vel) - yaw), 0 below
    v_min m/s. Swerves put up to 24 deg between yaw and velocity in Isaac."""
    vx, vy = float(vel[0]), float(vel[1])
    if gain <= 0 or math.hypot(vx, vy) < v_min:
        return 0.0
    e = math.atan2(vy, vx) - float(yaw)
    return float(gain) * math.atan2(math.sin(e), math.cos(e))


class StuckWatchdog:
    """Odometry-only stuck detector of the chunk executor (chunk_offboard
    --stuck; python sim --chunk-stuck, superfly_expert_sampler v8-chunk
    sim_episode.StuckWatchdog -- the same class, same numbers).

    At every decision the executor feeds it the distance to the goal
    (observe, before selection) and the speed it then commands (commanded,
    after the ensemble). It fires when, over the last `window` s, the goal
    distance fell by less than `progress` m while the mean commanded speed was
    at least `v_min` m/s and the goal is still more than `min_dist` m away (a
    hover at a stop goal is not "stuck"). Then it forces the best non-straight
    head by gate probability -- never the head in use, never one it already
    forced since the last healthy window -- for `hold` s, and releases; the
    next fire needs a fresh full window. Nothing here reads depth: it carries
    to any input modality."""

    def __init__(self, window: float = 3.0, progress: float = 0.5, v_min: float = 0.5,
                 hold: float = 2.5, min_dist: float = 1.5):
        self.window, self.progress, self.v_min = float(window), float(progress), float(v_min)
        self.hold, self.min_dist = float(hold), float(min_dist)
        self.reset()

    def reset(self):
        self.hist: list[list] = []          # [t, goal distance, commanded speed or None]
        self.forced = None
        self.until = None
        self.tried: list[int] = []
        self.fires: list[tuple] = []         # (t, forced head) per fire

    def observe(self, t: float, dist: float):
        self.hist.append([float(t), float(dist), None])
        while len(self.hist) > 1 and self.hist[1][0] <= t - self.window + 1e-9:
            del self.hist[0]                 # keep exactly one entry at or before t - window

    def commanded(self, speed: float):
        if self.hist:
            self.hist[-1][2] = float(speed)

    def force(self, t: float, probs, current, heads):
        """The head to fly at this decision, or None (not stuck / released)."""
        if self.forced is not None:
            if t < self.until - 1e-9:
                return self.forced
            self.forced = self.until = None
            self.hist = self.hist[-1:]       # a fresh window before the next fire
            return None
        if len(self.hist) < 2 or t - self.hist[0][0] < self.window - 1e-9:
            return None
        d_now = self.hist[-1][1]
        sp = [h[2] for h in self.hist[:-1] if h[2] is not None]
        stuck = (d_now > self.min_dist and self.hist[0][1] - d_now < self.progress
                 and bool(sp) and float(np.mean(sp)) >= self.v_min)
        if not stuck:
            self.tried = []
            return None
        side = [i for i, h in enumerate(heads) if h != "straight" and i != current and i < len(probs)]
        cand = [i for i in side if i not in self.tried]
        if not cand:
            self.tried, cand = [], side
        if not cand:
            return None
        f = max(cand, key=lambda i: (float(probs[i]), -i))
        self.tried.append(f)
        self.forced, self.until = f, float(t) + self.hold
        self.fires.append((round(float(t), 3), int(f)))
        return f


class ChunkPolicy:
    # the executor guards' "off" values, for instances built without __init__ (tests)
    heads = list(HEADS)
    side_dwell, side_margin, flip_margin = 0.0, 1.0, 0.0
    stuck = None
    v_cap = None

    def __init__(self, path, lead: float = 0.5, hysteresis: float = 0.15,
                 ensemble: int = 4, decay: float = 0.5, same_head: bool = True,
                 goal_speed: float = 0.0, threads: int = 4, dwell: float = 0.0,
                 dwell_margin: float = 0.3, side_dwell: float = 0.0, side_margin: float = 1.0,
                 flip_margin: float = 0.0, stuck_window: float = 0.0, stuck_progress: float = 0.5,
                 stuck_v: float = 0.5, stuck_hold: float = 2.5, stuck_min_dist: float = 1.5,
                 v_cap: float | None = None):
        import onnxruntime as ort
        self.path = str(path)
        self.sidecar = read_sidecar(path)
        if self.sidecar.get("arch", "chunk_v1") != "chunk_v1":
            raise ValueError(f"{path}: sidecar arch {self.sidecar.get('arch')!r} "
                             f"is not chunk_v1")
        self.steps = int(self.sidecar.get("chunk_steps", CHUNK_STEPS))
        self.cdt = float(self.sidecar.get("chunk_dt", CHUNK_DT))
        self.heads = list(self.sidecar.get("heads", HEADS))
        self.lead, self.hysteresis = float(lead), float(hysteresis)
        self.ensemble, self.decay = int(ensemble), float(decay)
        self.same_head = bool(same_head)
        self.dwell, self.dwell_margin = float(dwell), float(dwell_margin)
        # Plan B track 3 (2026-09-30), off by default = unchanged; python sim twins
        # --chunk-side-dwell / --chunk-side-margin / --chunk-flip-margin / --chunk-stuck*
        self.side_dwell, self.side_margin = float(side_dwell), float(side_margin)
        self.flip_margin = float(flip_margin)
        self.stuck = (StuckWatchdog(stuck_window, stuck_progress, stuck_v, stuck_hold, stuck_min_dist)
                      if stuck_window and stuck_window > 0 else None)
        self.v_cap = None if v_cap is None else float(v_cap)   # the watchdog's commanded speed only
        self.goal_speed = float(goal_speed)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(self.path, opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name: list(i.shape) for i in self.sess.get_inputs()}
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.reset()
        self.forward_ms = self._time_forward()

    def reset(self):
        self.ring: list[dict] = []
        self.head = None
        self.prev_chunk = None          # (steps, 4) heading frame of its issue
        self.prev_yaw = None
        self.switches = 0
        self.head_t = None              # time selection last moved to self.head
        self.last_side = None           # the last executed left/right head (flip_margin)
        if getattr(self, "stuck", None) is not None:
            self.stuck.reset()
        self.last = {}

    # --- graph ---------------------------------------------------------------
    def _feed(self, depth_in, state_in, prev):
        feed = {}
        for name, shape in self.inputs.items():
            if "prev" in name or (len(shape) == 2 and shape[-1] == 4 * self.steps):
                feed[name] = np.asarray(prev, np.float32)[None]
            elif "depth" in name or "img" in name or len(shape) == 5:
                feed[name] = depth_in
            else:
                feed[name] = state_in
        return feed

    def _infer(self, feed):
        outs = dict(zip(self.out_names, self.sess.run(None, feed)))
        want = self.sidecar.get("outputs") or {}
        H = len(self.heads)

        def pick(key, last):
            n = want.get(key) if isinstance(want.get(key), str) else None
            if n in outs:
                return np.asarray(outs[n], float)
            for nm, v in outs.items():
                if key in nm:
                    return np.asarray(v, float)
            for v in outs.values():
                if np.asarray(v).shape[-1] == last:
                    return np.asarray(v, float)
            raise KeyError(f"no {key} output among {list(outs)}")
        return (pick("chunk", 4 * self.steps).reshape(H, 4 * self.steps),
                pick("gate", H).reshape(H))

    def _time_forward(self) -> float:
        feed = self._feed(encode_depth(None), np.zeros((1, 1, 22), np.float32),
                          np.zeros(4 * self.steps))
        for _ in range(2):
            self.sess.run(None, feed)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            self.sess.run(None, feed)
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))

    # --- executor --------------------------------------------------------------
    def prev_input(self, yaw: float) -> np.ndarray:
        if self.prev_chunk is None:
            return np.zeros(4 * self.steps)
        c = self.prev_chunk
        v = c[:, :3] @ (rz(yaw).T @ rz(self.prev_yaw)).T
        return np.concatenate([v[:, 0], v[:, 1], v[:, 2], c[:, 3]])

    def select_head(self, probs, t=None):
        best = int(np.argmax(probs))
        if self.head is None:
            return best, "gate"
        if self.stuck is not None and t is not None:
            f = self.stuck.force(float(t), probs, self.head, self.heads)
            if f is not None:
                if f != self.head:
                    self.switches += 1
                return f, "stuck"
        margin, why = self.hysteresis, "hysteresis"
        if (self.dwell > 0 and t is not None and self.head_t is not None
                and t - self.head_t < self.dwell - 1e-9):
            if self.dwell_margin > margin:
                margin, why = self.dwell_margin, "dwell"
        side = [i for i, h in enumerate(self.heads) if h in ("left", "right")]
        if (self.side_dwell > 0 and self.head in side and t is not None and self.head_t is not None
                and t - self.head_t < self.side_dwell - 1e-9 and self.side_margin > margin):
            margin, why = self.side_margin, "side_dwell"
        if (self.flip_margin > 0 and best in side and self.last_side is not None
                and best != self.last_side and self.flip_margin > margin):
            margin, why = self.flip_margin, "flip"
        if best != self.head and probs[best] - probs[self.head] >= margin - 1e-12:
            self.switches += 1
            return best, "switch"
        if best == self.head:
            return self.head, "gate"
        return self.head, why

    def decide(self, t, pos, R, vel, omega_body, goal, depth):
        """One decision at time t (the observation's time). Returns the record."""
        yaw = heading_yaw(R)
        Rh = rz(yaw)
        prev = self.prev_input(yaw)
        chunk, gate = self._infer(self._feed(
            encode_depth(depth), encode_state(pos, R, vel, omega_body, goal,
                                              self.goal_speed), prev))
        probs = softmax(gate)
        if self.stuck is not None:
            self.stuck.observe(float(t), float(np.linalg.norm(np.asarray(goal, float)
                                                              - np.asarray(pos, float))))
        sel, reason = self.select_head(probs, float(t))
        if self.same_head and self.head is not None and sel != self.head:
            self.ring.clear()
        if sel != self.head:
            self.head_t = float(t)
        self.head = sel
        if sel < len(self.heads) and self.heads[sel] in ("left", "right"):
            self.last_side = sel
        c = chunk[sel]
        S = self.steps
        vh = np.stack([c[0:S], c[S:2 * S], c[2 * S:3 * S]], 1)
        self.ring.insert(0, {"t": float(t), "v": vh @ Rh.T,
                             "yr": np.asarray(c[3 * S:4 * S], float)})
        del self.ring[self.ensemble:]
        self.prev_chunk = np.concatenate([vh, c[3 * S:4 * S, None]], 1)
        self.prev_yaw = yaw
        if self.stuck is not None:      # the speed the offboard will command (after --v-cap)
            sp = float(np.linalg.norm(self.command(t)[0]))
            self.stuck.commanded(min(sp, self.v_cap) if self.v_cap is not None else sp)
        self.last = {"head": sel, "reason": reason, "probs": probs,
                     "v1": vh[0] @ Rh.T, "switches": self.switches,
                     # the full network output of this decision (DecisionLog);
                     # references only, nothing is copied or recomputed
                     "chunk": chunk, "gate": gate, "yaw": yaw}
        return self.last

    def command(self, t):
        """Temporal ensemble at time t: (v world ENU (3,), yaw rate, n used)."""
        vs, ys, ws = [], [], []
        T = self.cdt * self.steps
        for age, e in enumerate(self.ring):
            tau = t - e["t"]
            s = tau + self.lead
            if s > T + 1e-9:
                continue
            x = np.clip(s / self.cdt, 1.0, float(self.steps)) - 1.0
            i0 = int(math.floor(x))
            i1 = min(i0 + 1, self.steps - 1)
            f = x - i0
            vs.append((1 - f) * e["v"][i0] + f * e["v"][i1])
            ys.append(e["yr"][min(max(int(math.floor(tau / self.cdt + 1e-9)), 0),
                                  self.steps - 1)])
            ws.append(self.decay ** age)
        if not ws:
            return np.zeros(3), 0.0, 0
        w = np.asarray(ws) / np.sum(ws)
        return (w[:, None] * np.asarray(vs)).sum(0), float(w @ np.asarray(ys)), len(ws)

    def command_acc(self, t):
        """d/dt of command()'s velocity at t (sim OnnxChunkPolicy.command_acc):
        same weights, each chunk's slope on the step it is interpolated on."""
        acc, ws = [], []
        T = self.cdt * self.steps
        for age, e in enumerate(self.ring):
            s = t - e["t"] + self.lead
            if s > T + 1e-9:
                continue
            x = np.clip(s / self.cdt, 1.0, float(self.steps)) - 1.0
            i0 = int(math.floor(x))
            i1 = min(i0 + 1, self.steps - 1)
            acc.append((e["v"][i1] - e["v"][i0]) / self.cdt if i1 > i0 else np.zeros(3))
            ws.append(self.decay ** age)
        if not ws:
            return np.zeros(3)
        w = np.asarray(ws) / np.sum(ws)
        return (w[:, None] * np.asarray(acc)).sum(0)
