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


class ChunkPolicy:
    def __init__(self, path, lead: float = 0.5, hysteresis: float = 0.15,
                 ensemble: int = 4, decay: float = 0.5, same_head: bool = True,
                 goal_speed: float = 0.0, threads: int = 4, dwell: float = 0.0,
                 dwell_margin: float = 0.3):
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
        margin = self.hysteresis
        if (self.dwell > 0 and t is not None and self.head_t is not None
                and t - self.head_t < self.dwell - 1e-9):
            margin = max(margin, self.dwell_margin)
        if best != self.head and probs[best] - probs[self.head] >= margin - 1e-12:
            self.switches += 1
            return best, "switch"
        if best == self.head:
            return self.head, "gate"
        return self.head, "dwell" if margin > self.hysteresis else "hysteresis"

    def decide(self, t, pos, R, vel, omega_body, goal, depth):
        """One decision at time t (the observation's time). Returns the record."""
        yaw = heading_yaw(R)
        Rh = rz(yaw)
        prev = self.prev_input(yaw)
        chunk, gate = self._infer(self._feed(
            encode_depth(depth), encode_state(pos, R, vel, omega_body, goal,
                                              self.goal_speed), prev))
        probs = softmax(gate)
        sel, reason = self.select_head(probs, float(t))
        if self.same_head and self.head is not None and sel != self.head:
            self.ring.clear()
        if sel != self.head:
            self.head_t = float(t)
        self.head = sel
        c = chunk[sel]
        S = self.steps
        vh = np.stack([c[0:S], c[S:2 * S], c[2 * S:3 * S]], 1)
        self.ring.insert(0, {"t": float(t), "v": vh @ Rh.T,
                             "yr": np.asarray(c[3 * S:4 * S], float)})
        del self.ring[self.ensemble:]
        self.prev_chunk = np.concatenate([vh, c[3 * S:4 * S, None]], 1)
        self.prev_yaw = yaw
        self.last = {"head": sel, "reason": reason, "probs": probs,
                     "v1": vh[0] @ Rh.T, "switches": self.switches}
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
