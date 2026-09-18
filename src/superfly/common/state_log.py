"""Per-tick control-state log, shared by the offboard scripts.

numpy-only by design (like the rest of superfly.common, which is imported
inside three mutually-incompatible venvs) and free of pymavlink, so it is
testable on a box that has no MAVLink stack -- gs2, where every post-mortem
of this data actually happens.
"""

from __future__ import annotations

import os

import numpy as np


class StateLog:
    """Per-tick CSV of what the controller SAW and what it COMMANDED.

    Enabled by the SUPERFLY_STATE_LOG environment variable, which
    compare/runner.py points at <trial_dir>/state.csv. Every row carries an
    absolute unix timestamp, and run_px4_sim's traj.npz carries t_unix0, so the
    control state lines up with the ground-truth trajectory exactly -- no
    anchor-guessing from 1 Hz stdout lines, which is what made the 2026-09-17
    speed audit take three rounds. It exists because a control law can only be
    judged against the state it was given: on the diffaero field the offboard's
    own speed tracks ground truth at the median and under-reads the excursions,
    and at 1 Hz that is unprovable.

    Cheap by construction: one preformatted line per control tick, buffered,
    flushed by the OS. ~100 rows/s, ~8 KB/s.
    """

    #: Columns, in order. Extend at the END only (parsers index by name).
    COLUMNS = ("t_unix", "elapsed_s", "phase", "tracker",
               "px", "py", "pz", "vx", "vy", "vz", "speed",
               "tilt_cmd_deg", "tilt_meas_deg", "thrust", "alt_sp",
               "net_hz", "mode_idx")

    def __init__(self, path):
        self._f = open(path, "w", buffering=1 << 16)
        self._f.write(",".join(self.COLUMNS) + "\n")

    @classmethod
    def from_env(cls, var="SUPERFLY_STATE_LOG"):
        """The log named by `var`, or None when it is unset/unwritable."""
        path = os.environ.get(var)
        if not path:
            return None
        try:
            return cls(path)
        except OSError as exc:                     # never fail a flight for a log
            print(f"[state-log] cannot write {path}: {exc}", flush=True)
            return None

    @staticmethod
    def row(t_unix, elapsed, phase, tracker, pos, vel, tilt_cmd, tilt_meas,
            thrust, alt_sp, net_hz, mode_idx):
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        v = np.asarray(vel, dtype=np.float64).reshape(3)

        def _f(x):
            return "" if x is None else f"{float(x):.4f}"

        return ",".join([
            f"{float(t_unix):.4f}", f"{float(elapsed):.4f}", str(phase), str(tracker),
            _f(p[0]), _f(p[1]), _f(p[2]), _f(v[0]), _f(v[1]), _f(v[2]),
            _f(float(np.linalg.norm(v))),
            _f(tilt_cmd), _f(tilt_meas), _f(thrust), _f(alt_sp), _f(net_hz),
            "" if mode_idx is None else str(int(mode_idx)),
        ])

    def write(self, *args, **kwargs):
        try:
            self._f.write(self.row(*args, **kwargs) + "\n")
        except (OSError, ValueError):
            pass

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass
