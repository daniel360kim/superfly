"""Sentinel files coupling the offboard, sim, and comparison-harness processes.

These paths were previously declared 6x (every offboard, the sim launcher,
and the harness); a mismatch silently breaks --auto-stop or the per-phase
flight budgets, so they live here exactly once.
"""

import time
from pathlib import Path

# Touched by every offboard script on exit (any reason: landed, Ctrl-C,
# crash) so run_px4_sim --auto-stop can detect "the offboard process ended"
# and exit its own loop normally instead of relying on a manual Ctrl-C.
OFFBOARD_DONE_FILE = "/tmp/superfly_offboard_done"

# Phase sentinel for the comparison harness: "start <ts>" is appended the
# moment control hands off to the POLICY phase, "end <ts>" when it hands off
# to LANDING, so --timeout can budget the policy flight only (not
# arming/climb/landing). The harness deletes it before each trial.
POLICY_PHASE_FILE = "/tmp/superfly_policy_phase"


def mark_policy_phase(event: str):
    """Append '<event> <ts>' to POLICY_PHASE_FILE; never let it kill the loop."""
    try:
        with open(POLICY_PHASE_FILE, "a") as f:
            f.write(f"{event} {time.time()}\n")
    except Exception:
        pass


def mark_offboard_done():
    """Write the done-file; never let it kill an exiting offboard."""
    try:
        Path(OFFBOARD_DONE_FILE).write_text(str(time.time()))
    except Exception:
        pass


def read_policy_phase():
    """Parse POLICY_PHASE_FILE into (t_start, t_end) wall-clock timestamps
    (None where the event hasn't happened yet)."""
    t_start = t_end = None
    try:
        for line in Path(POLICY_PHASE_FILE).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "start":
                t_start = float(parts[1])
            elif len(parts) == 2 and parts[0] == "end":
                t_end = float(parts[1])
    except (FileNotFoundError, ValueError):
        pass
    return t_start, t_end
