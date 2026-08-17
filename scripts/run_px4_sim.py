#!/usr/bin/env python
"""Launcher shim for the Isaac Sim / Pegasus + PX4 SITL harness.

Run under Isaac's Python. superfly.sim.px4_sim boots SimulationApp at import
time (Isaac requires it before any other omni import), so all the work
happens in the import below; see that module for the CLI and examples.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from superfly.sim.px4_sim import main  # noqa: E402  (boots SimulationApp)

if __name__ == "__main__":
    main()
