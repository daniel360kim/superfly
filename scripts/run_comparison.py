#!/usr/bin/env python
"""Launcher shim for the comparison harness (superfly.compare.runner).

Needs only numpy in the launching interpreter for --dry-run / --report-only;
real runs additionally launch Isaac (via --sim-python) and each method's own
venv per the registry. See superfly/compare/runner.py for the CLI.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from superfly.compare.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
