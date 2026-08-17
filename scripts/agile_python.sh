#!/bin/bash
# Interpreter shim for the Agile Autonomy (Loquercio) method.
#
# acados' generated OCP solver is a C shared library that dlopens libacados.so;
# both ACADOS_SOURCE_DIR and LD_LIBRARY_PATH must be set in the process
# environment BEFORE Python starts (acados_template reads them at import and
# the dynamic linker resolves the .so at load), so a plain venv python cannot
# work as the method interpreter. This script sets them and execs the agile
# venv python (TF-cpu + casadi + acados_template + pymavlink), making it usable
# anywhere an interpreter path is expected (e.g. the method registry). The
# agile venv lives at the REPO ROOT (superfly/.venv): TF-cpu + casadi +
# acados_template + pymavlink, plus `pip install -e . --no-deps` for the
# superfly package itself.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACADOS_DIR="${ACADOS_SOURCE_DIR:-$HOME/acados}"
export ACADOS_SOURCE_DIR="$ACADOS_DIR"
export LD_LIBRARY_PATH="$ACADOS_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$HERE/../.venv/bin/python" "$@"
