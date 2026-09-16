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

# The agile venv is the default target. On a box where it was never built
# (airstation03, 2026-09-16: no ~/superfly/.venv -- acados_template and TF live
# in Isaac Sim's own interpreter instead), export AGILE_PYTHON to point this
# shim at an interpreter that has acados_template + TF/onnxruntime, e.g.
#   AGILE_PYTHON=$HOME/isaacsim/kit/python/bin/python3
# Doing it here rather than via --agile-python keeps the ACADOS env exports,
# which acados needs in the process environment BEFORE Python starts.
PY="${AGILE_PYTHON:-$HERE/../.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "agile_python.sh: no interpreter at $PY." >&2
    echo "  Build the agile venv (see INFRASTRUCTURE.md) or set AGILE_PYTHON to" >&2
    echo "  an interpreter with acados_template + tensorflow-cpu/onnxruntime." >&2
    exit 127
fi
exec "$PY" "$@"
