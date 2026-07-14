#!/bin/bash
# GPU interpreter shim for both Agile depth and Agile+CL4Nav RGB policies.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the GPU environment that trained/validated the CL4Nav policy. The
# container runs with host networking and the host home mounted at /home/jason,
# so MAVLink/UDP ports and this checkout are visible unchanged.
CONTAINER="${SUPERFLY_AGILE_CONTAINER:-agile-autonomy-run}"
PYTHON="${SUPERFLY_AGILE_PYTHON:-/opt/conda/envs/tf_gpu/bin/python}"
ACADOS_DIR="${ACADOS_SOURCE_DIR:-$HOME/SousVide/FiGS/acados}"
ACADOS_PY="$ACADOS_DIR/interfaces/acados_template"
ACADOS_LD="$ACADOS_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "${SUPERFLY_AGILE_RUNTIME:-tf_gpu}" == "local" ]]; then
    export ACADOS_SOURCE_DIR="$ACADOS_DIR"
    export PYTHONPATH="$ACADOS_PY${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$ACADOS_LD"
    exec "$HERE/.venv/bin/python" "$@"
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Agile GPU container '$CONTAINER' does not exist." >&2
    exit 1
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
    echo "Agile GPU container '$CONTAINER' is stopped; run: docker start $CONTAINER" >&2
    exit 1
fi

TOKEN="superfly-agile-$$"
docker_pid=""
cleanup() {
    if [[ -n "$docker_pid" ]]; then
        kill "$docker_pid" >/dev/null 2>&1 || true
    fi
    docker exec "$CONTAINER" pkill -TERM -f "$TOKEN" >/dev/null 2>&1 || true
}
trap 'cleanup; exit 143' INT TERM
trap cleanup EXIT

docker exec -i \
    -w "$HERE" \
    -e ACADOS_SOURCE_DIR="$ACADOS_DIR" \
    -e PYTHONPATH="$ACADOS_PY${PYTHONPATH:+:$PYTHONPATH}" \
    -e LD_LIBRARY_PATH="$ACADOS_LD" \
    -e SUPERFLY_EXEC_TOKEN="$TOKEN" \
    "$CONTAINER" bash -c 'exec -a "$SUPERFLY_EXEC_TOKEN" "$@"' \
    superfly-exec "$PYTHON" "$@" &
docker_pid=$!

set +e
wait "$docker_pid"
status=$?
set -e
docker_pid=""
trap - INT TERM EXIT
cleanup
exit "$status"
