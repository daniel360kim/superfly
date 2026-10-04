#!/usr/bin/env bash
# Train one of the three DiffAero vx/vz/yaw-rate policies (dynamics
# pmv_yawrate: forward + up velocity and yaw rate, no lateral velocity) on
# Starling 2 Max parameters: the PX4 velocity loop refitted on the Starling
# USD (plant=px4_fit), yaw rate <= 1 rad/s, collision radius 0.26 m.
# Recipes and rationale: docs/DIFFAERO_YAWRATE.md.
#
#   band s  cruise 0.8-1.5 m/s  (Starling low-speed; dynamics defaults)
#   band m  cruise 0.5-2.0 m/s
#   band f  cruise 2.0-5.0 m/s
#
# Runs wherever a GPU is -- never on gs2. On airstation03, from gs2:
#   airstation run --no-sync superfly -- bash scripts/train_diffaero_yawrate.sh s
# Extra Hydra overrides go after the band (later ones win), e.g.
#   ... train_diffaero_yawrate.sh f n_updates=500 env.obs_noise.enabled=true
# Output: checkpoints/DiffAero/vel_yawrate_<band>_<YYYYmmdd_HHMM>/ (override
# with OUT=...); interpreter: $DIFFAERO_PYTHON, else the airstation03 overlay
# venv from scripts/airstation_diffaero_venv.sh, else methods/diffaero/.venv.
set -euo pipefail

BAND="${1:-}"; shift || true
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMMON=(env=oa algo=sha2c dynamics=pmv_yawrate sensor=camera network=mlp
        n_updates=2000 save_freq=200 env.oob_terminates=true env.r_drone=0.26)
case "$BAND" in
  s) RECIPE=(env.min_target_vel=0.8 env.max_target_vel=1.5 env.max_time=60) ;;
  m) RECIPE=(env.min_target_vel=0.5 env.max_target_vel=2.0 env.max_time=80
             dynamics.max_vel.x.default=2.5 dynamics.max_vel.x.min=2.0 dynamics.max_vel.x.max=3.0) ;;
  f) RECIPE=(env.min_target_vel=2.0 env.max_target_vel=5.0 env.max_time=40
             dynamics.max_vel.x.default=6.0 dynamics.max_vel.x.min=5.0 dynamics.max_vel.x.max=7.0
             dynamics.max_vel.z.default=2.0 dynamics.max_vel.z.min=1.5 dynamics.max_vel.z.max=2.5) ;;
  *) echo "usage: $0 {s|m|f} [hydra overrides...]"; exit 2 ;;
esac

PY="${DIFFAERO_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$HOME/.cache/superfly/venv_diffaero/bin/python" "$REPO/methods/diffaero/.venv/bin/python"; do
    [ -x "$c" ] && { PY="$c"; break; }
  done
fi
[ -n "$PY" ] || { echo "no DiffAero interpreter: run scripts/airstation_diffaero_venv.sh or set DIFFAERO_PYTHON"; exit 1; }

OUT="${OUT:-checkpoints/DiffAero/vel_yawrate_${BAND}_$(date +%Y%m%d_%H%M)}"
cd "$REPO"
exec "$PY" scripts/train_diffaero.py --python "$PY" --out "$OUT" \
    --config "${COMMON[@]}" "${RECIPE[@]}" "$@"
