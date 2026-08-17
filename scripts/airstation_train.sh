#!/usr/bin/env bash
# Train a method on airstation03's 5090 from gs2, through the umbrella's
# dispatch loop (anyanything/bin/airstation) -- never train on gs2 itself.
#
# This wraps `airstation run superfly -- ...` and encodes MACHINES.md's rules:
#   * git is the only sync path (airstation sync runs first unless --no-sync);
#   * never edit on airstation03;
#   * airstation03's root filesystem sits ~97% full and is shared by ~27
#     accounts -- `airstation run` already refuses above 95% disk, and
#     training output goes to the repo's checkpoints/ (committed, small),
#     never to bulk scratch under ~.
#
# Usage:
#   scripts/airstation_train.sh diffaero [--out checkpoints/DiffAero/<run>] [...]
#   scripts/airstation_train.sh depthnav ...      (train_depthnav.py, Phase 3)
#   scripts/airstation_train.sh agile ...         (train_agile.py, Phase 3)
#
# Everything after the method name is passed through to
# scripts/train_<method>.py; a default --out of checkpoints/<Method>/<date>
# is added if none is given.
set -euo pipefail

METHOD="${1:-}"; shift || true
case "$METHOD" in
  diffaero) DIR=DiffAero ;;
  depthnav) DIR=DepthNav ;;
  agile)    DIR=AgileAutonomy ;;
  *) echo "usage: $0 {diffaero|depthnav|agile} [train args...]"; exit 2 ;;
esac

TRAIN="scripts/train_${METHOD}.py"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/$TRAIN" ] || { echo "$TRAIN does not exist yet (Phase 3)"; exit 1; }

ARGS=("$@")
case " ${ARGS[*]-} " in
  *" --out "*) ;;
  *) ARGS+=(--out "checkpoints/$DIR/$(date +%Y%m%d_%H%M)") ;;
esac

exec airstation run superfly -- python "$TRAIN" "${ARGS[@]}"
