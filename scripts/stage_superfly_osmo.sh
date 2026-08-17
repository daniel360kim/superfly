#!/usr/bin/env bash
# Build + upload the staging tarball that lets an OSMO container run the
# superfly comparison harness (Isaac Sim + PX4 SITL + the policy methods).
# Moved here from gs_drone_sim/scripts/ (it always staged THIS project).
#
# Why a tarball: rsync is disabled for this account (403 "Rsync is not enabled
# for this workflow"), so code ships to S3 and is pulled via `inputs: url:` at
# task start.
#
# What is NOT in here, deliberately:
#   * Isaac Sim      -- `pip install isaacsim[all,extscache]` inside the job
#                       takes ~3m20s (measured, gsds-isaacboot3-1). Shipping the
#                       20 GB local install would be strictly slower.
#   * method venvs   -- torch+CUDA stacks are rebuilt in-container
#                       (osmo/superfly-eval.yaml).
#   * PX4 flight logs -- build/px4_sitl_default/rootfs/log is GBs of past SITL
#                       runs; the job needs the binary + etc/ (~43 MB).
#
# Usage:  scripts/stage_superfly_osmo.sh [tag] [--code-only]
#         (default tag: date +%m%d)
# --code-only: stage just the superfly repo (code + checkpoints + submodule
# working trees) -- all a TRAINING job needs. The eval extras (PX4 binary,
# Pegasus, USD stages, acados) are skipped, so it runs on boxes without
# them (gs2).
set -euo pipefail

CODE_ONLY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --code-only) CODE_ONLY=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
TAG="${ARGS[0]:-$(date +%m%d)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # superfly/
PEGASUS="${PEGASUS_DIR:-$HOME/PegasusSimulator}"
PX4="${PX4_DIR:-$HOME/PX4-Autopilot}"
ACADOS="${ACADOS_DIR:-$HOME/acados}"
USD_STAGES="${SUPERFLY_USD_STAGES:-$REPO/usd_stages}"

if [ "$CODE_ONLY" = 0 ]; then
  for d in "$PEGASUS" "$PX4"; do
    [ -d "$d" ] || { echo "MISSING_INPUT_ABORT $d"; exit 1; }
  done

  PX4_BIN="$PX4/build/px4_sitl_default/bin/px4"
  PX4_ETC="$PX4/build/px4_sitl_default/etc"
  [ -x "$PX4_BIN" ] || { echo "PX4_NOT_BUILT_ABORT $PX4_BIN (run: make px4_sitl none_iris)"; exit 1; }
  [ -d "$PX4_ETC" ] || { echo "PX4_ETC_MISSING_ABORT $PX4_ETC"; exit 1; }
fi

STAGE="$(mktemp -d)/superfly_stage"
trap 'rm -rf "$(dirname "$STAGE")"' EXIT
mkdir -p "$STAGE"

echo "== staging the superfly repo (code + checkpoints + configs, no venvs) =="
mkdir -p "$STAGE/superfly"
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'results' --exclude 'methods/*/.venv' --exclude 'outputs' \
  "$REPO"/ "$STAGE/superfly"/
# results/mesh_cache IS needed: clearance scoring for USD scenes reads it, and
# regenerating a cache entry boots a headless Kit (SCENE_MESH_TIMEOUT 1200 s).
if [ -d "$REPO/results/mesh_cache" ]; then
  mkdir -p "$STAGE/superfly/results"
  rsync -a "$REPO/results/mesh_cache" "$STAGE/superfly/results"/
fi

if [ "$CODE_ONLY" = 1 ]; then
  TAR="superfly_stage_${TAG}.tar"
  OUT="${SUPERFLY_TAR_DIR:-${TMPDIR:-/tmp}}/$TAR"
  echo "== building $OUT (code-only) =="
  tar -C "$(dirname "$STAGE")" -cf "$OUT" superfly_stage
  ls -lh "$OUT"
  echo "== uploading to s3://superfly/tmp_data/$TAR =="
  [ -n "${SUPERFLY_S3_KEY_ID:-}${GSDS_S3_KEY_ID:-}" ] || { [ -f ~/.s3env ] && . ~/.s3env; }
  PYTHONPATH="$REPO/src" "${SUPERFLY_PYTHON:-python3}" -m superfly.remote_store upload "$OUT" "tmp_data/$TAR"
  rm -f "$OUT"
  echo "STAGED $TAR (code-only)"
  exit 0
fi

echo "== staging Pegasus extension =="
mkdir -p "$STAGE/PegasusSimulator"
rsync -a --exclude '.git' --exclude '__pycache__' \
  "$PEGASUS/extensions" "$STAGE/PegasusSimulator"/

echo "== staging PX4 SITL (bin + etc only, no logs) =="
mkdir -p "$STAGE/px4/build/px4_sitl_default" \
         "$STAGE/px4/build/px4_sitl_default/rootfs"
# The whole bin/ dir, not just the px4 binary: it also holds ~85 tiny px4-*
# shims, and etc/init.d-posix/rcS line 11 does `. px4-alias.sh` (expected on
# PATH). Shipping only bin/px4 gets you
#   "rcS: 11: .: px4-alias.sh: not found" / "Startup script returned 512".
# It costs nothing -- bin/ is 42 MB and the px4 binary alone is 41 MB of that.
rsync -a "$PX4/build/px4_sitl_default/bin" "$STAGE/px4/build/px4_sitl_default"/
rsync -a "$PX4_ETC" "$STAGE/px4/build/px4_sitl_default"/
# rootfs/etc is a symlink to ../etc locally; recreate it relative so it
# resolves wherever the tar is unpacked.
ln -sfn ../etc "$STAGE/px4/build/px4_sitl_default/rootfs/etc"

echo "== staging USD stages (local copies of the Nucleus stages) =="
# omniverse:// URLs cannot authenticate inside OSMO containers, so ship the
# collected stages and let the workflow export GSDS_USD_STAGE_ROOT so
# run_comparison.py retargets the JSONs' omniverse:// refs at the point of
# use (JSONs and mesh-cache keys untouched). Optional: absent dir => same
# tarball as before, no failure.
# -L (copy-links): stage dirs may be symlinks into a big mount; plain -a
# would ship dangling absolute symlinks into the container.
#   SUPERFLY_USD_APPEND=<dir>: skip the rsync here and instead tar-append
#   <dir>/usd_stages straight into the tarball after the main build (the
#   full stage set doesn't fit twice under /, and sshfs mounts cannot hold
#   the px4 symlinks -- rsync err 23, 2026-08-01).
if [ -n "${SUPERFLY_USD_APPEND:-}" ]; then
  echo "   (usd_stages deferred: will tar-append from $SUPERFLY_USD_APPEND)"
elif [ -d "$USD_STAGES" ]; then
  rsync -aL "$USD_STAGES" "$STAGE"/
else
  echo "WARN: no USD stages at $USD_STAGES -- omniverse:// scenario" \
       "cells will compose empty on OSMO (usd-guard aborts them)"
fi

echo "== staging acados (agile method) =="
if [ -d "$ACADOS" ]; then
  mkdir -p "$STAGE/acados"
  rsync -a --exclude '.git' "$ACADOS/lib" "$ACADOS/include" "$STAGE/acados"/ 2>/dev/null || \
    echo "WARN: acados lib/include not found -- agile method will be unavailable"
else
  echo "WARN: no acados at $ACADOS -- agile method will be unavailable"
fi

TAR="superfly_stage_${TAG}.tar"
# SUPERFLY_TAR_DIR: where the tarball lands (default TMPDIR). For big-stage
# builds: leave TMPDIR local, point SUPERFLY_TAR_DIR at the large mount, and
# set SUPERFLY_USD_APPEND so the stages stream straight from the mount into
# the tar without ever being copied.
OUT="${SUPERFLY_TAR_DIR:-${TMPDIR:-/tmp}}/$TAR"
echo "== building $OUT =="
tar -C "$(dirname "$STAGE")" -cf "$OUT" superfly_stage
if [ -n "${SUPERFLY_USD_APPEND:-}" ]; then
  echo "== appending usd_stages from $SUPERFLY_USD_APPEND =="
  tar -rf "$OUT" -C "$SUPERFLY_USD_APPEND" \
    --transform='s|^usd_stages|superfly_stage/usd_stages|' usd_stages
fi
ls -lh "$OUT"

echo "== uploading to s3://superfly/tmp_data/$TAR =="
[ -n "${SUPERFLY_S3_KEY_ID:-}${GSDS_S3_KEY_ID:-}" ] || { [ -f ~/.s3env ] && . ~/.s3env; }
PYTHONPATH="$REPO/src" "${SUPERFLY_PYTHON:-python3}" -m superfly.remote_store upload "$OUT" "tmp_data/$TAR"
# The S3 copy is the one OSMO reads -- the local one is pure scratch, so drop
# it as soon as the upload succeeds (tarballs have filled / mid-build before).
rm -f "$OUT"
echo "STAGED $TAR"
