#!/usr/bin/env bash
# Build (idempotently) the DiffAero training venv on airstation03.
#
# airstation03's disk is shared and ~97 % full, so this does NOT install a
# second torch: it is an overlay on ~/.cache/superfly/venv_torch (torch
# 2.11+cu128, sm_120-ready), linked in through a .pth file, plus DiffAero's
# light deps. pytorch3d is the pure-python copy the OSMO train job uses
# (osmo/superfly-train-diffaero.yaml): DiffAero only imports
# pytorch3d.transforms. Footprint is a few hundred MB.
#
# Usage (from gs2):
#   airstation run --no-sync superfly -- bash scripts/airstation_diffaero_venv.sh
# Prints the interpreter path; scripts/train_diffaero_yawrate.sh defaults to it.
set -euo pipefail

BASE="${VENV_TORCH:-$HOME/.cache/superfly/venv_torch}"
VENV="${VENV_DIFFAERO:-$HOME/.cache/superfly/venv_diffaero}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -x "$BASE/bin/python" ] || { echo "no base torch venv at $BASE"; exit 1; }
if [ ! -x "$VENV/bin/python" ]; then
    "$BASE/bin/python" -m venv "$VENV"   # same interpreter version as the base
fi
SITE="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
BASE_SITE="$("$BASE/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$BASE_SITE" > "$SITE/_venv_torch_overlay.pth"

"$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print("torch", torch.__version__)'

# requirements.txt minus what the overlay already has (torch, numpy, onnxruntime,
# opencv) and what training never imports (open3d: racing env only; wandb,
# gpustat, moviepy, torch-tb-profiler: optional extras).
"$VENV/bin/pip" install --no-cache-dir -q \
    tensordict taichi tqdm hydra-core hydra-joblib-launcher hydra_colorlog \
    welford_torch einops line_profiler tensorboard tensorboardX \
    imageio matplotlib onnx onnxscript

if ! "$VENV/bin/python" -c 'import pytorch3d.transforms' 2>/dev/null; then
    "$VENV/bin/python" - "$SITE" <<'P3D'
import glob, io, shutil, sys, tarfile, tempfile, urllib.request
url = "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/stable.tar.gz"
with tempfile.TemporaryDirectory() as tmp:
    tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())).extractall(tmp)
    shutil.copytree(glob.glob(tmp + "/pytorch3d-*/pytorch3d")[0], sys.argv[1] + "/pytorch3d")
P3D
fi

"$VENV/bin/pip" install --no-cache-dir -q --no-deps -e "$REPO/methods/diffaero"
"$VENV/bin/pip" install --no-cache-dir -q --no-deps -e "$REPO"
"$VENV/bin/python" -c 'import pytorch3d.transforms, tensordict, hydra, taichi, diffaero; print("diffaero venv OK")'
echo "$VENV/bin/python"
