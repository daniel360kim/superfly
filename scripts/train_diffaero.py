#!/usr/bin/env python
"""Train a DiffAero policy and land its deployable artifact where the
comparison-harness registry expects it.

Uniform training entrypoint shape (same for every method):
    scripts/train_<method>.py --out checkpoints/<Method>/<run> [--config ...]

For diffaero that wraps the submodule's own Hydra trainer -- the incantation
that produced the committed sha2c runs was
    python script/train.py env=oa algo=sha2c dynamics=pmc sensor=camera
-- with hydra.run.dir pointed at --out, then script/export.py to emit
checkpoints/exported_actor.pt2 (the self-contained TorchScript actor that
checkpoint_ready() tests for and the offboard loads).

Interface verified against the submodule source (2026-08-17):
  * train.py writes final weights to <hydra.run.dir>/checkpoints/actor.pth
    (utils/runner.py) and hydra drops .hydra/config.yaml in the run dir.
  * export.py wants checkpoint=<run>/checkpoints -- the DIRECTORY holding
    actor.pth: agent.load() joins "actor.pth" onto it, the run config is
    read from <run>/.hydra/config.yaml (ckpt_path.parent/.hydra), and
    PolicyExporter writes exported_actor.pt2/.onnx INTO that directory --
    exactly the committed checkpoints/DiffAero/<run>/ layout.

Runs under the method venv: methods/diffaero/.venv/bin/python. GPU required
(train on OSMO via osmo/superfly-train-diffaero.yaml or on airstation03 via
scripts/airstation_train.sh) -- never on gs2. Not yet exercised end to end
(needs a GPU box); it fails loudly if the expected artifact is missing.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import socket

REPO = Path(__file__).resolve().parents[1]
DIFFAERO = REPO / "methods" / "diffaero"

DEFAULT_OVERRIDES = ["env=oa", "algo=sha2c", "dynamics=pmc", "sensor=camera"]


def sh(cmd, cwd):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="Run dir, e.g. checkpoints/DiffAero/<run>. The "
                         "deployable artifact lands at "
                         "<out>/checkpoints/exported_actor.pt2.")
    ap.add_argument("--config", nargs="*", default=DEFAULT_OVERRIDES,
                    help=f"Hydra overrides (default: {' '.join(DEFAULT_OVERRIDES)})")
    ap.add_argument("--python", default=None,
                    help="Interpreter (default: methods/diffaero/.venv/bin/python, "
                         "falling back to this one)")
    args = ap.parse_args()

    if not (DIFFAERO / "script" / "train.py").exists():
        raise SystemExit(f"methods/diffaero not present at {DIFFAERO} -- "
                         "clone the submodules first (git submodule update "
                         "--init) and build its venv.")
    py = args.python or str(DIFFAERO / ".venv" / "bin" / "python")
    if not Path(py).exists():
        print(f"[warn] {py} missing; using {sys.executable}", file=sys.stderr)
        py = sys.executable

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    sh([py, "script/train.py", *args.config, f"hydra.run.dir={out}"], cwd=DIFFAERO)
    # Export the deployable TorchScript actor into <out>/checkpoints/
    # (export.py reads <out>/.hydra/config.yaml via checkpoint dir's parent).
    sh([py, "script/export.py", f"checkpoint={out / 'checkpoints'}"], cwd=DIFFAERO)

    artifact = out / "checkpoints" / "exported_actor.pt2"
    if not artifact.exists():
        raise SystemExit(f"TRAIN_DONE_BUT_NO_ARTIFACT: {artifact} missing -- "
                         "check export.py's expected run-dir layout before "
                         "trusting this wrapper.")

    try:
        # git is absent in the OSMO training image; the artifact matters
        # more than the provenance sha (superfly-train-diffaero-13 trained
        # to sr 0.93 and then died right here).
        sha = subprocess.run(["git", "-C", str(DIFFAERO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        sha = ""
    meta = dict(schema="superfly-run-meta-v1", method="diffaero",
                artifact="checkpoints/exported_actor.pt2",
                produced_by=f"scripts/train_diffaero.py --config {' '.join(args.config)}",
                submodule_sha=sha or "unknown",
                host=socket.gethostname(),
                date=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK: {artifact} + run_meta.json written")


if __name__ == "__main__":
    main()
