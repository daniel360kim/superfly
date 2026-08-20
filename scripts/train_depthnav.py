#!/usr/bin/env python
"""Train a DepthNav policy and land its deployable artifact where the
comparison-harness registry expects it.

Uniform training entrypoint shape (same for every method):
    scripts/train_depthnav.py --out checkpoints/DepthNav/<run> [--vel] [...]

Wraps the submodule's two-level curriculum runner:
    examples/navigation/run_nav_level1.py        (thrust,  small_yaw)
    examples/navigation/run_nav_level1_vel.py    (--vel,   small_yaw_vel)
level0 learns bare target-reaching in an empty box, level1 adds the obstacle
set; the runner chains them via --weight (depthnav/scripts/runner.py) and
train_bptt saves <run_name>_<iter_total>.pth at the end of each level plus
_iteration_<i>.pth every checkpoint_interval.

The registry artifact is a single .pth file (ckpt_kind="file"):
    --vel  -> <out>/level1_vel.pth   (registry "depthnav_vel")
    thrust -> <out>/level1.pth       (a new thrust run; the committed
              legacy run keeps its own level1_4/... name)
We copy the newest final-level checkpoint there, and keep the whole
logs/ tree (tfevents, iteration ckpts, merged run configs) beside it for
provenance -- the depthnav analog of diffaero's hydra run dir.

Requires the dataset (methods/depthnav/datasets/depthnav_dataset, via
datasets/get_dataset.sh) and a habitat-sim build in the interpreter's env.
GPU required (train on OSMO via osmo/superfly-train-depthnav.yaml or on
airstation03) -- never on gs2.
"""

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPTHNAV = REPO / "methods" / "depthnav"


def sh(cmd, cwd):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def newest_final_ckpt(log_dir: Path, run_name: str) -> Path:
    """The runner's convention: final saves are <run_name>_<total_iter>.pth
    (no 'iteration'); pick the highest-numbered one."""
    def last_digits(p: Path) -> int:
        m = re.search(r"_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    finals = [p for p in log_dir.glob(f"{run_name}_*.pth")
              if "iteration" not in p.name]
    if not finals:
        raise SystemExit(
            f"TRAIN_DONE_BUT_NO_ARTIFACT: no {run_name}_*.pth in {log_dir} -- "
            "check the runner's save naming before trusting this wrapper.")
    return max(finals, key=last_digits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True,
                    help="Run dir, e.g. checkpoints/DepthNav/<run>. The "
                         "deployable artifact lands at <out>/level1_vel.pth "
                         "(--vel) or <out>/level1.pth.")
    ap.add_argument("--vel", action="store_true",
                    help="Train the velocity-command variant (small_yaw_vel, "
                         "VELOCITY_YAW, 0.8-1.5 m/s starling velocity loop)")
    ap.add_argument("--planar", action="store_true",
                    help="Train the PLANAR velocity-command variant "
                         "(small_yaw_vel_planar, vz forced to zero -- the "
                         "depthnav analog of diffaero pmv_planar; implies --vel)")
    ap.add_argument("--level0-iters", type=int, default=500)
    ap.add_argument("--level1-iters", type=int, default=20000)
    ap.add_argument("--python", default=None,
                    help="Interpreter (default: methods/depthnav/.venv/bin/python, "
                         "falling back to this one)")
    args = ap.parse_args()

    if args.planar:
        args.vel = True
    entry = ("examples/navigation/run_nav_level1_vel_planar.py" if args.planar
             else "examples/navigation/run_nav_level1_vel.py" if args.vel
             else "examples/navigation/run_nav_level1.py")
    if not (DEPTHNAV / entry).exists():
        raise SystemExit(f"methods/depthnav not present/complete at {DEPTHNAV} "
                         "-- clone the submodules first (git submodule update "
                         "--init).")
    dataset = DEPTHNAV / "datasets" / "depthnav_dataset"
    if not dataset.is_dir():
        raise SystemExit(f"DATASET_MISSING_ABORT: {dataset} -- run "
                         "datasets/get_dataset.sh (or restore the S3 mirror) "
                         "first; habitat has no scenes without it.")
    py = args.python or str(DEPTHNAV / ".venv" / "bin" / "python")
    if not Path(py).exists():
        print(f"[warn] {py} missing; using {sys.executable}", file=sys.stderr)
        py = sys.executable

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [py, entry]
    if args.vel:
        cmd += ["--level0-iters", str(args.level0_iters),
                "--level1-iters", str(args.level1_iters)]
    elif (args.level0_iters, args.level1_iters) != (500, 20000):
        print("[warn] --level*-iters are only wired into the _vel entry; "
              "the thrust runner uses its built-in 500/20000", file=sys.stderr)
    sh(cmd, cwd=DEPTHNAV)

    variant = ("level1_vel_planar" if args.planar
               else "level1_vel" if args.vel else "level1")
    log_dir = DEPTHNAV / "examples" / "navigation" / "logs" / variant
    final = newest_final_ckpt(log_dir, variant)

    artifact = out / f"{variant}.pth"
    shutil.copy2(final, artifact)
    # keep the full training record beside the artifact (tfevents, iteration
    # ckpts, the merged per-level run configs the runner wrote)
    dst_logs = out / "logs"
    if dst_logs.exists():
        shutil.rmtree(dst_logs)
    shutil.copytree(log_dir, dst_logs)

    try:
        # git is absent in the OSMO training image; the artifact matters
        # more than the provenance sha.
        sha = subprocess.run(["git", "-C", str(DEPTHNAV), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        sha = ""
    meta = dict(schema="superfly-run-meta-v1", method="depthnav",
                artifact=artifact.name,
                produced_by=("scripts/train_depthnav.py "
                             + ("--planar " if args.planar
                                else "--vel " if args.vel else "")
                             + f"--level0-iters {args.level0_iters} "
                             + f"--level1-iters {args.level1_iters}"),
                submodule_sha=sha or "unknown",
                host=socket.gethostname(),
                date=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"OK: {artifact} (from {final.name}) + logs/ + run_meta.json written")


if __name__ == "__main__":
    main()
