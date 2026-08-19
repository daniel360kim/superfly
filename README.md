# superfly

Three learned depth-based drone-navigation baselines — **DepthNav**,
**DiffAero**, **Agile Autonomy** — each trained in its own native simulator,
then flown through the **same Isaac Sim + PX4-SITL obstacle course** and
scored on the **same metrics**, so they can be compared head-to-head (and
against other policies) on equal terms.

The repo is two things at once:

- **`src/superfly/`** — an installable package: a dependency-light shared
  layer (MAVLink offboard I/O, frames, depth transport), one policy core per
  method, and the comparison harness (runner, registry, metrics, plots).
- **the harness around it** — scenario configs, committed checkpoints, and
  the three method repos as git submodules.

## Requirements

| | |
|---|---|
| Python | ≥ 3.9 (3.12 on the lab boxes) |
| Scoring / dry-runs | numpy, scipy, pymavlink only — runs on any CPU box |
| Real flights | NVIDIA GPU + Isaac Sim, [Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator), and a built PX4-Autopilot SITL |
| Agile Autonomy only | acados (built from source) + TensorFlow + casadi |

You do **not** need Isaac or a GPU to install the package, inspect scenarios,
dry-run a comparison, or re-score existing results.

## Install

Clone with submodules — the three method forks live under `methods/`:

```bash
git clone --recursive https://github.com/daniel360kim/superfly.git
cd superfly
# already cloned without --recursive?
git submodule update --init
```

The checkpoints are committed in git (~68 MB, no LFS, no lab credentials), so
a fresh clone is already flyable.

### 1. Harness venv (enough to dry-run and re-score)

```bash
python -m venv .venv
.venv/bin/pip install -e .            # numpy, scipy, pymavlink
.venv/bin/pip install -e '.[viz]'     # + matplotlib, for --report plots
```

### 2. Per-method venvs

Each method runs its policy in **its own interpreter** — the comparison
registry resolves `methods/<name>/.venv/bin/python` per method, so their
torch/TF stacks never have to coexist. For each method you intend to fly:

```bash
cd methods/diffaero                   # or methods/depthnav
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../.. --no-deps    # make superfly.* importable
```

`--no-deps` is deliberate: the shared layer must not drag torch or TF into a
venv that already has its own pinned stack. On sm_120 GPUs (RTX 50-series),
install the cu128 torch wheels.

### 3. Agile Autonomy (optional, extra steps)

The agile venv is the **repo-root `.venv`** (TF-cpu + casadi +
`acados_template` + pymavlink). Its MPC solver is a generated C library that
dlopens `libacados.so`, so the environment must be set before the interpreter
starts — use the shim rather than the venv python directly:

```bash
.venv/bin/pip install -e '.[agile]'
export ACADOS_SOURCE_DIR=~/acados          # defaults to ~/acados
scripts/agile_python.sh -c 'import acados_template'
```

Without an acados build the other two methods still work; agile is skipped
with a notice.

### 4. PX4 and Isaac (real flights only)

```bash
git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
cd ~/PX4-Autopilot && make px4_sitl none_iris     # one-time build
```

The harness launches the bare `px4` binary once per trial (clean EKF, arming
and battery state) with a sanitized environment. Point it at your setup with
`--px4-dir` / `$PX4_DIR`, `--sim-python` / `$ISAACSIM_PYTHON`, and put
`PegasusSimulator/extensions/pegasus.simulator` on `$PYTHONPATH` for the sim
process.

## Layout

```
src/superfly/       the installable package
  common/           dependency-light shared layer (numpy/scipy/pymavlink):
                    MAVLink offboard I/O, frames, sentinels, depth transport
  policies/         per-method policy cores (Obs -> Cmd)
  perception/       intrinsics helpers, mesh surface sampling, occupancy
  sim/              Isaac/Pegasus launcher (import under Isaac only) + fields
  compare/          runner, method registry, metrics, plots
  remote_store.py   S3 upload/download/list client
scripts/            verb-first entrypoints: run_comparison.py, run_px4_sim.py,
                    *_offboard.py, train_diffaero.py, scene_audit.py, ...
methods/            git submodules: depthnav / diffaero / agile_autonomy forks
configs/scenarios/  suites/ (multi-scene benchmarks), probes/ (single-scene)
configs/vehicles/   airframe params (iris, starling2max)
checkpoints/        <Method>/<run>/ + run_meta.json — committed weights
results/            harness output (gitignored, regenerated per run)
```

Per-directory detail: [`methods/README.md`](methods/README.md) (the three
forks, their venvs and training entrypoints) and
[`checkpoints/README.md`](checkpoints/README.md) (which artifact each
registry method loads).

## Running a comparison

```bash
# What would this run do? No GPU, no Isaac needed.
.venv/bin/python scripts/run_comparison.py \
    configs/scenarios/suites/poster_suite_v1.json \
    --methods depthnav diffaero agile --dry-run

# Fly it for real (on the box with Isaac + GPU + PX4):
<isaacsim>/python.sh scripts/run_comparison.py \
    configs/scenarios/suites/poster_suite_v1.json \
    --headless --record-video --report \
    --sim-python <isaacsim>/python.sh

# Re-score / re-aggregate existing results (pure NumPy, any box):
.venv/bin/python -m superfly.compare.metrics \
    results/<run>/<scenario>/<method>/traj.npz
.venv/bin/python scripts/run_comparison.py --report-only --results-dir results/<run>
```

Useful flags: `--resume` (skip trials already on disk), `--methods` (subset),
`--vehicle {iris,starling2max}`, `--max-speed`, `--goal-radius`,
`--timeout`, `--<method>-checkpoint` (override a single method's weights).
`--help` lists the rest.

Registry methods: `depthnav`, `depthnav_vel`, `diffaero`, `diffaero_vel`,
`diffaero_vel_planar`, `agile`. Each is gated on its checkpoint actually
existing — a method with a missing artifact is skipped with a notice instead
of flying garbage.

## Training

Training happens on a GPU box, never on a dispatch/CPU machine:

```bash
scripts/train_diffaero.py --out checkpoints/DiffAero/<run>   # wraps the
                    # fork's script/train.py + script/export.py
```

Inside the lab, `scripts/airstation_train.sh <method> [args...]` dispatches
the same thing to the GPU box (it syncs the repo over git first — it does not
copy working-tree state). `train_depthnav.py` and `train_agile.py` are not
written yet; `airstation_train.sh` will tell you so rather than fail
obscurely.

## Results and storage

A run writes `results/<run>/<scenario>/<method>/` — `traj.npz` (trajectory +
per-step telemetry), metrics JSON, optional video, and aggregate plots under
`--report`. `results/` is gitignored; nothing there is recoverable from a
clone.

For sharing runs between machines there is an S3 client:

```bash
python -m superfly.remote_store upload <local> <key>
python -m superfly.remote_store download <key> <local>
python -m superfly.remote_store list [prefix]
```

It reads `SUPERFLY_S3_KEY_ID` / `SUPERFLY_S3_KEY` from the environment,
falling back to `~/.s3env`. Lab-internal; the bucket is not public.

## Working on the method forks

`methods/*` are **forks**, tracked as submodules. Any deploy or training
patch goes on the fork — commit, push, then bump the submodule pointer here
in a normal commit. Editing a loose checkout instead is how earlier work got
lost. See [`methods/README.md`](methods/README.md) for what each fork
contains and which entrypoints are wired up.
