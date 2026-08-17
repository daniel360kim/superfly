# superfly

Three learned depth-based drone-navigation baselines — **DepthNav**,
**DiffAero**, **Agile Autonomy** — trained in their own native simulators,
flown through the **same Isaac-Sim + PX4-SITL obstacle course**, and scored
on the **same metrics** so they (and `gs_drone_sim` policies) can be compared
head-to-head.

**Agents / new sessions: read [`ORCHESTRATOR.md`](ORCHESTRATOR.md) first.**

## One-clone reproducibility

```bash
git clone --recursive https://github.com/daniel360kim/superfly.git
```

gets you the harness, the three method repos (forks, as submodules under
`methods/`), and flyable checkpoints (committed in git — no lab credentials,
no LFS). Then, per venv you need:

```bash
python -m venv .venv && .venv/bin/pip install -e .      # harness / agile base
# each method venv additionally: pip install -e . --no-deps  (superfly package)
```

## Layout

```
src/superfly/       the installable package
  common/           dependency-light shared layer (numpy/scipy/pymavlink):
                    MAVLink offboard I/O, frames, sentinels, depth transport
  policies/         per-method policy cores (Obs -> Cmd)
  perception/       intrinsics helpers + mesh surface sampling
  sim/              Isaac/Pegasus launcher (import under Isaac only) + fields
  compare/          runner, method registry, metrics, plots
scripts/            verb-first entrypoints: run_px4_sim.py, run_comparison.py,
                    *_offboard.py, train_*.py, stage_superfly_osmo.sh, ...
methods/            git submodules: depthnav / diffaero / agile_autonomy forks
configs/scenarios/  suites/ (benchmarks) and probes/ (single-scene checks)
checkpoints/        <Method>/<run>/ + run_meta.json — committed weights
osmo/               superfly-* OSMO workflow YAMLs
docs/               harness manual + archive of superseded docs
```

## The 30-second tour

```bash
# What would a comparison run do? (no GPU, no Isaac needed)
python scripts/run_comparison.py configs/scenarios/suites/poster_suite_v1.json \
    --methods depthnav diffaero agile --dry-run

# Fly it for real (airstation03 — the box with Isaac + GPU + PX4):
<isaacsim-python> scripts/run_comparison.py \
    configs/scenarios/suites/poster_suite_v1.json \
    --headless --record-video --report --sim-python <isaacsim>/python.sh

# Re-score / re-aggregate existing results (pure NumPy, any box):
python -m superfly.compare.metrics results/<run>/<scenario>/<method>/traj.npz
python scripts/run_comparison.py --report-only --results-dir results/<run>

# Train (OSMO or airstation03, never gs2):
scripts/airstation_train.sh diffaero
osmo workflow submit osmo/superfly-train-diffaero.yaml --pool default ...
```

Full harness manual: [`docs/comparison_harness.md`](docs/comparison_harness.md).
Compute/storage specifics: [`INFRASTRUCTURE.md`](INFRASTRUCTURE.md).
What's been tried: [`ATTEMPTS.md`](ATTEMPTS.md).
