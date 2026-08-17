> **ARCHIVED 2026-08-17.** Pre-reorganization README; describes the old
> `starling-deployment/` layout, the removed DiffPhysDrone method, and a
> `--methods/--seeds/--obstacles` CLI that no longer exists. Current
> entry point: the repo-root `ORCHESTRATOR.md` and `README.md`.

# superfly — comparing DiffPhysDrone, DiffAero & DepthNav in one environment

Three learned depth-based drone-navigation methods, trained in their own
simulators, flown through the **same Isaac-Sim + PX4-SITL obstacle course** and
scored on the **same metrics** so they can be compared head-to-head.

| Method | Trainer (native sim) | Deploy policy | Control rate |
|---|---|---|---|
| **DiffPhysDrone** | `DiffPhysDrone/` — custom CUDA sim | `starling-deployment/diffdrone_offboard.py` | 15 Hz |
| **DiffAero** | `diffaero/` — Taichi GPU sim (Hydra) | `starling-deployment/diffaero_offboard.py` | 30 Hz |
| **DepthNav** | `depthnav/` — Habitat-sim | `starling-deployment/depthnav_offboard.py` | 50 Hz |

The three cannot share a *training* simulator (different physics, obs/action
spaces, timesteps, and conflicting pinned deps), so each keeps its native trainer
and **its own venv**. They *do* share an *evaluation* substrate: the
`starling-deployment/` Isaac-Sim + PX4-SITL harness. That is where the comparison
happens.

## Environments

There are two different senses of "environment" in this repo — training envs
(where each policy learns) and evaluation envs (the shared Isaac-Sim scenes
used to compare them). Don't confuse the two: a policy never sees the
evaluation env during training.

### 1. Training environments (per method, native sim)

| Env | Sim backend | Obs / action | Where |
|---|---|---|---|
| **DiffPhysDrone** | custom CUDA kernel sim (`env_cuda.py`) | depth image + kinematic state → body-rate/thrust | `DiffPhysDrone/` |
| **DiffAero** | Taichi GPU sim, Hydra-configured (`env=oa` obstacle-avoidance, `env=pc` position-control, `env=racing`) | depth/lidar/relpos sensor (configurable) → velocity or accel cmd | `diffaero/diffaero/env/` |
| **DepthNav** | Habitat-sim (photorealistic scenes + physics) | depth image → velocity cmd | `depthnav/depthnav/envs/navigation_env.py` |

Each has its own obstacle distribution, timestep, and reward — that's *why* they
can't share a trainer. `DiffPhysDrone/env_cuda.py`'s procedural field (balls +
voxels + cylinders, see below) is what `obstacle_field.generate()` replicates
for evaluation; DiffAero's is replicated by `obstacle_field.generate_diffaero()`.

### 2. Evaluation environments (shared, Isaac Sim + PX4 SITL)

This is what `run_px4_sim.py` builds and what `--environment` / `--obstacles`
in `run_comparison.py` actually select. Two independent axes:

- **Background scene** (`--environment`, a Pegasus `SIMULATION_ENVIRONMENTS` key
  — e.g. `"Box Room"` (default), `"Warehouse"`, `"Warehouse with Shelves"`) or
  an arbitrary USD stage (`--usd-environment <omniverse:// path>`, with
  `--env-scale` for unit conversion, e.g. `0.01` for a cm-authored stage like
  ConiferForest). This is just the visual/collision backdrop.
- **Obstacles** (`--obstacles {diffphys, diffaero, none}`) — procedural
  primitives spawned *on top of* the background scene, generated deterministically
  from `--seed` + `--scale` by `obstacle_field.py`:
  - `diffphys`: replicates `DiffPhysDrone/env_cuda.py`'s field — 30 spheres, 30
    axis-aligned boxes + 10 ground boxes, 30 vertical cylinders, 2 horizontal
    cylinders, y-stretched and x-scaled by `--scale`, rotated so the corridor
    (start → goal, ~9.5×`scale` m apart) aligns with the drone's actual
    magnetometer-locked heading (ENU 90°) in the Pegasus sim.
  - `diffaero`: replicates DiffAero's training distribution instead.
  - `none`: scene geometry only (no procedural primitives) — requires an
    explicit `--spawn X Y Z` (sim side) / `--start`+`--goal` (comparison side).
  - `--obstacle-assets` additionally swaps each procedural primitive for a
    scaled-to-fit USD mesh from `OBSTACLE_ASSETS`, for a training-distribution
    layout with realistic-looking geometry instead of raw primitives.

Same `seed` + `scale` + `obstacles` choice ⇒ byte-identical field every time
(`np.random.default_rng(seed)`), which is what lets every method fly the exact
same course.

## Layout

```
DiffPhysDrone/          # trainer (CUDA); deploy ckpt: checkpoints/DiffPhysDrone/*.pth
diffaero/               # trainer (Taichi, Hydra); deploy ckpt dir: checkpoints/DiffAero/<run>/
depthnav/               # trainer (Habitat); deploy ckpt: depthnav/.../logs/level1/*.pth
checkpoints/            # deployable policy artifacts per method
starling-deployment/    # shared Isaac-Sim + PX4-SITL flight harness
  run_px4_sim.py        #   one launcher, --policy {diffphys,diffaero,depthnav}
  diffdrone_offboard.py #   per-method PX4 offboard controllers (CLIMB->[YAW]->POLICY)
  diffaero_offboard.py
  depthnav_offboard.py
  obstacle_field.py     #   analytic obstacle distributions (diffphys / diffaero)
  compare/              #   the comparison harness (this repo's deliverable)
    run_comparison.py   #     orchestrator + per-method registry + aggregation
    metrics.py          #     offline scoring (success / collision-clearance / time-speed)
SAFE_Benchmark/         # (unused placeholder — left untouched)
```

## Per-method training (native, unchanged)

```bash
# DiffPhysDrone (CUDA kernels must be built: pip install -e DiffPhysDrone/src)
cd DiffPhysDrone && python main_cuda.py $(cat configs/single_agent.args)

# DiffAero (Hydra)
cd diffaero && python script/train.py env=oa algo=sha2c dynamics=pmc sensor=camera

# DepthNav (Habitat, curriculum)
cd depthnav && python examples/navigation/run_nav_level1.py
```

Each runs in its own environment: `diffaero/.venv` (Taichi + torch), a DepthNav
venv (`depthnav/requirements.txt`: torch 2.2.1 / numpy 1.23.5 + Habitat-sim), and
a torch env for DiffPhysDrone.

## Comparing the three (single environment, single command)

The comparison flies each trained policy through the identical obstacle field
(same seed → same layout, same start/goal) and scores the logged ground-truth
trajectory. Everything lives in `starling-deployment/compare/`.

**Prerequisites at run time**

- **PX4 SITL running** (both the sim and offboard talk to it over MAVLink
  `udp:localhost:14550`).
- Interpreters: `run_px4_sim.py` runs under **Isaac Sim's Python** (set
  `--sim-python` or `$ISAAC_PYTHON`); each offboard runs under its method's venv
  (`--diffphys-python` / `--diffaero-python` / `--depthnav-python`, defaulting to
  `<method>/.venv/bin/python`).
- Deployable **checkpoints** in place (see below).

**Preview the exact commands (no Isaac needed):**

```bash
cd starling-deployment
python compare/run_comparison.py --methods diffphys diffaero \
    --obstacles diffphys --seeds 0 1 2 --dry-run
```

**Run the comparison and print the report** (PX4 SITL must be up):

```bash
python compare/run_comparison.py --methods diffphys diffaero \
    --obstacles diffphys --seeds 0 1 2 --headless --report
```

**Re-aggregate existing results only:**

```bash
python compare/run_comparison.py --report-only --results-dir compare/results
```

Per-trial trajectories land in `compare/results/<method>_seed<n>.npz` (+ `.json`
scores); `--report` writes `compare/results/summary.csv` and prints:

```
    method   n  success  collide  clear[m]  t_goal[s]   v_avg    v_pk
  diffaero   3     0.67     0.00      0.93       9.25    2.97    4.03
  diffphys   3     0.67     0.33      0.40      12.17    2.50    3.17
```

### What `run_comparison.py` actually does

For each `(method, seed)` pair in `--methods` × `--seeds` it runs one **trial**:

1. **Resolve the scenario.** `field_for(obstacles, seed, scale)` calls
   `obstacle_field.generate[_diffaero](seed, scale)` to get the deterministic
   `(start, goal)` for that seed (skipped if `--obstacles none`, which requires
   explicit `--start`/`--goal`). `--goal` can still override the field's goal.
2. **Build two argv lists** (`build_commands`): a `sim_cmd` for the shared
   `run_px4_sim.py --policy <method> --obstacles ... --seed ... --auto-stop
   --log-traj <npz>` and an `off_cmd` for that method's own
   `*_offboard.py --checkpoint ... --goal ...`. Interpreters and checkpoints
   come from the per-method **registry** (`method_registry()`) — native venv +
   default checkpoint path per method — overridable per-trial via
   `--<method>-python` / `--<method>-checkpoint`. Speed flags are mapped to
   each method's own CLI spelling (`speed_args`: `--max-speed` for diffphys,
   `--max-vel` for diffaero, `--target-speed` for depthnav), all driven by one
   shared `--max-speed`.
3. **Launch and synchronize** (`run_trial`): clears the shared sentinel file
   `/tmp/superfly_offboard_done`, starts the sim subprocess, sleeps `--warmup`
   seconds for Isaac to boot, then starts the offboard subprocess and waits up
   to `--timeout` for it to finish (killing it on timeout). It then force-writes
   the sentinel so the sim's `--auto-stop` watcher exits even if the offboard
   was killed, and gives the sim `--sim-grace` seconds to shut down cleanly
   before terminating it.
4. **Score** the resulting `compare/results/<method>_seed<n>.npz` trajectory
   via `metrics.score_trajectory()` (or records a failed/empty result if no
   `.npz` was written) and writes the per-trial `<method>_seed<n>.json`.
5. **Checkpoint gating**: before any trial for a method, `checkpoint_ready()`
   checks its artifact exists (a directory containing
   `checkpoints/exported_actor.pt2` for diffaero, a file for the others) — if
   missing, that method is skipped entirely with a message, so
   `--methods diffphys diffaero depthnav` silently runs only what's ready.
6. **`--dry-run`** prints the two commands per trial and launches nothing
   (useful for sanity-checking flags/paths without Isaac running).
7. **Aggregation** (`--report` after running, or `--report-only` to just
   re-aggregate): `aggregate()` globs every `<method>_seed*.json` in
   `--results-dir`, groups by method, and computes success rate, collision
   rate, mean min-clearance, mean time-to-goal (over trials that reached the
   goal), mean and peak speed. `print_report()` prints the table shown above
   and writes `summary.csv`.

**Metrics** (chosen for this comparison):
- **success rate** — reached the goal (within `--goal-radius`) and never collided.
- **collision / clearance** — collision rate + minimum clearance, where clearance
  = distance from the drone centre to the nearest obstacle surface minus
  `--drone-radius` (analytic SDFs vs. the known obstacle field).
- **time & speed** — time-to-goal, mean and peak speed.

### Checkpoints

| Method | Deployable artifact | Status |
|---|---|---|
| DiffPhysDrone | `checkpoints/DiffPhysDrone/checkpoint0004.pth` | present |
| DiffAero | `checkpoints/DiffAero/sha2c_pmc/checkpoints/exported_actor.pt2` (export via `diffaero/script/export.py`) | present |
| DepthNav | `depthnav/examples/navigation/logs/level1/level1_4_iteration_13500.pth` | **pending** |

`run_comparison.py` **auto-skips** any method whose checkpoint is missing (with a
message), so `--methods diffphys diffaero depthnav` runs the available two today
and picks up DepthNav automatically once its checkpoint lands — no code change.

### Fairness note

Each policy was trained on a different obstacle distribution, so flying them
through one shared field is an out-of-distribution **generalization** comparison —
the honest way to benchmark three methods in a single world. The shared field is
explicit (`--obstacles`); use `--obstacles diffaero` (or a method's own
distribution) to compare under a different shared world.
