# Policy Comparison Harness

Flies **DepthNav**, **DiffAero**, and **Agile Autonomy** (plus their
velocity-command variants) through the *same* Isaac-Sim + PX4-SITL scenarios and scores them on the same metrics, without
retraining or modifying any method. Each trial launches the shared simulator
(`scripts/run_px4_sim.py --log-traj`) plus the method's own
`scripts/*_offboard.py`, waits for
the flight to finish, and scores the logged ground-truth trajectory offline.

```
src/superfly/compare/
├── runner.py              # the harness: trials, PX4 lifecycle, aggregation
├── registry.py            # per-method launch config (interpreter, ckpt_kind)
├── metrics.py             # offline scoring of one logged flight (pure NumPy)
└── plot.py                # trajectory figures
scripts/run_comparison.py  # launcher shim
scripts/extract_scene_mesh.py  # USD scene -> surface samples for clearance
configs/scenarios/{suites,probes}/   # scenario JSONs
results/<timestamp>/       # one folder per run (never overwritten)
results/mesh_cache/        # cached scene-mesh .npz files (per usd+scale)
```

## Quick start

```bash
# Preview exactly what would run (no Isaac, no PX4 needed):
python scripts/run_comparison.py configs/scenarios/suites/example_scenarios.json --dry-run

# Real headless run with per-trial videos + summary table (airstation03):
<isaacsim>/kit/python/bin/python3 scripts/run_comparison.py \
    configs/scenarios/suites/my_scenarios.json --headless --record-video --report \
    --sim-python <isaacsim>/python.sh

# Re-aggregate an existing run without flying anything:
python scripts/run_comparison.py --report-only \
    --results-dir results/20260703_134005
```

The harness itself needs NumPy (+ SciPy for scene-mesh clearance); running it
under Isaac's kit python with `setup_python_env.sh` sourced provides both.

## How a trial works

For every `(method, scenario)` pair:

1. **PX4 SITL is killed and relaunched** (unless `--no-px4-manage`) so each
   trial starts from a clean flight-controller state (EKF home, arming
   latches, simulated battery). Battery failsafes are disabled via boot params.
2. **Isaac Sim** is launched via `run_px4_sim.py` with the scenario's
   environment, spawn, goal, and (optional) procedural obstacle field, plus
   `--log-traj` to record the ground-truth ENU pose/velocity every physics
   step and `--auto-stop` to exit when the offboard script finishes.
3. **The method's offboard script** is launched with its own interpreter
   (each method has its own venv; see *Interpreters* below), the checkpoint,
   and the goal converted to PX4's spawn-relative local frame.
4. **Phase-aware timeouts**: the offboard scripts write handoff events to
   `/tmp/superfly_policy_phase`, so the pre-policy phase (heartbeat/arm/climb/
   yaw), the policy flight, and the landing each get their own budget
   (`--pre-policy-timeout`, `--timeout`, `--landing-timeout`, all overridable
   per scenario). A slow PX4 boot can never eat the policy's flight budget.
5. **Scoring**: `metrics.py` scores the logged trajectory (see *Metrics*).
   Results land in `results/<run>/<scenario>/<method>/`:
   `traj.npz`, `metrics.json` (self-describing: scores + hyperparams +
   scenario + exact commands), and with `--record-video` also `depth.mp4`
   (the policy's depth input) and `rgb.mp4` (onboard RGB, same viewpoint).

## Scenario files

A run is driven by a JSON list of scenarios; every method flies every entry.

```jsonc
[
  {
    "name": "english_college",       // unique label (default scenario<i>)
    // environment: EITHER a named Pegasus scene ...
    "environment": "Box Room",
    // ... OR an arbitrary USD stage (takes precedence when set):
    "usd_environment": "omniverse://.../EnglishCollege.stage.usd",
    "env_scale": 0.01,               // uniform scale for the USD stage
    "obstacles": "none",             // none | diffphys | diffaero (procedural field)
    "seed": 0,                       // procedural-field RNG seed
    "scale": 5.0,                    // procedural-field size
    "start": [46.9, 127.7, 68.6],    // world-frame spawn (required unless a
                                     // procedural field supplies it)
    "goal": [-43.3, 145.9, 71.0],    // world-frame goal (always required for
                                     // scene-only; overrides the field's target)
    "climb_alt": 5,                  // climb height [m] ABOVE the spawn altitude
    "timeout": 300,                  // policy-phase budget [s]
    "pre_policy_timeout": 240,       // arm/climb/yaw budget [s]
    "landing_timeout": 90            // post-policy landing budget [s]
  }
]
```

Notes:

- **Procedural fields** (`"obstacles": "diffphys" | "diffaero"`) are
  deterministic in `(seed, scale)` and define the spawn themselves; `start`
  is ignored for them.
- **USD stages** are loaded under `/World/layout` with a single uniform
  `env_scale` (no offset/rotation, no `metersPerUnit` conversion — a
  centimetre-authored stage needs `env_scale: 0.01`). Lighting and static
  triangle-mesh colliders are added automatically by `run_px4_sim.py`.
- Scenario names must be unique — results are keyed by them.

## Metrics

All metrics are computed offline by `metrics.py` from `traj.npz` (pure NumPy,
no Isaac needed). Takeoff is auto-detected (first sustained motion) so the
long parked prefix (Isaac warmup, arming) never pollutes time/speed numbers.

| metric | meaning |
|---|---|
| `success` | reached the goal AND never collided. "Reached" = ground-truth position within `--goal-radius` (default 1 m) of the world goal, **or** the policy's own goal-reached handoff (`policy_reported_reached`; the thrust-variant depthnav never reports one — it has no landing phase). |
| `collided` | clearance dropped below 0 at any scored tick |
| `min_clearance_m` | min over the flight of (distance from drone centre to nearest obstacle surface) − `--drone-radius` (default 0.2 m) |
| `time_to_goal_s` | takeoff → first sample inside the goal radius |
| `mean_speed_mps` | mean speed over takeoff → first goal hit (or log end) |
| `peak_speed_mps` | max speed over the whole log |

`--report` / `--report-only` pool every trial's `metrics.json` into a
per-method table and `summary.csv` (`mean_min_clearance_m` averages
`min_clearance_m` over trials where it exists; `mean_time_to_goal_s` averages
only over trials that reached the goal).

### Clearance: procedural fields vs USD scenes

Clearance needs obstacle geometry to measure distance against. There are two
sources, recorded per trial as `clearance_source` in `metrics.json`:

- **`analytic_field`** — procedural-field scenarios. `run_px4_sim.py` dumps
  the field's exact primitives (spheres/boxes/cylinders) into `traj.npz` and
  `metrics.py` evaluates exact signed-distance functions against them. The
  ground is not part of the field and is never counted.
- **`scene_mesh`** — USD-environment scenarios. There is no analytic field,
  so after each trial the harness extracts the scene's geometry
  (`scripts/extract_scene_mesh.py`, cached in `results/mesh_cache/` keyed on
  usd + env_scale + flight corridor + extraction params) and scores clearance
  against a dense point-sampling of its surfaces. Details and caveats:
  - Extraction must run under **Isaac's python** (`--sim-python`): opening an
    `omniverse://` stage needs the Nucleus resolver, which only exists once
    Kit boots. It runs headless *after* the flight, so it never competes with
    the trial's own Isaac instance, and the ~1 min boot is paid once per
    scene, not per trial.
  - The extractor mirrors exactly how `run_px4_sim.py` places the stage
    (composed world transforms × `env_scale`), including instanced geometry.
    `PointInstancer` prims are *not* expanded (a warning is printed).
  - **Extraction is cropped to the flight corridor** — the start/goal box
    plus a 50 m horizontal margin (city stages are km-scale; sampling the
    whole stage at 5 cm is intractable, and the EnglishCollege stage alone
    has ~9 M triangles). Clearance to geometry outside the crop is
    unmeasured; if the drone strays within 10 m of a crop face,
    `clearance_bounds_exceeded: true` is recorded in that trial's
    `metrics.json`. Different start/goal pairs in the same scene get their
    own cache entries.
  - **Ground-like faces are excluded**: faces within 30° of horizontal
    (terrain, floors — but also rooftops/ceilings) are dropped so the metric
    matches the analytic-field semantics, where the ground is not an obstacle
    and takeoff/landing don't count as collisions. Consequence: skimming low
    over a flat roof registers no clearance signal; passing close to walls,
    trees, poles, or facades does.
  - Clearance is measured **from takeoff onward** — the parked drone sits
    legitimately ON the scene surface.
  - Surface sampling is a 5 cm lattice (`SCENE_MESH_SAMPLE_H`), so distances
    *overestimate* the true surface distance by at most ~5 cm; `collided`
    means the scene surface came within `drone_radius` of the drone's centre.
  - If extraction fails (e.g. Nucleus unreachable), the trial is still scored
    — just with empty clearance/collision columns, as before.

### Retrofitting clearance into old USD runs

Runs recorded before scene-mesh scoring existed have empty clearance columns.
Rescore them in place (flights are not re-run; `traj.npz` is rescored and each
`metrics.json` + `summary.csv` updated):

```bash
python scripts/run_comparison.py --report-only --rescore-clearance \
    --results-dir results/20260703_134005 \
    --sim-python <isaacsim>/python.sh
```

Note that `collided`/`success` may change: a trial that "succeeded" before
(collision unscored) can turn into a failure if it actually clipped scene
geometry.

## Interpreters & checkpoints

Each method runs in its own venv (`methods/<repo>/.venv`; agile via
`scripts/agile_python.sh` -> the repo-root `.venv`); `run_px4_sim.py` needs
Isaac's python. Defaults live in `superfly.compare.registry` (one data entry
per method, `ckpt_kind` file/hydra_dir/tf_prefix) and every interpreter/
checkpoint is overridable with `--<method>-python` / `--<method>-checkpoint`;
`--sim-python` defaults to `$ISAACSIM_PYTHON`.

Methods whose checkpoint is missing are skipped with a message (see
`checkpoints/README.md` for which artifacts exist today). Select a subset
with `--methods diffaero depthnav`.

## Key options

| flag | default | meaning |
|---|---|---|
| `--max-speed` | 3.0 | cruise speed, mapped to each method's own flag |
| `--drone-radius` | 0.2 | collision radius for clearance scoring |
| `--goal-radius` | 1.0 | reached-the-goal distance |
| `--climb-alt` | 2.0 | default climb height above spawn (per-scenario override) |
| `--warmup` | 45 | seconds for Isaac to boot before the offboard launches |
| `--headless` | off | no GUI viewport (recommended for batch runs) |
| `--record-video` | off | per-trial depth.mp4 + rgb.mp4 (headless-safe) |
| `--px4-dir` | `$PX4_DIR` or `~/PX4-Autopilot` | PX4 checkout for SITL |
| `--no-px4-manage` | off | you run PX4 SITL yourself in another terminal |

## Results layout

```
results/<YYYYMMDD_HHMMSS>/
├── command.txt          # exact harness invocation
├── scenarios.json       # verbatim copy of the scenario file
├── run_manifest.json    # parsed args + resolved scenarios
├── px4_sitl.log         # all PX4 boots of this run
├── summary.csv          # per-method aggregate (written by --report)
└── <scenario>/<method>/
    ├── traj.npz         # ground-truth trajectory (+ field or goal/start)
    ├── metrics.json     # scores + hyperparams + scenario + exact commands
    ├── depth.mp4        # with --record-video
    └── rgb.mp4          # with --record-video
```

## Troubleshooting

- **"Waiting for heartbeat" hangs** — a stale `px4` process is holding the
  instance-0 lock: `pkill -9 -f bin/px4`. (The harness does this itself
  before every trial unless `--no-px4-manage`.)
- **`Errno 98 Address already in use`** — a leftover `run_px4_sim.py` holds
  TCP 4560; the harness kills stale sims automatically, or
  `pkill -f run_px4_sim.py`.
- **Every `make px4_sitl` dies instantly with an OpenSSL/cmake error** — the
  shell had Isaac's `LD_LIBRARY_PATH` exported; the harness launches PX4 with
  a sanitized environment, so use the harness-managed PX4 (default).
- **Empty clearance columns on a USD run** — the run predates scene-mesh
  scoring (fix with `--rescore-clearance`, see above) or extraction failed
  (check the `[scene-mesh]` lines in the harness output; Nucleus must be
  reachable).
