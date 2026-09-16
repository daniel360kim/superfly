# Policy Comparison Harness

Flies **DepthNav**, **DiffAero**, and **Agile Autonomy** (plus their
velocity-command variants, plus the `anyanything` ONNX student as
`agile_student`) through the *same* Isaac-Sim + PX4-SITL scenarios and scores them on the same metrics, without
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

## `agile_student` — flying an anyanything ONNX student

`agile_student` is the `anyanything` end-to-end student (test-5 recipe-v2
labels) flown through the *same* offboard, acados MPC and PX4 plumbing as
`agile`. Nothing is duplicated: the `.onnx` extension of `--checkpoint` is
what switches `superfly.policies.agile.core` over, and `ckpt_kind: onnx` is
what makes `checkpoint_ready` gate it. Three things differ from `agile`:

| | `agile` (ckpt-50) | `agile_student` (.onnx) |
|---|---|---|
| state | 21-dim, **de-yawed** R, goal as a **unit direction** to a point `future_time * max_vel` ahead on the mission line | 22-dim, **raw** R, goal as the **metric** body-frame vector clamped to 10 m, plus `v_goal` (arrival speed, `--goal-speed`) |
| plan | 10 waypoints at 0.1 s, rescaled by `max_vel / 7` | M x N waypoints (both read off the graph) in **absolute metres** at 0.5 s — **never** rescaled |
| mode | always the lowest `\|alpha\|` | depth veto + argmin cost (`--mode-select`) |
| altitude | PD hold on the climb altitude | **follows the plan's z** (`--alt-follow`, forced), clipped to the reference's `Z_REF` = 0.5-4.0 m |
| omega | body rate | `R^T` x body rate — what test-5 fed (see below) |
| reference | unconstrained cubic, position integrated from it | `sim_episode.fit_cubic`: pinned to `p(0)=p`, `p'(0)=v`, evaluated at `tau = t - t_decision` |

**Rate / resampling.** The student's waypoints are *not* resampled. The MPC's
`build_reference` uses `dt_wp` only to build the time base `t = arange(nwp) *
dt_wp` for a cubic, which it then samples at the solver's own nodes (N=10 x
0.1 s = a 1.0 s horizon), so passing the true 0.5 s spacing is exact. What is
changed is *how many* points the cubic is fitted through: a least-squares
cubic over the whole 5 s plan is a poor local fit for the first second, so the
student feeds it the current position (t=0, the waypoint the student does not
emit) plus waypoints 1-3 (t = 0.5, 1.0, 1.5 s) — four points, one
exactly-determined cubic, the same window the test-5 evaluation harness fits
(`sim_episode.fit_cubic`). `STUDENT_MPC_WAYPOINTS` in `policies/agile/core.py`.

**Depth veto** (`--mode-select veto`, the default for a student): each mode's
next 3.5 m of body-frame path is projected into the 224x224 depth frame the
policy was just given, with the render pinhole (640x480 @ 91 deg, principal
point at `(w-1)/2`, anisotropically resized 640->224 / 480->224); a mode with
a sample behind the depth surface by more than 0.15 m is given an infinite
cost. `--mode-select cost` restores upstream's always-mode-0 rule.
`tests/test_agile_student.py` asserts this agrees with the reference
implementation (`superfly_expert_sampler.sim_episode.DepthVetoPolicy.blocked`)
on a rendered wall, and that the state encoder is byte-identical to
`OnnxPolicy.encode_state`.

**Altitude.** `--alt-follow` is **forced** for `agile_student`. Without it the
MPC reference sets `vz = 0, z = cruise_alt` and thrust becomes an altitude PD:
the student's vertical plan is deleted while the depth veto still clears modes
on their vertical geometry, so it can pick a climb-over and fly flat into the
obstacle. World-frame plan z is clipped to `sim_episode.Z_REF` = (0.5, 4.0) m
in `_adopt_plan`, before mode selection, exactly as `run_episode` does.

That band only means anything if it *contains the vehicle*, and 12 shipped
scenarios climb to 5 m (4 to 3 m): clipping a level plan to 4.0 m while the
prepended origin sits at 5.0 makes the cubic command a ~2 m/s dive from the
first decision. Two guards:

* `agile_student`'s registry row carries **`climb_alt = 2.0`**, which
  `build_commands` lets override the scenario's — the student hands over inside
  the band its labels cover. This was chosen over a `--student-climb-alt` CLI
  flag because it is registry *data* (how CLAUDE.md says a method is
  configured), needs no new CLI surface, and applies to every scenario file
  automatically. The override is printed at dry-run/launch and lands in each
  trial's `metrics.json` `commands` like any other argument. Disable it with
  `AGILE_STUDENT_CLIMB_ALT=scenario`, or set a number to change it. **Caveat:
  `agile_student` then flies a given scenario lower than the other methods** —
  deliberate, but say so when comparing.
* `core.student_z_band()` widens the ceiling to `max(4.0, handover + 2.0)`
  anyway, so a scenario forced above the band still never flies a commanded
  dive; the startup log says loudly when that happened.

**Altitude hold, and the hover throttle.** `--alt-follow` alone is not enough:
the evaluation simulator applies `f = a_cmd + g` exactly, so it has no
throttle-mapping error, while PX4 has one. The offboard used to assume
`hover_thrust = G / MAX_ACCEL = 0.490`; the Iris in the 2026-09-16 airstation03
trial hovers at **0.577**. With an `alt_target` re-pinned to the vehicle at every
replan, `alt_err` is ~0 by construction and that 18 % gap can only be balanced by
`-kd_alt * vz` — a permanent sink of
`-G(h_true/h_assumed - 1)/kd_alt = -0.43 m/s`, which is why the first trial flew
the whole field at 0.1-0.6 m. Two halves, both in:

* `core._advance_alt_setpoint`: an **absolute** setpoint `_alt_sp`, initialised
  at the handover altitude and advanced each control tick by the *plan's* own
  vertical velocity, clipped to `z_band()`. It is advanced by the waypoint-z
  profile, **not** by the cubic's `vz` — the cubic is pinned to
  `p'(0) = v_current`, so using it would let the setpoint chase the vehicle
  again (measured: a level plan ratcheted it up 0.28 m during a recovery climb).
* `--hover-thrust` / `AGILE_HOVER_THRUST` (registry default **0.577**),
  `ki_alt` 0.4 → 1.5, an integrator limit of ±4 m/s² (±2 saturates at an
  `h_true/h_assumed` ratio of 1.20, and this airframe is already at 1.18), and
  an **online estimate**: at equilibrium `thrust * cos_tilt` *is* the true hover
  throttle whatever parameter was assumed, so it is EMA'd over consecutive
  settled samples (|vz| < 0.05, |alt_err| < 0.05, 2 s worth) and adopted, with
  the integrator rebased so the swap is bumpless. Logged once on adoption, and
  `alt sp=` / `hover=` appear in each verbose line.

Verified against the reviewer's PX4 plant (`a_z = G*thrust/h_true - G`) from a
1.96 m handover, `tests/test_agile_student.py`:

| case | z after 20 s | vz |
|---|---|---|
| level plan, assumed 0.490 vs true 0.577 (**was 0.24 m / −0.45 m/s**) | 1.960 | +0.000 |
| same, estimator disabled | 1.960 | +0.000 |
| same, parameter calibrated to 0.577 | 1.960 | +0.000 |
| a true 0.63 airframe, estimator off | 1.960 | +0.000 |
| the net's own −0.11 m/s sink | tracks the setpoint to 0.016 m (both walk to the band floor) | |
| a genuine +0.8 m/s climb-over | follows it to the band ceiling | |

**Reference.** The student's MPC reference is `sim_episode.fit_cubic`, not
`np.polyfit`: it pins `p(0) = p_current` **and `p'(0) = v_current`` and
least-squares only the quadratic/cubic terms through waypoints 1-3. The
unconstrained polyfit the legacy path uses commands 1.36 m/s at t = 0 from
rest, a step demand every time the vehicle is slower than the plan's mean
speed. `mpc.build_reference_cubic` samples it at the solver's nodes, offset by
the plan's age so the manoeuvre is consumed rather than dragged along
(`tau = t - t_decision`).

**Rate.** `--net-thread` is **forced**. One forward pass of the 5 s student
costs 120-220 ms of CPU (measured on gs2, 32 cores under load average 39,
median of 15-20 runs: 174 ms default session options, 124 ms at 8 intra-op
threads, 118 ms at 12-16 — returns flatten past 8, so `OnnxStudentBackend`
pins 8, `AGILE_ONNX_THREADS` overrides). Run inline that blocks the 100 Hz
attitude stream to PX4 and collapses the decision rate to ~5 Hz against the
evaluation harness's 15. Threaded, the worker is latest-only and rate-gated to
`STUDENT_DECISION_HZ = 15` so a fast box cannot out-run the rate the policy was
scored at; the achieved rate is printed in the offboard's verbose line with a
`<-- BELOW 15 Hz` marker, and the *measured single-forward time* is logged once
at startup (`[agile] onnx forward NNN ms ... -> at most N.N Hz of decisions on
this box`). **Check that line on the target box before trusting any result**:
loaded gs2 measures 140 ms / 7.1 Hz ceiling and achieves 5.4-7.3 Hz closed loop.

The plan's clock starts at the state the net input was built from, not at
adoption — in threaded mode those differ by one forward pass, which at 3 m/s
would park the reference's t=0 point ~0.4 m behind the vehicle for the plan's
whole life (chronic braking).

A non-finite net output keeps the previous plan; a non-finite *first* plan
raises and the trial is scored as an error.

**omega.** The port feeds `R^T omega_body`, not `omega_body`. The evaluation
harness's `Obs.omega` is already a body rate (`run_episode` integrates it from
`dR = R^T R_new`) and `encode_state` applies `R^T` again, so `R^T omega_body` is
what test-5 measured. It is also harmless to match: the TRAINING data's omega
column is drawn noise, not a measured rate
(`draw_states.synthesize_attitude`: `omega = rng.normal(0, 0.3, 3)`,
independent of `R`), and the loader passes it through unrotated (it rotates
velocity only), so the net learned nothing frame-dependent from that channel.

**Speed.** `--max-vel` is pinned to 3.5 = `sim_episode.V_CAP`, not the harness
default 3.0: the real student plans above 3.0 m/s (measured 3.08) and the MPC's
speed cap would otherwise clip it chronically and bias arrival time. It never
rescales the plan.

**Interpreter.** Needs `acados_template` **and** `onnxruntime` in one
interpreter. `scripts/agile_python.sh` targets the repo-root `.venv`; where
that was never built, point it elsewhere with `AGILE_PYTHON=<python>` (it
still exports the ACADOS env vars, which a bare `--agile_student-python`
would not).

```bash
python scripts/run_comparison.py configs/scenarios/probes/diffaero_field.json \
    --methods agile_student --headless --record-video --report \
    --sim-python <isaacsim>/python.sh
# a different student:
#   --agile_student-checkpoint checkpoints/Student/t5_m2_r2/student.onnx
# a non-zero arrival speed:  AGILE_STUDENT_GOAL_SPEED=2.0
# upstream mode rule:        AGILE_STUDENT_MODE_SELECT=cost
```

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
