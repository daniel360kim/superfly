# ATTEMPTS — ledger of what's been tried

Condensed record of approaches and their verdicts. Check before re-trying
anything. Verdicts: `REJECTED` / `ESTABLISHED-NEGATIVE` / `SHIPPED` /
`SUPERSEDED` / `PENDING`.

## 2026-08-20 — Cylinder.generate z-offset bug: planar spawns 0.5 m below target — ESTABLISHED-NEGATIVE (fixed @762a2e2)

`Cylinder.generate` computed `z = half*rand - 0.5 + mean` — the 0.5 is NOT
scaled by half, which cancels only for the stock half=1.0 configs. The
planar variant's pinned spawn (half=0) therefore spawned at mean-0.5 =
1.0 m while targets sat at 1.5 m: with vz forced to 0 and success radius
0.1, success_rate was mathematically pinned at 0 (superfly-train-depthnav-8,
cancelled; the 3D run scored 0.86+ at the same phase, which is what flagged
it). Fixed to `half*(rand-0.5)` (Uniform semantics; byte-identical
distribution for half=1) + planar spawn-vz noise zeroed. Retrained as
superfly-train-depthnav-9: sr 0.86 within 15 min of level1. Lesson: when a
variant pins a distribution to a point, verify the rng actually emits the
mean — Uniform and Cylinder had different offset conventions.

## 2026-08-19 — depthnav velocity-command variant (VELOCITY_YAW) re-implemented in the fork — SHIPPED (code); OSMO training PENDING

Mirrors the diffaero pmv re-implementation: the vel variant existed only in
the lost checkout, so it was rebuilt in `methods/depthnav` @ `4f845f5`
(pushed), schema-matched to what `DepthNavPolicy(action_mode="velocity")`
reads. Dynamics `velocity_world_frame` = explicit PX4 velocity loop (P on
vel error, MPC_*_VEL_P_ACC defaults 1.8/4.0, nominal-9.81 hover feedforward
so the randomized-gravity mismatch stands in for hover-thrust estimation
error) -> starling accel box (xy 17.0 / z 0.5..19.6, from
`configs/vehicles/starling2max.yaml`) -> first-order rotor lag (lmbda
[11.8, 18.2] per env, = sys-ID tau 85..55 ms) -> the existing thrust
integrator, so orientation/omega/jerk fall out unchanged.
`VelocityBoundedYaw` bounds xy on the norm (tanh, 2.5) and z componentwise
(1.5) — MUST stay aligned with the registry's --max-vel-xy/--max-vel-z.
Targets 0.8–1.5 m/s (`target_speed` mean 1.15 half 0.7; Uniform spans mean
± half/2 — same trap as the thrust variant's [2,4]). Verified on gs2 CPU:
setpoint tracking exact, 0.56 s rise, grad/detach/indexed-reset, activation
bounds, state_dict identical to level1_4's, deploy wrapper loads + steps it.
Training: `scripts/train_depthnav.py [--vel]` (artifact
`checkpoints/DepthNav/level1_vel/level1_vel.pth`) via
`osmo/superfly-train-depthnav.yaml` — habitat-sim builds FROM SOURCE in-job
(--headless --with-cuda; gpu2gpu default is True) with an S3 deps-cache of
the built site-packages entries; the 4.6 GB scene dataset ships once as
`deps/depthnav_dataset_v1.tar` (stage_superfly_osmo.sh now excludes
`datasets/depthnav_dataset`), and a 30-min background loop mirrors logs/ to
S3 mid-run. Untested risks, in order: habitat source build against the
image's python/toolchain, EGL headless render on the pool (the env
preflight in the YAML fails fast on both), torch-2.9-vs-2.2 API drift in
the trainer.

## 2026-08-18 — Scene vetting pipeline (audit + planar goal mining) — SHIPPED (pilot); Nucleus sweep PENDING on auth

`scripts/scene_audit.py` (airstation03, Isaac python) + `scripts/
mine_scene_goals.py` / `superfly.perception.occupancy` (gs2) + `docs/
scene_vetting.md`. Verified three ways before touching real scenes:
- **Known-answer tests** (gs2, `usd-core`, no GPU): a synthetic stage
  (cm units, nested xform, instanceable reference, PointInstancer) through
  the REAL `mesh_sampling` -> extract-npz -> `metrics.score_trajectory`
  path reproduces hand-computed numbers (gap clearance 0.8 m ±h, wall
  crossing collides, 0.15 m ground skim does NOT — filter semantics).
  19 tests in `tests/`, run with the repo venv.
- **Pilot** on `isaac_Full_Warehouse` (public NVIDIA S3, no auth): full
  audit 37 s warm; drop test 3/3 rest at exactly floor+radius (colliders
  verified physically — `scene_setup.add_colliders` is now the SHARED impl
  the flight harness also calls); depth probe 100% finite 15–32 m;
  mining found 122 gated pairs, 3 selected across difficulty percentiles.
- **Pilot caught a real bug**: dominant-ground-mode picked the warehouse
  CEILING (9.1 m) — in any roofed scene ceiling+roof out-sample the floor.
  Fixed: ground = lowest bin holding ≥25% of the max bin. Also: stock
  NVIDIA stages ship ~3.4k authored colliders + their own PhysicsScene
  (FLAGged); the negative control must STRIP authored collision APIs or
  it is vacuous.
Blockers/next: **Nucleus auth on airstation03 is EXPIRED** (omni.client
Auth error 5) — the airlab Library/Stages sweep needs an OMNI_API_TOKEN
(Navigator) or interactive re-login; the token would also decide the OSMO
fan-out probe (see 2026-07 omniverse://-in-OSMO ESTABLISHED-NEGATIVE — the
token path postdates it and is untested). Disk on airstation03 at 97%
(pilot ran with `--force`, outputs capped ~60 MB/scene).

## 2026-08-17 — pmv (velocity-command) dynamics re-implemented in the fork — SHIPPED

Goal shift: deployments now want **PX4 velocity setpoints, planar at
0.8–1.5 m/s on the Starling 2 Max** (not attitude+thrust). The pre-reorg
`velocity_pointmass` (pmv) dynamics + planar option existed only in the dead
checkout; re-implemented in `methods/diffaero` @ `dfbb935` schema-matched to
the surviving `sha2c_vel_cmd*` hydra configs and to what `DiffAeroVelPolicy`
reads: first-order velocity lag `alpha=1-exp(-lmbda*dt)` (identical to the
deploy-side `_apply_velocity_lag`), level attitude with rate-limited yaw
slew toward the vel EMA (deploy `slew_yaw_ned_cmd`), planar = 2-dim [vx,vy]
action with vz≡0, targets flattened to spawn altitude (with a reachable
planar min-init-dist — the 3-D half-diagonal is NOT reachable same-altitude
from a centered spawn), episodes start goal-facing (deploy YAW phase).
`cfg/dynamics/pmv_planar.yaml` is the Starling low-speed config (action
clamp 2.0 m/s, rand 1.5–2.5; cruise band set by env target vels). Verified
on gs2 CPU: tiny sha2c train + jit/onnx export for pmv_planar AND pmv, and
the exported planar actor loaded + stepped through `DiffAeroVelPolicy`
(planar flag, clamps, vz==0, yaw bridge all exercised). Registry entry
`diffaero_vel_planar` -> `checkpoints/DiffAero/pmv_planar_starling_v1`;
eval suite `suites/planar_lowvel_v1.json` (diffphys+diffaero fields,
climb_alt 2, budgets sized for ~1 m/s). NOTE: fork push to GitHub was
blocked in-session — `daniel360kim/diffaero` main is ahead of origin
locally; push before relying on GitHub state.

## 2026-08-18 — Planar low-vel deployment PASSES the diffaero+diffphys fields — SHIPPED

`diffaero_vel_planar` (checkpoint `pmv_planar_starling_v1` + the offboard
arming-retry and grounded-recovery fixes) flies
`suites/planar_lowvel_prims_v1.json` **6/6, zero collisions** (eval
`planar5`): clearances 0.40–1.27 m at ~1.0 m/s cruise on all four
standard diffphys/diffaero fields, and both dense stretch fields pass
too (s112_dense squeaks by at 0.01 m, 104 s). Peak speed 1.43 m/s —
fully inside the 2 m/s clamp.

What the five eval campaigns established on the way:
- **Asset-mesh suites (obstacle_assets: true) are stochastic for this
  policy**: planar1/3/4 scored 4/6, 3/6, 2/6 with the same policy. RTF
  was healthy (0.66–0.85) every time; the failures are physical grazes
  with USD tree meshes that extend past the analytic primitives —
  contact happens OUTSIDE the 86° camera cone during lateral dodges,
  then PX4's land detector latches (now recoverable in the offboard,
  but a graze near a trunk is still a tumble). Analytic clearance was
  held (0.1–0.6 m) in every one of those "failures". Documented gap:
  margin vs. unseen canopy is a forward-camera limitation to attack
  later (wider margin training recipe, or camera-cone-aware costs).
- **v2 (r_drone 0.3 + n_obstacles 40) REGRESSED**: 1/6 on the asset
  suite, collisions on fields v1 passed. Reverted; committed for
  reference. Don't reuse that recipe as the margin lever.
- The offboard now **retries arming** every 2 s in CLIMB (first-trial
  shader-compile stall made PX4 reject the single arm attempt —
  deterministic "s89 failure" was purely an order artifact) and
  **re-runs CLIMB/YAW if grounded+motionless mid-policy** (land-detector
  latch), capped at 3 recoveries.

Caveats on the passing claim: eval vehicle is **iris** (starling2max
USD lives on Nucleus, unusable on OSMO; the velocity-loop deploy
contract abstracts the airframe, but fly `--vehicle starling2max` on
airstation03 before hardware). The diffaero fork's pmv commit is
**unpushed** (session permission); push `daniel360kim/diffaero` main
before cloning anywhere fresh.

## 2026-08-18 — pmv_planar + pmc-starling trained on OSMO — SHIPPED

`checkpoints/DiffAero/pmv_planar_starling_v1` (train-17, **sr 0.92 /
survive 0.96**, targets 0.8–1.5 m/s, max_time 60, 2000 updates) and
`sha2c_pmc_starling2max_v1` (train-18, sr 0.78 at 3–6 m/s) — both
committed. It took runs 12→17 to get there; each failure is now fixed in
`osmo/superfly-train-diffaero.yaml` + `scripts/train_diffaero.py`:
- **cpu8/32Gi/100Gi was unschedulable for hours with 11/12 pool GPUs
  idle** — the shared nodes are CPU/mem-starved by other tenants. This
  trainer is a fully on-GPU sim: 2cpu/16Gi/40Gi schedules in ~2 min.
  (The 8/32Gi umbrella-doc guidance is a ceiling for heavy jobs, not a
  floor.)
- taichi dlopens libX11 → apt set added (~25 s, eval-job pattern).
- run_meta's `git rev-parse` died: no git in the image (try/except now).
- the S3 mirror needs boto3 (not a diffaero dep; pip'd in the workflow).
Eval: `superfly-eval-2`, suite planar_lowvel_v1, methods
diffaero_vel_planar, --max-speed 1.2, tarball superfly_stage_0817f.tar
(= 0817e sim payload + repo + the new checkpoint).

## 2026-08-17 — diffaero requirements.txt was missing einops — ESTABLISHED-NEGATIVE

`import diffaero.algo` pulls `dreamerv3` → `einops`, but einops was never in
`requirements.txt`, so ANY fresh-venv training job (incl. OSMO
superfly-train-diffaero-4, cancelled) dies at import after a clean deps
install. Fixed in the same fork commit (also added `onnxscript`, required by
torch>=2.13 ONNX export). If a diffaero job fails at import, check the
requirements install log before suspecting the code.

## 2026-08-17 — DiffAero planar checkpoints are unrecoverable — ESTABLISHED-NEGATIVE

`checkpoints/DiffAero/planar_{cnn,mlp,rcnn}_sr0.9*` were committed as
**symlinks** pointing at `/home/ubuntu/superfly/diffaero/outputs/train/2026-07-03/...`.
Only the symlink text ever entered git; the targets lived in a gitignored
Hydra output tree on a checkout that no longer exists. The weights are gone.
The three symlinks and the `diffaero_vel_planar` registry entry that pointed
at them are deleted. Any future planar run must be retrained from scratch
(`scripts/train_diffaero.py`, planar config) — do not go looking for these
files on other boxes; the 2026-07-03 output tree survives nowhere.

## 2026-08-17 — Workspace reorganization — SHIPPED

`starling-deployment/` -> installable `src/superfly` package + `scripts/`
entrypoints; common MAVLink/frames/sentinel/transport layer extracted (was
declared 4-6x per offboard); sim + registry table-driven; gsds + DiffPhysDrone
methods removed (the `diffphys` *obstacle field* stays — suites use it as a
shared world); checkpoints normalized to `<Method>/<run>/` + `run_meta.json`;
OSMO workflows renamed `superfly-*`. Regression gate: `superfly.compare.metrics`
re-scored two pre-reorg trajectories byte-identically, `--report-only`
aggregation identical on two results dirs, `--dry-run` argv correct. Flight
code is only statically verified — first hardware trial per method on
airstation03 is the outstanding proof. A shared offboard PhaseMachine was
deliberately NOT introduced (unverifiable without hardware; the per-method
phase differences are load-bearing and documented in CLAUDE.md).

## 2026-08-17 — Baseline method sources lost from git — ESTABLISHED-NEGATIVE

`agile_autonomy` and `depthnav` were never in git; `diffaero`/`DiffPhysDrone`
were broken gitlinks with no `.gitmodules` (purged in 5400a1f); the only real
submodule in history, `mapnav`, is stranded on `main` (not an ancestor of
`triage`). The only surviving local copy is `/home/ubuntu/aa_build/agile_autonomy`
(non-git, missing `planner_learning/`). Recovery path: fork upstream repos on
GitHub (user action; `gh` absent on gs2) and add as `methods/` submodules.
**RESOLVED 2026-08-17**: `daniel360kim/{depthnav,diffaero,agile_autonomy}`
added as submodules; agile_autonomy's fork includes `planner_learning/`.

## 2026-08-17 — depthnav_vel checkpoint — PENDING (code half RESOLVED 2026-08-19, see the VELOCITY_YAW entry)

`depthnav_vel` has ALWAYS pointed at `level1_vel.pth`, which was never
produced by anyone; it is checkpoint-gated out of the harness until Phase 3
delivers it. 2026-08-17 update: the fork also lacks
`examples/navigation/policy_cfg/small_yaw_vel.yaml` (the config
`superfly.policies.depthnav` VELOCITY_CFG points at) — it existed only in
the lost checkout, so Phase 3 must recreate the config too (the docstring in
`policies/depthnav.py` records its known training parameters). (An earlier draft of this entry also claimed
`sha2c_vel_cmd_oa` lacked its `exported_actor.pt2` — false, a truncated
directory listing; the artifact is committed and `diffaero_vel` is ready.)
2026-08-19: config + VELOCITY_YAW support recreated in the fork (@4f845f5);
only the trained `.pth` itself is still missing.

## 2026-08 — Agile training dataset: sampler renders no images — PENDING

`superfly_expert_sampler` labels are byte-compatible with upstream's training
format (verified: `trajectories_bf_*.csv` = 11 waypoints x 17 state fields +
`rel_cost`, body frame, 5 candidates/frame; 21 s and 8.2 MB per flight) but
contain NO depth images — it plans against a point cloud, never a camera.
`scripts/build_agile_dataset.py` (to write) must render 224x224 / 91 deg /
20 m depth at each `odometry.csv` pose from the rollout's
`pointcloud-unity.ply`. **Flagged risk: train/deploy camera mismatch** is the
most likely way to get a policy that trains cleanly and flies badly — build
against the sim's agile camera row (POLICY_CAMERAS) and validate by rendering
one rollout both ways (point-cloud raycast vs Isaac) before training on any.

## 2026-07-29/30 — Isaac eval on OSMO works; the workarounds are load-bearing — SHIPPED

Full measured detail lives in `osmo/superfly-eval.yaml`'s comments (kept
verbatim on purpose). One-line index of what each failure cost:
- **carb import**: `--sim-python` isn't isaacsim's python.sh; a sitecustomize
  puts kit on sys.path (without it every trial dies at import).
- **/rtx/dataWindowNDC unset** -> `None - None` TypeError killed every trial
  (gsds-superfly-5); identity window (0,0,1,1) fixes it.
- **root user**: the harness's own `pgrep -f run_px4_sim.py` sweep SIGTERMed
  container PID 1 (gsds-superfly-2); run as non-root `flyer`.
- **root-owned Isaac install + non-root user**: renderer cache init fails
  (gsds-superfly-7); chown the install.
- **libGL.so.1 absent** from the pytorch image: Kit loads, frames come back
  empty; ~25 s apt set fixes (no deps cache needed for eval).
- **isaacsim pip install downgrades torch to cu126 (no sm_120)**: separate
  /opt/policy venv on cu128.
- **missing torchvision made a dead policy look like a bad one** (rc=0,
  success 0.00, gsds-superfly-9): hence the per-method import preflight.

## 2026-07-30 — asyncRendering=false / waitIdle=true on OSMO — REJECTED

Tried to fix empty frames (gsds-superfly-9/10); serialized the render loop,
RTF 0.308 vs 0.798 local. The sim is NOT lockstepped, so lower RTF = fewer
control updates per simulated second: the drone crawled (mean_speed 0.54 vs
2.48) and never reached goals. **Throughput is a correctness property here.**
The empty frames were really the root-owned-install bug. Never re-try.

## 2026-07 — omniverse:// Nucleus URLs inside OSMO containers — ESTABLISHED-NEGATIVE

Cannot authenticate headless (no cached credential, no browser SSO); stages
compose EMPTY silently and trials fly in a void recording plausible numbers
(jobs 32/33). Fix shipped: tarball ships `usd_stages/`, `GSDS_USD_STAGE_ROOT`
retargets scenario refs at point of use, `GSDS_REQUIRE_STAGE=1` aborts a
trial whose stage still composes empty.

## 2026-07 — Obstacle-asset surface sampling on OSMO — ESTABLISHED-NEGATIVE

`sample_subtree` of PointInstancer foliage group-OOM-kills the 32Gi container
(jobs 32/33) and has never produced usable samples anywhere; scoring always
fell back to the analytic field. `GSDS_SKIP_OBST_SAMPLING=1` on OSMO, always.

## 2026-08-17 — Starling 2 Max spec-matched training (diffaero first) — PENDING

Source of truth: `configs/vehicles/starling2max.yaml` (from the AirLab
sys-ID doc, Slite dj1982T35dt0qG). Key mappings, all derived there:
- **Motor lag**: pmc's `lmbda` IS a first-order lag rate
  (`a_dot = (1-exp(-lmbda*dt))/dt * (a_cmd - a)`, dynamics/pointmass.py), so
  lmbda = 1/tau. Sys-ID tau 72 ms -> lmbda 13.9, randomized [11.8, 18.2]
  (tau 85..55 ms, the measured down/up asymmetry). **The pmc default
  lmbda=2.6 corresponds to tau ~385 ms — the baseline plant was ~5x more
  sluggish than the real motors.**
- **Accel box**: T/W ASSUMED = 2.0 (sys-ID has no max thrust; assumption
  matches the harness's MAX_ACCEL=20 deploy convention within 2%). z-max
  = 2g = 19.6, xy-max = g*sqrt(3) = 17.0 (horizontal component at full
  thrust with 1 g held). Retighten when a measured max thrust lands.
- **Deploy bounds**: the exported actor rescales by min/max_action passed at
  call time; DiffAeroPolicy now reads them from the ckpt run dir's hydra
  config (legacy ckpts: reads their own 20/40, byte-identical regression).
- **Eval vehicle**: `--vehicle starling2max` (sim + harness) flies the lab
  starling2max.usd with measured rotor constants and a NEW
  FirstOrderQuadraticThrustCurve (55/85 ms asymmetric rotor lag; Pegasus
  stock curve is instantaneous). Not yet flown — validate rotor order/spin
  dirs + PX4 gains on airstation03 before any campaign.
- Not spec-mapped (no sys-ID data): drag (kept pmc defaults), camera (kept
  the method's own training sensor).

Training run: OSMO `superfly-train-diffaero-3`, tag sha2c_pmc_starling2max_v1
(=-2 failed instantly: pytorch3d has no PyPI distribution; fixed to the
GitHub stable tag). depthnav/agile spec-matching deferred by scope decision
(diffaero first).
