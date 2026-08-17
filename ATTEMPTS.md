# ATTEMPTS — ledger of what's been tried

Condensed record of approaches and their verdicts. Check before re-trying
anything. Verdicts: `REJECTED` / `ESTABLISHED-NEGATIVE` / `SHIPPED` /
`SUPERSEDED` / `PENDING`.

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

## 2026-08-17 — depthnav_vel checkpoint — PENDING

`depthnav_vel` has ALWAYS pointed at `level1_vel.pth`, which was never
produced by anyone; it is checkpoint-gated out of the harness until Phase 3
delivers it. (An earlier draft of this entry also claimed
`sha2c_vel_cmd_oa` lacked its `exported_actor.pt2` — false, a truncated
directory listing; the artifact is committed and `diffaero_vel` is ready.)

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
