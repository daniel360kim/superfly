# checkpoints/ — committed policy weights

Layout: `checkpoints/<Method>/<run>/`, one `run_meta.json` beside each run
(method, producing command, provenance; runs written by
`scripts/train_<method>.py` also record the submodule SHA and host).

**These are committed in git on purpose** — anyone who clones gets flyable
weights with no lab credentials and no LFS. Every new run adds permanently;
the pressure valve is pruning superseded runs (delete the run dir in a
normal commit), never rewriting history. `checkpoint_ready()`
(`superfly.compare.registry`) gates each method on its artifact's existence
per `ckpt_kind`, so a missing artifact skips the method with a notice
instead of flying garbage.

TFLite/ONNX exports sit beside their source checkpoint where they exist:
`scripts/export_tflite.py` (DiffAero, ONNX→TFLite) and
`scripts/export_depthnav_tflite.py` (DepthNav, .pth→ONNX→TFLite; the GRU
makes it a different pipeline). Each `.tflite` has a `.tflite.json` sidecar
with sha256s, tool versions, and verification error stats.

## Index

Speeds are the training band. "6/6" results are the
`suites/planar_lowvel_prims_v1.json` benchmark at `--max-speed 1.2`
(deterministic primitives, zero collisions unless noted); see ATTEMPTS.md
for the campaigns.

### DepthNav (habitat-trained, 72×128 depth, GRU policy)

| run | registry method | artifact | what it is |
|---|---|---|---|
| `level1_4` | `depthnav` | `level1_4_iteration_13500.pth` | Legacy thrust-command policy (2–4 m/s, `small_yaw.yaml`). Pre-reorg; exact training config unrecorded. |
| `level1_vel` | `depthnav_vel` | `level1_vel.pth` + `.onnx` + `.tflite` | Velocity-command, 0.8–1.5 m/s starling velocity-loop dynamics. OSMO run superfly-train-depthnav-7, harvested at plateau iter 9500 (training-eval sr 0.94–0.98, collision 0). **Benchmark 6/6.** Full 20k-iter final: `s3://superfly/runs/level1_vel/`. |
| `level1_vel_planar` | `depthnav_vel_planar` | `level1_vel_planar.pth` + `.onnx` + `.tflite` | As `level1_vel` but vz forced ≡ 0 (diffaero-pmv_planar analog). Run superfly-train-depthnav-9, harvested iter 9000 (sr 0.93–0.96). **Benchmark 6/6.** Deploy MUST pass `--policy-cfg small_yaw_vel_planar.yaml` (the vz head is untrained). Full final: `s3://superfly/runs/level1_vel_planar/`. |

Each vel run dir also carries the merged training config (`.yaml`) and the
training-repo eval curve (`.csv`).

### DiffAero (taichi-sim-trained; deployable artifact is `checkpoints/exported_actor.pt2` in each run)

| run | registry method | what it is |
|---|---|---|
| `sha2c_pmc` | `diffaero` | Legacy thrust/attitude policy (pmc dynamics, default plant). The baseline "diffaero" method. |
| `sha2c_pmc_lag_2026-06-22` | — | Legacy pmc variant with first-order velocity lag (pmclag). Kept for reference; not in the registry. |
| `sha2c_pmc_starling2max_v1` | — | pmc with measured starling motor lag (lmbda 13.9, tau 72 ms), 3–6 m/s, sr 0.78 (OSMO train-18). Starling spec-matching experiment; not yet a registry method. |
| `sha2c_vel_cmd` | — | Legacy velocity-command actor, env=pc (no depth). Superseded by `_oa`. |
| `sha2c_vel_cmd_oa` | `diffaero_vel` | Legacy velocity-command actor consuming 9×16 depth. |
| `pmv_planar_starling_v1` | `diffaero_vel_planar` | **The deployed planar low-speed policy** (0.8–1.5 m/s, sr 0.92): flew `planar_lowvel_prims_v1` 6/6 zero-collision (eval planar5). |
| `pmv_planar_starling_v2` | — | v1 recipe + r_drone 0.3 / n_obstacles 40, meant to buy clearance margin — **REGRESSED at deploy** (1/6 vs v1's pass; ATTEMPTS 2026-08-18). Kept for reference — do not redeploy. Carries the ONNX + TFLite export (`exported_actor.tflite`, verified 1.9e-06). |

### AgileAutonomy

| run | registry method | what it is |
|---|---|---|
| `ckpt-50` | `agile` | Legacy TF2 checkpoint prefix (`.index` + `.data-*`, no pointer file) for the Keras re-implementation in `superfly.policies.agile`. `scripts/train_agile.py` (Phase 3) will produce successors. |

Deleted, deliberately: `DiffPhysDrone/` (method removed from the harness)
and the three `DiffAero/planar_*` symlinks (targets never committed,
unrecoverable — see ATTEMPTS.md 2026-08-17).
