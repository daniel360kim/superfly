# checkpoints/ — committed policy weights

Layout: `checkpoints/<Method>/<run>/`, one `run_meta.json` beside each run
(method, producing command, config notes; new runs written by
`scripts/train_<method>.py` also record the submodule SHA and host).

**These are committed in git on purpose** — anyone who clones gets flyable
weights with no lab credentials and no LFS. The repo carries ~68 MB of
weights today and every new run adds to it permanently; the pressure valve,
if it becomes one, is pruning superseded runs (delete the run dir in a
normal commit), never rewriting history.

| method (registry) | deployable artifact | produced by |
|---|---|---|
| `depthnav` | `DepthNav/level1_4/level1_4_iteration_13500.pth` | `scripts/train_depthnav.py` (legacy run predates it) |
| `depthnav_vel` | `DepthNav/level1_vel/level1_vel.pth` | `scripts/train_depthnav.py --vel` (OSMO superfly-train-depthnav-7, harvested at plateau iter 9500 — see run_meta.json) |
| `depthnav_vel_planar` | `DepthNav/level1_vel_planar/level1_vel_planar.pth` | `scripts/train_depthnav.py --planar` (OSMO superfly-train-depthnav-9, harvested at plateau iter 9000; vz forced to zero; deploy with `--policy-cfg small_yaw_vel_planar.yaml`) |
| `diffaero` | `DiffAero/sha2c_pmc/checkpoints/exported_actor.pt2` | `scripts/train_diffaero.py` (wraps `script/train.py` + `script/export.py`) |
| `diffaero_vel` | `DiffAero/sha2c_vel_cmd_oa/checkpoints/exported_actor.pt2` | as above |
| `agile` | `AgileAutonomy/ckpt-50/ckpt-50` (TF2 prefix: `.index` + `.data-*`, no pointer file) | `scripts/train_agile.py` |

`checkpoint_ready()` (superfly.compare.registry) gates each method on its
artifact's existence per `ckpt_kind`, so a missing artifact skips the method
with a notice instead of flying garbage. Verified 2026-08-17 on a fresh
clone: every method except `depthnav_vel` reports ckpt_ready=True.

Deleted, deliberately: `DiffPhysDrone/` (method removed from the harness)
and the three `DiffAero/planar_*` symlinks (targets never committed,
unrecoverable — see ATTEMPTS.md 2026-08-17).
