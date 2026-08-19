# methods/ — the three baseline repos (git submodules, forks only)

Added 2026-08-17. `git clone --recursive` (or `git submodule update --init`)
brings them in; each builds its own venv at `methods/<name>/.venv`, which is
where the comparison registry resolves interpreters from. After building a
venv, also `pip install -e ../.. --no-deps` in it so `superfly.*` imports.

| submodule | fork | upstream | notes | venv |
|---|---|---|---|---|
| `depthnav/` | `daniel360kim/depthnav` | (Habitat-sim depth-nav trainer) | training entries `examples/navigation/run_nav_level1.py` (thrust) and `run_nav_level1_vel.py` (velocity variant, added 2026-08-19 @4f845f5 along with `VELOCITY_YAW`/`velocity_world_frame` support and `small_yaw_vel.yaml`); deploy configs `examples/navigation/policy_cfg/small_yaw{,_vel}.yaml` — `small_yaw` matches the committed `level1_4` checkpoint, `small_yaw_vel` the `level1_vel` run (see ATTEMPTS.md). Training needs `datasets/get_dataset.sh` (4.6 GB, mirrored at `s3://superfly/deps/depthnav_dataset_v1.tar`) and a habitat-sim source build. | `methods/depthnav/.venv` (torch, habitat-sim, gymnasium; `pip install -r requirements.txt`) |
| `diffaero/` | `daniel360kim/diffaero` | (Taichi GPU sim, Hydra) | `script/train.py` writes `<run>/checkpoints/actor.pth`; `script/export.py checkpoint=<run>/checkpoints` emits `exported_actor.pt2` beside it — wrapped by `scripts/train_diffaero.py` (interface verified against source) | `methods/diffaero/.venv` (`pip install -r requirements.txt`; torch cu128 on sm_120 boxes) |
| `agile_autonomy/` | `daniel360kim/agile_autonomy` | `uzh-rpg/agile_autonomy` | **includes `planner_learning/`** (the training half the old local copy lacked: `dagger_training.py`, `config/*.yaml`, `models/`). Deploy side never imports this repo — `superfly.policies.agile` is a self-contained Keras re-implementation | trains via `scripts/train_agile.py` (Phase 3); ROS/data_generation half not used (labels come from `superfly_expert_sampler`) |

Fork policy: any deploy/training patches go on the fork (commit + push +
bump the submodule pointer here), never in a loose checkout — that is how
the originals were lost (ATTEMPTS.md 2026-08-17).

Related but NOT a method: `mapnav` (`JamesEmi/mapnav`) was a submodule on
the old `main` branch only; it is not part of the three-method panel and is
not restored here.
