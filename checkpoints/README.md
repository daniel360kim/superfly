# checkpoints/ — committed policy weights

One folder per training run, `<Method>/<run>/`. Weights are committed on
purpose: a fresh clone flies with no credentials. Full provenance for any
run (training command, metrics, former name) is in its `run_meta.json`.

Naming: `thrust_*` policies command attitude/thrust, `vel_*` command PX4
velocity setpoints, `vel_planar_*` are velocity with vz fixed to 0.
`starling` = trained for the Starling 2 Max at 0.8–1.5 m/s. A `.tflite` /
`.onnx` next to a `.pth` is a verified export of that same policy.

## DepthNav

| run | what it is |
|---|---|
| `thrust_level1_4` | Legacy thrust policy, 2–4 m/s. The baseline `depthnav` method. |
| `vel_starling_v1` | Velocity commands, 0.8–1.5 m/s. Benchmarked 6/6, no collisions. → `depthnav_vel` |
| `vel_planar_starling_v1` | Same but horizontal-only (vz=0). Benchmarked 6/6, no collisions. → `depthnav_vel_planar` |

## DiffAero

| run | what it is |
|---|---|
| `thrust_pmc` | Legacy thrust policy. The baseline `diffaero` method. |
| `thrust_pmc_lag` | Old experiment (motor-lag dynamics). Reference only. |
| `thrust_pmc_starling_v1` | Thrust policy with measured starling motor lag, 3–6 m/s. |
| `vel_nodepth` | Old velocity policy, no depth input. Superseded by `vel_depth`. |
| `vel_depth` | Velocity commands with depth input. → `diffaero_vel` |
| `vel_planar_starling_v1` | Horizontal-only velocity, 0.8–1.5 m/s. Benchmarked 6/6. → `diffaero_vel_planar` |
| `vel_planar_starling_v2` | Retrain of v1 that flew worse. **Do not deploy** — kept for reference. |

## AgileAutonomy

| run | what it is |
|---|---|
| `ckpt-50` | Legacy TF2 checkpoint for the `agile` method (dir name = TF prefix). |

`→ name` is the method entry in `superfly.compare.registry` that flies the
run. Runs without an arrow aren't wired into the comparison harness.
