# checkpoints/ — committed policy weights

One folder per training run, `<Method>/<run>/`. Weights are committed on
purpose: a fresh clone flies with no credentials. Full provenance for any
run (training command, metrics, former name) is in its `run_meta.json`.

Naming: `thrust_*` policies command attitude/thrust, `vel_*` command PX4
velocity setpoints, `vel_planar_*` are velocity with vz fixed to 0.
`vel_yawrate_*` command forward velocity, vertical velocity and yaw rate (no
sideways velocity, so the camera always faces the direction of travel).
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
| `vel_yawrate_s_starling_v1` | Forward/vertical velocity + yaw rate, Starling PX4-fitted dynamics, 0.8–1.5 m/s. Isaac tier C 39/53. → `diffaero_vel_yawrate_s` |
| `vel_yawrate_m_starling_v1` | Same, 0.5–2.0 m/s. Isaac tier C 38/53. → `diffaero_vel_yawrate_m` |
| `vel_yawrate_f_starling_v1` | Same, 2–5 m/s (cruises ~2.4 m/s in practice). Isaac tier C 34/53. → `diffaero_vel_yawrate_f` |

## AgileAutonomy

| run | what it is |
|---|---|
| `ckpt-50` | Legacy TF2 checkpoint for the `agile` method (dir name = TF prefix). |

`→ name` is the method entry in `superfly.compare.registry` that flies the
run. Runs without an arrow aren't wired into the comparison harness.

## Student

The `anyanything` end-to-end student (agile_autonomy `planner_learning` with
the `ours:` block), exported to ONNX. I/O contract:
`~/anyanything/agile_student/INPUTS.md` — `imu` (1,1,22), `depth`
(1,1,224,224,3), output (1, M, 1+3N) = per mode `[alpha, x_1..N, y_1..N,
z_1..N]` in **absolute body-frame metres** at t = 0.5 j s. M and N are read
off the graph, so all three run here fly through the same code path.

| run | what it is |
|---|---|
| `t5fix_s_r1` | 3 modes x 10 waypoints (5 s horizon). The deployed `agile_student`. → `agile_student` |
| `t5_m2_r2` | Two-mode ablation, 10 waypoints. Reference only. |
| `t5_h25_local` | 2.5 s horizon ablation (3 modes x 5 waypoints). Reference only. |

Fly a non-default one with `--agile_student-checkpoint
checkpoints/Student/<run>/student.onnx`.
