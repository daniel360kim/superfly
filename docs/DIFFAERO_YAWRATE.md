# DiffAero vx / vz / yaw-rate policies (`pmv_yawrate`)

DiffAero velocity policies whose action is **forward velocity, up velocity and
yaw rate**. There is no lateral velocity, so the forward depth camera always
faces the commanded motion. To dodge sideways, the policy has to turn.

Three policies, one recipe each (`scripts/train_diffaero_yawrate.sh {s|m|f}`):

| band | cruise (env target vel) | vx clamp (train randomization) | vz clamp | yaw-rate clamp | max_time |
|---|---|---|---|---|---|
| `s` | 0.8–1.5 m/s (Starling low-speed) | 2.0 (1.5–2.5), reverse 0.5 | 1.0 (0.8–1.2) | 60 °/s (45–80) | 60 s |
| `m` | 0.5–2.0 m/s | 2.5 (2.0–3.0), reverse 0.5 | 1.0 (0.8–1.2) | 75 °/s (60–90) | 80 s |
| `f` | 2.0–5.0 m/s | 6.0 (5.0–7.0), reverse 0.5 | 2.0 (1.5–2.5) | 90 °/s (70–120) | 40 s |

The other settings follow `vel_planar_starling_v1`, the deployed planar policy:
sha2c, MLP on the 9×16 depth image, 2000 updates, r_drone 0.2, 30 obstacles.
v1 used the defaults because v2's r_drone 0.3 / 40 obstacles regressed at
deploy (registry note on `diffaero_vel_planar`). max_time is raised for the
slow bands because starts are ≥ 34 m from the goal (the arena is 50 × 50 m).

## Model (fork `daniel360kim/diffaero`, branch `vxvz-yawrate`)

`dynamics=pmv_yawrate` is `VelocityPointMassModel` with
`action_space: vx_vz_yawrate`:

- action `[vx, vz, r_cmd]`, yaw-local; vx ∈ [−`reverse_vel_x`, `max_vel.x`]
- state `[p, v, yaw, r]`; `r ← lerp(r, r_cmd, 1−e^(−lmbda_yaw·dt))`, then
  yaw integrates r (trapezoidal). Yaw keeps its gradient, so the velocity
  and position losses reach the yaw-rate action.
- `v_cmd = Rz(yaw_k) [vx, 0, vz]` (heading at command time, as the bridge
  does with the measured yaw), then the existing first-order velocity lag
  in the world frame (`lmbda`)
- observation `[target_vel_local(3), v_local(3), r]`, 7-dim
- loss: the pmv loss, plus `loss_weights.pointmass.yaw_rate · r_cmd²` (0.05)
- episodes start facing the goal ± `init_yaw_jitter_deg` (30°), so the
  policy learns to turn toward the goal
- export: `vel_yawrate_cmd = [vx_w, vy_w, vz_w, r]`, with the local velocity
  already rotated by the `Rz` input

The lags (`lmbda` 1.2, `lmbda_yaw` 6.0, each randomized) are placeholders
until they are fitted from ulogs (below).

## Deploy

`DiffAeroVelPolicy` reads `dynamics.action_space` from the run's hydra
config. In yaw-rate mode it:

1. appends the measured world yaw rate to the state (`DiffAeroObs.yaw_rate_enu`)
2. clamps the output in the yaw-local frame, with lateral forced to 0
3. applies the same software lags the training model has (velocity: `lmbda`;
   yaw rate: `lmbda_yaw`)

`scripts/diffaero_vel_offboard.py` sends the result as velocity + yaw-rate
setpoints (`send_velocity_yawrate_target_ned`: the yaw field is masked and
the yaw rate is negated, ENU→NED). The climb/yaw/landing phases are
unchanged. Planar and 3-D checkpoints behave exactly as before.

## Train

```bash
# once: overlay venv on airstation03 (reuses venv_torch, no second torch)
airstation run --no-sync superfly -- bash scripts/airstation_diffaero_venv.sh
# per band; extra hydra overrides go after the band
airstation run --no-sync superfly -- bash scripts/train_diffaero_yawrate.sh s
```

Output: `checkpoints/DiffAero/vel_yawrate_<band>_<stamp>/` with
`exported_actor.pt2` + `run_meta.json`. `airstation sync superfly` pushes
gs2's main checkout (`triage`), but airstation03 is on `reorg`. So commit in
the `superfly_reorg` worktree, `git push origin reorg`, and `git pull` there.

## Sensor noise from PX4 ulogs (next step)

`env.obs_noise` (in the fork's `cfg/env/oa.yaml`, off by default) corrupts
only what the policy sees. The dynamics, losses and critic state stay clean.
Each field, and where its value should come from:

| field | meaning | ulog source |
|---|---|---|
| `vel_std` | white noise on velocity, per axis | `vehicle_local_position.{vx,vy,vz}`: residual vs a zero-phase low-pass, hover + cruise segments |
| `vel_bias_std` | per-episode velocity bias | needs ground truth (mocap / `vehicle_visual_odometry`); otherwise keep the prior |
| `yaw_rate_std` | white noise on yaw rate | `vehicle_angular_velocity.xyz[2]`: residual vs a low-pass |
| `pos_std`, `pos_bias_std` | position-estimate error (enters through the goal vector) | `vehicle_local_position.{x,y,z}` residual; drift vs ground truth if available |
| `depth_std_rel`, `depth_dropout` | depth range noise, holes | **not in ulogs**: needs recorded depth frames of known geometry |

The same logs also fit the dynamics placeholders: `lmbda` from
`trajectory_setpoint.velocity` → `vehicle_local_position.v*`, and `lmbda_yaw`
from `trajectory_setpoint.yawspeed` → `vehicle_angular_velocity.xyz[2]`
(first-order fit, plus the command latency).

Planned tool: `scripts/ulog_noise_fit.py` (pyulog) writes a hydra override
file per vehicle. Retraining is then the band recipe plus
`env.obs_noise.enabled=true` and the fitted values. Action latency is not
modelled yet. If the fit shows a delay that matters, add a k-step action
buffer to the dynamics.
