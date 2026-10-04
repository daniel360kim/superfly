# DiffAero vx / vz / yaw-rate policies (`pmv_yawrate`)

DiffAero velocity policies whose action is **forward velocity, up velocity and
yaw rate**. There is no lateral velocity, so the forward depth camera always
faces the commanded motion. To dodge sideways, the policy has to turn.

Three policies, one recipe each (`scripts/train_diffaero_yawrate.sh {s|m|f}`):

| band | cruise (env target vel) | vx clamp (train randomization) | vz clamp | yaw-rate clamp | max_time |
|---|---|---|---|---|---|
| `s` | 0.8–1.5 m/s (Starling low-speed) | 2.0 (1.5–2.5), reverse 0.5 | 1.0 (0.8–1.2) | 57.3 °/s = 1 rad/s (45–57.3) | 60 s |
| `m` | 0.5–2.0 m/s | 2.5 (2.0–3.0), reverse 0.5 | 1.0 (0.8–1.2) | same | 80 s |
| `f` | 2.0–5.0 m/s | 6.0 (5.0–7.0), reverse 0.5 | 2.0 (1.5–2.5) | same | 40 s |

All bands train on **Starling 2 Max parameters**: the PX4 velocity loop refitted
on the real Starling USD in Isaac (`plant: px4_fit`, below), yaw rate capped at
the 1 rad/s Starling feasibility target, and collision radius 0.26 m
(`configs/vehicles/starling2max.yaml` `collision_radius_m`, the prop-tip reach).
The sys-ID's rigid-body constants (mass, inertia, k_T, motor lag) are already
inside that PX4 fit; a velocity point mass has no rotor layer to put them in.
Other settings follow `vel_planar_starling_v1`: sha2c, MLP on the 9×16 depth
image, 2000 updates, 30 obstacles, plus `env.oob_terminates=true`. max_time is
raised for the slow bands because starts are ≥ 34 m from the goal (the arena is
50 × 50 m).

## Model (fork `daniel360kim/diffaero`, branch `vxvz-yawrate`)

`dynamics=pmv_yawrate` is `VelocityPointMassModel` with
`action_space: vx_vz_yawrate`:

- action `[vx, vz, r_cmd]`, yaw-local; vx ∈ [−`reverse_vel_x`, `max_vel.x`]
- state `[p, v, yaw, r]`; `r ← lerp(r, r_cmd, 1−e^(−lmbda_yaw·dt))`, then
  yaw integrates r (trapezoidal). Yaw keeps its gradient, so the velocity
  and position losses reach the yaw-rate action.
- `v_cmd = Rz(yaw_k) [vx, 0, vz]` (heading at command time, as the bridge
  does with the measured yaw), then the velocity plant in the world frame
- `plant: px4_fit` (default for `pmv_yawrate`): the command is delayed 4 env
  steps (0.13 s); its horizontal part is scaled by 0.9; then
  `a_cmd = 4 (v_sp − v)`, clamped like the sampler's `clamp_command`
  (a_xy ≤ 11.35, thrust 2–15 m/s²), reaches the vehicle through a 0.05 s
  acceleration lag at 5 substeps (150 Hz). Each constant is randomized in a band
  around the fit. Source: `superfly_expert_sampler` `wt/v8-chunk`
  `scripts/fit_px4_plant.py` (40 Isaac trials), constants `PX4_*` in its
  `sim_episode.py`. Step check: first motion at 0.13 s, 1.35 m/s flown for a
  1.5 command, t63 = 0.37 s. The yaw-rate command shares the delay, then a
  first-order lag `lmbda_yaw` 10/s. That lag is not fitted: the Isaac fit is
  velocity-only. `plant: first_order` keeps the old `lmbda` lag.
- observation `[target_vel_local(3), v_local(3), r]`, 7-dim
- loss: the pmv loss, plus `loss_weights.pointmass.yaw_rate · r_cmd²` (0.05)
  and an optional heading term `heading · (1 − cos(yaw − bearing))`, off by
  default (see the rounds below)
- `env.oob_terminates`: leaving the arena ends the episode with the collision
  penalty instead of a truncation the critic bootstraps for free
- episodes start facing the goal ± `init_yaw_jitter_deg` (30°), so the
  policy learns to turn toward the goal
- export: `vel_yawrate_cmd = [vx_w, vy_w, vz_w, r]`, with the local velocity
  already rotated by the `Rz` input

## Training rounds (2026-10-04, 5090, 2000 updates each)

Evaluated in the training env: 512 episodes, deterministic actor.

| round | change | s | m | f |
|---|---|---|---|---|
| 1 | first-order plant 1.2/s, yaw 60/75/90 °/s | 0.22 | 0.33 | 0.45 |
| 2 | + heading loss 1.0, oob_terminates | 0.20 | 0.22 | 0.44 |
| 3 | Starling plant, yaw 1 rad/s, r 0.26, heading 0.25 | 0.31 | 0.39 | 0.75 |
| 3b | as 3, heading 0 | **0.51** | (training) | (training) |

Success rates. For reference, the planar v1 recipe trains to 0.92 and the 3-D
holonomic `vel_depth` to 0.85. What the eval breakdown showed:

- Round 1: yaw-rate commands averaged 4–7 °/s with 20–60° heading error. Every
  out-of-bounds exit (12–24 %) went through the 6.25 m ceiling, mid-flight:
  without lateral velocity, climbing is the cheap dodge.
- Round 2: heading 1.0 locks the nose on the goal (error 2–8°, yaw rate
  0.03 rad/s). 169 of the slow band's 215 terminations were still ceiling exits.
- Round 3: the Starling plant turns a heading change into velocity about 3× sooner
  than 1.2/s did, and success rises in every band (f most: 0.44 → 0.75). The
  heading term still hurts the slow band (0.31 vs 0.51 without it), so its
  default is now 0.
- Remaining failure mode: timeouts (35–55 % in s/m). The policy stalls in front
  of an obstacle, backing up (reverse 29 % of timeout ticks), at final distances
  around 21 m.

## Deploy

`DiffAeroVelPolicy` reads `dynamics.action_space` from the run's hydra
config. In yaw-rate mode it:

1. appends the measured world yaw rate to the state (`DiffAeroObs.yaw_rate_enu`)
2. clamps the output in the yaw-local frame, with lateral forced to 0
3. sends the setpoints unfiltered for `plant: px4_fit` checkpoints, because the
   model already is PX4's response. A `first_order` checkpoint gets the software
   lags its model assumes (`lmbda`, `lmbda_yaw`).

`scripts/diffaero_vel_offboard.py` sends the result as velocity + yaw-rate
setpoints (`send_velocity_yawrate_target_ned`: the yaw field is masked and
the yaw rate is negated, ENU→NED). The climb/yaw/landing phases are
unchanged. Planar and 3-D checkpoints behave exactly as before.

In the Isaac harness the rows are `diffaero_vel_yawrate_{s,m,f}`. With
`DIFFAERO_VEL_EXTRA_ARGS="--clock px4 --policy-timeout <s>"` the offboard runs
on sim time and writes `diffaero_cmds.npz`: per tick, the state, the depth
input, the action and the setpoint.

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
