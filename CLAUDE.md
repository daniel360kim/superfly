# superfly charter

> Agents: entry point is [`ORCHESTRATOR.md`](ORCHESTRATOR.md) — goal, phase,
> read order. This file is the *why*.

## Scope

The **baseline panel** for the `anyanything` umbrella (lightweight end-to-end
onboard drone navigation): three published depth-based methods —
`depthnav`, `diffaero`, `agile_autonomy` — plus the Isaac Sim + PX4 SITL
harness that trains, deploys, and scores them under identical scenarios and
metrics. `gs_drone_sim` policies are evaluated *by* this harness but live in
their own repo. Exactly three methods: DiffPhysDrone and the gsds in-repo
method wiring were removed in the 2026-08-17 reorg.

## Why it's built the way it is

- **The three methods cannot share a training simulator** (different physics,
  obs/action spaces, timesteps, conflicting pinned deps), so each keeps its
  native trainer in its own fork (`methods/<repo>`, git submodule) with its
  own venv. What they *do* share is the evaluation substrate and the deploy
  contract.
- **`src/superfly/common` is dependency-light by contract** (numpy + scipy +
  pymavlink only): it is imported inside three mutually-incompatible venvs.
  Heavy stacks live in per-method extras and the method repos. Each venv
  installs the package with `pip install -e . --no-deps`.
- **Per-method phase loops stay in `scripts/*_offboard.py`** rather than a
  shared state machine: the differences are documented and load-bearing
  (depthnav's thrust variant has no YAW/LANDING phase — which is *why* its
  `policy_reported_reached` is always False; agile requests 50 Hz stream
  rates because its MPC limit-cycles on stale state; the vel variants climb
  on velocity setpoints). Only literally-identical code was extracted.
- **Adding a method = data, not code paths**: one entry in
  `superfly.compare.registry` (with `ckpt_kind`) + one row in each of
  `POLICY_CAMERAS` / `POLICY_DEPTH_PUBLISH` (`superfly.sim.px4_sim`).
- **Checkpoints are committed in git** so `git clone --recursive` is the
  whole setup: no lab creds, no LFS (neither exists on every box). Pressure
  valve is pruning superseded runs, never history rewrites.
- **Procedural obstacle fields are method-agnostic worlds**: scenario
  `"obstacles": "diffphys"` names a *field distribution* (kept from the
  removed method because suites depend on it), not a method.
- **Sim ↔ offboard is UDP on localhost** (depth frames in metres at each
  method's native training resolution; sentinel files for lifecycle) so the
  wire format matches what a real depth-camera driver hands the Starling/
  VOXL2 — the offboard scripts deploy unchanged on hardware.

## Machine roles

gs2 edits/commits/dispatches (CPU-only); airstation03 is the only box with
Isaac + GPU + Docker together (agile eval, expert labels, hardware-ish
validation); OSMO fans out training + diffaero/depthnav eval campaigns.
Rules and the dispatch loop: `anyanything/MACHINES.md`,
`INFRASTRUCTURE.md` here, `anyanything/INFRASTRUCTURE_OSMO.md` for the pool.
