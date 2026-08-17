# ORCHESTRATOR — read this first, every session

Same operating model as `gs_drone_sim` (the convention the `anyanything`
umbrella assumes): no standing work queue — check what's been tried, check
the infrastructure, decide what to do next. Read in this order:

1. **This file** — goal, current phase, read order.
2. **[`ATTEMPTS.md`](ATTEMPTS.md)** — ledger of everything tried, with
   verdicts (`REJECTED` / `ESTABLISHED-NEGATIVE` / `SHIPPED` / `SUPERSEDED` /
   `PENDING`). **Check before trying anything** — the OSMO workarounds in
   particular encode failures that each cost a real job.
3. **[`INFRASTRUCTURE.md`](INFRASTRUCTURE.md)** — where things run (gs2 /
   airstation03 / OSMO), storage, credentials, venv layout. Read before
   touching compute or storage.
4. **[`CLAUDE.md`](CLAUDE.md)** — charter: scope and why the architecture is
   what it is.
5. [`docs/comparison_harness.md`](docs/comparison_harness.md) — the harness
   manual, when actually running comparisons.

## The goal

A fair, reproducible head-to-head of the three depth-based navigation
baselines (**depthnav**, **diffaero**, **agile_autonomy**, plus their
velocity-command variants) in one Isaac-Sim + PX4-SITL evaluation substrate —
including **training** any of them from scratch with one uniform command —
as the baseline panel for `anyanything`'s end-to-end onboard navigation goal
and `gs_drone_sim`'s policies.

## Current phase (2026-08-17, post-reorg)

The 2026-08-17 reorganization (`reorg` branch) landed: installable
`superfly` package, deduped common layer, data-driven registry, normalized
checkpoints, `superfly-*` OSMO workflows, this doc set. What remains, in
dependency order:

1. ~~Method forks~~ **DONE 2026-08-17**: submodules live under `methods/`
   (`daniel360kim/{depthnav,diffaero,agile_autonomy}`, the last WITH
   `planner_learning/`) — see `methods/README.md` for the per-fork notes.
   Method venvs still need building per machine (`methods/<name>/.venv`).
2. **S3 (BLOCKED on user, one-time)**: create the `superfly` bucket
   (`source ~/.s3env && PYTHONPATH=<repo>/src <python-with-boto3> -m
   superfly.remote_store create-bucket`), register the credential with OSMO
   for `s3://superfly`, and optionally add `SUPERFLY_S3_*` names to
   `~/.s3env`.
3. **Training pipelines (Phase 3)**: `scripts/train_diffaero.py` is written
   and its train/export interface verified against the submodule source
   (still needs one GPU run). `train_depthnav.py` (entry:
   `examples/navigation/run_nav_level1.py`; the vel variant also needs
   `policy_cfg/small_yaw_vel.yaml` recreated — it exists nowhere),
   `train_agile.py` + `build_agile_dataset.py` still to write. The agile
   labels come from `superfly_expert_sampler` on airstation03; the one
   genuine gap is depth-image rendering for its rollouts (see ATTEMPTS.md
   "agile dataset" entry — camera-match risk).
4. **Hardware validation**: one comparison trial per method on airstation03
   against the committed checkpoints — the offboard refactor is verified
   only statically on gs2 (byte-identical scoring + dry-run argv).

## Operating rules

- **No GPU work on gs2**; dispatch to OSMO (`osmo/`) or airstation03
  (`scripts/airstation_train.sh`, `anyanything/bin/airstation`).
- Git is the only sync path between boxes; artifacts are gitignored except
  `checkpoints/` (committed deliberately — see `checkpoints/README.md`).
- Every training run writes `checkpoints/<Method>/<run>/` + `run_meta.json`.
- Record verdicts in `ATTEMPTS.md` as you go; archive superseded docs under
  `docs/archive/` rather than deleting.
