# INFRASTRUCTURE — where superfly runs, stores, and authenticates

Umbrella-level rules live in `anyanything/MACHINES.md` (gs2/airstation03
roles, the `airstation` dispatch helper) and
`anyanything/INFRASTRUCTURE_OSMO.md` (pool rules, quotas, deps cache,
credentials). This file is only what's superfly-specific.

## Compute

| what | where | how |
|---|---|---|
| edit / commit / dry-run / re-score | `gs2` | plain python + numpy |
| comparison flights (all methods, esp. agile) | `airstation03` | `<isaacsim>/python.sh` + `scripts/run_comparison.py` |
| comparison campaigns (depthnav, diffaero) | OSMO | `osmo/superfly-eval.yaml` |
| training | OSMO (`osmo/superfly-train-*.yaml`) or airstation03 (`scripts/airstation_train.sh`) | never `gs2` |
| agile expert labels | `airstation03` (needs Docker) | `superfly_expert_sampler` (Jason's repo, branch `expert-sampler`) |

Agile on OSMO is **not wired**: it needs TF + casadi + an in-container acados
build that has never been attempted. Run agile trials on airstation03.

airstation03 cautions (from `MACHINES.md`, they bite here): never edit code
there; its root disk sits ~97% full and is shared by ~27 accounts — the
expert labeller writes under `~/.cache/superfly_expert_sampler` and can fill
it; repos live flat in `~` (`~/superfly`, `~/superfly-depthnav_vel`, ...).

## Venv layout (per machine)

- repo root `.venv` — the **agile** venv (TF-cpu, casadi, acados_template,
  pymavlink) reached through `scripts/agile_python.sh`, which exports
  `ACADOS_SOURCE_DIR`/`LD_LIBRARY_PATH` before exec'ing python (acados'
  generated `.so` needs them at interpreter start).
- `methods/depthnav/.venv`, `methods/diffaero/.venv` — the method venvs the
  registry resolves interpreters from.
- Every venv: `pip install -e <repo> --no-deps` so `superfly.*` imports
  without PYTHONPATH/cwd tricks (the `scripts/` entrypoints also fall back
  to inserting `../src` themselves).

## Storage / S3

- Container: **`s3://superfly`** (path-style, endpoint
  `https://airlab-cloud.andrew.cmu.edu:8080`; vhost-style DNS does not
  resolve). Prefixes: `tmp_data/` staging tarballs, `deps/` the per-project
  OSMO deps cache, `runs/<tag>/` results + checkpoint mirrors.
- Client: `python -m superfly.remote_store {upload,download,list}`.
- Credentials: `SUPERFLY_S3_KEY_ID`/`SUPERFLY_S3_KEY` env, falling back to
  the `GSDS_*` pair in `~/.s3env` — the same Keystone EC2 credential covers
  every bucket in the account.
- One-time setup **done 2026-08-17** on gs2: bucket created
  (`python -m superfly.remote_store create-bucket`), OSMO DATA credential
  `superfly` registered for `s3://superfly`
  (`osmo credential set superfly --type DATA --payload access_key_id=...
  access_key=... endpoint=s3://superfly override_url=... region=us-east-1`
  — note the server rejects an `addressing_style` payload key), and
  `SUPERFLY_S3_*` aliases appended to `~/.s3env`. Both superfly workflow
  YAMLs pass `osmo workflow validate --pool default`.

## Checkpoints

Committed in git, `checkpoints/<Method>/<run>/` + `run_meta.json` — layout,
per-method artifact kinds, and the one known gap (depthnav_vel's
never-produced `.pth`):
[`checkpoints/README.md`](checkpoints/README.md). The repo carries ~68 MB of
weights; each new run adds permanently — prune superseded runs, never
rewrite history.

## PX4 / Isaac (local runs)

- PX4 checkout: `--px4-dir` / `$PX4_DIR` (default `~/PX4-Autopilot`), built
  once with `make px4_sitl none_iris`. The harness launches the bare `px4`
  binary per trial (clean EKF/arming/battery state) with a sanitized
  environment (Isaac's `LD_LIBRARY_PATH` breaks cmake/OpenSSL in children).
- Isaac interpreter: `--sim-python` / `$ISAACSIM_PYTHON`.
- Pegasus: `$PYTHONPATH` must include
  `PegasusSimulator/extensions/pegasus.simulator` for the sim process.

## OSMO specifics

`osmo/superfly-eval.yaml` header comments are the authoritative record of
every measured workaround (carb sitecustomize, dataWindowNDC, non-root
`flyer` user, libGL, the *rejected* asyncRendering=false — throughput is a
correctness property in a non-lockstep sim). Do not simplify them away.
The eval image needs no apt deps cache (~25 s installs); a training image
likely does (`deps/deps_cache_v1.tar`, see the umbrella doc's 58-minute
apt-tax measurement).
