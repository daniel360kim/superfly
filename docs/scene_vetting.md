# Scene vetting pipeline: Nucleus library → benchmark suite

Automates growing the benchmark scene library: sweep the Nucleus stage
library, vet every scene (composes / scale / ground / colliders / lighting /
depth), mine planar start–goal pairs with a documented difficulty score, and
emit a `run_comparison.py` suite — with a human review checkpoint before
anything flies.

```
gs2                                airstation03 (Isaac python)
---                                ---------------------------
tests/test_scene_metrics_*  ─┐
tests/test_occupancy.py      ├ 1. offline verification (no GPU)
                             ┘
                                   2. scripts/scene_audit.py --out results/scene_audit
                                      (inventory + per-scene audit + samples.npz)
airstation fetch superfly results/scene_audit
3. scripts/mine_scene_goals.py <fetched> --out ... --suite-out configs/scenarios/suites/...
4. review the report/maps, trim the suite
                                   5. run_comparison.py <suite> --sim-python ...
```

## 1. Offline verification (gs2, `pip install usd-core matplotlib` in `.venv`)

`python -m unittest discover -s tests -t .`

- `test_scene_metrics_synthetic.py` — authors a synthetic USD (cm units,
  nested xforms, instanceable reference, PointInstancer) and asserts
  `mesh_sampling` → extract-format npz → `metrics.score_trajectory` produce
  hand-computed clearances/collisions (gap pass 0.8 m, wall hit, ground skim).
- `test_occupancy.py` — the mining gates on synthetic grids (wall-with-gap
  keeps detour pairs; open field / sealed wall / canopy are rejected).

## 2. Scene audit (airstation03)

```bash
airstation sync superfly
airstation run superfly -- ~/isaacsim/python.sh scripts/scene_audit.py \
    --out results/scene_audit --resume [--isaac-environments] [--only 'Pat*']
```

One Kit boot for the whole catalog; `--resume` skips scenes already done, so
rerun after any crash. `--list-only` prints the catalog without auditing.
Per-scene verdict PASS / FLAG(reasons) / FAIL(reasons) in `report.json`;
what each check means is in `scene_audit.py`'s docstring. The collider drop
test runs the *same* `scene_setup.add_colliders` the flight harness uses —
a scene that passes it will not repeat the probe3 fall-through.

Nucleus auth: interactive login is cached on airstation03. For headless
boxes/OSMO, `OMNI_API_TOKEN` (Omniverse Navigator API token) is honored, but
omniverse:// on OSMO is ESTABLISHED-NEGATIVE until a probe proves the token
path (ATTEMPTS 2026-07).

## 3. Mining (gs2, no Isaac)

```bash
airstation fetch superfly results/scene_audit    # → scratch path
python3 scripts/mine_scene_goals.py <fetched>/scene_audit --out results/mining \
    --suite-out configs/scenarios/suites/nucleus_planar_v1.json
```

Gates per pair (superfly/perception/occupancy.py; select_seeds.py semantics,
scene edition): start/goal columns clear (≥1.5 m + climb column unobstructed),
straight line NOT safely flyable (min clearance < 0.5 m), a flyable detour
exists (Dijkstra on the inflated grid), corridor min passage ≥ 0.6 m EDT.
Pairs are planar: same dominant ground level, constant cruise altitude
`z0 + climb_alt`.

Difficulty `D ∈ [0,1]`, pool-normalized across all scenes (comparable):
`D = 0.4·detour + 0.3·blockage + 0.3·tightness` (components stored per pair
in `mined_pairs.json`). Selection picks `--pairs-per-scene` spread over the
25th–75th difficulty percentiles with non-overlapping corridors.

Review artifacts: `<scene>_map.png` (occupancy, candidate corridors,
selected pairs + D), `mined_pairs.json`.

## Gotchas

- Audit outputs live under `results/` (gitignored). Sizes are capped
  (`--max-samples`); watch `airstation status` disk regardless.
- `scene_setup.py` is the single source of truth for lighting + colliders —
  `run_px4_sim.py` and `scene_audit.py` both call it. Change it once, both
  change.
- Suite `start` z is spawn height (`z0 + 0.3`, drops to ground at sim start);
  `goal` z is cruise altitude — matches `construction.json` conventions.
- The audit's `samples.npz` (h≈0.2 m) is for mining only; fly-time clearance
  scoring still extracts at 5 cm via `run_comparison.py`'s `extract_scene_mesh`
  path, unchanged.
