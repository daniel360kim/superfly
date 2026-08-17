# methods/ — the three baseline repos (git submodules, forks only)

**Status: PENDING (Phase 1).** No submodules exist yet — the original
checkouts were lost from git entirely (see ATTEMPTS.md 2026-08-17,
"Baseline method sources lost"). The forks must be created on GitHub first
(`gh` is not installed on gs2, so via the web UI), then:

```bash
git submodule add https://github.com/daniel360kim/<fork>.git methods/<name>
```

| submodule | fork of (upstream) | why forked / what changes on the fork | venv |
|---|---|---|---|
| `depthnav/` | *(fork URL needed)* | pin the exact training/deploy revision; any deploy patches live on the fork, never in a loose checkout | `methods/depthnav/.venv` (torch, habitat-sim, gymnasium) |
| `diffaero/` | *(the fork exists — URL needed)* | as above; Hydra trainer + `script/export.py` produce the deployable `.pt2` | `methods/diffaero/.venv` (torch, taichi) |
| `agile_autonomy/` | `uzh-rpg/agile_autonomy` — fork **WITH `planner_learning/`** (the local `~/aa_build` copy on gs2 lacks the training half) | training half only: the deploy wrapper (`superfly.policies.agile`) is a self-contained Keras re-implementation that never imports this repo at runtime | trains via `scripts/train_agile.py` (Phase 3) |

Once a submodule lands, fill in its row (upstream URL, fork URL, what was
changed, exact venv build steps) — this table is the contract for anyone
re-creating a machine.

Related but NOT a method: `mapnav` (`JamesEmi/mapnav`) was a submodule on
the old `main` branch only; it is not part of the three-method panel and is
not restored here.
