#!/usr/bin/env python3
"""Mine planar start/goal pairs for USD benchmark scenes from scene_audit.py
output (runs on gs2 -- numpy/scipy/matplotlib only, no Isaac).

Input: an audit directory (one subdir per scene holding samples.npz +
report.json, fetched from airstation03 via `airstation fetch superfly
results/scene_audit`), or explicit .npz paths. For every scene that passed
the audit it builds the planar occupancy slice at z0 + climb_alt, mines
candidate pairs through the four hard gates (superfly.perception.occupancy:
endpoint columns clear / straight line blocked / flyable detour exists /
corridor passable), scores difficulty over the POOLED candidate list (so D is
comparable across scenes), and picks `--pairs-per-scene` spread over the
difficulty percentiles with non-overlapping corridors.

Outputs (under --out):
    mined_pairs.json          all survivors + selection, per scene, with stats
    <scene>_map.png           overhead map: occupancy, candidates, selection
    suite JSON (--suite-out)  ready for scripts/run_comparison.py

Example:
    python3 scripts/mine_scene_goals.py results/fetched/scene_audit \\
        --out results/mining --suite-out configs/scenarios/suites/nucleus_planar_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from superfly.perception import occupancy as oc            # noqa: E402


def load_scene(npz_path: Path):
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return (np.asarray(z["samples"], np.float32),
            np.asarray(z["ground_samples"], np.float32), meta)


def plot_scene(sl, cands, picked, name, out_png):
    fig, ax = plt.subplots(figsize=(10, 10 * sl.occ.shape[0] / max(sl.occ.shape[1], 1)))
    H, W = sl.occ.shape
    ext = [sl.origin[0], sl.origin[0] + W * sl.cell,
           sl.origin[1], sl.origin[1] + H * sl.cell]
    img = np.full(sl.occ.shape, 1.0)
    img[~sl.known] = 0.85
    img[sl.occ] = 0.0
    ax.imshow(img, cmap="gray", origin="lower", extent=ext, vmin=0, vmax=1,
              interpolation="nearest")
    for c in cands:
        if any(c is p for p in picked):
            continue
        xy = np.array([sl.cell_to_world(p) for p in c.path_cells])
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", lw=0.6, alpha=0.25)
    colors = ["tab:green", "tab:orange", "tab:red", "tab:purple"]
    for i, c in enumerate(picked):
        col = colors[i % len(colors)]
        xy = np.array([sl.cell_to_world(p) for p in c.path_cells])
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=2.0,
                label=f"p{i}: D={c.difficulty:.2f} detour={c.detour:.2f} "
                      f"len={c.path_m:.0f}m gap={2 * c.path_min_edt_m:.1f}m")
        ax.plot(*c.start_xy, marker="o", color=col, ms=9, mec="k")
        ax.plot(*c.goal_xy, marker="*", color=col, ms=14, mec="k")
        ax.plot([c.start_xy[0], c.goal_xy[0]], [c.start_xy[1], c.goal_xy[1]],
                color=col, lw=0.8, ls="--", alpha=0.6)
    ax.set_title(f"{name} -- z_fly={sl.z_fly:.1f} m (ground {sl.z0:.1f} m), "
                 f"{len(cands)} candidates")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit", help="scene_audit output dir (subdir per scene), "
                                  "or a single samples .npz")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--suite-out", default=None,
                    help="also write a run_comparison scenario suite JSON here "
                         "(the curated --pairs-per-scene spread)")
    ap.add_argument("--suite-all-out", default=None,
                    help="also write EVERY gate-surviving pair as a scenario "
                         "suite (dense harvest; curate per experiment)")
    ap.add_argument("--goals-per-source", type=int, default=60,
                    help="goal candidates evaluated per Dijkstra source")
    ap.add_argument("--max-keep", type=int, default=400,
                    help="survivor cap per scene")
    ap.add_argument("--climb-alt", type=float, default=2.0,
                    help="cruise altitude above the dominant ground [m] (default 2)")
    ap.add_argument("--pairs-per-scene", type=int, default=3)
    ap.add_argument("--len-range", type=float, nargs=2, default=(25.0, 80.0),
                    metavar=("MIN", "MAX"), help="straight-line length band [m]")
    ap.add_argument("--cell", type=float, default=oc.DEFAULT_CELL_M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", type=int, default=12,
                    help="Dijkstra sources per scene (default 12)")
    ap.add_argument("--include-flagged", action="store_true",
                    help="also mine scenes whose audit verdict is FLAG (default: "
                         "PASS only; scenes without report.json are always mined)")
    args = ap.parse_args()

    audit = Path(args.audit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenes = []          # (name, npz_path, report|None)
    if audit.is_file():
        scenes.append((audit.stem, audit, None))
    else:
        for d in sorted(audit.iterdir()):
            npz = d / "samples.npz"
            if not d.is_dir() or not npz.exists():
                continue
            rep = None
            if (d / "report.json").exists():
                rep = json.loads((d / "report.json").read_text())
                verdict = rep.get("verdict", "PASS")
                if verdict == "FAIL" or (verdict == "FLAG" and not args.include_flagged):
                    print(f"[skip] {d.name}: audit verdict {verdict}")
                    continue
            scenes.append((d.name, npz, rep))
    if not scenes:
        raise SystemExit(f"no scenes found under {audit}")

    per_scene = {}       # name -> dict(sl, cands, stats, meta, report)
    for name, npz, rep in scenes:
        S, G, meta = load_scene(npz)
        print(f"[scene] {name}: {S.shape[0]} lateral / {G.shape[0]} ground samples")
        sl = oc.build_slice(S, G, climb_alt=args.climb_alt, cell=args.cell)
        if sl is None:
            print(f"[scene] {name}: no usable ground -- skipped")
            continue
        cands, stats = oc.mine_pairs(sl, n_sources=args.sources, seed=args.seed,
                                     len_range=tuple(args.len_range),
                                     goals_per_source=args.goals_per_source,
                                     max_keep=args.max_keep)
        per_scene[name] = dict(sl=sl, cands=cands, stats=stats, meta=meta, report=rep)

    # pool-normalized difficulty, then per-scene selection
    pool = [c for v in per_scene.values() for c in v["cands"]]
    ranges = oc.score_difficulty(pool)
    print(f"[difficulty] pool={len(pool)} ranges={ranges}")

    def scenario_entry(scen_name, c, usd, scale, sl):
        return {
            "name": scen_name,
            "usd_environment": usd,
            "env_scale": scale,
            "start": [c.start_xy[0], c.start_xy[1], round(sl.z0 + 0.3, 2)],
            "goal": [c.goal_xy[0], c.goal_xy[1], round(sl.z_fly, 2)],
            "climb_alt": args.climb_alt,
            "timeout": int(max(240, 60 + 2 * c.path_m)),
            "pre_policy_timeout": 600,
            "difficulty": round(c.difficulty, 3),
        }

    result, suite, suite_all = {}, [], []
    for name, v in per_scene.items():
        picked = oc.select_pairs(v["cands"], k=args.pairs_per_scene)
        plot_scene(v["sl"], v["cands"], picked, name, out / f"{name}_map.png")
        sl = v["sl"]
        result[name] = {
            "stats": v["stats"],
            "z0": sl.z0, "z_fly": sl.z_fly, "cell": sl.cell,
            "n_candidates": len(v["cands"]),
            "selected": [c.to_json() for c in picked],
            "candidates": [c.to_json() for c in v["cands"]],
        }
        rep = v["report"] or {}
        usd = rep.get("usd") or v["meta"].get("usd")
        scale = rep.get("env_scale") or v["meta"].get("env_scale", 1.0)
        for i, c in enumerate(picked):
            suite.append(scenario_entry(f"{name}_p{i}", c, usd, scale, sl))
        for i, c in enumerate(sorted(v["cands"], key=lambda c: c.difficulty)):
            suite_all.append(scenario_entry(f"{name}_c{i:03d}", c, usd, scale, sl))

    (out / "mined_pairs.json").write_text(json.dumps({
        "params": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(args).items() if k not in ("audit", "out")},
        "difficulty_ranges": {k: list(map(float, r)) for k, r in ranges.items()},
        "scenes": result}, indent=1))
    print(f"[out] {out / 'mined_pairs.json'} + {len(per_scene)} map PNGs")
    for path, entries, label in ((args.suite_out, suite, "curated"),
                                 (args.suite_all_out, suite_all, "all-survivors")):
        if not path:
            continue
        if entries:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(entries, indent=1))
            print(f"[out] {label} suite: {path} ({len(entries)} scenarios)")
        else:
            print(f"[out] no scenarios survived -- {label} suite not written")


if __name__ == "__main__":
    main()
