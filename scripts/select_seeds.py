#!/usr/bin/env python3
"""Pick obstacle-field seeds whose spawn/goal are genuinely clear, and whose
corridor is genuinely blocked.

`obstacle_field.generate*` places the start and goal at FIXED offsets
(diffphys: start (-1.5*scale, -3.0), goal (8*scale, 3.0)) and never checks them
against the obstacles it just sampled. A seed can therefore put a tree on the
goal, which scores as a collision no policy could have avoided -- or leave the
straight line start->goal completely open, which scores as a success no policy
had to earn. Both are silent; both poison a comparison table.

This script rejects those seeds BEFORE anything flies, using metrics.py's own
SDFs so the selection can never disagree with the scorer.

Three gates per (generator, seed, scale):

  1. start column clear   -- min clearance over the vertical climb column at the
                             spawn XY, z in [z_field, climb_alt], >= MARGIN
  2. goal column clear    -- same at the goal XY. run_px4_sim protects the SPAWN
                             from oversized assets (GSDS_ASSET_SPAWN_CLEAR) but
                             NOT the goal, so this gate is what covers the goal.
  3. corridor blocked     -- the straight line start->goal at cruise altitude
                             must not be SAFELY flyable, i.e. its min clearance
                             must be < TRIVIAL_M. Otherwise a blind go-straight
                             controller solves the cell and it measures nothing.

                             Note the threshold is body-aware, not zero: the
                             drone is a 0.2 m sphere, so a corridor with 0.24 m
                             of clearance is arithmetically "open" and
                             physically impassable. Scoring it as open
                             understates the suite badly (64% vs 7% of seeds).

Usage:
    python3 compare/select_seeds.py                       # survey + pick
    python3 compare/select_seeds.py --n-seeds 200 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from superfly.sim import obstacle_field                            # noqa: E402
from superfly.compare.metrics import clearance_along_traj          # noqa: E402

#: Clearance the drone body needs at an endpoint. The drone is modelled with a
#: 0.2 m radius; assets can exceed the primitive they replace, so this is
#: deliberately well above that rather than merely non-colliding.
MARGIN_M = 1.5

#: Drone collision radius used by metrics.score_trajectory.
DRONE_RADIUS_M = 0.2

#: A straight-line corridor with at least this much surface clearance is
#: comfortably flyable by a DRONE_RADIUS_M body, so the cell is trivially
#: solvable and discriminates nothing. Anything tighter counts as blocked.
TRIVIAL_M = 0.5

#: Cruise altitude. Must match the scenario's climb_alt -- the drone does not
#: fly at the field's z=1.0 endpoint altitude, it climbs first.
CLIMB_ALT_M = 5.0

GENERATORS = {
    "diffphys": obstacle_field.generate,
    "diffaero": obstacle_field.generate_diffaero,
}


def _field_dict(fld) -> dict:
    """ObstacleField -> the array dict metrics.clearance_along_traj expects."""
    def arr(seq, width):
        if not seq:
            return np.zeros((0, width), dtype=np.float64)
        return np.asarray(seq, dtype=np.float64).reshape(len(seq), -1)

    boxes = arr(fld.boxes, 6)
    return {
        "spheres": arr(fld.spheres, 4),
        "boxes": boxes,
        "cyl_v": arr(fld.cyl_v, 3),
        "cyl_h": arr(fld.cyl_h, 4),
    }


def _column(xy, z_lo, z_hi, n=40) -> np.ndarray:
    """Vertical sample column at an XY position (the climb-out / descent path)."""
    z = np.linspace(z_lo, z_hi, n)
    return np.stack([np.full(n, xy[0]), np.full(n, xy[1]), z], axis=1)


def goal_for(scale: float) -> np.ndarray:
    """The goal the scenarios actually fly, which is NOT the generator's
    `p_target`.

    Scenario files carry an explicit `"goal": [-3.0, 50.0, 2.0]`, which for
    scale 5 is 10 m beyond the far edge of the field: after the heading-90
    rotation an obstacle's y is its pre-scale x (in [0, 8]) times `scale`, so
    the field ends at y = 8*scale = 40 and the goal sits at 50.

    That constant is scale-dependent and was hardcoded. At scale 6.5 the field
    reaches y = 52, so a goal at y = 50 lands INSIDE the obstacle region --
    the existing dense scenarios (diffaero_dense_s10/s11) have this bug. Keep
    the 10 m standoff instead of the literal 50.
    """
    return np.array([-3.0, 8.0 * scale + 10.0, 2.0])


def evaluate(gen_name: str, seed: int, scale: float,
             climb_alt: float = CLIMB_ALT_M) -> dict:
    fld = GENERATORS[gen_name](seed=seed, scale=scale)
    field = _field_dict(fld)
    start, goal = np.asarray(fld.p_init), goal_for(scale)

    z_lo = min(float(start[2]), float(goal[2]))
    start_col = clearance_along_traj(_column(start[:2], z_lo, climb_alt), field).min()
    goal_col = clearance_along_traj(_column(goal[:2], z_lo, climb_alt), field).min()

    # Straight line at cruise altitude: the blind-controller probe.
    t = np.linspace(0.0, 1.0, 400)[:, None]
    line = start[None, :] * (1 - t) + goal[None, :] * t
    line[:, 2] = climb_alt
    line_min = clearance_along_traj(line, field).min()

    dist = float(np.linalg.norm(goal[:2] - start[:2]))
    return {
        "generator": gen_name, "seed": seed, "scale": scale,
        "start_clear_m": float(start_col),
        "goal_clear_m": float(goal_col),
        "corridor_min_m": float(line_min),
        "course_len_m": dist,
        "start_ok": bool(start_col >= MARGIN_M),
        "goal_ok": bool(goal_col >= MARGIN_M),
        "blocked_ok": bool(line_min < TRIVIAL_M),
    }


def passes(r: dict) -> bool:
    return r["start_ok"] and r["goal_ok"] and r["blocked_ok"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--scales", type=float, nargs="+", default=[5.0, 6.5])
    ap.add_argument("--climb-alt", type=float, default=CLIMB_ALT_M)
    ap.add_argument("--pick", type=int, default=3,
                    help="Seeds to pick per (generator, scale).")
    ap.add_argument("--json", type=Path, help="Write full survey + picks here.")
    args = ap.parse_args()

    rows = [evaluate(g, s, sc, args.climb_alt)
            for g in GENERATORS for sc in args.scales
            for s in range(args.n_seeds)]

    print(f"surveyed {len(rows)} (generator, seed, scale) combinations, "
          f"margin {MARGIN_M} m, climb_alt {args.climb_alt} m\n")

    picks = {}
    for g in GENERATORS:
        for sc in args.scales:
            sub = [r for r in rows if r["generator"] == g and r["scale"] == sc]
            ok = [r for r in sub if passes(r)]
            bad_start = sum(1 for r in sub if not r["start_ok"])
            bad_goal = sum(1 for r in sub if not r["goal_ok"])
            open_line = sum(1 for r in sub if not r["blocked_ok"])
            print(f"{g:9s} scale {sc:4.1f}: {len(ok):3d}/{len(sub)} pass  "
                  f"(start-blocked {bad_start}, goal-blocked {bad_goal}, "
                  f"corridor-open {open_line})")
            # Prefer the most obstructed corridors among the valid ones: those
            # discriminate between policies instead of saturating.
            ok.sort(key=lambda r: r["corridor_min_m"])
            picks[f"{g}_s{sc}"] = ok[:args.pick]

    print("\nPicked cells:")
    for key, sel in picks.items():
        seeds = [r["seed"] for r in sel]
        print(f"  {key:16s} seeds {seeds}")
        for r in sel:
            print(f"      seed {r['seed']:3d}  start {r['start_clear_m']:6.2f}  "
                  f"goal {r['goal_clear_m']:6.2f}  corridor {r['corridor_min_m']:6.2f}  "
                  f"len {r['course_len_m']:5.1f}")

    if args.json:
        args.json.write_text(json.dumps({"margin_m": MARGIN_M,
                                         "climb_alt_m": args.climb_alt,
                                         "survey": rows, "picks": picks}, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
