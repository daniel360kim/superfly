"""Synthetic-grid tests for superfly.perception.occupancy (numpy/scipy only).

The scene is built directly as sample clouds (the extract_scene_mesh npz
contract: `samples` = lateral surfaces, `ground_samples` = ground faces):
a 60x40 m ground plane with a wall at x=30 crossed only by a 4 m gap at
y in [18,22], plus a low canopy slab over a corner. Hand-checkable outcomes:

  - pairs whose straight line fits through the gap are TOO EASY (rejected),
  - pairs crossing the wall elsewhere must detour through the gap (kept),
  - with the gap sealed there is no flyable path (rejected),
  - endpoints under the canopy are rejected (climb column blocked).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from superfly.perception import occupancy as oc      # noqa: E402


def _grid_pts(x0, x1, y0, y1, z, h=0.4):
    xs = np.arange(x0, x1 + 1e-9, h)
    ys = np.arange(y0, y1 + 1e-9, h)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))], 1)


def _wall_x(x, y0, y1, z0=0.0, z1=4.0, h=0.2):
    ys = np.arange(y0, y1 + 1e-9, h)
    zs = np.arange(z0, z1 + 1e-9, h)
    Y, Z = np.meshgrid(ys, zs)
    return np.stack([np.full(Y.size, float(x)), Y.ravel(), Z.ravel()], 1)


def _scene(gap=(18.0, 22.0), canopy=True):
    ground = _grid_pts(0, 60, 0, 40, 0.0)
    if canopy:  # ground-like slab 1.5 m over the far corner: blocks the climb
        ground = np.concatenate([ground, _grid_pts(50, 60, 30, 40, 1.5)])
    walls = [_wall_x(30.0, 0.0, gap[0]), _wall_x(30.0, gap[1], 40.0)] \
        if gap else [_wall_x(30.0, 0.0, 40.0)]
    return np.concatenate(walls), ground


class TestBuildSlice(unittest.TestCase):

    def setUp(self):
        S, G = _scene()
        self.sl = oc.build_slice(S, G, climb_alt=2.0)

    def test_ground_and_altitude(self):
        self.assertIsNotNone(self.sl)
        self.assertAlmostEqual(self.sl.z0, 0.0, delta=0.2)
        self.assertAlmostEqual(self.sl.z_fly, self.sl.z0 + 2.0)

    def test_wall_occupied_gap_free(self):
        for y, expect in ((10.0, True), (20.0, False), (30.0, True)):
            iy, ix = self.sl.world_to_cell((30.0, y))
            self.assertEqual(bool(self.sl.occ[iy, ix]), expect, f"y={y}")

    def test_canopy_blocks_climb_column(self):
        m = oc.endpoint_mask(self.sl)
        iy, ix = self.sl.world_to_cell((55.0, 35.0))    # under the canopy
        self.assertFalse(m[iy, ix])
        iy, ix = self.sl.world_to_cell((10.0, 20.0))    # open field
        self.assertTrue(m[iy, ix])

    def test_outside_map_not_passable(self):
        iy, ix = self.sl.world_to_cell((30.0, 41.5))    # past the ground edge
        if 0 <= iy < self.sl.known.shape[0] and 0 <= ix < self.sl.known.shape[1]:
            self.assertFalse(self.sl.known[iy, ix])


class TestMining(unittest.TestCase):

    def _mine(self, gap):
        S, G = _scene(gap=gap, canopy=False)
        sl = oc.build_slice(S, G, climb_alt=2.0)
        cands, stats = oc.mine_pairs(sl, n_candidates=6000, seed=7,
                                     len_range=(25.0, 60.0), max_evals=200,
                                     verbose=False)
        return sl, cands, stats

    def test_wall_with_gap_yields_detour_pairs(self):
        sl, cands, stats = self._mine(gap=(18.0, 22.0))
        self.assertGreater(len(cands), 0, stats)
        for c in cands:
            self.assertLess(c.line_min_edt_m, oc.TRIVIAL_M)      # gate 2
            self.assertGreaterEqual(c.path_min_edt_m, oc.PASSAGE_MIN_M)  # gate 4
            self.assertGreater(c.detour, 1.0)
            # the only opening is the gap: every surviving path crosses it
            xy = np.array([sl.cell_to_world(p) for p in c.path_cells])
            crossing = xy[np.abs(xy[:, 0] - 30.0) < 1.0]
            self.assertTrue((np.abs(crossing[:, 1] - 20.0) < 3.0).all())

    def test_straight_through_gap_is_rejected_as_trivial(self):
        sl, cands, stats = self._mine(gap=(14.0, 26.0))  # wide-open 12 m gap
        # pairs whose line goes through the middle are filtered as line_open
        self.assertGreater(stats["line_open"], 0)

    def test_sealed_wall_yields_nothing(self):
        sl, cands, stats = self._mine(gap=None)
        self.assertEqual(len(cands), 0)
        self.assertGreater(stats["no_path"], 0, stats)

    def test_open_field_yields_nothing(self):
        S, G = _scene(gap=(18.0, 22.0), canopy=False)
        sl = oc.build_slice(S[:0], G, climb_alt=2.0)      # no obstacles at all
        cands, stats = oc.mine_pairs(sl, n_candidates=2000, seed=3,
                                     len_range=(25.0, 60.0), verbose=False)
        self.assertEqual(len(cands), 0)


class TestDifficultyAndSelection(unittest.TestCase):

    def setUp(self):
        S, G = _scene(canopy=False)
        self.sl = oc.build_slice(S, G, climb_alt=2.0)
        self.cands, _ = oc.mine_pairs(self.sl, n_candidates=6000, seed=7,
                                      len_range=(25.0, 60.0), max_evals=300,
                                      verbose=False)
        oc.score_difficulty(self.cands)

    def test_difficulty_normalized_and_ordered(self):
        self.assertGreater(len(self.cands), 2)
        d = np.array([c.difficulty for c in self.cands])
        self.assertTrue(((d >= 0.0) & (d <= 1.0)).all())
        # a bigger detour at equal weights must not score lower than a
        # candidate it dominates on every component
        for a in self.cands:
            for b in self.cands:
                if (a.components["detour_n"] >= b.components["detour_n"] and
                        a.components["blockage_n"] >= b.components["blockage_n"] and
                        a.components["tightness_n"] >= b.components["tightness_n"]):
                    self.assertGreaterEqual(a.difficulty + 1e-9, b.difficulty)

    def test_selection_spread_and_disjoint(self):
        picked = oc.select_pairs(self.cands, k=3)
        self.assertLessEqual(len(picked), 3)
        self.assertGreater(len(picked), 0)
        self.assertEqual(len({id(p) for p in picked}), len(picked))


if __name__ == "__main__":
    unittest.main()
