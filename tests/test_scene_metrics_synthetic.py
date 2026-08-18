"""Known-answer verification of the USD-scene scoring path, no Isaac needed.

Authors a tiny USD stage with pip's usd-core (`pip install usd-core`; provides
`pxr` for LOCAL files -- omniverse:// still needs Kit) whose geometry has
hand-computable clearances, runs it through the REAL production code
(`superfly.perception.mesh_sampling` -> extract-format .npz ->
`superfly.compare.metrics.score_trajectory`), and asserts the numbers.

The stage deliberately exercises every composition feature that has bitten
before (ATTEMPTS 2026-07-29, probe3 2026-08-07):
  - authored in CENTIMETERS (all coords x100), scored after env_scale=0.01,
  - a wall placed through a NESTED Xform (translate),
  - a second wall as an INSTANCEABLE REFERENCE to a separate layer,
  - two poles via a PointInstancer (prototype authored invisible, per
    convention),
  - a ground quad that must be excluded by the 30-degree ground-face filter.

Layout (meters, after scale) -- flight corridor along +X at y=0:

    ground   z=0 quad, x in [-10,30], y in [-12,12]
    wallL    x=10 plane, y in [-12,-1], z in [0,5]   (nested Xform)
    wallR    x=10 plane, y in [  1,12], z in [0,5]   (instanceable reference)
    gap      y in (-1,1) at x=10  ->  true clearance at gap center = 1.0 m
    poles    0.4x0.4 m sides, z in [0,3], at (20,+-3) (PointInstancer)

Trajectories are hand-built in the run_px4_sim --log-traj npz format with
explicit phase-handoff timestamps (t_unix0/policy_start_unix/policy_end_unix)
so scoring uses the exact policy window, not the motion heuristics.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from superfly.compare import metrics                      # noqa: E402
from superfly.perception import mesh_sampling             # noqa: E402

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    HAVE_PXR = True
except ImportError:
    HAVE_PXR = False

ENV_SCALE = 0.01          # stage is authored in cm
SAMPLE_H = 0.1            # [m] surface sample spacing (after scale)
DRONE_R = 0.2
CM = 100.0                # meters -> authored units


def _quad(stage, path, corners_m):
    """A single-quad Mesh from 4 corners given in METERS (authored x100)."""
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(
        [Gf.Vec3f(x * CM, y * CM, z * CM) for x, y, z in corners_m]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    return m


def _wall_x(stage, path, x, y0, y1, z0=0.0, z1=5.0):
    """Vertical quad in the x=const plane (meters)."""
    return _quad(stage, path, [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])


def _author_stage(tmp: Path) -> str:
    """Write proto layer + main stage; return the main stage path."""
    # -- separate layer holding the instanceable wall prototype (y in [0,11] local)
    proto_path = str(tmp / "wall_proto.usda")
    pstage = Usd.Stage.CreateNew(proto_path)
    UsdGeom.SetStageMetersPerUnit(pstage, ENV_SCALE)
    UsdGeom.SetStageUpAxis(pstage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(pstage, "/wall")
    pstage.SetDefaultPrim(root.GetPrim())
    _wall_x(pstage, "/wall/m", x=0.0, y0=0.0, y1=11.0)
    pstage.Save()

    stage_path = str(tmp / "synthetic_scene.usda")
    stage = Usd.Stage.CreateNew(stage_path)
    UsdGeom.SetStageMetersPerUnit(stage, ENV_SCALE)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    _quad(stage, "/World/ground",
          [(-10, -12, 0), (30, -12, 0), (30, 12, 0), (-10, 12, 0)])

    # nested Xform: /World/env translated to x=10; wallL authored at local x=0
    env = UsdGeom.Xform.Define(stage, "/World/env")
    env.AddTranslateOp().Set(Gf.Vec3d(10.0 * CM, 0.0, 0.0))
    _wall_x(stage, "/World/env/wallL", x=0.0, y0=-12.0, y1=-1.0)

    # instanceable reference: wallR = proto layer, translated to y=+1 (local,
    # under the same /World/env translate)
    wall_r = UsdGeom.Xform.Define(stage, "/World/env/wallR")
    wall_r.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0 * CM, 0.0))
    wall_r.GetPrim().GetReferences().AddReference(proto_path, "/wall")
    wall_r.GetPrim().SetInstanceable(True)

    # PointInstancer: two poles at (20, +-3); prototype invisible (convention)
    pi = UsdGeom.PointInstancer.Define(stage, "/World/pi")
    proto = UsdGeom.Xform.Define(stage, "/World/pi/protos/pole")
    for i, (x0, y0, x1, y1) in enumerate([
            (-0.2, -0.2, 0.2, -0.2), (0.2, -0.2, 0.2, 0.2),
            (0.2, 0.2, -0.2, 0.2), (-0.2, 0.2, -0.2, -0.2)]):
        _quad(stage, f"/World/pi/protos/pole/side{i}",
              [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, 3.0), (x0, y0, 3.0)])
    UsdGeom.Imageable(proto.GetPrim()).MakeInvisible()
    pi.CreatePrototypesRel().AddTarget(Sdf.Path("/World/pi/protos/pole"))
    pi.CreateProtoIndicesAttr(Vt.IntArray([0, 0]))
    pi.CreatePositionsAttr(Vt.Vec3fArray(
        [Gf.Vec3f(20.0 * CM, 3.0 * CM, 0.0), Gf.Vec3f(20.0 * CM, -3.0 * CM, 0.0)]))

    stage.Save()
    return stage_path


def _extract(stage_path: str, out_npz: Path):
    """Mirror scripts/extract_scene_mesh.py's pipeline on a local stage:
    gather -> env_scale -> ground split -> sample both channels -> npz."""
    stage = Usd.Stage.Open(stage_path)
    V, F = mesh_sampling.gather_triangles(stage, root_path=None, verbose=False)
    V = V * ENV_SCALE
    lat, gnd = mesh_sampling.split_ground_faces(V, F, ground_deg=30.0)
    S = mesh_sampling.dedupe_samples(
        mesh_sampling.sample_surface(V, F[lat], SAMPLE_H), SAMPLE_H / 2.0)
    Sg = mesh_sampling.dedupe_samples(
        mesh_sampling.sample_surface(V, F[gnd], SAMPLE_H * 2), SAMPLE_H)
    meta = dict(usd=stage_path, env_scale=ENV_SCALE, sample_h=SAMPLE_H,
                ground_deg=30.0, bounds=None, n_triangles=int(F.shape[0]),
                extractor_version=2)
    np.savez_compressed(out_npz, samples=S.astype(np.float32),
                        ground_samples=Sg.astype(np.float32),
                        meta=json.dumps(meta))
    return V, F, lat, gnd, S


DT = 0.016
T0_UNIX = 1000.0


def _traj_npz(path: Path, cruise_z: float, goal_y: float):
    """Parked 5 s at (0,0,0.1) -> climb at 1 m/s to cruise_z -> straight cruise
    at 2 m/s to (20, goal_y, cruise_z) -> parked 1 s. Wall clock == sim time."""
    rows, t = [], 0.0

    def seg(p0, p1, v):
        nonlocal t
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = np.linalg.norm(p1 - p0)
        n = max(int(round(d / (v * DT))), 1) if d > 0 else int(round(v / DT))
        vel = (p1 - p0) / (n * DT) if d > 0 else np.zeros(3)
        for k in range(n):
            rows.append([t, *(p0 + vel * (k * DT) if d > 0 else p0), *vel])
            t += DT
        return p1

    p = np.array([0.0, 0.0, 0.1])
    for _ in range(int(5.0 / DT)):                     # parked (sim warmup)
        rows.append([t, *p, 0.0, 0.0, 0.0]); t += DT
    p = seg(p, [0.0, 0.0, cruise_z], 1.0)              # scripted climb
    t_policy_start = t
    goal = np.array([20.0, goal_y, cruise_z])
    p = seg(p, goal, 2.0)                              # policy cruise
    t_policy_end = t
    for _ in range(int(1.0 / DT)):                     # parked at goal
        rows.append([t, *p, 0.0, 0.0, 0.0]); t += DT

    traj = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(
        path, traj=traj, t_sim=traj[:, 0].copy(), goal=goal,
        policy="synthetic", seed=0, t_unix0=T0_UNIX,
        policy_start_unix=T0_UNIX + t_policy_start,
        policy_end_unix=T0_UNIX + t_policy_end)
    return goal


@unittest.skipUnless(HAVE_PXR, "usd-core not installed (pip install usd-core)")
class TestSceneMetricsSynthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.stage_path = _author_stage(tmp)
        cls.mesh_npz = tmp / "scene_mesh.npz"
        cls.V, cls.F, cls.lat, cls.gnd, cls.S = _extract(cls.stage_path, cls.mesh_npz)
        cls.tmp = tmp

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ---- geometry / composition ----------------------------------------- #

    def test_all_composition_arcs_captured(self):
        S = self.S
        # nested-Xform wallL: samples at x~10, y in [-12,-1]
        self.assertTrue(((np.abs(S[:, 0] - 10) < 0.05) & (S[:, 1] < -1.01)).any(),
                        "wallL (nested Xform) missing from lateral samples")
        # instanceable-reference wallR: samples at x~10, y in [1,12]
        self.assertTrue(((np.abs(S[:, 0] - 10) < 0.05) & (S[:, 1] > 1.01)).any(),
                        "wallR (instanceable reference) missing -- instance "
                        "proxies not traversed?")
        # PointInstancer poles at (20, +-3)
        for ysign in (1, -1):
            near = (np.abs(S[:, 0] - 20) < 0.3) & (np.abs(S[:, 1] - 3 * ysign) < 0.3)
            self.assertTrue(near.any(),
                            f"pole at (20,{3 * ysign}) missing -- PointInstancer "
                            "not expanded?")

    def test_scale_and_bounds(self):
        # env_scale applied: whole scene must be meter-plausible, not cm
        self.assertLess(self.V.max(), 50.0)
        self.assertGreater(self.V[:, 0].max(), 25.0)   # ground reaches x=30

    def test_ground_filter(self):
        # the ground quad is ground-like; every lateral sample is off-ground
        self.assertGreater(self.gnd.sum(), 0)
        self.assertTrue((self.S[:, 2] > -0.01).all())
        z = np.load(self.mesh_npz)
        self.assertGreater(np.asarray(z["ground_samples"]).shape[0], 0)
        self.assertLess(np.abs(np.asarray(z["ground_samples"])[:, 2]).max(), 0.01)

    def test_gap_edge_sampled_densely(self):
        # a sample must sit within ~SAMPLE_H of the wallL gap edge (10,-1,2):
        # clearance overestimation bound depends on it
        d = np.linalg.norm(self.S - np.array([10.0, -1.0, 2.0]), axis=1)
        self.assertLess(d.min(), SAMPLE_H)

    # ---- end-to-end scoring --------------------------------------------- #

    def _score(self, name, cruise_z, goal_y):
        p = self.tmp / f"{name}.npz"
        _traj_npz(p, cruise_z, goal_y)
        return metrics.score_trajectory(str(p), drone_radius=DRONE_R,
                                        goal_radius=1.0,
                                        scene_mesh=str(self.mesh_npz))

    def test_flight_through_gap(self):
        res = self._score("gap", cruise_z=2.0, goal_y=0.0)
        self.assertEqual(res["clearance_source"], "scene_mesh")
        self.assertEqual(res["policy_window_source"], "phase_file")
        self.assertTrue(res["reached"])
        self.assertFalse(res["collided"])
        self.assertTrue(res["success"])
        # true clearance = 1.0 (gap half-width) - 0.2 (radius) = 0.8;
        # sampling overestimates by <= ~SAMPLE_H
        self.assertGreaterEqual(res["min_clearance_m"], 0.8 - 0.02)
        self.assertLessEqual(res["min_clearance_m"], 0.8 + 1.5 * SAMPLE_H)

    def test_flight_through_wall(self):
        res = self._score("wall", cruise_z=2.0, goal_y=-6.0)
        self.assertTrue(res["reached"])          # it does arrive over the goal
        self.assertTrue(res["collided"])         # ...through the wall
        self.assertFalse(res["success"])
        self.assertLess(res["min_clearance_m"], -0.1)

    def test_ground_skim_is_not_a_collision(self):
        # cruise at z=0.15 with drone radius 0.2: if ground faces leaked into
        # the clearance samples this would score a collision; the filter's
        # whole point is that it must not.
        res = self._score("skim", cruise_z=0.15, goal_y=0.0)
        self.assertFalse(res["collided"])
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["min_clearance_m"], 0.8 - 0.02)


if __name__ == "__main__":
    unittest.main()
