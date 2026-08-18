#!/usr/bin/env python
"""Vet Nucleus / Isaac-assets USD scenes for the superfly benchmark harness.

Runs under Isaac Sim's python on airstation03 (Nucleus resolver + auth live
there; omniverse:// inside OSMO is ESTABLISHED-NEGATIVE without a token --
ATTEMPTS 2026-07). One Kit boot audits the whole catalog.

Per scene it answers, without a human opening Isaac:
  1. composes?           usd-guard semantics from run_px4_sim.py
  2. axis/scale?         metersPerUnit + AABB -> env_scale in {1, mPU, 0.01}
  3. geometry            gather_triangles (instancing expanded) -> coarse
                         dual-channel samples.npz (extract_scene_mesh format;
                         mining input for scripts/mine_scene_goals.py)
  4. ground?             dominant ground z + coverage (occupancy module)
  5. colliders work?     scene_setup.add_colliders (the SAME code the flight
                         harness runs) + rigid-sphere drop test
  6. visible?            RGB luminance + depth-camera sanity at flight
                         altitude, after scene_setup.spawn_lighting
  7. static? conflicts?  time-sampled xforms, authored PhysicsScene/rigid
                         bodies, load-time/triangle-count cost telemetry

Outputs, per scene, under --out/<scene>/: report.json, samples.npz,
thumb_fpv.png, thumb_overhead.png; plus --out/scene_catalog.json and
--out/summary.json. Fetch to gs2 with `airstation fetch superfly
results/scene_audit`.

Usage (airstation03):
    ~/isaacsim/python.sh scripts/scene_audit.py --out results/scene_audit \\
        [--list-only] [--only PATTERN] [--resume] [--limit N] \\
        [--nucleus omniverse://airlab-nucleus.andrew.cmu.edu/Library/Stages] \\
        [--isaac-environments] [--no-drop-test] [--catalog FILE]
"""
import argparse
import fnmatch
import json
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", required=True)
parser.add_argument("--nucleus", action="append", default=None,
                    help="Nucleus folder(s) to sweep recursively (default: "
                         "airlab Library/Stages). Repeatable.")
parser.add_argument("--isaac-environments", action="store_true",
                    help="also audit the stock Isaac/NVIDIA environments "
                         "(Pegasus SIMULATION_ENVIRONMENTS catalog)")
parser.add_argument("--no-nucleus", action="store_true",
                    help="skip the Nucleus roots (stock environments only -- "
                         "e.g. while Nucleus auth is expired)")
parser.add_argument("--catalog", default=None,
                    help="reuse an existing scene_catalog.json instead of listing")
parser.add_argument("--list-only", action="store_true",
                    help="write scene_catalog.json and exit (no audits)")
parser.add_argument("--only", default=None, help="fnmatch pattern on scene name")
parser.add_argument("--limit", type=int, default=None, help="audit at most N scenes")
parser.add_argument("--resume", action="store_true",
                    help="skip scenes that already have a report.json")
parser.add_argument("--no-drop-test", action="store_true",
                    help="geometry-only audit (no World/physics/rendering)")
parser.add_argument("--negative-control", action="store_true",
                    help="SKIP collider application before the drop test: every "
                         "sphere must fall through and the scene must FAIL. "
                         "Proves the drop test can catch broken colliders. "
                         "Writes reports under <out>/_negative_control/.")
parser.add_argument("--sample-h", type=float, default=0.2,
                    help="lateral sample spacing [m] for the mining npz (default 0.2)")
parser.add_argument("--max-samples", type=float, default=3e6,
                    help="lateral sample budget per scene (h coarsens to fit)")
parser.add_argument("--climb-alt", type=float, default=2.0,
                    help="probe/flight altitude above ground [m]")
args = parser.parse_args()

DEFAULT_NUCLEUS = ["omniverse://airlab-nucleus.andrew.cmu.edu/Library/Stages"]

# Non-interactive Nucleus auth MUST be configured BEFORE Kit boots: omni.client
# honors OMNI_USER='$omni-api-token' + OMNI_PASS=<token> read at init. (The
# register_authentication_callback route that extract_scene_mesh.py originally
# used does NOT work against airlab-nucleus -- the server still pushes browser
# SSO and the list fails ERROR_CONNECTION; verified 2026-08-18.) When the env
# var is missing, ~/.omni_env ("export OMNI_API_TOKEN=...") is parsed directly
# -- dispatch layers (airstation run -> ssh -> nice) mangle `bash -c 'source
# ...'` quoting too easily.
import os                                              # noqa: E402
if not os.environ.get("OMNI_API_TOKEN"):
    _envf = Path.home() / ".omni_env"
    if _envf.exists():
        for _line in _envf.read_text().splitlines():
            _line = _line.strip().removeprefix("export ").strip()
            if _line.startswith("OMNI_API_TOKEN="):
                os.environ["OMNI_API_TOKEN"] = _line.split("=", 1)[1].strip().strip("'\"")
                print("[auth] OMNI_API_TOKEN loaded from ~/.omni_env")
                break
if os.environ.get("OMNI_API_TOKEN") and not os.environ.get("OMNI_USER"):
    os.environ["OMNI_USER"] = "$omni-api-token"
    os.environ["OMNI_PASS"] = os.environ["OMNI_API_TOKEN"]

from isaacsim import SimulationApp                     # noqa: E402
simulation_app = SimulationApp({"headless": True})

import numpy as np                                     # noqa: E402
import omni.client                                     # noqa: E402
import omni.usd                                        # noqa: E402
import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics               # noqa: E402
from scipy.spatial.transform import Rotation           # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from superfly.perception import mesh_sampling, occupancy as oc   # noqa: E402
from superfly.sim import scene_setup                   # noqa: E402

GROUND_DEG = 30.0          # must match extract_scene_mesh.py / metrics semantics
PLAUSIBLE_XY = (15.0, 3000.0)   # horizontal extent [m] a benchmark scene can have
PLAUSIBLE_Z = (0.5, 500.0)


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #

def _list_dir(url):
    res, entries = omni.client.list(url)
    if res != omni.client.Result.OK:
        print(f"[inventory] cannot list {url}: {res}", file=sys.stderr)
        return []
    return entries


def _walk_usd_files(root, depth=4):
    """(url, size) of every .usd* file under root, depth-capped."""
    out = []
    stack = [(root.rstrip("/"), 0)]
    while stack:
        url, d = stack.pop()
        for e in _list_dir(url):
            name = e.relative_path.strip("/")
            child = f"{url}/{name}"
            if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                if d < depth and not name.startswith("."):
                    stack.append((child, d + 1))
            elif name.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
                out.append((child, int(getattr(e, "size", 0) or 0)))
    return out


_SKIP_PAT = ("/materials/", "/textures/", "/props/", "/looks/", "/skies/",
             "/assets/", "/subusds/", "/archive/", "/engine/", "_mat.",
             "_material")


def build_catalog(nucleus_roots, include_isaac_envs):
    """ONE entry per top-level scene folder under each root (personal work
    dirs like Muyang/ or Dmytro/ hold dozens of asset .usd files in subtrees
    -- those are parts, not scenes). Within a folder prefer, in order:
    '*stage*' basenames, shallower depth, larger size."""
    catalog = []
    for root in nucleus_roots:
        rootp = root.rstrip("/")
        files = _walk_usd_files(root)
        groups = {}
        for url, size in files:
            if any(p in url.lower() for p in _SKIP_PAT):
                continue
            rel = url[len(rootp):].strip("/")
            top = rel.split("/", 1)[0]          # scene folder, or root-level file
            groups.setdefault(top, []).append((url, size, rel.count("/")))
        for top, fs in sorted(groups.items()):
            def rank(f):
                url, size, depth = f
                staged = "stage" in url.rsplit("/", 1)[1].lower()
                return (not staged, depth, -size)
            pick = sorted(fs, key=rank)[0]
            name = top[:-len(Path(top).suffix)] if "." in top else top
            catalog.append({"name": name, "usd": pick[0], "size": pick[1],
                            "source": root,
                            "siblings": [f[0] for f in fs if f[0] != pick[0]][:20]})
    if include_isaac_envs:
        from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
        for name, url in sorted(SIMULATION_ENVIRONMENTS.items()):
            catalog.append({"name": f"isaac_{name.replace(' ', '_')}",
                            "usd": url, "size": None, "source": "pegasus",
                            "siblings": []})
    return catalog


# --------------------------------------------------------------------------- #
# per-scene audit
# --------------------------------------------------------------------------- #

def detect_env_scale(V, meters_per_unit):
    """Pick env_scale from {1.0, metersPerUnit, 0.01} by extent plausibility.
    Extent is the 0.5-99.5% vertex quantile span, NOT min/max: UE exports
    carry km-scale sky spheres / backdrop geometry that blow up the raw AABB
    at every candidate scale (AbandonedFactory & friends, sweep 2026-08-18).
    Returns (scale or None, [plausible scales], reason, robust extent)."""
    ext = np.quantile(V, 0.995, axis=0) - np.quantile(V, 0.005, axis=0)
    cands = []
    for s in dict.fromkeys([1.0, float(meters_per_unit or 1.0), 0.01]):
        xy = float(max(ext[0], ext[1])) * s
        z = float(ext[2]) * s
        if PLAUSIBLE_XY[0] <= xy <= PLAUSIBLE_XY[1] and \
                PLAUSIBLE_Z[0] <= z <= PLAUSIBLE_Z[1]:
            cands.append(s)
    if not cands:
        return None, [], f"no plausible scale for extent {ext.round(1).tolist()}", ext
    # prefer the authored metersPerUnit, then 1.0
    for pref in (float(meters_per_unit or 1.0), 1.0):
        if pref in cands:
            return pref, cands, "authored" if pref != 1.0 else "unit", ext
    return cands[0], cands, "fallback", ext


def scan_stage_flags(stage, max_prims=60_000):
    """(n_time_sampled, physics dict) -- animated geometry + authored physics."""
    n_anim, n_scanned = 0, 0
    phys = {"physics_scene": 0, "rigid_bodies": 0, "colliders": 0}
    for prim in stage.Traverse():
        n_scanned += 1
        if n_scanned > max_prims:
            break
        if prim.IsA(UsdPhysics.Scene):
            phys["physics_scene"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            phys["rigid_bodies"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            phys["colliders"] += 1
        if prim.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetNumTimeSamples() > 1:
                    n_anim += 1
                    break
    return n_anim, phys, n_scanned


def geometry_audit(entry, rep, scene_dir):
    """Stage-only checks (no World). Fills rep; returns (V*scale, F, lat, gnd)
    or None on FAIL."""
    t0 = time.time()
    stage = Usd.Stage.Open(entry["usd"])
    if stage is None:
        rep["reasons"].append("FAIL:stage_open")
        return None
    rep["load_time_s"] = round(time.time() - t0, 1)
    rep["meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))
    rep["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
    if rep["up_axis"] != "Z":
        rep["reasons"].append(f"FLAG:up_axis_{rep['up_axis']}")

    t0 = time.time()
    V, F = mesh_sampling.gather_triangles(stage, root_path=None, verbose=True)
    rep["gather_time_s"] = round(time.time() - t0, 1)
    rep["n_triangles"] = int(F.shape[0])
    if F.shape[0] == 0:
        rep["reasons"].append("FAIL:no_mesh_geometry")
        return None

    scale, cands, why, ext_units = detect_env_scale(V, rep["meters_per_unit"])
    rep["env_scale"], rep["scale_candidates"], rep["scale_reason"] = scale, cands, why
    rep["robust_extent_units"] = ext_units.round(1).tolist()
    if scale is None:
        rep["reasons"].append("FAIL:implausible_scale")
        return None
    if len(cands) > 1:
        rep["reasons"].append(f"FLAG:scale_ambiguous_{cands}")
    V = V * scale
    # robust bounds (sky spheres excluded) -- the drop-test / camera code
    # aims at this box's center, so it must be the SCENE, not the skybox
    lo = np.quantile(V, 0.005, axis=0)
    hi = np.quantile(V, 0.995, axis=0)
    rep["aabb_m"] = [lo.round(1).tolist(), hi.round(1).tolist()]
    rep["aabb_raw_m"] = [V.min(0).round(1).tolist(), V.max(0).round(1).tolist()]

    n_anim, phys, n_scanned = scan_stage_flags(stage)
    rep["time_sampled_xforms"], rep["authored_physics"] = n_anim, phys
    if n_anim:
        rep["reasons"].append(f"FLAG:animated_xforms_{n_anim}")
    if phys["physics_scene"] or phys["rigid_bodies"]:
        rep["reasons"].append(
            f"FLAG:authored_physics_scene={phys['physics_scene']}"
            f"_rigid={phys['rigid_bodies']}")

    # Crop triangles to the robust box (+margin) BEFORE sampling: UE sky
    # spheres/kill-planes have ~10^5x the scene's surface area, so without the
    # crop they eat the whole sample budget -- CityPark's npz was 94% skybox
    # with 41 samples on the actual terrain (sweep finding 2026-08-18). Same
    # rationale as extract_scene_mesh --bounds. Chunked: V[F] on 30M tris is
    # multi-GB at once.
    box_lo = lo - np.array([20.0, 20.0, 10.0])
    box_hi = hi + np.array([20.0, 20.0, 10.0])
    keep = np.zeros(F.shape[0], dtype=bool)
    for s in range(0, F.shape[0], 5_000_000):
        tv = V[F[s:s + 5_000_000]]
        keep[s:s + 5_000_000] = ((tv.max(axis=1) >= box_lo) &
                                 (tv.min(axis=1) <= box_hi)).all(axis=1)
    rep["n_triangles_cropped"] = int((~keep).sum())
    F = F[keep]
    if F.shape[0] == 0:
        rep["reasons"].append("FAIL:no_geometry_in_robust_box")
        return None

    # dual-channel coarse sampling (extract_scene_mesh format, mining input)
    lat, gnd = mesh_sampling.split_ground_faces(V, F, GROUND_DEG)
    h = args.sample_h
    est = mesh_sampling.estimated_samples(V, F[lat], h)
    if est > args.max_samples:
        h *= float(np.sqrt(est / args.max_samples))
    S = mesh_sampling.dedupe_samples(
        mesh_sampling.sample_surface(V, F[lat], h), h / 2.0).astype(np.float32)
    gh = max(2 * h, 0.4)
    g_est = mesh_sampling.estimated_samples(V, F[gnd], gh)
    if g_est > args.max_samples / 2:
        gh *= float(np.sqrt(g_est / (args.max_samples / 2)))
    Sg = mesh_sampling.dedupe_samples(
        mesh_sampling.sample_surface(V, F[gnd], gh), gh / 2.0).astype(np.float32)
    rep["sample_h"], rep["n_samples"], rep["n_ground_samples"] = \
        round(h, 3), int(S.shape[0]), int(Sg.shape[0])

    z0, frac = oc.dominant_ground_z(Sg) if Sg.shape[0] else (None, 0.0)
    rep["ground_z"], rep["ground_frac"] = z0, round(frac, 3)
    if z0 is None:
        rep["reasons"].append("FAIL:no_ground")
        return None
    if frac < 0.2:
        rep["reasons"].append(f"FLAG:ground_fragmented_{frac:.2f}")

    meta = dict(usd=entry["usd"], env_scale=scale, sample_h=h, ground_deg=GROUND_DEG,
                bounds=None, ground_sample_h=gh, n_triangles=int(F.shape[0]),
                extractor_version=2, audit=True)
    np.savez_compressed(scene_dir / "samples.npz", samples=S, ground_samples=Sg,
                        meta=json.dumps(meta))

    # mining preview: how much of this scene is even eligible as an endpoint
    sl = oc.build_slice(S, Sg, climb_alt=args.climb_alt)
    if sl is not None:
        rep["endpoint_cells"] = int(oc.endpoint_mask(sl).sum())
        rep["occupied_frac_at_alt"] = round(float(sl.occ.mean()), 4)
        if rep["endpoint_cells"] < 50:
            rep["reasons"].append("FLAG:few_endpoint_cells")
    return V, F, z0


def _quat_wxyz(yaw_deg, pitch_deg):
    q = Rotation.from_euler("ZYX", [yaw_deg, pitch_deg, 0.0], degrees=True).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def sim_audit(entry, rep, scene_dir, z0):
    """World-level checks: usd-guard, colliders + drop test, lighting/depth
    probes, thumbnails. Mirrors run_px4_sim.py's load path (reference under
    /World/layout, uniform env_scale, scene_setup lighting+colliders)."""
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicSphere
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.camera import Camera
    from pxr import Gf

    World.clear_instance()          # drop any World left by a previous scene
    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    scale = rep["env_scale"]

    add_reference_to_stage(entry["usd"], "/World/layout")
    layout = stage.GetPrimAtPath("/World/layout")
    if not layout.IsValid() or not layout.GetChildren():   # usd-guard
        rep["reasons"].append("FAIL:stage_empty_usd_guard")
        return
    if scale != 1.0:
        UsdGeom.XformCommonAPI(layout).SetScale(Gf.Vec3f(scale, scale, scale))
    scene_setup.spawn_lighting()
    if args.negative_control:
        # deliberately broken setup: no colliders -- including any AUTHORED
        # collision APIs the stage ships with (stock NVIDIA envs carry
        # thousands; without stripping them the control is vacuous).
        # The drop test MUST fail afterwards.
        n_stripped = 0
        for prim in stage.Traverse():
            try:
                for api in (UsdPhysics.MeshCollisionAPI, UsdPhysics.CollisionAPI):
                    if prim.HasAPI(api):
                        prim.RemoveAPI(api)
                        n_stripped += 1
            except Exception:
                pass  # instance proxies etc. -- best effort
        rep["n_colliders"] = 0
        rep["reasons"].append(f"NOTE:negative_control_stripped_{n_stripped}")
    else:
        t0 = time.time()
        rep["n_colliders"] = scene_setup.add_colliders(stage, "/World/layout")
        rep["collider_time_s"] = round(time.time() - t0, 1)
        if not rep["n_colliders"]:
            rep["reasons"].append("FAIL:zero_colliders")
            return

    # drop test: rigid spheres over 3 spread ground points must come to rest
    # near z0 instead of falling through (the probe3 failure mode)
    z = np.load(scene_dir / "samples.npz")
    G = np.asarray(z["ground_samples"])
    g = G[np.abs(G[:, 2] - z0) <= 0.5]
    qs = [0.3, 0.5, 0.7]
    pts = [g[np.argmin(np.abs(g[:, 0] - np.quantile(g[:, 0], q)) +
                       np.abs(g[:, 1] - np.quantile(g[:, 1], q)))] for q in qs]
    R = 0.25
    spheres = []
    for i, p in enumerate(pts):
        spheres.append(world.scene.add(DynamicSphere(
            prim_path=f"/World/audit_drop_{i}", name=f"audit_drop_{i}",
            position=np.array([p[0], p[1], z0 + 3.0]), radius=R, mass=1.0)))
    world.reset()
    for _ in range(240):                     # 4 s at 1/60
        world.step(render=False)
    drops = []
    for s, p in zip(spheres, pts):
        rest = float(s.get_world_pose()[0][2])
        ok = abs(rest - (z0 + R)) <= 0.6
        through = rest < z0 - 1.0
        drops.append({"xy": [round(float(p[0]), 1), round(float(p[1]), 1)],
                      "rest_z": round(rest, 2), "ok": bool(ok),
                      "fell_through": bool(through)})
    rep["drop_test"] = drops
    if any(d["fell_through"] for d in drops):
        rep["reasons"].append("FAIL:drop_fell_through")
    elif not any(d["ok"] for d in drops):
        rep["reasons"].append("FLAG:drop_rest_off_ground")

    # camera probes at flight altitude from the middle drop point
    try:
        p = pts[1]
        center = 0.5 * (np.array(rep["aabb_m"][0]) + np.array(rep["aabb_m"][1]))
        yaw = float(np.degrees(np.arctan2(center[1] - p[1], center[0] - p[0])))
        cam = Camera(prim_path="/World/audit_cam", resolution=(400, 300),
                     position=np.array([p[0], p[1], z0 + args.climb_alt]),
                     orientation=_quat_wxyz(yaw, 0.0))
        cam.initialize()
        cam.add_distance_to_image_plane_to_frame()
        for _ in range(45):
            world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            rgb = np.asarray(rgba)[..., :3].astype(np.float32)
            rep["luminance"] = round(float(rgb.mean()) / 255.0, 3)
            plt.imsave(scene_dir / "thumb_fpv.png", np.asarray(rgba)[..., :3])
            if rep["luminance"] < 0.03:
                rep["reasons"].append("FLAG:dark_render_interior")
        depth = cam.get_current_frame().get("distance_to_image_plane")
        if depth is not None and getattr(depth, "size", 0):
            d = np.asarray(depth, np.float32)
            finite = np.isfinite(d) & (d > 0.05)
            rep["depth_finite_frac"] = round(float(finite.mean()), 3)
            if finite.any():
                rep["depth_p5_p95_m"] = [round(float(np.percentile(d[finite], 5)), 1),
                                         round(float(np.percentile(d[finite], 95)), 1)]
            if finite.mean() < 0.05:
                rep["reasons"].append("FLAG:depth_degenerate")

        # overhead thumbnail for the human review report. Roofed scenes:
        # stay BELOW the ceiling (aabb top) or the shot is just the roof
        # exterior (pilot finding, Full Warehouse).
        ext = np.array(rep["aabb_m"][1]) - np.array(rep["aabb_m"][0])
        alt = float(np.clip(0.7 * max(ext[0], ext[1]), 20.0, 400.0))
        top = float(rep["aabb_m"][1][2])
        cam_z = min(z0 + alt, top - 0.5) if top - 0.5 > z0 + 2.0 else z0 + alt
        cam.set_world_pose(position=np.array([center[0], center[1], cam_z]),
                           orientation=_quat_wxyz(90.0, -90.0))
        for _ in range(30):
            world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            plt.imsave(scene_dir / "thumb_overhead.png", np.asarray(rgba)[..., :3])
    except Exception:
        rep["reasons"].append("FLAG:camera_probe_failed")
        traceback.print_exc()

    world.clear_instance()


def verdict(rep):
    if any(r.startswith("FAIL") for r in rep["reasons"]):
        return "FAIL"
    if any(r.startswith("FLAG") for r in rep["reasons"]):
        return "FLAG"
    return "PASS"


def audit_scene(entry, out_dir):
    if args.negative_control:
        out_dir = out_dir / "_negative_control"
    scene_dir = out_dir / entry["name"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    rep = {"name": entry["name"], "usd": entry["usd"], "reasons": [],
           "audited_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    try:
        geo = geometry_audit(entry, rep, scene_dir)
        if geo is not None and not args.no_drop_test:
            _, _, z0 = geo
            sim_audit(entry, rep, scene_dir, z0)
    except Exception as e:
        rep["reasons"].append(f"FAIL:exception_{type(e).__name__}")
        rep["error"] = traceback.format_exc(limit=8)
        traceback.print_exc()
    rep["audit_time_s"] = round(time.time() - t0, 1)
    rep["verdict"] = verdict(rep)
    (scene_dir / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"[audit] {entry['name']}: {rep['verdict']} {rep['reasons']} "
          f"({rep['audit_time_s']}s)", flush=True)
    return rep


def main():
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.catalog:
        catalog = json.loads(Path(args.catalog).read_text())
    else:
        roots = [] if args.no_nucleus else (args.nucleus or DEFAULT_NUCLEUS)
        catalog = build_catalog(roots, args.isaac_environments)
        (out_dir / "scene_catalog.json").write_text(json.dumps(catalog, indent=1))
        print(f"[inventory] {len(catalog)} scenes -> {out_dir/'scene_catalog.json'}")
    if args.list_only:
        for e in catalog:
            print(f"  {e['name']:40s} {e['usd']}")
        return

    todo = [e for e in catalog
            if not args.only or fnmatch.fnmatch(e["name"], args.only)]
    if args.resume:
        todo = [e for e in todo if not (out_dir / e["name"] / "report.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[audit] {len(todo)} scene(s) to audit")

    reports = []
    for entry in todo:
        reports.append(audit_scene(entry, out_dir))
    # summary over EVERY report present (incl. previous resumed runs)
    all_reports = []
    for d in sorted(out_dir.iterdir()):
        rj = d / "report.json"
        if d.is_dir() and rj.exists():
            all_reports.append(json.loads(rj.read_text()))
    counts = {}
    for r in all_reports:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    (out_dir / "summary.json").write_text(json.dumps(
        {"counts": counts, "n_catalog": len(catalog),
         "scenes": {r["name"]: {"verdict": r["verdict"], "reasons": r["reasons"]}
                    for r in all_reports}}, indent=1))
    print(f"[audit] summary: {counts} -> {out_dir/'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
