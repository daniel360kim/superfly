#!/usr/bin/env python
"""
Extract the collision-relevant surface of a USD environment into a point-sample
.npz that compare/metrics.py can score trajectories against (min clearance /
collision for scenes where there is no analytic obstacle field).

Must run under Isaac Sim's python (e.g. ~/isaacsim/python.sh): opening an
omniverse:// stage needs the Nucleus resolver, which is only registered once
Kit boots -- so this script boots a headless SimulationApp first, exactly like
run_px4_sim.py. run_comparison.py invokes it automatically (with --sim-python)
after any USD-environment trial and caches the result per (usd, scale, params),
so the ~1 min Kit boot is paid once per scene, not per trial.

Geometry semantics (must mirror how run_px4_sim.py places the scene):
run_px4_sim references the USD under /World/layout and applies a single uniform
env_scale on that prim -- no offset, no rotation, no metersPerUnit conversion.
So world-frame geometry = (stage-composed world points) * env_scale, which is
what this script computes.

Ground filtering: the analytic-field clearance metric never counts the ground
(the field has no ground shape, and the drone legitimately touches it at
takeoff/landing). To match, near-horizontal faces -- |normal_z| >= cos(
--ground-deg), i.e. terrain, floors, and also rooftops/ceilings -- are dropped
before sampling; clearance is measured to the remaining "lateral" geometry
(walls, trees, poles, facades). Distances to the sampled surface OVERESTIMATE
the true surface distance by at most ~--sample-h, so keep h small relative to
the drone radius.

CLI:
    ~/isaacsim/python.sh extract_scene_mesh.py <usd-url> --env-scale 0.01 \
        --out mesh_cache/english_college.npz [--sample-h 0.05] [--ground-deg 30]
"""

import argparse
import json
import sys
from pathlib import Path

# SimulationApp must boot before pxr/omni imports (same pattern as run_px4_sim.py).
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("usd", help="USD stage URL/path (omniverse:// or local file)")
parser.add_argument("--env-scale", type=float, default=1.0,
                    help="Uniform scale run_px4_sim applied to the stage (--env-scale).")
parser.add_argument("--out", required=True, help="Output .npz path.")
parser.add_argument("--sample-h", type=float, default=0.05,
                    help="Surface sample spacing [m] AFTER scaling. Clearance "
                         "overestimates by at most ~h (default 0.05).")
parser.add_argument("--ground-deg", type=float, default=30.0,
                    help="Faces within this tilt of horizontal (|normal_z| >= "
                         "cos(ground_deg)) are treated as ground-like (terrain/"
                         "floors/rooftops) and excluded from clearance (default 30).")
parser.add_argument("--bounds", type=float, nargs=6, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                    help="World-frame crop box [m] (after env_scale): only geometry "
                         "whose AABB intersects it is sampled. Essential for large "
                         "scenes -- pass the flight corridor plus a generous margin, "
                         "since clearance to geometry outside the box is unmeasured.")
parser.add_argument("--max-samples", type=int, default=20_000_000,
                    help="Cap on total surface samples; h is coarsened to fit.")
args = parser.parse_args()

# Optional non-interactive Nucleus auth via an Omniverse Navigator API token
# (OMNI_API_TOKEN env var) -- no-op if unset, so existing username/password /
# interactive-login paths are untouched. Must be configured BEFORE Kit boots:
# omni.client reads OMNI_USER/OMNI_PASS at init, and the
# register_authentication_callback route used previously does NOT work against
# airlab-nucleus (server pushes browser SSO; verified 2026-08-18).
import os                                              # noqa: E402
if not os.environ.get("OMNI_API_TOKEN"):
    _envf = Path.home() / ".omni_env"                  # same fallback as scene_audit.py
    if _envf.exists():
        for _line in _envf.read_text().splitlines():
            _line = _line.strip().removeprefix("export ").strip()
            if _line.startswith("OMNI_API_TOKEN="):
                os.environ["OMNI_API_TOKEN"] = _line.split("=", 1)[1].strip().strip("'\"")
                break
if os.environ.get("OMNI_API_TOKEN") and not os.environ.get("OMNI_USER"):
    os.environ["OMNI_USER"] = "$omni-api-token"
    os.environ["OMNI_PASS"] = os.environ["OMNI_API_TOKEN"]

from isaacsim import SimulationApp                     # noqa: E402
simulation_app = SimulationApp({"headless": True})

import numpy as np                                     # noqa: E402
from pxr import Usd, UsdGeom, Gf                       # noqa: E402

# v2: PointInstancer prototypes are expanded (mesh_sampling) -- EnglishCollege
# tree canopy was invisible to v1 clearance scoring (ATTEMPTS 2026-07-29).
EXTRACTOR_VERSION = 2

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from superfly.perception import mesh_sampling          # noqa: E402


def gather_world_triangles(stage):
    """All triangles of visible, default/render-purpose UsdGeom.Mesh prims, in
    the stage's composed world frame (instance proxies traversed AND
    PointInstancer prototypes expanded per instance, so instanced vegetation
    is captured). Returns (V (N,3) float64, F (M,3) int64)."""
    return mesh_sampling.gather_triangles(stage, root_path=None, verbose=True)


def triangle_normals_areas(V, F):
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(e1, e2)
    nlen = np.linalg.norm(n, axis=1)
    area = 0.5 * nlen
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nlen[:, None]
    return n, area


def _max_edge(V, F):
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.maximum.reduce([np.linalg.norm(B - A, axis=1),
                              np.linalg.norm(C - B, axis=1),
                              np.linalg.norm(A - C, axis=1)])


def sample_surface(V, F, h):
    """Surface samples with spacing <= h, so any surface point is within ~h of a
    sample: triangles smaller than h get their centroid (dense scan/vegetation
    meshes have millions of sub-h faces; 1 point each, not 3), larger ones a
    barycentric lattice. Grouped by subdivision level to stay vectorized."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    max_edge = _max_edge(V, F)
    out = []
    tiny = max_edge <= h
    if tiny.any():
        out.append((A[tiny] + B[tiny] + C[tiny]) / 3.0)
    k = np.ceil(max_edge / h).astype(np.int64)
    for kk in np.unique(k[~tiny]):
        sel = ~tiny & (k == kk)
        a, b, c = A[sel], B[sel], C[sel]
        # lattice (i, j): i + j <= kk; barycentric (1 - u - v, u, v), u=i/kk, v=j/kk
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        keep = (i + j) <= kk
        u = (i[keep] / kk).astype(np.float64)     # (L,)
        v = (j[keep] / kk).astype(np.float64)
        w = 1.0 - u - v
        # (T,1,3)*(1,L,1) sums -> (T,L,3)
        pts = (a[:, None, :] * w[None, :, None] +
               b[:, None, :] * u[None, :, None] +
               c[:, None, :] * v[None, :, None])
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


def estimated_samples(V, F, h):
    max_edge = _max_edge(V, F)
    tiny = max_edge <= h
    k = np.ceil(max_edge[~tiny] / h)
    return float(tiny.sum()) + float(((k + 1) * (k + 2) / 2).sum())


def dedupe_samples(S, cell):
    """Collapse samples to one per `cell`-sized voxel (dense scan meshes place
    many sub-h faces in the same spot; their centroids are redundant)."""
    q = np.floor(S / cell).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return S[np.sort(idx)]


def main():
    print(f"[extract] opening stage {args.usd} ...")
    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise SystemExit(f"[extract] could not open stage {args.usd}")

    V, F = gather_world_triangles(stage)
    if F.shape[0] == 0:
        raise SystemExit("[extract] no mesh geometry found in the stage.")
    V = V * args.env_scale
    print(f"[extract] {F.shape[0]} triangles, bounds min={V.min(0).round(1)} "
          f"max={V.max(0).round(1)} (after env_scale={args.env_scale})")

    if args.bounds is not None:
        lo = np.asarray(args.bounds[:3]); hi = np.asarray(args.bounds[3:])
        tv = V[F]                                       # (M,3,3)
        inside = ((tv.max(axis=1) >= lo) & (tv.min(axis=1) <= hi)).all(axis=1)
        F = F[inside]
        print(f"[extract] crop box {lo.tolist()}..{hi.tolist()} keeps "
              f"{F.shape[0]} triangles")
        if F.shape[0] == 0:
            raise SystemExit("[extract] no geometry inside --bounds; wrong box?")

    normals, area = triangle_normals_areas(V, F)
    ok = area > 1e-12                                   # drop degenerate faces
    ground_cos = float(np.cos(np.radians(args.ground_deg)))
    ground_like = np.abs(normals[:, 2]) >= ground_cos
    keep = ok & ~ground_like
    print(f"[extract] ground-like faces excluded: {int((ok & ground_like).sum())} "
          f"({area[ok & ground_like].sum():.0f} m^2); "
          f"obstacle faces kept: {int(keep.sum())} ({area[keep].sum():.0f} m^2)")
    if not keep.any():
        raise SystemExit("[extract] every face was filtered as ground-like; "
                         "raise --ground-deg or check the stage.")

    h = args.sample_h
    est = estimated_samples(V, F[keep], h)
    if est > args.max_samples:
        h = h * float(np.sqrt(est / args.max_samples))
        print(f"[extract] {est:.2e} samples at h={args.sample_h} exceeds "
              f"--max-samples; coarsening to h={h:.3f} (clearance accuracy "
              f"degrades accordingly -- consider a tighter --bounds)")
    samples = sample_surface(V, F[keep], h)
    n_raw = samples.shape[0]
    samples = dedupe_samples(samples, h / 2.0).astype(np.float32)
    print(f"[extract] {samples.shape[0]} surface samples at h={h:.3f} m "
          f"({n_raw} before voxel dedupe)")

    # Ground channel (2026-08-06): the dropped ground-like faces, sampled at
    # a coarser spacing into a SEPARATE array. Clearance/mining consumers of
    # `samples` are unchanged; ground-aware consumers (corridor
    # qualification ESDF, true-AGL checks) opt in by reading
    # `ground_samples`. Rooftops/ceilings are ground-like too — the
    # consumer decides how to treat them.
    ground_keep = ok & ground_like
    ground_samples = np.zeros((0, 3), dtype=np.float32)
    gh = max(h * 2.0, 0.2)
    if ground_keep.any():
        g_est = estimated_samples(V, F[ground_keep], gh)
        if g_est > args.max_samples // 2:
            gh = gh * float(np.sqrt(g_est / (args.max_samples // 2)))
        ground_samples = dedupe_samples(
            sample_surface(V, F[ground_keep], gh), gh / 2.0
        ).astype(np.float32)
    print(f"[extract] {ground_samples.shape[0]} ground samples at "
          f"gh={gh:.3f} m")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta = dict(usd=args.usd, env_scale=args.env_scale, sample_h=h,
                requested_sample_h=args.sample_h, ground_deg=args.ground_deg,
                bounds=args.bounds, ground_sample_h=gh,
                n_triangles=int(F.shape[0]), n_obstacle_triangles=int(keep.sum()),
                extractor_version=EXTRACTOR_VERSION)
    np.savez_compressed(args.out, samples=samples,
                        ground_samples=ground_samples, meta=json.dumps(meta))
    print(f"[extract] saved {args.out}")


if __name__ == "__main__":
    # simulation_app.close() may os._exit before an exception propagates to the
    # default excepthook, silently eating tracebacks -- print them first.
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
