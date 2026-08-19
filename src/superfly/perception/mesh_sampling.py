"""Surface-sampling primitives shared by extract_scene_mesh.py (whole USD
stages) and run_px4_sim.py (the spawned /World/obstacles subtree).

Why this module exists: with --obstacle-assets, each analytic primitive is
replaced by a USD mesh scaled to the primitive's bounding box. A tree that
bbox-matches a cylinder has a thin trunk and a wide canopy, so scoring
clearance against the CYLINDER declares solid a volume the drone can legally
fly through. metrics.py's docstring says the scene mesh is "ignored when the
analytic field is present", which is exactly the asset case -- so asset trials
were being scored against geometry that was never spawned.

Sampling the real spawned prims fixes that at the source.

pxr is imported lazily: both callers only have a USD runtime after Kit has
booted, and this module may be imported before that.
"""
from __future__ import annotations

import sys

import numpy as np


# PointInstancer expansion is capped so a leaf-instanced forest cannot OOM the
# extractor: past this many instanced triangles per instancer, instances are
# strided evenly (logged). Clearance then slightly UNDER-covers that canopy.
MAX_INSTANCED_TRIS = 30_000_000
# ...and capped ACROSS instancers too: DerelicitCorridor has ~30 pipe
# instancers of ~31M triangles each -- individually under the cap after
# striding, together ~450M and an OOM kill (sweep 2026-08-18). The budget is
# shared over the whole gather; once spent, later instancers stride harder.
MAX_TOTAL_INSTANCED_TRIS = 60_000_000
_MAX_INSTANCER_DEPTH = 3


def _fan_triangulate(counts, idx):
    tri, pos = [], 0
    for c in counts:
        if c >= 3:
            fan = idx[pos:pos + c]
            for k in range(1, c - 1):
                tri.append((fan[0], fan[k], fan[k + 1]))
        pos += c
    return tri


def _mesh_local_triangles(prim, check_visibility=True):
    """(P_local (N,3), tri (M,3)) of one Mesh prim in its own prim frame, or
    None if skipped/empty. Purpose is always enforced; visibility only when
    `check_visibility` (PointInstancer prototype scopes are conventionally
    authored invisible so they don't ALSO render at the origin -- their
    geometry is still instanced, so prototype subtrees skip that check)."""
    from pxr import UsdGeom

    img = UsdGeom.Imageable(prim)
    if check_visibility and img.ComputeVisibility() == UsdGeom.Tokens.invisible:
        return None
    if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
        return None
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        return None
    tri = _fan_triangulate(np.asarray(counts, dtype=np.int64),
                           np.asarray(idx, dtype=np.int64))
    if not tri:
        return None
    return np.asarray(pts, dtype=np.float64), np.asarray(tri, dtype=np.int64)


def _instance_xforms(pi, time):
    """Instance->instancer-local transforms INCLUDING the proto root xform,
    across USD binding generations: ComputeInstanceTransformsAtTime (current),
    ComputeInstanceTransforms (newer), else manual composition from
    positions/orientations/scales (+ proto root local xform in the caller's
    frame convention is NOT needed -- manual path composes it per prototype).
    Returns a list of Gf.Matrix4d-compatible 4x4s (row-vector convention)."""
    from pxr import Gf, UsdGeom

    for name in ("ComputeInstanceTransformsAtTime", "ComputeInstanceTransforms"):
        fn = getattr(pi, name, None)
        if fn is None:
            continue
        try:
            xf = fn(time, time)
            if xf is not None:
                return list(xf)
        except Exception:
            continue
    # manual: T = S * R * T_pos per instance, then proto-root local xform
    pos = pi.GetPositionsAttr().Get(time)
    if pos is None:
        return []
    n = len(pos)
    ori = pi.GetOrientationsAttr().Get(time)
    scl = pi.GetScalesAttr().Get(time)
    proto_idx = np.asarray(pi.GetProtoIndicesAttr().Get(time), dtype=np.int64)
    stage = pi.GetPrim().GetStage()
    proto_local = {}
    for i, tgt in enumerate(pi.GetPrototypesRel().GetTargets()):
        root = stage.GetPrimAtPath(tgt)
        m = Gf.Matrix4d(1.0)
        if root and root.IsValid():
            m = UsdGeom.Xformable(root).GetLocalTransformation(time)
        proto_local[i] = m
    out = []
    for k in range(n):
        m = Gf.Matrix4d(1.0)
        if scl is not None and k < len(scl):
            m = m * Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(*scl[k]))
        if ori is not None and k < len(ori):
            q = ori[k]
            m = m * Gf.Matrix4d(1.0).SetRotate(
                Gf.Quatd(q.GetReal(), Gf.Vec3d(*q.GetImaginary())))
        m = m * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*pos[k]))
        ip = int(proto_idx[k]) if k < len(proto_idx) else 0
        out.append(proto_local.get(ip, Gf.Matrix4d(1.0)) * m)
    return out


def _instancer_triangles(pi_prim, verbose=True, _depth=0,
                         cap=MAX_INSTANCED_TRIS):
    """World-frame (verts_list, tris_list) for every instance of a
    PointInstancer. Nested instancers inside prototypes recurse up to
    _MAX_INSTANCER_DEPTH. Row-vector convention throughout:
    p_world = p_proto @ X_instance @ M_instancer_world."""
    from pxr import Usd, UsdGeom

    if _depth >= _MAX_INSTANCER_DEPTH:
        if verbose:
            print(f"[mesh_sampling] instancer nesting deeper than "
                  f"{_MAX_INSTANCER_DEPTH} at {pi_prim.GetPath()}; skipped.",
                  file=sys.stderr)
        return [], []
    pi = UsdGeom.PointInstancer(pi_prim)
    if UsdGeom.Imageable(pi_prim).ComputeVisibility() == UsdGeom.Tokens.invisible \
            and _depth == 0:
        return [], []
    targets = pi.GetPrototypesRel().GetTargets()
    proto_idx = pi.GetProtoIndicesAttr().Get()
    if not targets or proto_idx is None or len(proto_idx) == 0:
        return [], []
    time = Usd.TimeCode.Default()
    xforms = _instance_xforms(pi, time)                 # incl. proto root xform
    if xforms is None or len(xforms) == 0:
        return [], []
    proto_idx = np.asarray(proto_idx, dtype=np.int64)
    M_pi = np.asarray(
        UsdGeom.Xformable(pi_prim).ComputeLocalToWorldTransform(time),
        dtype=np.float64)
    stage = pi_prim.GetStage()

    # Prototype geometry in the frame BELOW the proto root's own xform op
    # (ComputeInstanceTransforms already includes the proto root xform).
    protos = []          # per prototype: (verts_local, tris) pairs + tri count
    for tgt in targets:
        root = stage.GetPrimAtPath(tgt)
        pieces, n_tri = [], 0
        if root and root.IsValid():
            M_root = np.asarray(
                UsdGeom.Xformable(root).ComputeLocalToWorldTransform(time),
                dtype=np.float64)
            M_root_inv = np.linalg.inv(M_root)
            rng = Usd.PrimRange(
                root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
            for prim in rng:
                if prim != root and prim.IsA(UsdGeom.PointInstancer):
                    # nested instancer: expand in world, pull back to proto frame
                    nv, nf = _instancer_triangles(prim, verbose, _depth + 1, cap)
                    for V, F in zip(nv, nf):
                        Vl = V @ M_root_inv[:3, :3] + M_root_inv[3, :3]
                        pieces.append((Vl, F))
                        n_tri += len(F)
                    continue
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                got = _mesh_local_triangles(prim, check_visibility=False)
                if got is None:
                    continue
                P, tri = got
                M_rel = np.asarray(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time),
                    dtype=np.float64) @ M_root_inv
                pieces.append((P @ M_rel[:3, :3] + M_rel[3, :3], tri))
                n_tri += len(tri)
        protos.append((pieces, n_tri))

    tris_per_inst = np.array([protos[i][1] if 0 <= i < len(protos) else 0
                              for i in proto_idx], dtype=np.int64)
    total = int(tris_per_inst.sum())
    stride = 1
    cap = max(int(cap), 100_000)     # never stride down to literally nothing
    if total > cap:
        stride = int(np.ceil(total / cap))
        if verbose:
            print(f"[mesh_sampling] {pi_prim.GetPath()}: {total} instanced "
                  f"triangles > cap {cap}; keeping every "
                  f"{stride}th instance.", file=sys.stderr)
    verts_out, tris_out = [], []
    for n, (i_proto, X) in enumerate(zip(proto_idx, xforms)):
        if n % stride or not (0 <= i_proto < len(protos)):
            continue
        pieces, n_tri = protos[i_proto]
        if not n_tri:
            continue
        M = np.asarray(X, dtype=np.float64) @ M_pi
        for P, tri in pieces:
            verts_out.append(P @ M[:3, :3] + M[3, :3])
            tris_out.append(tri)
    return verts_out, tris_out


def gather_triangles(stage, root_path: str | None = None, verbose: bool = True):
    """Triangles of visible, default/render-purpose UsdGeom.Mesh prims in the
    composed world frame. Restricted to the `root_path` subtree when given.

    Returns (V (N,3) float64, F (M,3) int64). Instance proxies are traversed so
    instanced vegetation/props are captured, and PointInstancer prototypes are
    expanded per instance (v2: previously skipped with a warning -- EnglishCollege
    tree canopy was absent from clearance scoring, see ATTEMPTS 2026-07-29).
    """
    from pxr import Usd, UsdGeom

    if root_path is None:
        rng = Usd.PrimRange.Stage(
            stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    else:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            if verbose:
                print(f"[mesh_sampling] no prim at {root_path}", file=sys.stderr)
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        rng = Usd.PrimRange(
            root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))

    verts, faces, v_off = [], [], 0
    n_mesh = n_skipped = n_instancers = 0
    inst_budget = MAX_TOTAL_INSTANCED_TRIS      # shared across ALL instancers
    it = iter(rng)
    for prim in it:
        if prim.IsA(UsdGeom.PointInstancer):
            n_instancers += 1
            iv, itr = _instancer_triangles(
                prim, verbose, cap=min(MAX_INSTANCED_TRIS, inst_budget))
            for P, tri in zip(iv, itr):
                verts.append(P)
                faces.append(tri + v_off)
                v_off += P.shape[0]
                inst_budget -= tri.shape[0]
            # prototypes live under the instancer; don't ALSO walk them as
            # plain (origin-frame) meshes
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        got = _mesh_local_triangles(prim, check_visibility=True)
        if got is None:
            n_skipped += 1
            continue
        P, tri = got
        # Gf matrices are row-vector convention: p_world = p_local @ M[:3,:3] + M[3,:3]
        M = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64)
        verts.append(P @ M[:3, :3] + M[3, :3])
        faces.append(tri + v_off)
        v_off += P.shape[0]
        n_mesh += 1

    if n_instancers and verbose:
        print(f"[mesh_sampling] {n_instancers} PointInstancer prim(s) expanded "
              "into instanced geometry (extractor v2).", file=sys.stderr)
    if verbose:
        print(f"[mesh_sampling] {n_mesh} mesh prims used, {n_skipped} skipped "
              f"(invisible/guide/proxy){'' if root_path is None else f' under {root_path}'}.")
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(verts), np.concatenate(faces)


def _max_edge(V, F):
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.maximum.reduce([np.linalg.norm(B - A, axis=1),
                              np.linalg.norm(C - B, axis=1),
                              np.linalg.norm(A - C, axis=1)])


def sample_surface(V, F, h):
    """Surface samples with spacing <= h, so any surface point is within ~h of a
    sample: triangles smaller than h get their centroid, larger ones a
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
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        keep = (i + j) <= kk
        u = (i[keep] / kk).astype(np.float64)
        v = (j[keep] / kk).astype(np.float64)
        w = 1.0 - u - v
        pts = (a[:, None, :] * w[None, :, None] +
               b[:, None, :] * u[None, :, None] +
               c[:, None, :] * v[None, :, None])
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


def triangle_normals_areas(V, F):
    """Unit face normals (M,3) + areas (M,) of a triangle soup. Degenerate
    faces get nan normals and ~0 area (filter on area > eps before using the
    normals). Same math as extract_scene_mesh.py's local copy."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    n = np.cross(e1, e2)
    nlen = np.linalg.norm(n, axis=1)
    area = 0.5 * nlen
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nlen[:, None]
    return n, area


def split_ground_faces(V, F, ground_deg: float = 30.0):
    """(lateral_mask, ground_mask) over F, using the exact semantics the
    clearance metric is documented with (docs/comparison_harness.md): faces
    within `ground_deg` of horizontal (|normal_z| >= cos(ground_deg) --
    terrain, floors, and also rooftops/ceilings) are 'ground-like' and
    excluded from clearance; everything else is 'lateral' obstacle surface
    (walls, trees, poles, facades). Degenerate faces land in neither mask."""
    normals, area = triangle_normals_areas(V, F)
    ok = area > 1e-12
    ground_like = np.abs(normals[:, 2]) >= float(np.cos(np.radians(ground_deg)))
    return ok & ~ground_like, ok & ground_like


def estimated_samples(V, F, h):
    """Predicted sample_surface count at spacing h (same formula as
    extract_scene_mesh.py) -- used to coarsen h to a sample budget."""
    max_edge = _max_edge(V, F)
    tiny = max_edge <= h
    k = np.ceil(max_edge[~tiny] / h)
    return float(tiny.sum()) + float(((k + 1) * (k + 2) / 2).sum())


def dedupe_samples(S, cell):
    """Collapse samples to one per `cell`-sized voxel."""
    if S.shape[0] == 0:
        return S
    q = np.floor(S / cell).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return S[np.sort(idx)]


def sample_subtree(stage, root_path: str, sample_h: float = 0.05,
                   dedupe_cell: float | None = None, verbose: bool = True):
    """Convenience: gather + sample + dedupe one subtree.

    Returns (samples (N,3) float32, meta dict). No ground filtering -- the
    caller scopes this to obstacle prims, and an obstacle's horizontal faces
    (crate lids, canopy tops) are as solid as its vertical ones.
    """
    V, F = gather_triangles(stage, root_path, verbose=verbose)
    if F.shape[0] == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                {"n_tris": 0, "sample_h": sample_h, "root": root_path})
    S = sample_surface(V, F, sample_h)
    if dedupe_cell is None:
        dedupe_cell = sample_h
    S = dedupe_samples(S, dedupe_cell)
    if verbose:
        print(f"[mesh_sampling] {root_path}: {F.shape[0]} tris -> "
              f"{S.shape[0]} samples at h={sample_h}")
    return S.astype(np.float32), {"n_tris": int(F.shape[0]),
                                  "sample_h": float(sample_h),
                                  "dedupe_cell": float(dedupe_cell),
                                  "root": root_path}
