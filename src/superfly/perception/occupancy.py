"""Planar occupancy slice + start/goal mining for USD benchmark scenes.

Consumes the surface-sample .npz produced by scripts/extract_scene_mesh.py /
scripts/scene_audit.py (`samples` = lateral obstacle surfaces, ground faces
excluded; `ground_samples` = the ground-like faces, coarser) and mines planar
(constant-altitude) start/goal pairs whose corridor is genuinely blocked but
genuinely flyable -- the scene generalization of scripts/select_seeds.py's
three gates, plus the gate scenes need that procedural fields don't: a
*flyable detour must exist* (A* on the inflated grid).

Gate semantics follow select_seeds.py (MARGIN_M / TRIVIAL_M, body-aware
thresholds) and the clearance metric's ground handling (metrics.py: the
ground is never an obstacle; only lateral geometry is).

Everything here is numpy + scipy only -- it runs on gs2 against fetched
audit npz files; nothing imports Isaac/pxr.

Coordinate convention: grids are indexed [iy, ix]; `origin` is the world xy
of cell (0,0)'s center; world = origin + index * cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# ---- gate thresholds (select_seeds.py lineage; distances are SURFACE
# distances in meters, not radius-subtracted clearances) ---------------------
DRONE_RADIUS_M = 0.2
ENDPOINT_CLEAR_M = 1.5      # select_seeds MARGIN_M: start/goal columns
TRIVIAL_M = 0.5             # straight line with >= this min clearance = too easy
INFLATE_M = 0.4             # A* passable cells: radius + 0.2 margin
PASSAGE_MIN_M = 0.6         # min EDT along the A* path (>= 3x drone radius)

DEFAULT_CELL_M = 0.25
BAND_HALF_M = 0.5           # occupancy slice: z_fly +- this
MAX_GRID_CELLS = 40_000_000  # coarsen cell size beyond this


@dataclass
class PlanarSlice:
    origin: np.ndarray          # (2,) world xy of cell (0,0) center
    cell: float
    occ: np.ndarray             # (H,W) bool -- lateral geometry in the band
    known: np.ndarray           # (H,W) bool -- cell is inside the mapped scene
    col_blocked: np.ndarray     # (H,W) bool -- climb column obstructed
    ground_ok: np.ndarray       # (H,W) bool -- ground within tol of z0
    edt: np.ndarray             # (H,W) float -- distance [m] to nearest occ cell
    z0: float                   # dominant ground elevation
    z_fly: float                # cruise altitude (z0 + climb_alt)
    climb_alt: float
    meta: dict = field(default_factory=dict)

    def world_to_cell(self, xy):
        return np.round((np.asarray(xy, float) - self.origin) / self.cell).astype(int)[::-1]

    def cell_to_world(self, iyx):
        return self.origin + np.asarray(iyx, float)[::-1] * self.cell


def dominant_ground_z(ground_samples: np.ndarray, bin_m: float = 0.25):
    """The scene's dominant ground elevation: mode of the ground-sample z
    histogram (ground_samples include rooftops/ceilings; the *dominant* level
    is the walkable ground). Returns (z0, coverage_fraction_of_ground_samples
    within +-0.5 m of z0)."""
    z = np.asarray(ground_samples[:, 2], float)
    if z.size == 0:
        return None, 0.0
    zmin = float(z.min())
    idx = np.floor((z - zmin) / bin_m).astype(np.int64)
    counts = np.bincount(idx)
    z0 = zmin + (int(np.argmax(counts)) + 0.5) * bin_m
    frac = float(np.mean(np.abs(z - z0) <= 0.5))
    return float(z0), frac


def build_slice(samples: np.ndarray, ground_samples: np.ndarray,
                climb_alt: float = 2.0, cell: float = DEFAULT_CELL_M,
                band_half: float = BAND_HALF_M, ground_tol: float = 0.5,
                z0: float | None = None) -> PlanarSlice | None:
    """Planar occupancy at cruise altitude z0 + climb_alt.

    - occ: lateral samples with z in [z_fly - band_half, z_fly + band_half]
    - known: cells with ground evidence (closed + slightly dilated so sampling
      holes don't fragment the map); A* never leaves the mapped scene
    - col_blocked: any sample above the ground and up to ~1 m over z_fly --
      canopy/ceiling over a spawn blocks the scripted vertical climb
    - ground_ok: cell's lowest ground sample within ground_tol of z0 (planar
      pairs must start and end on the dominant ground level)
    """
    if z0 is None:
        z0, _ = dominant_ground_z(ground_samples)
    if z0 is None:
        return None
    z_fly = z0 + float(climb_alt)

    G = np.asarray(ground_samples, float)
    S = np.asarray(samples, float)
    pts = G[np.abs(G[:, 2] - z0) <= 3.0 * max(ground_tol, 1.0)]  # map extent from ground
    if pts.shape[0] < 10:
        return None
    lo = pts[:, :2].min(0) - 2.0
    hi = pts[:, :2].max(0) + 2.0
    span = hi - lo
    n_cells = (span[0] / cell) * (span[1] / cell)
    if n_cells > MAX_GRID_CELLS:
        cell = float(cell * np.sqrt(n_cells / MAX_GRID_CELLS))
    W = int(np.ceil(span[0] / cell)) + 1
    H = int(np.ceil(span[1] / cell)) + 1

    def _grid_count(P):
        ij = np.floor((P[:, :2] - lo) / cell).astype(np.int64)
        keep = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        ij = ij[keep]
        g = np.zeros((H, W), dtype=np.int32)
        np.add.at(g, (ij[:, 1], ij[:, 0]), 1)
        return g

    in_band = np.abs(S[:, 2] - z_fly) <= band_half
    occ = _grid_count(S[in_band]) > 0

    # climb column: anything (lateral or ground-like, e.g. a ceiling) between
    # just above the ground and just above cruise altitude
    zlo, zhi = z0 + 0.3, z_fly + 1.0
    col = (_grid_count(S[(S[:, 2] > zlo) & (S[:, 2] <= zhi)]) +
           _grid_count(G[(G[:, 2] > zlo) & (G[:, 2] <= zhi)])) > 0

    # ground maps
    g_near = G[np.abs(G[:, 2] - z0) <= ground_tol]
    ground_ok = _grid_count(g_near) > 0
    known = _grid_count(G) > 0
    known = ndimage.binary_closing(known, iterations=2)
    known = ndimage.binary_dilation(known, iterations=1)
    ground_ok = ndimage.binary_closing(ground_ok, iterations=2)

    edt = ndimage.distance_transform_edt(~occ) * cell
    return PlanarSlice(origin=lo.copy(), cell=float(cell), occ=occ, known=known,
                       col_blocked=col, ground_ok=ground_ok, edt=edt,
                       z0=float(z0), z_fly=float(z_fly), climb_alt=float(climb_alt),
                       meta={"H": H, "W": W})


# ---- geometry on the grid -------------------------------------------------- #

def line_cells(a_iyx, b_iyx, step: float = 0.5):
    """Integer (iy, ix) cells sampled along the segment a->b, ~`step` cells apart."""
    a = np.asarray(a_iyx, float)
    b = np.asarray(b_iyx, float)
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
    t = np.linspace(0.0, 1.0, n + 1)
    pts = np.round(a[None, :] + (b - a)[None, :] * t[:, None]).astype(int)
    return pts

def line_min_edt(sl: PlanarSlice, a_iyx, b_iyx) -> float:
    pts = line_cells(a_iyx, b_iyx)
    return float(sl.edt[pts[:, 0], pts[:, 1]].min())


def _grid_graph(passable: np.ndarray, edt: np.ndarray, cell: float):
    """8-connected sparse graph over passable cells (C-speed shortest paths via
    scipy.sparse.csgraph). Edge weight = step length x (1 + wall-hug penalty):
    inside 1.5 m of an obstacle, up to +40% -- routes prefer corridor centers,
    which matches how any competent policy actually flies. Node id = iy*W+ix."""
    from scipy.sparse import csr_matrix

    H, W = passable.shape
    idx = np.arange(H * W).reshape(H, W)
    pen = 1.0 + 0.4 * np.clip(1.0 - edt / 1.5, 0.0, 1.0)
    rows, cols, wts = [], [], []
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        sy, sx = slice(max(0, -dy), H - max(0, dy)), slice(max(0, -dx), W - max(0, dx))
        ty, tx = slice(max(0, dy), H + min(0, dy) or H), slice(max(0, dx), W + min(0, dx) or W)
        a, b = idx[sy, sx].ravel(), idx[ty, tx].ravel()
        m = passable[sy, sx].ravel() & passable[ty, tx].ravel()
        a, b = a[m], b[m]
        ww = w * cell * 0.5 * (pen.ravel()[a] + pen.ravel()[b])
        rows.append(a); cols.append(b); wts.append(ww)
    rows = np.concatenate(rows); cols = np.concatenate(cols); wts = np.concatenate(wts)
    return csr_matrix((wts, (rows, cols)), shape=(H * W, H * W))


def shortest_paths(passable: np.ndarray, edt: np.ndarray, cell: float,
                   sources_iyx: np.ndarray):
    """Dijkstra shortest-path trees from each source cell (C-speed, one tree
    serves every goal). Returns (dist (S, H*W) weighted-m, pred (S, H*W) int32);
    unreachable = inf / -9999."""
    from scipy.sparse.csgraph import dijkstra

    W = passable.shape[1]
    graph = _grid_graph(passable, edt, cell)
    src_ids = sources_iyx[:, 0] * W + sources_iyx[:, 1]
    dist, pred = dijkstra(graph, directed=False, indices=src_ids,
                          return_predecessors=True)
    return np.atleast_2d(dist), np.atleast_2d(pred)


def extract_path(pred_row: np.ndarray, goal_iyx, W: int, max_len: int = 500_000):
    """Cells (N,2) of the tree path source->goal from one predecessor row, or
    None if the goal is unreachable."""
    node = int(goal_iyx[0]) * W + int(goal_iyx[1])
    if pred_row[node] < 0:
        return None
    out = []
    while node >= 0 and len(out) < max_len:
        out.append(node)
        node = int(pred_row[node])
    p = np.asarray(out[::-1], dtype=np.int64)
    return np.stack([p // W, p % W], axis=1)


# ---- mining ---------------------------------------------------------------- #

@dataclass
class Candidate:
    start_xy: tuple
    goal_xy: tuple
    straight_m: float
    path_m: float
    detour: float               # path_m / straight_m
    line_min_edt_m: float       # min surface distance along the straight line
    path_min_edt_m: float       # tightest passage along the A* path
    path_mean_edt_m: float
    blocked_frac: float         # fraction of straight-line samples inside INFLATE_M
    path_cells: np.ndarray      # (N,2) int, for overlap dedupe + plotting
    difficulty: float = np.nan  # filled by score_difficulty (pool-normalized)
    components: dict = field(default_factory=dict)

    def to_json(self):
        d = {k: getattr(self, k) for k in
             ("straight_m", "path_m", "detour", "line_min_edt_m", "path_min_edt_m",
              "path_mean_edt_m", "blocked_frac", "difficulty")}
        d = {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                 else round(float(v), 3)) for k, v in d.items()}
        d["start_xy"] = [round(float(v), 2) for v in self.start_xy]
        d["goal_xy"] = [round(float(v), 2) for v in self.goal_xy]
        d["components"] = {k: round(float(v), 3) for k, v in self.components.items()}
        return d


def endpoint_mask(sl: PlanarSlice) -> np.ndarray:
    """Cells eligible as a start or goal: on dominant ground, inside the map,
    climb column clear, and ENDPOINT_CLEAR_M of lateral clearance at altitude."""
    return (sl.known & sl.ground_ok & ~sl.col_blocked &
            (sl.edt >= ENDPOINT_CLEAR_M))


def mine_pairs(sl: PlanarSlice, n_sources: int = 12, goals_per_source: int = 60,
               seed: int = 0, len_range=(25.0, 80.0), max_keep: int = 400,
               n_candidates: int = None, max_evals: int = None,
               verbose: bool = True):
    """Mine start/goal pairs: pick `n_sources` well-separated endpoint cells,
    grow one Dijkstra shortest-path tree per source (C-speed; serves every
    goal at once), then gate `goals_per_source` random endpoint cells per
    source. Returns (survivors, stats); difficulty is NOT yet normalized --
    call score_difficulty on the pooled list. Deterministic in `seed`.
    (`n_candidates`/`max_evals` kept for API compat; unused.)"""
    rng = np.random.default_rng(seed)
    ok = endpoint_mask(sl)
    iy, ix = np.nonzero(ok)
    stats = {"endpoint_cells": int(iy.size), "sources": 0, "evaluated": 0,
             "line_open": 0, "no_path": 0, "too_tight": 0, "passed": 0}
    if iy.size < 2:
        return [], stats
    passable = sl.known & (sl.edt >= INFLATE_M)
    lmin, lmax = len_range
    lmin_c, lmax_c = lmin / sl.cell, lmax / sl.cell
    cells = np.stack([iy, ix], 1)

    # well-separated sources: greedy accept at >= lmin/3 from those accepted
    order = rng.permutation(cells.shape[0])
    sources = []
    for k in order:
        c = cells[k]
        if all(np.linalg.norm((c - s).astype(float)) >= lmin_c / 3.0 for s in sources):
            sources.append(c)
        if len(sources) >= n_sources:
            break
    sources = np.asarray(sources, int)
    stats["sources"] = int(sources.shape[0])
    if sources.shape[0] == 0:
        return [], stats
    dist, pred = shortest_paths(passable, sl.edt, sl.cell, sources)
    W = passable.shape[1]

    out = []
    for si, A in enumerate(sources):
        d = np.linalg.norm((cells - A).astype(float), axis=1)
        pool = np.nonzero((d >= lmin_c) & (d <= lmax_c))[0]
        for k in rng.permutation(pool)[:goals_per_source]:
            B = cells[k]
            stats["evaluated"] += 1
            # gate 2: the straight line must NOT be safely flyable
            lmin_edt = line_min_edt(sl, A, B)
            if lmin_edt >= TRIVIAL_M:
                stats["line_open"] += 1
                continue
            # gate 3: a flyable detour exists (finite tree distance)
            if not np.isfinite(dist[si, B[0] * W + B[1]]):
                stats["no_path"] += 1
                continue
            path = extract_path(pred[si], B, W)
            if path is None:
                stats["no_path"] += 1
                continue
            p_edt = sl.edt[path[:, 0], path[:, 1]]
            if p_edt.min() < PASSAGE_MIN_M:          # gate 4: passable corridor
                stats["too_tight"] += 1
                continue
            straight_m = float(d[k] * sl.cell)
            lc = line_cells(A, B)
            blocked = sl.edt[lc[:, 0], lc[:, 1]] < INFLATE_M
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            path_m = float(seg.sum() * sl.cell)      # geometric, penalty-free
            out.append(Candidate(
                start_xy=tuple(np.round(sl.cell_to_world(A), 2)),
                goal_xy=tuple(np.round(sl.cell_to_world(B), 2)),
                straight_m=straight_m, path_m=path_m,
                detour=path_m / straight_m,
                line_min_edt_m=float(lmin_edt),
                path_min_edt_m=float(p_edt.min()),
                path_mean_edt_m=float(p_edt.mean()),
                blocked_frac=float(blocked.mean()),
                path_cells=path))
            stats["passed"] += 1
            if len(out) >= max_keep:
                break
        if len(out) >= max_keep:
            break
    if verbose:
        print(f"[mine] {stats}")
    return out, stats


def score_difficulty(cands: list, w_detour: float = 0.4, w_block: float = 0.3,
                     w_tight: float = 0.3):
    """Fill .difficulty in-place over the POOLED candidate list (all scenes
    together, so D is comparable across scenes). Components:
      detour    -- A* length / straight-line length          (>1; capped at 3)
      blockage  -- fraction of the straight line inside obstacles
      tightness -- mean(1/passage width) along the A* path   (capped at 1/0.6)
    Each min-max normalized over the pool. Returns the pool ranges used."""
    if not cands:
        return {}
    detour = np.array([min(c.detour, 3.0) for c in cands])
    block = np.array([c.blocked_frac for c in cands])
    tight = np.array([min(1.0 / max(2 * c.path_mean_edt_m, 1e-6), 1.0 / 0.6)
                      for c in cands])

    def norm(v):
        lo, hi = float(v.min()), float(v.max())
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v), (lo, hi)

    nd, rd = norm(detour)
    nb, rb = norm(block)
    nt, rt = norm(tight)
    D = w_detour * nd + w_block * nb + w_tight * nt
    for c, d, a1, a2, a3 in zip(cands, D, nd, nb, nt):
        c.difficulty = float(d)
        c.components = {"detour_n": float(a1), "blockage_n": float(a2),
                        "tightness_n": float(a3)}
    return {"detour": rd, "blockage": rb, "tightness": rt}


def select_pairs(cands: list, k: int = 3, overlap_max: float = 0.3):
    """Pick up to k pairs from one scene's candidates: spread over the pool
    difficulty (25th/50th/75th percentile targets) with corridors that don't
    overlap (IoU of path cell sets < overlap_max). Returns the picked list."""
    if not cands:
        return []
    ranked = sorted(cands, key=lambda c: c.difficulty)
    targets = np.percentile([c.difficulty for c in ranked],
                            np.linspace(25, 75, k))
    picked = []

    def cells_set(c):
        return set(map(tuple, np.floor(c.path_cells / 8).astype(int)))

    for tgt in targets:
        order = sorted(ranked, key=lambda c: abs(c.difficulty - tgt))
        for c in order:
            if any(c is p for p in picked):
                continue
            cs = cells_set(c)
            if any(len(cs & cells_set(p)) / max(len(cs | cells_set(p)), 1)
                   > overlap_max for p in picked):
                continue
            picked.append(c)
            break
    return picked
