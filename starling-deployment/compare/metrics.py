#!/usr/bin/env python
"""
Offline scoring for a single flight logged by run_px4_sim.py --log-traj.

run_px4_sim dumps an .npz holding the drone's ground-truth ENU trajectory plus
the analytic obstacle field and goal/start (see run_px4_sim._save_trajectory).
This module scores that run on the three metric families the comparison harness
reports -- all pure NumPy, no Isaac/torch, so it runs anywhere and is unit-testable:

  - success rate     : reached the goal (within goal_radius) AND never collided
  - collision/clear  : collided (clearance < 0 at any tick) + minimum clearance,
                       clearance = (distance from drone centre to nearest obstacle
                       surface) - drone_radius, via per-shape signed-distance fns
  - time & speed     : time-to-goal, mean and peak speed over the flight

Obstacle geometry matches what run_px4_sim spawns from obstacle_field.ObstacleField:
  spheres (cx,cy,cz,r) | boxes (cx,cy,cz,hx,hy,hz[,roll,pitch,yaw]) |
  vertical cylinders (cx,cy,r), axis +Z, treated as spanning the flight band |
  horizontal cylinders (cx,cy,cz,r), axis +X, finite length CYLH_LEN.

For USD-scene scenarios there is no analytic field in the log; pass a scene
mesh .npz produced by compare/extract_scene_mesh.py (surface point samples of
the scene's non-ground geometry) and clearance/collision are scored against it
instead: clearance = (distance to nearest surface sample) - drone_radius,
measured from takeoff onward (the parked prefix sits legitimately ON the scene).
Sampled distances overestimate the true surface distance by at most the mesh's
sample spacing h (stored in the npz), so clearances are accurate to ~h.

CLI:
    python metrics.py run.npz [--drone-radius 0.2] [--goal-radius 1.0] \
        [--scene-mesh mesh_cache/english_college.npz]
"""

import argparse
import json
import math

import numpy as np

# Geometry constants must match run_px4_sim.py's obstacle spawning.
CYLH_LEN = 6.0   # horizontal-cylinder length [m] (run_px4_sim CYLH_LEN)

INF = float("inf")


# --------------------------------------------------------------------------- #
# Per-shape signed distance from a set of points to the obstacle SURFACE.
# Each returns an (N,) array; negative => the point is inside the solid.
# --------------------------------------------------------------------------- #
def _sdf_spheres(P, spheres):
    """P: (N,3). spheres: (M,4) = cx,cy,cz,r. Returns (N,) min surface distance."""
    if len(spheres) == 0:
        return np.full(P.shape[0], INF)
    c = spheres[:, :3]                      # (M,3)
    r = spheres[:, 3]                       # (M,)
    d = np.linalg.norm(P[:, None, :] - c[None, :, :], axis=2) - r[None, :]  # (N,M)
    return d.min(axis=1)


def _euler_xyz_matrix(roll, pitch, yaw):
    """Intrinsic XYZ-euler rotation matrix (radians); matches obstacle_field."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rx @ ry @ rz


def _sdf_box_single(P, box):
    """Signed distance to one (possibly rotated) box. box: 6- or 9-vector."""
    c = np.asarray(box[:3], float)
    h = np.asarray(box[3:6], float)
    q = P - c                                # (N,3) into box-centre frame
    if len(box) >= 9:
        R = _euler_xyz_matrix(box[6], box[7], box[8])
        q = q @ R                            # world->box: R^T @ q == q @ R
    d = np.abs(q) - h                        # (N,3)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


def _sdf_boxes(P, boxes):
    if len(boxes) == 0:
        return np.full(P.shape[0], INF)
    return np.min(np.stack([_sdf_box_single(P, b) for b in boxes], axis=1), axis=1)


def _sdf_cyl_v(P, cyl_v):
    """Vertical cylinders (axis +Z), treated as infinite over the flight band.
    cyl_v: (M,3) = cx,cy,r. Distance uses only the XY radial offset."""
    if len(cyl_v) == 0:
        return np.full(P.shape[0], INF)
    c = cyl_v[:, :2]                         # (M,2)
    r = cyl_v[:, 2]
    rad = np.linalg.norm(P[:, None, :2] - c[None, :, :], axis=2) - r[None, :]
    return rad.min(axis=1)


def _sdf_cyl_h(P, cyl_h):
    """Horizontal cylinders (axis +X, finite length CYLH_LEN).
    cyl_h: (M,4) = cx,cy,cz,r. Finite-cylinder SDF (capsule-style end caps)."""
    if len(cyl_h) == 0:
        return np.full(P.shape[0], INF)
    out = np.full(P.shape[0], INF)
    half = CYLH_LEN / 2.0
    for cx, cy, cz, r in cyl_h:
        dx = np.abs(P[:, 0] - cx) - half            # axial overrun (>0 past caps)
        rr = np.hypot(P[:, 1] - cy, P[:, 2] - cz)   # radial offset in YZ
        side = rr - r
        inside_axial = dx <= 0.0
        d = np.where(inside_axial, side,
                     np.where(rr <= r, dx, np.hypot(np.maximum(dx, 0.0),
                                                    np.maximum(side, 0.0))))
        out = np.minimum(out, d)
    return out


def clearance_along_traj(P, field):
    """Min distance from each trajectory point to ANY obstacle surface. (N,)."""
    return np.minimum.reduce([
        _sdf_spheres(P, field["spheres"]),
        _sdf_boxes(P, field["boxes"]),
        _sdf_cyl_v(P, field["cyl_v"]),
        _sdf_cyl_h(P, field["cyl_h"]),
    ])


def _load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    return out


# --------------------------------------------------------------------------- #
# Scene-mesh clearance (USD environments, no analytic field): distance from
# trajectory points to the nearest surface sample of the scene geometry
# (extract_scene_mesh.py). scipy's cKDTree when available; otherwise a chunked
# brute-force fallback so metrics.py stays runnable anywhere.
# --------------------------------------------------------------------------- #
def load_scene_mesh(mesh_npz):
    """Returns (samples (N,3) float32, meta dict) from an extract_scene_mesh npz."""
    z = np.load(mesh_npz)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return np.asarray(z["samples"], dtype=np.float32), meta


def _nearest_sample_dist(P, samples):
    """Min distance from each point in P (N,3) to the sample cloud (M,3). (N,)."""
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(samples).query(np.asarray(P, np.float32), workers=-1)
        return np.asarray(d, np.float64)
    except ImportError:
        d = np.full(P.shape[0], INF)
        P32 = np.asarray(P, np.float32)
        chunk = max(1, int(2e7) // max(P.shape[0], 1))
        for i in range(0, samples.shape[0], chunk):
            s = samples[i:i + chunk]
            dd = np.linalg.norm(P32[:, None, :] - s[None, :, :], axis=2).min(axis=1)
            d = np.minimum(d, dd)
        return d


# Takeoff detection: the log starts at sim-loop start, ~45 s before the offboard
# script even launches (run_comparison --warmup), so the drone sits parked at
# the spawn for a long prefix. "Takeoff" = first sample of a sustained run of
# TAKEOFF_HOLD_N consecutive samples above TAKEOFF_SPEED_MPS (a single-sample
# threshold would trigger on physics-settling jitter at spawn).
TAKEOFF_SPEED_MPS = 0.3
TAKEOFF_HOLD_N = 5


def takeoff_index(speed):
    """Index of the first sustained-motion sample, or None if the drone never
    moved (e.g. PX4 was down and the trial logged a parked drone)."""
    moving = speed > TAKEOFF_SPEED_MPS
    if moving.size < TAKEOFF_HOLD_N:
        return None
    sustained = np.convolve(moving.astype(int),
                            np.ones(TAKEOFF_HOLD_N, dtype=int), "valid") == TAKEOFF_HOLD_N
    idx = np.argmax(sustained)
    return int(idx) if sustained[idx] else None


def score_trajectory(npz_path, drone_radius=0.2, goal_radius=1.0, scene_mesh=None):
    """Score one logged flight. Returns a flat dict of metrics (JSON-friendly).

    scene_mesh: optional path to an extract_scene_mesh.py .npz; used for
    clearance/collision when the log carries no analytic obstacle field
    (USD-scene scenarios). Ignored when the analytic field is present."""
    z = _load(npz_path)
    traj = np.asarray(z["traj"], dtype=np.float64)
    method = str(z["policy"]) if "policy" in z else "unknown"
    seed = int(z["seed"]) if "seed" in z else -1

    res = {"method": method, "seed": seed, "n_poses": int(traj.shape[0])}
    if traj.shape[0] == 0:
        res.update(success=False, reached=False, collided=False,
                   duration_s=0.0, flight_time_s=None, t_takeoff_s=None,
                   min_clearance_m=None, min_dist_to_goal_m=None,
                   time_to_goal_s=None, mean_speed_mps=0.0, peak_speed_mps=0.0)
        return res

    t = traj[:, 0]
    P = traj[:, 1:4]
    V = traj[:, 4:7]
    speed = np.linalg.norm(V, axis=1)
    res["duration_s"] = float(t[-1] - t[0])   # full log, incl. pre-offboard warmup

    # --- takeoff: strip the parked prefix (sim warmup + arming) from the
    # time/speed metrics; collision & goal checks still use the whole log ---
    tko = takeoff_index(speed)
    res["t_takeoff_s"] = float(t[tko] - t[0]) if tko is not None else None
    res["flight_time_s"] = float(t[-1] - t[tko]) if tko is not None else None

    # --- collision / clearance (only if a field was logged) ---
    have_field = all(k in z for k in ("spheres", "boxes", "cyl_v", "cyl_h"))
    if have_field:
        field = {k: np.asarray(z[k], float).reshape(-1, n)
                 for k, n in (("spheres", 4), ("boxes", z["boxes"].shape[1] if z["boxes"].size else 6),
                              ("cyl_v", 3), ("cyl_h", 4))}
        clr = clearance_along_traj(P, field) - drone_radius
        res["min_clearance_m"] = float(np.min(clr))
        res["collided"] = bool(np.any(clr < 0.0))
        res["clearance_source"] = "analytic_field"
    elif scene_mesh is not None:
        # USD-scene scenario: score against the extracted scene geometry.
        # Only the post-takeoff segment counts -- the parked drone sits ON the
        # scene surface, so pre-takeoff proximity is not an avoidance failure.
        samples, mesh_meta = load_scene_mesh(scene_mesh)
        if tko is not None and samples.shape[0]:
            clr = _nearest_sample_dist(P[tko:], samples) - drone_radius
            res["min_clearance_m"] = float(np.min(clr))
            res["collided"] = bool(np.any(clr < 0.0))
        else:
            res["min_clearance_m"] = None
            res["collided"] = False
        res["clearance_source"] = "scene_mesh"
        res["scene_mesh_sample_h"] = mesh_meta.get("sample_h")
        # The mesh is cropped to the planned flight corridor; if the drone
        # strayed within 10 m of (or past) the crop faces, clearance near
        # those samples may be against missing geometry -- flag it.
        if mesh_meta.get("bounds") and tko is not None:
            b = np.asarray(mesh_meta["bounds"], float)
            outside = np.any((P[tko:] < b[:3] + 10.0) | (P[tko:] > b[3:] - 10.0),
                             axis=1)
            res["clearance_bounds_exceeded"] = bool(outside.any())
    else:
        # No analytic obstacle field in the log (USD-scene scenario) and no
        # scene mesh provided: collision against scene geometry is not scored,
        # only goal/time/speed.
        res["min_clearance_m"] = None
        res["collided"] = False
        res["clearance_source"] = None

    # --- goal reached + time-to-goal (takeoff -> first sample inside radius) ---
    first_hit = None
    if "goal" in z:
        goal = np.asarray(z["goal"], float).reshape(3)
        dist_goal = np.linalg.norm(P - goal[None, :], axis=1)
        res["min_dist_to_goal_m"] = float(np.min(dist_goal))
        hit = np.where(dist_goal <= goal_radius)[0]
        if hit.size and tko is not None:
            first_hit = int(hit[0])
            res["reached"] = True
            res["time_to_goal_s"] = float(t[first_hit] - t[tko])
        else:
            res["reached"] = bool(hit.size)  # reached-without-moving can't happen
            res["time_to_goal_s"] = None
    else:
        res["min_dist_to_goal_m"] = None
        res["reached"] = None
        res["time_to_goal_s"] = None

    # --- speed over the productive flight segment: takeoff -> first goal hit
    # (or log end if the goal was never reached) ---
    flight = slice(tko, first_hit + 1 if first_hit is not None else traj.shape[0]) \
        if tko is not None else slice(0, 0)
    res["mean_speed_mps"] = float(np.mean(speed[flight])) if speed[flight].size else 0.0
    res["peak_speed_mps"] = float(np.max(speed)) if speed.size else 0.0

    # success = reached the goal AND never collided
    reached = res.get("reached")
    res["success"] = bool(reached) and not res["collided"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="trajectory .npz from run_px4_sim.py --log-traj")
    ap.add_argument("--drone-radius", type=float, default=0.2,
                    help="Drone collision radius [m] for clearance (default 0.2).")
    ap.add_argument("--goal-radius", type=float, default=1.0,
                    help="Distance [m] to the goal that counts as reached (default 1.0).")
    ap.add_argument("--scene-mesh", default=None,
                    help="extract_scene_mesh.py .npz for clearance/collision when the "
                         "log has no analytic obstacle field (USD-scene runs).")
    args = ap.parse_args()
    res = score_trajectory(args.npz, drone_radius=args.drone_radius,
                           goal_radius=args.goal_radius, scene_mesh=args.scene_mesh)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
