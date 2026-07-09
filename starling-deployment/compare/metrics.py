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
  - time & speed     : time-to-goal (POLICY flight only: policy handoff ->
                       first goal hit, excluding the scripted takeoff/climb/yaw
                       and the post-goal landing), mean and peak speed.
                       All durations are in SIMULATED seconds (matching the
                       recorded videos and the sim-frame velocities), taken
                       from the npz's per-pose t_sim or reconstructed from
                       the fixed per-pose render dt -- NOT wall clock, which
                       runs faster than a headless Isaac (see sim_timebase)

Obstacle geometry matches what run_px4_sim spawns from obstacle_field.ObstacleField:
  spheres (cx,cy,cz,r) | boxes (cx,cy,cz,hx,hy,hz[,roll,pitch,yaw]) |
  vertical cylinders (cx,cy,r), axis +Z, treated as spanning the flight band |
  horizontal cylinders (cx,cy,cz,r), axis +X, finite length CYLH_LEN.

For USD-scene scenarios there is no analytic field in the log; pass a scene
mesh .npz produced by compare/extract_scene_mesh.py (surface point samples of
the scene's non-ground geometry) and clearance/collision are scored against it
instead: clearance = (distance to nearest surface sample) - drone_radius,
measured over the POLICY flight only: climb/yaw -> policy handoff until first
goal hit (the parked prefix sits legitimately ON the scene, the scripted
climb-out measures spawn geometry identically for every method, and the
post-goal landing descent touches down intentionally). The policy window comes
from the phase-file timestamps run_px4_sim embeds in the npz when available,
else from a horizontal-motion heuristic (see policy_start_index).
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


def sim_timebase(z, traj):
    """Per-pose SIMULATED-time axis (zeroed at the first pose) for duration
    metrics, plus its source ("logged" | "reconstructed" | "wall").

    traj[:, 0] is wall-clock, but Isaac's render loop runs slower than
    realtime (headless RTX), so wall durations overstate flight times while
    the logged velocities are sim-frame m/s -- durations must be scored in
    sim time. Prefer the npz's t_sim (run_px4_sim logs world.current_time per
    pose); for older logs reconstruct it as index * dt, exploiting that the
    sim advances a FIXED render dt per logged pose: dt = median(|dP| / |V|)
    over moving samples (measured 16.00 ms on every existing trial). Fall
    back to wall clock only if the drone never moved enough to estimate dt
    (durations are meaningless for such a trial anyway)."""
    t_wall = traj[:, 0]
    n = traj.shape[0]
    if "t_sim" in z and np.asarray(z["t_sim"]).shape == (n,):
        ts = np.asarray(z["t_sim"], np.float64)
        return ts - ts[0], "logged"
    dp = np.linalg.norm(np.diff(traj[:, 1:4], axis=0), axis=1)
    vmid = 0.5 * (np.linalg.norm(traj[1:, 4:7], axis=1) +
                  np.linalg.norm(traj[:-1, 4:7], axis=1))
    moving = vmid > 0.5
    if np.count_nonzero(moving) >= 100:
        dt = float(np.median(dp[moving] / vmid[moving]))
        return np.arange(n, dtype=np.float64) * dt, "reconstructed"
    return t_wall - t_wall[0], "wall"


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
#
# The spawn settle itself can be sustained motion: scenarios that spawn the
# drone above the scene surface (construction --spawn z=1.0, rooftop spawns)
# free-fall for ~1 s at sim start, peaking near 4.8 m/s -- identically for
# every method. Takeoff must therefore be the first sustained-motion run AFTER
# the first parked stretch of >= SETTLE_PARKED_S (the drone always sits
# disarmed at the spawn for the sim warmup + arming, far longer than that).
TAKEOFF_SPEED_MPS = 0.3
TAKEOFF_HOLD_N = 5
SETTLE_PARKED_S = 2.0


def takeoff_index(t, speed):
    """Index of the first sustained-motion sample of the actual flight (spawn
    free-fall/settling excluded), or None if the drone never took off (e.g.
    PX4 was down and the trial logged a parked -- or merely settling -- drone)."""
    moving = speed > TAKEOFF_SPEED_MPS
    n = moving.size
    if n < TAKEOFF_HOLD_N:
        return None
    # Start of the search window: just past the first parked stretch lasting
    # >= SETTLE_PARKED_S. If there is none (log starts mid-flight), search the
    # whole log; if there is one, motion before it is spawn settling, and a
    # drone that never moves after it never took off.
    search_from = 0
    i = 0
    while i < n:
        if moving[i]:
            i += 1
            continue
        j = i
        while j < n and not moving[j]:
            j += 1
        if t[min(j, n - 1)] - t[i] >= SETTLE_PARKED_S:
            search_from = j
            break
        i = j
    sustained = np.convolve(moving[search_from:].astype(int),
                            np.ones(TAKEOFF_HOLD_N, dtype=int), "valid") == TAKEOFF_HOLD_N
    if not sustained.any():
        return None
    return int(search_from + np.argmax(sustained))


# Policy-phase start detection (heuristic fallback for logs that predate the
# policy_start_unix field in the .npz): the pre-policy phase is a scripted
# vertical climb + in-place yaw at the spawn, so the policy flight starts at
# the first sustained horizontal motion -- but only counted once the climb is
# at least half done, because liftoff itself can jitter sideways past the
# speed threshold for a few samples (observed 0.31 m/s on the construction
# scene, faking a policy start at the moment of takeoff).
POLICY_HSPEED_MPS = 0.3
DEFAULT_CLIMB_ALT = 2.0     # run_comparison --climb-alt default


def policy_start_index(t, P, V, tko, climb_alt):
    """Heuristic index of the climb/yaw -> policy handoff: first sustained
    horizontal-motion run after the drone has climbed climb_alt/2 above its
    resting altitude. None if never found (e.g. the climb never finished)."""
    if tko is None:
        return None
    gate_z = P[tko, 2] + 0.5 * float(climb_alt)
    gated = np.nonzero(P[tko:, 2] >= gate_z)[0]
    if not gated.size:
        return None
    frm = tko + int(gated[0])
    moving = np.linalg.norm(V[frm:, :2], axis=1) > POLICY_HSPEED_MPS
    sustained = np.convolve(moving.astype(int),
                            np.ones(TAKEOFF_HOLD_N, dtype=int), "valid") == TAKEOFF_HOLD_N
    if not sustained.any():
        return None
    return frm + int(np.argmax(sustained))


def _policy_window(z, t_wall, P, V, tko, climb_alt):
    """(start_idx, end_idx, source) of the policy flight within the log.

    Exact when the .npz carries the offboard's phase-file handoff timestamps
    (t_unix0 + policy_start/end_unix, written by run_px4_sim) -- those are
    unix times, so they map to indices through the WALL-clock column;
    otherwise the heuristic above for the start and None for the end. Either
    index is None when unknown."""
    if "t_unix0" in z and "policy_start_unix" in z:
        t0 = float(z["t_unix0"])   # traj wall column is time.time() - t0
        start = int(np.searchsorted(t_wall, float(z["policy_start_unix"]) - t0))
        end = (int(np.searchsorted(t_wall, float(z["policy_end_unix"]) - t0))
               if "policy_end_unix" in z else None)
        if start < t_wall.size:
            return start, end, "phase_file"
    return policy_start_index(t_wall, P, V, tko, climb_alt), None, "heuristic"


def score_trajectory(npz_path, drone_radius=0.2, goal_radius=1.0, scene_mesh=None,
                     climb_alt=DEFAULT_CLIMB_ALT):
    """Score one logged flight. Returns a flat dict of metrics (JSON-friendly).

    scene_mesh: optional path to an extract_scene_mesh.py .npz; used for
    clearance/collision when the log carries no analytic obstacle field
    (USD-scene scenarios). Ignored when the analytic field is present.
    climb_alt: the scenario's scripted climb height [m], used only by the
    heuristic policy-start detection on logs without phase timestamps."""
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

    t_wall = traj[:, 0]
    # All duration metrics are in SIMULATED seconds (see sim_timebase) -- wall
    # clock runs faster than the sim, and speeds are sim-frame m/s.
    t, res["timebase"] = sim_timebase(z, traj)
    P = traj[:, 1:4]
    V = traj[:, 4:7]
    speed = np.linalg.norm(V, axis=1)
    res["duration_s"] = float(t[-1] - t[0])   # full log, incl. pre-offboard warmup
    res["wall_duration_s"] = float(t_wall[-1] - t_wall[0])
    res["realtime_factor"] = (float((t[-1] - t[0]) / (t_wall[-1] - t_wall[0]))
                              if t_wall[-1] > t_wall[0] else None)

    # --- takeoff: strip the parked prefix (sim warmup + arming) from the
    # time/speed metrics; collision & goal checks still use the whole log ---
    tko = takeoff_index(t, speed)
    res["t_takeoff_s"] = float(t[tko] - t[0]) if tko is not None else None
    res["flight_time_s"] = float(t[-1] - t[tko]) if tko is not None else None

    # --- policy window: exact from the phase-file timestamps in the npz when
    # present, else the horizontal-motion heuristic. Used to clip scene-mesh
    # clearance to the policy flight (the scripted climb-out happens at the
    # spawn, so its clearance is spawn geometry, identical for every method).
    p_start, p_end, p_src = _policy_window(z, t_wall, P, V, tko, climb_alt)
    res["policy_start_s"] = float(t[p_start] - t[0]) if p_start is not None else None
    res["policy_end_s"] = (float(t[min(p_end, t.size - 1)] - t[0])
                           if p_end is not None else None)
    res["policy_window_source"] = p_src if p_start is not None else None

    # --- goal reached + time-to-goal (policy handoff -> first sample inside
    # radius); computed before collision scoring because the scene-mesh
    # clearance segment ends at the first goal hit ---
    first_hit = None
    if "goal" in z:
        goal = np.asarray(z["goal"], float).reshape(3)
        dist_goal = np.linalg.norm(P - goal[None, :], axis=1)
        res["min_dist_to_goal_m"] = float(np.min(dist_goal))
        hit = np.where(dist_goal <= goal_radius)[0]
        if hit.size and tko is not None:
            first_hit = int(hit[0])
            res["reached"] = True
            # time-to-goal measures the POLICY flight ONLY: from the climb/yaw
            # -> policy handoff to the first goal hit. This excludes the
            # scripted pre-policy takeoff/climb/yaw (which starts at tko, ~5-8 s
            # earlier and varies with climb height) and -- ending at goal
            # contact -- the post-goal landing descent, so it reports how long
            # the policy itself flew. Falls back to takeoff only when the policy
            # handoff couldn't be located (p_start is None).
            seg_start = p_start if p_start is not None else tko
            res["time_to_goal_s"] = float(t[first_hit] - t[seg_start])
        else:
            res["reached"] = bool(hit.size)  # reached-without-moving can't happen
            res["time_to_goal_s"] = None
    else:
        res["min_dist_to_goal_m"] = None
        res["reached"] = None
        res["time_to_goal_s"] = None

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
        # USD-scene scenario: score against the extracted scene geometry over
        # the POLICY flight only: policy handoff (falling back to takeoff if
        # the handoff is unknown) -> first goal hit (or the policy -> landing
        # handoff / log end if the goal was never reached). Excluded on
        # purpose: the parked drone sits ON the scene surface, the scripted
        # climb-out measures spawn geometry identically for every method, and
        # the post-goal landing intentionally descends to touch down
        # (observed: agile's landing at the construction goal scored
        # clearance -0.10 m against ground-adjacent samples on an otherwise
        # clean flight).
        samples, mesh_meta = load_scene_mesh(scene_mesh)
        seg0 = p_start if p_start is not None else tko
        if first_hit is not None:
            end = first_hit + 1
        elif p_end is not None:
            end = min(p_end + 1, P.shape[0])
        else:
            end = P.shape[0]
        if seg0 is not None and seg0 < end and samples.shape[0]:
            clr = _nearest_sample_dist(P[seg0:end], samples) - drone_radius
            res["min_clearance_m"] = float(np.min(clr))
            res["collided"] = bool(np.any(clr < 0.0))
            res["clearance_seg_s"] = [float(t[seg0] - t[0]), float(t[end - 1] - t[0])]
        else:
            res["min_clearance_m"] = None
            res["collided"] = False
        res["clearance_source"] = "scene_mesh"
        res["scene_mesh_sample_h"] = mesh_meta.get("sample_h")
        # The mesh is cropped to the planned flight corridor; if the drone
        # strayed within 10 m of (or past) the crop faces, clearance near
        # those samples may be against missing geometry -- flag it.
        if mesh_meta.get("bounds") and seg0 is not None and seg0 < end:
            b = np.asarray(mesh_meta["bounds"], float)
            outside = np.any((P[seg0:end] < b[:3] + 10.0) | (P[seg0:end] > b[3:] - 10.0),
                             axis=1)
            res["clearance_bounds_exceeded"] = bool(outside.any())
    else:
        # No analytic obstacle field in the log (USD-scene scenario) and no
        # scene mesh provided: collision against scene geometry is not scored,
        # only goal/time/speed.
        res["min_clearance_m"] = None
        res["collided"] = False
        res["clearance_source"] = None

    # --- speed over the productive flight segment: takeoff -> first goal hit
    # (or log end if the goal was never reached) ---
    flight = slice(tko, first_hit + 1 if first_hit is not None else traj.shape[0]) \
        if tko is not None else slice(0, 0)
    res["mean_speed_mps"] = float(np.mean(speed[flight])) if speed[flight].size else 0.0
    res["peak_speed_mps"] = float(np.max(speed[flight])) if speed[flight].size else 0.0

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
    ap.add_argument("--climb-alt", type=float, default=DEFAULT_CLIMB_ALT,
                    help="Scenario's scripted climb height [m]; only used by the "
                         "heuristic policy-start detection on logs without phase "
                         f"timestamps (default {DEFAULT_CLIMB_ALT}).")
    args = ap.parse_args()
    res = score_trajectory(args.npz, drone_radius=args.drone_radius,
                           goal_radius=args.goal_radius, scene_mesh=args.scene_mesh,
                           climb_alt=args.climb_alt)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
