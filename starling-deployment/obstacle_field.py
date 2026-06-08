#!/usr/bin/env python
"""
Generate a DiffPhysDrone training-distribution obstacle field as analytic
primitives, for spawning into Isaac Sim.

DiffPhysDrone has no USD/mesh world — obstacles are spheres, axis-aligned boxes,
and cylinders, and depth is CUDA ray-casting against them (env_cuda.py,
quadsim_kernel.cu). This module reproduces the SAME sampling + transforms so the
Isaac depth camera renders a statistically-matched scene.

Ranges (val = rand*w + b, uniform [b, b+w]) and transforms mirror
env_cuda.py:52-198 for the single_agent config (--single --speed_mtp 4
--ground_voxels). We fix the world `scale` (training randomizes it via max_speed)
so the eval is deterministic.

All positions/sizes are in metres, world ENU (z up), matching how the offboard
script and Isaac scene use ENU.
"""

import math
from dataclasses import dataclass, field
from typing import List
import numpy as np

# --- single_agent config constants (env_cuda.py) ---
N_BALLS = 30
N_VOXELS = 30
N_CYL = 30
N_CYL_H = 2
N_GROUND_VOXELS = 10

# raw range tensors [w] (width) and [b] (bias): val = rand*w + b
BALL_W = np.array([8., 18., 6., 0.2]);            BALL_B = np.array([0., -9., -1., 0.4])
VOXEL_W = np.array([8., 18., 6., 0.1, 0.1, 0.1]); VOXEL_B = np.array([0., -9., -1., 0.2, 0.2, 0.2])
GVOX_W = np.array([8., 18., 0., 2.9, 2.9, 1.9]);  GVOX_B = np.array([0., -9., -1., 0.1, 0.1, 0.1])
CYL_W = np.array([8., 18., 0.35]);                CYL_B = np.array([0., -9., 0.05])
CYLH_W = np.array([8., 6., 0.1]);                 CYLH_B = np.array([0., 0., 0.05])


@dataclass
class ObstacleField:
    # DiffPhysDrone boxes are 6-tuples (cx,cy,cz, hx,hy,hz); DiffAero boxes are
    # 9-tuples (cx,cy,cz, hx,hy,hz, roll,pitch,yaw) with rpy in radians (XYZ euler).
    spheres: List[tuple] = field(default_factory=list)   # (cx,cy,cz, r)
    boxes: List[tuple] = field(default_factory=list)      # (cx,cy,cz, hx,hy,hz[, r,p,y])
    cyl_v: List[tuple] = field(default_factory=list)      # (cx,cy, r)  vertical (axis=Z), full height
    cyl_h: List[tuple] = field(default_factory=list)      # (cx,cz, r)  horizontal (axis=X)
    p_init: np.ndarray = None      # drone start (ENU)
    p_target: np.ndarray = None    # goal (ENU)
    scale: float = 5.0

    def summary(self) -> str:
        return (f"ObstacleField(scale={self.scale}): "
                f"{len(self.spheres)} spheres, {len(self.boxes)} boxes, "
                f"{len(self.cyl_v)} vert-cyl, {len(self.cyl_h)} horiz-cyl | "
                f"start={np.round(self.p_init,2)} target={np.round(self.p_target,2)}")


# --- DiffAero outdoor obstacle distribution (diffaero/cfg/env/obstacles/outdoor.yaml
#     + diffaero/utils/assets.py ObstacleManager) ---
DA_N_OBSTACLES = 30
DA_SPHERE_PCT = 0.33                       # n_spheres = int(30*0.33) = 9, n_cubes = 21
DA_SPHERE_R = (0.6, 2.0, 0.4)              # arange(min, max, step) -> [0.6,1.0,1.4,1.8]
DA_CUBE_LW = (0.6, 1.4, 0.2)              # -> [0.6,0.8,1.0,1.2]
DA_CUBE_H = (10.0, 15.0, 1.0)            # -> [10,11,12,13,14]
DA_CUBE_RP_DEG = 15.0                      # cube roll/pitch range (deg); yaw uniform +-pi
DA_RANDPOS_STD = (6.0, 7.0)               # (min, max) perpendicular std along the path
DA_SAFETY_RANGE = 1.7
DA_HEIGHT_SCALE = 0.25                     # vertical std = std * height_scale
DA_GROUND_Z = 0.0                          # Isaac Box Room ground plane (ENU z up)


def _euler_xyz_matrix(roll, pitch, yaw):
    """Rotation matrix for intrinsic XYZ euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def _grounded_box_cz(hx, hy, hz, roll, pitch, yaw, ground_z=DA_GROUND_Z):
    """Center z so the lowest vertex of the rotated box rests on ground_z."""
    rot = _euler_xyz_matrix(roll, pitch, yaw)
    min_local_z = float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = rot @ np.array([sx * hx, sy * hy, sz * hz])
                min_local_z = min(min_local_z, corner[2])
    return ground_z - min_local_z


def generate(seed: int = 0, scale: float = 5.0, heading_deg: float = 90.0) -> ObstacleField:
    """Build a deterministic obstacle field for the given seed and world scale.

    `scale` corresponds to env_cuda's `scale = clamp(max_speed-0.5, min=1)`. With
    speed_mtp=4, scale in [2.5, 12.5]; we fix it. The y-stretch factor uses the
    matching max_speed for that scale: scale = max_speed - 0.5  =>  max_speed = scale + 0.5.

    heading_deg rotates the whole field+start+goal about world Z. The native
    corridor runs +X; in the Pegasus HIL sim the mag-driven EKF locks the drone's
    heading to ~+Y (ENU yaw 90°), so heading_deg=90 aligns the corridor with the
    drone's actual forward direction (camera sees the obstacles it flies into).
    """
    rng = np.random.default_rng(seed)
    max_speed = scale + 0.5
    y_stretch = (max_speed + 4.0) / scale  # env_cuda.py:160-162

    def sample(n, w, b):
        return rng.random((n, len(w))) * w + b

    balls = sample(N_BALLS, BALL_W, BALL_B)      # (30,4): x,y,z,r
    voxels = sample(N_VOXELS, VOXEL_W, VOXEL_B)  # (30,6): x,y,z,hx,hy,hz
    cyl = sample(N_CYL, CYL_W, CYL_B)            # (30,3): x,y,r
    cylh = sample(N_CYL_H, CYLH_W, CYLH_B)       # (2,3):  x,z,r

    # ground voxels (single_agent has --ground_voxels): 10 boxes on the ground.
    gvox = sample(N_GROUND_VOXELS, GVOX_W, GVOX_B)  # (10,6)
    gvox[:, 2] = gvox[:, 5] - 1.0                   # env_cuda.py:157: z = hz - 1
    voxels = np.concatenate([voxels, gvox], axis=0)

    # --- X clamp so obstacles stay in [r, 8 - r] BEFORE x*=scale (env_cuda.py:136-139)
    balls[:, 0] = np.clip(balls[:, 0], balls[:, 3] + 0.3 / scale, 8 - 0.3 / scale - balls[:, 3])
    voxels[:, 0] = np.clip(voxels[:, 0], voxels[:, 3] + 0.3 / scale, 8 - 0.3 / scale - voxels[:, 3])
    cyl[:, 0] = np.clip(cyl[:, 0], cyl[:, 2] + 0.3 / scale, 8 - 0.3 / scale - cyl[:, 2])
    cylh[:, 0] = np.clip(cylh[:, 0], cylh[:, 2] + 0.3 / scale, 8 - 0.3 / scale - cylh[:, 2])

    # --- y stretch (env_cuda.py:160-162) then x scale (env_cuda.py:181-184)
    for arr in (balls, voxels, cyl):
        arr[:, 1] *= y_stretch
    balls[:, 0] *= scale
    voxels[:, 0] *= scale
    cyl[:, 0] *= scale
    cylh[:, 0] *= scale

    # Rotate the whole field about world Z by heading_deg so the corridor aligns
    # with the drone's actual forward heading in the sim.
    cs, sn = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))

    def rot_xy(x, y):
        return cs * x - sn * y, sn * x + cs * y

    fld = ObstacleField(scale=scale)
    for b in balls:
        x, y = rot_xy(b[0], b[1]); fld.spheres.append((x, y, b[2], b[3]))
    for v in voxels:
        x, y = rot_xy(v[0], v[1]); fld.boxes.append((x, y, v[2], v[3], v[4], v[5]))
    for cc in cyl:  # vertical cylinder stays vertical; only its XY position rotates
        x, y = rot_xy(cc[0], cc[1]); fld.cyl_v.append((x, y, cc[2]))
    for ch in cylh:  # horizontal cylinder (cx,cz,r): only cx is in the XY plane
        x, y = rot_xy(ch[0], 0.0); fld.cyl_h.append((x, y, ch[1], ch[2]))

    pix, piy = rot_xy(-1.5 * scale, -3.0)
    ptx, pty = rot_xy(8.0 * scale, 3.0)
    fld.p_init = np.array([pix, piy, 1.0])
    fld.p_target = np.array([ptx, pty, 1.0])
    return fld


def generate_diffaero(seed: int = 0, scale: float = 5.0, heading_deg: float = 90.0) -> ObstacleField:
    """Build a DiffAero training-distribution obstacle field.

    Replicates diffaero's `outdoor` ObstacleManager (diffaero/utils/assets.py):
    30 obstacles, 33% spheres (9) + 67% cubes (21), with sizes drawn from the
    configured ranges and rotated cubes (tall pillars). Obstacles are scattered
    around the straight line from the drone start to the goal in XY, with a
    perpendicular Gaussian spread (std 6-7 m) and pushed outside a safety radius
    of the start/goal, exactly as in `randomize_obstacles_positions`.

    All obstacles are grounded on the Isaac ground plane (z=0): spheres sit on
    the floor, tall cubes rise from it (including roll/pitch tilt). The drone
    start/goal XY mirror the DiffPhysDrone corridor; metadata z matches diffphys
    (1 m) while the offboard script cruises at `flight_alt`.
    """
    rng = np.random.default_rng(seed)

    cs, sn = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))

    def rot_xy(x, y):
        return cs * x - sn * y, sn * x + cs * y

    # Same start/goal XY as the DiffPhysDrone field.
    pix, piy = rot_xy(-1.5 * scale, -3.0)
    ptx, pty = rot_xy(8.0 * scale, 3.0)
    p_init = np.array([pix, piy, 1.0], dtype=np.float64)
    p_target = np.array([ptx, pty, 1.0], dtype=np.float64)

    n_spheres = int(DA_N_OBSTACLES * DA_SPHERE_PCT)   # 9
    n_cubes = DA_N_OBSTACLES - n_spheres               # 21

    # --- sizes ---
    sphere_choices = np.arange(*DA_SPHERE_R)
    r_spheres = rng.choice(sphere_choices, size=n_spheres)
    lw_choices = np.arange(*DA_CUBE_LW)
    h_choices = np.arange(*DA_CUBE_H)
    cube_l = rng.choice(lw_choices, size=n_cubes)
    cube_w = rng.choice(lw_choices, size=n_cubes)
    cube_h = rng.choice(h_choices, size=n_cubes)
    lwh = np.stack([cube_l, cube_w, cube_h], axis=-1)               # (n_cubes, 3)
    r_cubes = np.linalg.norm(lwh / 2.0, axis=-1)                    # bounding radius
    r_obstacles = np.concatenate([r_spheres, r_cubes])             # (30,)

    # --- cube poses (rpy, radians; XYZ euler) ---
    rp = math.radians(DA_CUBE_RP_DEG)
    roll = rng.uniform(-rp, rp, size=n_cubes)
    pitch = rng.uniform(-rp, rp, size=n_cubes)
    yaw = rng.uniform(-math.pi, math.pi, size=n_cubes)
    rpy = np.stack([roll, pitch, yaw], axis=-1)                     # (n_cubes, 3)

    # --- positions, scattered around the start->goal line in XY only ---
    p_init_xy = p_init[:2]
    p_target_xy = p_target[:2]
    rel_pos_xy = p_target_xy - p_init_xy
    rel_len = np.linalg.norm(rel_pos_xy)
    target_axis_xy = rel_pos_xy / rel_len
    horizontal_axis_xy = np.array([-target_axis_xy[1], target_axis_xy[0]])

    n = DA_N_OBSTACLES
    minstd, maxstd = DA_RANDPOS_STD
    target_axis_ratio = rng.random((n, 1))
    target_axis_pos_xy = target_axis_ratio * rel_pos_xy                   # (n,2)
    std = np.abs(target_axis_ratio - 0.5) * 2 * (maxstd - minstd) + minstd  # (n,1)
    h_ratio = rng.standard_normal((n, 1)) * std
    relpos2target_axis_xy = h_ratio * horizontal_axis_xy                  # (n,2)
    relpos2drone_xy = target_axis_pos_xy + relpos2target_axis_xy          # (n,2)

    # push obstacles outside a safety radius of the start and goal (XY)
    relpos2target_xy = relpos2drone_xy - rel_pos_xy
    dist2drone = np.linalg.norm(relpos2drone_xy, axis=-1)
    dist2target = np.linalg.norm(relpos2target_xy, axis=-1)
    safety = r_obstacles + DA_SAFETY_RANGE
    tooclose = (dist2drone < safety) | (dist2target < safety)
    if np.any(tooclose):
        perp = relpos2target_axis_xy[tooclose]
        perp_n = perp / np.clip(np.linalg.norm(perp, axis=-1, keepdims=True), 1e-6, None)
        relpos2drone_xy[tooclose] += perp_n * DA_SAFETY_RANGE

    obstacle_xy = p_init_xy + relpos2drone_xy                           # (n,2)

    fld = ObstacleField(scale=scale)
    for i in range(n_spheres):
        cx, cy = obstacle_xy[i]
        r = float(r_spheres[i])
        fld.spheres.append((float(cx), float(cy), r + DA_GROUND_Z, r))
    for j in range(n_cubes):
        cx, cy = obstacle_xy[n_spheres + j]
        hx, hy, hz = lwh[j] / 2.0
        rr, pp, yy = rpy[j]
        cz = _grounded_box_cz(float(hx), float(hy), float(hz), float(rr), float(pp), float(yy))
        fld.boxes.append((float(cx), float(cy), cz, float(hx), float(hy), float(hz),
                          float(rr), float(pp), float(yy)))
    fld.p_init = p_init
    fld.p_target = p_target
    return fld


if __name__ == "__main__":
    print("=== DiffPhysDrone distribution ===")
    for s in (0, 1, 2):
        print(generate(seed=s, scale=5.0).summary())
    print("=== DiffAero distribution ===")
    for s in (0, 1, 2):
        print(generate_diffaero(seed=s, scale=5.0).summary())
