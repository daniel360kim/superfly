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
    spheres: List[tuple] = field(default_factory=list)   # (cx,cy,cz, r)
    boxes: List[tuple] = field(default_factory=list)      # (cx,cy,cz, hx,hy,hz)
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


if __name__ == "__main__":
    for s in (0, 1, 2):
        print(generate(seed=s, scale=5.0).summary())
