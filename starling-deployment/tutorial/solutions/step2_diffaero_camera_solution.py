#!/usr/bin/env python
"""
STEP 2 SOLUTION — DiffAero camera + depth publish pipeline
==========================================================

Reference while you edit run_px4_sim.py. NOT imported by the sim.

Step 2 goal: when --policy diffaero, Isaac Sim must publish a 9×16 float32
EUCLIDEAN range image (metres) over UDP. The offboard policy (Step 3) will
apply: perception = 1 - clamp(range, 0, 5) / 5.

DiffPhysDrone path stays unchanged (--policy diffphys default).

Run the math self-test (no Isaac):
    python3 starling-deployment/tutorial/solutions/step2_diffaero_camera_solution.py
"""

from __future__ import annotations

import math
import numpy as np

# =============================================================================
# PART A — constants to add near top of run_px4_sim.py (after DiffPhysDrone block)
# =============================================================================

# From checkpoints/DiffAero/sha2c_pmc/hydra/config.yaml → sensor:
#   width: 16, height: 9, horizontal_fov: 86.0, max_dist: 5.0
#   onboard_position: [0.2, 0, 0.05], pitch: 0 (forward)

DA_OUT_W, DA_OUT_H = 16, 9          # policy grid: cols × rows
DA_POOL = 4                         # render 4× denser, min-pool down
DA_RENDER_W = DA_OUT_W * DA_POOL    # 64
DA_RENDER_H = DA_OUT_H * DA_POOL    # 36
DA_FOV_X_DEG = 86.0
DA_FOV_Y_DEG = DA_FOV_X_DEG * DA_OUT_H / DA_OUT_W   # 48.375° (DiffAero definition)
DA_CAM_ANGLE_DEG = 0.0              # no downward pitch
DA_MAX_DIST = 5.0


# =============================================================================
# PART B — _setup_camera_diffaero()  (extract from existing inline camera code)
# =============================================================================

def setup_camera_diffaero_intrinsics():
    """
    Returns (fx, fy, cx, cy, euclid_scale) for the DiffAero render grid.

    Pegasus MonocularCamera config dict:
        position:     [0.20, 0.0, 0.05]   body frame, metres
        orientation:  [0, -DA_CAM_ANGLE_DEG, 180]  ZYX euler deg; 180° = forward
        resolution:   (DA_RENDER_W, DA_RENDER_H) = (64, 36)  NOTE: (width, height)
        frequency:    30
    """
    fx = 0.5 * DA_RENDER_W / math.tan(0.5 * math.radians(DA_FOV_X_DEG))
    fy = 0.5 * DA_RENDER_H / math.tan(0.5 * math.radians(DA_FOV_Y_DEG))
    cx = 0.5 * DA_RENDER_W
    cy = 0.5 * DA_RENDER_H

    u = np.arange(DA_RENDER_W, dtype=np.float32)
    v = np.arange(DA_RENDER_H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # row=v (height), col=u (width)
    xn = (uu - cx) / fx
    yn = (vv - cy) / fy
    euclid_scale = np.sqrt(1.0 + xn * xn + yn * yn).astype(np.float32)
    return fx, fy, cx, cy, euclid_scale


# =============================================================================
# PART C — _publish_depth_diffaero()  (pure numpy version for testing)
# =============================================================================

def publish_depth_diffaero_pipeline(planar_depth: np.ndarray,
                                    euclid_scale: np.ndarray) -> np.ndarray:
    """
    Input:  planar Z-depth from Isaac get_depth(), shape (DA_RENDER_H, DA_RENDER_W)
    Output: Euclidean range, shape (DA_OUT_H, DA_OUT_W) = (9, 16)
    """
    far = 1e3
    depth = np.nan_to_num(planar_depth, nan=far, posinf=far, neginf=far).astype(np.float32)

    if depth.shape != (DA_RENDER_H, DA_RENDER_W):
        yi = (np.linspace(0, depth.shape[0] - 1, DA_RENDER_H)).astype(np.int64)
        xi = (np.linspace(0, depth.shape[1] - 1, DA_RENDER_W)).astype(np.int64)
        depth = depth[yi][:, xi]

    euclid = depth * euclid_scale
    euclid = euclid.reshape(DA_OUT_H, DA_POOL, DA_OUT_W, DA_POOL).min(axis=(1, 3))
    return np.ascontiguousarray(euclid, dtype=np.float32)


def diffaero_perception(range_m: np.ndarray) -> np.ndarray:
    """What the offboard policy does in Step 3 (for reference)."""
    return 1.0 - np.clip(range_m, 0.0, DA_MAX_DIST) / DA_MAX_DIST


# =============================================================================
# PART D — edits checklist for run_px4_sim.py
# =============================================================================
#
# 1. PegasusApp.__init__: add policy/obstacles args; store self.policy
#
# 2. Replace inline camera block with:
#        if policy == "diffaero":
#            self._setup_camera_diffaero()
#        else:
#            self._setup_camera_diffphys()
#
# 3. Move existing camera code (lines ~72-95) into _setup_camera_diffphys()
#
# 4. Add _setup_camera_diffaero() from PART B (+ store self._da_euclid_scale)
#
# 5. Rename _publish_depth body → _publish_depth_diffphys; add dispatcher:
#        def _publish_depth(self):
#            if self.policy == "diffaero":
#                self._publish_depth_diffaero()
#            else:
#                self._publish_depth_diffphys()
#
# 6. Add _publish_depth_diffaero() from PART C (uses self._da_euclid_scale)
#
# 7. In _dump_depth_debug: vmax = DA_MAX_DIST if diffaero else 24.0
#
# 8. main(): add --policy and --obstacles argparse; pass to PegasusApp
#
# depth_transport.py: NO CHANGES (header already carries H, W dynamically)


if __name__ == "__main__":
    fx, fy, cx, cy, scale = setup_camera_diffaero_intrinsics()
    hfov = 2 * math.degrees(math.atan((DA_RENDER_W / 2) / fx))
    vfov = 2 * math.degrees(math.atan((DA_RENDER_H / 2) / fy))
    print(f"fx={fx:.2f} fy={fy:.2f}  hfov={hfov:.2f}° vfov={vfov:.2f}°")
    assert abs(hfov - 86.0) < 0.01
    assert abs(vfov - 48.375) < 0.01

    # Synthetic: left half 2 m planar, right half far
    planar = np.full((DA_RENDER_H, DA_RENDER_W), 1e3, dtype=np.float32)
    planar[:, : DA_RENDER_W // 2] = 2.0
    euclid = publish_depth_diffaero_pipeline(planar, scale)
    perc = diffaero_perception(euclid)
    print(f"output shape {euclid.shape}  left mean range {euclid[:, :8].mean():.2f} m")
    print(f"perception left mean {perc[:, :8].mean():.3f}  right mean {perc[:, 8:].mean():.3f}")
    assert euclid.shape == (9, 16)
    assert perc[:, :8].mean() > 0.5   # close obstacles → high perception
    assert perc[:, 8:].mean() < 0.01  # far → ~0
    print("Step 2 math checks passed.")
