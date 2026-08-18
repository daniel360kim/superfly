"""USD-environment setup shared by run_px4_sim.py (flight harness) and
scripts/scene_audit.py (scene vetting): lighting for stages with no authored
lights, and static colliders for stages with no authored physics.

Lifted verbatim from PegasusApp._spawn_lighting / _add_colliders (2026-08-18)
so the audit predicts exactly what the flight harness will do -- if you change
behavior here, both consumers change together, which is the point.

Everything imports Kit modules lazily: this module lives in superfly.sim but
must be importable for introspection before SimulationApp boots (same
convention as superfly.perception.mesh_sampling).
"""
from __future__ import annotations

import os

import numpy as np


def spawn_lighting(headlamp_parent: str | None = None):
    """Add outdoor lighting (a directional 'sun' + an ambient dome) so a USD
    stage with no authored lights is actually visible. DistantLight angle is in
    degrees down from horizontal-ish; intensities are in the UsdLux nits scale.

    Interiors (station halls, sewer tunnels) receive neither sun nor dome; RGB
    renders pitch black there (probe3 videos, 2026-08-07). GSDS_CAMERA_LIGHT=1
    parents an omnidirectional 'headlamp' to `headlamp_parent` (the vehicle in
    the flight harness) so the camera always sees. Opt-in: it changes
    appearance, so it must never silently apply to benchmark scenes."""
    import isaacsim.core.utils.prims as prim_utils
    from scipy.spatial.transform import Rotation

    # Directional sun, tilted 45 deg down so the forest casts shadows.
    prim_utils.create_prim(
        "/World/lighting/sun", "DistantLight",
        orientation=np.array(
            Rotation.from_euler("XYZ", [45.0, 0.0, 0.0], degrees=True).as_quat()[[3, 0, 1, 2]]),
        attributes={"inputs:intensity": 3000.0, "inputs:angle": 1.0,
                    "inputs:color": (1.0, 0.98, 0.95)},
    )
    # Ambient sky fill so shadowed areas aren't pure black.
    prim_utils.create_prim(
        "/World/lighting/sky", "DomeLight",
        attributes={"inputs:intensity": 1000.0, "inputs:color": (0.8, 0.85, 1.0)},
    )
    if headlamp_parent and os.environ.get("GSDS_CAMERA_LIGHT") == "1":
        prim_utils.create_prim(
            f"{headlamp_parent}/headlamp", "SphereLight",
            translation=np.array([0.0, 0.0, 0.05]),
            attributes={"inputs:intensity": 60000.0,
                        "inputs:radius": 0.05,
                        "inputs:color": (1.0, 1.0, 1.0)},
        )


def add_colliders(stage, root: str = "/World/layout", approximation: str = "none",
                  verbose: bool = True):
    """Give a loaded USD stage physics colliders so the drone collides with the
    ground/trees instead of falling through. Static environment geometry has no
    rigid body, so each Mesh just gets a CollisionAPI + MeshCollisionAPI with a
    triangle-mesh approximation ('none' = exact tris, correct for static scenes;
    thin trunks and ground stay solid). Other gprims get a plain CollisionAPI.
    The /World/layout scale flows into the colliders via the xform hierarchy.

    Returns the number of prims that received colliders."""
    import carb
    from pxr import Usd, UsdGeom, UsdPhysics

    root_prim = stage.GetPrimAtPath(root)
    if not root_prim or not root_prim.IsValid():
        return
    # UE-exported stages (TrainStation, Sewerage, Dmytro scenes, ...) are
    # built from INSTANCED meshes; Usd.PrimRange does not descend into
    # instance proxies and APIs cannot be applied to them, so those
    # stages silently got zero colliders and the drone fell through the
    # world (probe3 freefalls, 2026-08-07). De-instance under the layout
    # root first; repeat for nested instancing.
    for _ in range(4):
        inst = [p for p in Usd.PrimRange(root_prim) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
    if inst := sum(1 for p in Usd.PrimRange(root_prim) if p.IsInstance()):
        carb.log_warn(f"{inst} instance prims remain after de-instancing")
    n = 0
    for prim in Usd.PrimRange(root_prim):
        try:
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(approximation)
                n += 1
            elif prim.IsA(UsdGeom.Gprim):
                UsdPhysics.CollisionAPI.Apply(prim)
                n += 1
        except Exception as e:
            carb.log_warn(f"collider apply failed on {prim.GetPath()}: {e}")
    if verbose:
        print(f"[environment] applied colliders ('{approximation}') to {n} prims under {root}")
    return n
