#!/usr/bin/env python
"""Static camera probe for the RGB policy camera (pixel parity, 2026-10-01).

Builds the harness's own PegasusApp (superfly.sim.px4_sim: the same vehicle
USD, camera mount code, environment and box spawner a flight uses) with
SUPERFLY_POLICY_RGB=1 and the agile depth camera on the Starling contract
(SUPERFLY_CAM_CONTRACT=starling87), switches gravity OFF so the unpowered
vehicle hangs at the spawn pose (no PX4 needed: the MAVLink backend waits for
a heartbeat without blocking), steps the renderer, and saves

    rgb       (480, 640, 3) uint8   the policy RGB camera (exactly what RgbPublisher sends)
    depth     (480, 640) float32    the depth camera at the same geometry (planar z, m)
    body_T    (4, 4)  world <- /World/quadrotor/body
    rgbcam_T  (4, 4)  world <- the policy RGB camera prim (USD camera: looks -Z, +Y up)
    depthcam_T(4, 4)  world <- the depth camera prim
    cam_attrs json    USD camera attributes of the RGB camera (focal length, apertures, clipping, lens API)
    boxes, spawn, yaw_deg

to --out (.npz). Run under Isaac's python on airstation03:

    SUPERFLY_INSTANCE=7 ~/isaacsim/python.sh scripts/rgb_cam_probe.py --headless \\
        --boxes-json probe_boxes.json --spawn 0 0 1.5 --yaw 0 --out /tmp/probe.npz

SUPERFLY_INSTANCE keeps its ports (TCP 4560+i, UDP 15001/15003+10i) off any
running trial's. The vehicle is the one --vehicle names (default the lab
starling2max USD; Nucleus token from ~/.omni_env must be exported).
"""
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--boxes-json", required=True)
ap.add_argument("--spawn", type=float, nargs=3, default=(0.0, 0.0, 1.5))
ap.add_argument("--yaw", type=float, default=0.0, help="spawn yaw [deg] (PegasusApp spawn_yaw_deg)")
ap.add_argument("--environment", default="Box Room")
ap.add_argument("--vehicle", default="starling2max")
ap.add_argument("--steps", type=int, default=240)
ap.add_argument("--out", required=True)
args, _ = ap.parse_known_args()

os.environ.setdefault("SUPERFLY_POLICY_RGB", "1")
os.environ.setdefault("SUPERFLY_CAM_CONTRACT", "starling87")
os.environ.setdefault("SUPERFLY_CAM_PITCH_DEG", "0")

import superfly.sim.px4_sim as S  # noqa: E402  (boots SimulationApp)
import numpy as np  # noqa: E402
from pxr import UsdGeom, Usd  # noqa: E402  (Usd also used for the kinematic hold below)

app = S.PegasusApp(policy="agile", obstacles="boxes", boxes_json=args.boxes_json,
                   spawn_xyz=tuple(args.spawn), spawn_yaw_deg=args.yaw,
                   environment=args.environment, goal_xyz=(10.0, 0.0, 1.5),
                   vehicle=args.vehicle, debug_frames=False,
                   px4_instance=int(os.environ.get("SUPERFLY_INSTANCE", "7") or 7))
try:
    app.world.get_physics_context().set_gravity(0.0)
except Exception as e:  # older API
    print(f"[probe] set_gravity failed ({e}); trying the PhysicsScene prim", flush=True)
    from pxr import UsdPhysics
    for p in app.world.stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            UsdPhysics.Scene(p).CreateGravityMagnitudeAttr(0.0)
# hold the vehicle exactly at the spawn pose: kinematic rigid bodies (gravity off alone
# still let the unpowered airframe sink 0.3 m over the probe, 2026-10-01 first run)
from pxr import UsdPhysics  # noqa: E402
n_kin = 0
for prim in Usd.PrimRange(app.world.stage.GetPrimAtPath("/World/quadrotor")):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        n_kin += 1
print(f"[probe] {n_kin} rigid bodies under /World/quadrotor made kinematic", flush=True)
app.timeline.play()


def xf(path):
    prim = app.world.stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([[m[r][c] for c in range(4)] for r in range(4)]).T   # column vectors


rgb = depth = None
body_hist = []
cap_T = None
for k in range(args.steps):
    app.world.step(render=True)
    T = xf("/World/quadrotor/body")
    if T is not None:
        body_hist.append(T[:3, 3].copy())
    c = app._rgb_policy_camera
    if k > args.steps // 2 and getattr(c, "_camera_full_set", False):
        r = c._camera.get_rgb()
        d = app._camera._camera.get_depth() if getattr(app._camera, "_camera_full_set", False) else None
        if r is not None and np.asarray(r).size and d is not None and np.asarray(d).size:
            rgb, depth = np.asarray(r)[..., :3].copy(), np.asarray(d, np.float32).copy()
            cap_T = T

rcam = app._rgb_policy_camera._stage_prim_path
dcam = app._camera._stage_prim_path
attrs = {}
cp = app.world.stage.GetPrimAtPath(rcam)
for a in cp.GetAttributes():
    try:
        v = a.Get()
        attrs[a.GetName()] = v if isinstance(v, (int, float, str, bool)) else str(v)
    except Exception:
        pass
bh = np.asarray(body_hist)
print(f"[probe] body drift over {len(bh)} steps: {np.ptp(bh, axis=0) if len(bh) else None}", flush=True)
np.savez(args.out, rgb=rgb, depth=depth, body_T=cap_T, body_T_end=xf("/World/quadrotor/body"),
         body_hist=bh,
         rgbcam_T=xf(rcam), depthcam_T=xf(dcam), rgbcam_path=rcam, depthcam_path=dcam,
         cam_attrs=json.dumps(attrs, default=str), boxes=json.load(open(args.boxes_json)),
         spawn=np.asarray(args.spawn), yaw_deg=args.yaw, body_drift=np.ptp(bh, axis=0) if len(bh) else None,
         fx=app._rgb_policy_camera.fx, cx=app._rgb_policy_camera.cx, cy=app._rgb_policy_camera.cy)
print(f"[probe] wrote {args.out}: rgb {None if rgb is None else rgb.shape}, "
      f"depth {None if depth is None else depth.shape}", flush=True)
os._exit(0)
