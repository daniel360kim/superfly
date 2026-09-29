#!/usr/bin/env python
"""Airframe extents of a vehicle USD from the body centre, props as SWEPT discs,
plus what the policy depth camera (body (0.10, 0, 0), 13 deg down, 91 deg hfov,
640x480) can see of the airframe. pxr only (no Kit), so it runs in seconds.

On airstation03 (Isaac's pxr lives in extscache, not on python.sh's path):
    EC=~/isaacsim/extscache; U=$(ls -d $EC/omni.usd.libs-*); PH=$(ls -d $EC/omni.usd.schema.physx-*)
    PYTHONPATH=$U:$PH LD_LIBRARY_PATH=$U/bin:$PH/bin \
        ~/isaacsim/kit/python/bin/python3 scripts/vehicle_extents.py <usd> <label> \
        [--iris-as-starling] [--json out.json]
An omniverse:// USD: copy it to /tmp first with omni.client (OMNI_USER /
OMNI_PASS from ~/.omni_env), e.g. oc.copy(url, "file:/tmp/x.usd").
--iris-as-starling moves the Iris rotors exactly like
superfly.sim.vehicles._iris_frame_overrides (--vehicle starling2max_iris).

Results (2026-09-29) are in configs/vehicles/starling2max.yaml `geometry`.
"""
import sys, json, math
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

usd, label = sys.argv[1], sys.argv[2]
iris_as_starling = "--iris-as-starling" in sys.argv
out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
st = Usd.Stage.Open(usd)
root = st.GetDefaultPrim()
R = root.GetPath().pathString

if iris_as_starling:
    # exactly superfly.sim.vehicles.apply_vehicle_overrides' rotor move
    signs = [(1, -1), (-1, 1), (1, 1), (-1, -1)]
    for i, (sx, sy) in enumerate(signs):
        rotor = st.GetPrimAtPath(f"{R}/rotor{i}")
        xf = UsdGeom.Xformable(rotor)
        op = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate][0]
        old = op.Get()
        op.Set(type(old)(sx * 0.085, sy * 0.0625, old[2]))

tc = Usd.TimeCode.Default()
body_xf = UsdGeom.Xformable(st.GetPrimAtPath(f"{R}/body")).ComputeLocalToWorldTransform(tc)
to_body = body_xf.GetInverse()

def pts_body(mesh_prim):
    m = UsdGeom.Mesh(mesh_prim)
    p = np.asarray(m.GetPointsAttr().Get(tc), dtype=np.float64)
    M = np.asarray(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(tc) * to_body)
    ph = np.c_[p, np.ones(len(p))] @ M
    return ph[:, :3]

res = {"label": label, "usd": usd, "parts": {}}
static, swept = [], []
rot_info = []
for c in root.GetChildren():
    meshes = [p for p in Usd.PrimRange(c) if p.IsA(UsdGeom.Mesh)]
    if not meshes:
        continue
    P = np.vstack([pts_body(m) for m in meshes])
    name = c.GetName()
    if name.startswith("rotor"):
        # spin axis: the rotor prim's z axis through its origin (revolute joint axis Z)
        Mr = np.asarray(UsdGeom.Xformable(c).ComputeLocalToWorldTransform(tc) * to_body)
        ctr, ax = Mr[3, :3], Mr[2, :3] / np.linalg.norm(Mr[2, :3])
        d = P - ctr
        along = d @ ax
        radial = np.linalg.norm(d - np.outer(along, ax), axis=1)
        rp = float(radial.max())
        # swept disc sampled as the prop points rotated about the axis
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        u = np.cross(ax, [1, 0, 0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.cross(ax, [0, 1, 0])
        v = np.cross(ax, u)
        # keep the outer rim + a coarse subset (enough for extents / frustum)
        sub = d[np.argsort(-radial)[: max(200, len(d) // 50)]]
        S = []
        for a in ang:
            ca, sa = math.cos(a), math.sin(a)
            # Rodrigues rotation about ax
            Rm = ca * np.eye(3) + sa * np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]) + (1 - ca) * np.outer(ax, ax)
            S.append(sub @ Rm.T + ctr)
        S = np.vstack(S)
        swept.append(S)
        # spin direction lives in the thrust curve (rot_dir), not the USD;
        # record the prop mesh name (cw / ccw) when the asset says it
        rot_info.append(dict(name=name, centre=[round(float(x), 4) for x in ctr],
                             axis=[round(float(x), 3) for x in ax], prop_radius=round(rp, 4),
                             mesh=[m.GetName() for m in meshes],
                             reach_xy=round(float(np.hypot(ctr[0], ctr[1]) + rp), 4),
                             z=[round(float(P[:, 2].min()), 4), round(float(P[:, 2].max()), 4)]))
    else:
        static.append(P)
        r = np.hypot(P[:, 0], P[:, 1])
        k = int(np.argmax(r))
        res["parts"][name] = dict(n_pts=int(len(P)),
            x=[round(float(P[:, 0].min()), 4), round(float(P[:, 0].max()), 4)],
            y=[round(float(P[:, 1].min()), 4), round(float(P[:, 1].max()), 4)],
            z=[round(float(P[:, 2].min()), 4), round(float(P[:, 2].max()), 4)],
            r_xy_max=round(float(r[k]), 4), r_xy_max_at=[round(float(v), 4) for v in P[k]],
            r_xy_p99=round(float(np.percentile(r, 99)), 4),
            r_xy_p95=round(float(np.percentile(r, 95)), 4),
            r3_max=round(float(np.linalg.norm(P, axis=1).max()), 4))
res["rotors"] = rot_info
A = np.vstack(static + swept)
rxy = np.hypot(A[:, 0], A[:, 1])
r3 = np.linalg.norm(A, axis=1)
res["all"] = dict(r_xy_max=round(float(rxy.max()), 4), r3_max=round(float(r3.max()), 4),
                  z=[round(float(A[:, 2].min()), 4), round(float(A[:, 2].max()), 4)],
                  x=[round(float(A[:, 0].min()), 4), round(float(A[:, 0].max()), 4)],
                  y=[round(float(A[:, 1].min()), 4), round(float(A[:, 1].max()), 4)])
if swept:
    Sw = np.vstack(swept)
    res["props_swept"] = dict(r_xy_max=round(float(np.hypot(Sw[:, 0], Sw[:, 1]).max()), 4),
                              r3_max=round(float(np.linalg.norm(Sw, axis=1).max()), 4))
if static:
    St = np.vstack(static)
    res["static"] = dict(r_xy_max=round(float(np.hypot(St[:, 0], St[:, 1]).max()), 4),
                         r3_max=round(float(np.linalg.norm(St, axis=1).max()), 4))

# radial profile of the static airframe by azimuth (which direction sticks out)
if static:
    St = np.vstack(static)
    az = np.degrees(np.arctan2(St[:, 1], St[:, 0]))
    prof = []
    for a0 in range(-180, 180, 30):
        mk = (az >= a0) & (az < a0 + 30)
        if mk.any():
            prof.append((a0, round(float(np.hypot(St[mk, 0], St[mk, 1]).max()), 3)))
    res["static_rxy_by_azimuth30"] = prof

# --- policy depth camera: body (0.10, 0, 0), pitched 13 deg down, 91 deg hfov, 640x480
def cam_view(pos, pitch_deg, hfov=91.0, W=640, H=480):
    th = math.radians(pitch_deg)
    fwd = np.array([math.cos(th), 0.0, -math.sin(th)])
    left = np.array([0.0, 1.0, 0.0])
    up = np.cross(fwd, left)
    fx = 0.5 * W / math.tan(math.radians(hfov) / 2)
    out = {}
    for nm, P in (("static", np.vstack(static)), ("props_swept", np.vstack(swept) if swept else np.zeros((0, 3)))):
        d = P - np.asarray(pos)
        zc = d @ fwd
        xc = -(d @ left); yc = -(d @ up)
        ok = zc > 1e-4
        u = fx * xc[ok] / zc[ok] + W / 2; v = fx * yc[ok] / zc[ok] + H / 2
        inimg = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zz = zc[ok][inimg]
        out[nm] = dict(n_in_frustum=int(inimg.sum()),
                       max_depth=round(float(zz.max()), 4) if len(zz) else None,
                       min_depth=round(float(zz.min()), 4) if len(zz) else None,
                       v_rows=[int(v[inimg].min()), int(v[inimg].max())] if len(zz) else None)
    return out
res["camera_010_pitch13"] = cam_view((0.10, 0, 0), 13.0)
res["camera_010_pitch0"] = cam_view((0.10, 0, 0), 0.0)
# camera origin inside the static hull? (ray casts along +-x/+-y/+-z hit counts is overkill;
# report the nearest static points ahead of the camera on the optical axis corridor)
St = np.vstack(static)
corr = (np.abs(St[:, 1]) < 0.02) & (np.abs(St[:, 2]) < 0.02)
res["static_front_on_axis_x"] = round(float(St[corr, 0].max()), 4) if corr.any() else None
res["static_back_on_axis_x"] = round(float(St[corr, 0].min()), 4) if corr.any() else None

# physics authoring
phys = []
for p in Usd.PrimRange(root):
    if p.HasAPI(UsdPhysics.CollisionAPI):
        appr = p.GetAttribute("physics:approximation").Get() if p.HasAttribute("physics:approximation") else None
        en = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
        phys.append((p.GetPath().pathString, appr, en))
    if p.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(p)
        phys.append((p.GetPath().pathString, "MASS", m.GetMassAttr().Get(), str(m.GetDiagonalInertiaAttr().Get()),
                     str(m.GetCenterOfMassAttr().Get()), m.GetDensityAttr().Get()))
res["physics"] = phys
res["cameras"] = [p.GetPath().pathString for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Camera)]
res["all_prims"] = len(list(Usd.PrimRange(root)))
print(json.dumps(res, indent=1, default=str))
if out_json:
    json.dump(res, open(out_json, "w"), indent=1, default=str)
