# Porting the DiffAero policy into the PX4 sim — step-by-step guide

You're adding a **second** vision-based flight policy (DiffAero / SHA2C) next to the existing
**DiffPhysDrone** one, sharing the same Isaac Sim + PX4 SITL harness. This guide is meant to be
followed top to bottom: each step says *where* to edit, gives you the *complete* code to paste,
and then explains *what it does and why*. When a step fixes something already half-written on
your branch, I call that out.

Reference solution (peek only after you've tried a step):

```bash
git diff tutorial-practice origin/diffaero-solution -- starling-deployment/run_px4_sim.py
git show origin/diffaero-solution:starling-deployment/diffaero_offboard.py
```

---

## 0. The big picture — two processes, one UDP pipe

The system is **two OS processes** that talk over a UDP socket. This split is deliberate: it
mirrors the real Starling/VOXL2 drone, where the depth-camera driver and the flight controller
are separate programs.

```
┌─────────────────────────────────────────┐         ┌──────────────────────────────────────┐
│ Process A: run_px4_sim.py                │         │ Process B: *_offboard.py               │
│ (Isaac Sim + Pegasus + PX4 SITL)         │         │ (the policy / "offboard controller")   │
│                                          │         │                                        │
│  • spawns world + obstacle field         │ depth   │  • connects to PX4 over MAVLink        │
│  • spawns drone w/ forward depth camera  │ frames  │  • subscribes to depth frames          │
│  • every render tick:                    │  UDP    │  • 30 Hz: obs → policy → attitude cmd  │
│      render depth → publish ─────────────┼────────▶│  • sends SET_ATTITUDE_TARGET to PX4    │
│                                          │ :15001  │                                        │
└─────────────────────────────────────────┘         └──────────────────────────────────────┘
        ▲                                                          │
        │                MAVLink (PX4 SITL UDP :14550)             │
        └──────────────────────────────────────────────────────────┘
              state (pose, vel) flows up; attitude setpoints flow down
```

The pipe in the middle — [depth_transport.py](../depth_transport.py) — is **policy-agnostic**:
its wire header carries `(seq, height, width)` per frame, so a 48×64 DiffPhys frame and a 9×16
DiffAero frame both ride the same socket with **zero changes**. That's why you only ever touch
the two endpoints.

`--policy diffphys|diffaero` picks which camera config the sim uses and which offboard script
you run. The two policies are **parallel implementations of the same contract**. Keep this
table nearby — almost every step below is one row of it:

| Stage | DiffPhysDrone (`diffphys`) | DiffAero (`diffaero`) |
|---|---|---|
| Policy grid (after pooling) | 12×16, pitched **20° down** | 9×16, **forward** (0°) |
| Render resolution | 48×64 | 36×64 (4× the 9×16 grid) |
| Horizontal FOV | 78.6° | 86° |
| Depth value shipped on the wire | planar **Z-depth** (m) | Euclidean **range** (m) |
| Depth → network encoding | `3/clamp(d,0.3,24) − 0.6`, then `max_pool 4×4` | `1 − clamp(r,0,5)/5` |
| Policy output | net accel (gravity already cancelled) | **thrust** accel + attitude quaternion |
| Who builds the attitude quaternion | the offboard script | the exported actor (baked in) |

Two terms you'll see constantly:
- **Planar Z-depth**: distance along the camera's optical axis (what a depth sensor's "distance
  to image plane" gives). **Euclidean range**: straight-line distance to the surface. They differ
  for off-axis pixels — that mismatch drives Step 1.4.
- **ENU/FLU vs NED/FRD**: the policies think in ENU (x-East, y-North, z-Up) with an FLU body
  (x-Forward, y-Left, z-Up). PX4 thinks in NED/FRD. Every setpoint crosses that boundary and
  gets converted.

---

# PART 1 — Finish the sim side (`run_px4_sim.py`)

You've already written the DiffPhys camera and the `DA_*` constants block. Five edits remain.
Do them in this order; I note the bugs already in your tree as we reach them.

## Step 1.1 — Import the DiffAero obstacle generator

**Where:** the import near the top, currently:

```python
from obstacle_field import generate as generate_field
```

**Change it to:**

```python
from obstacle_field import generate as generate_field, generate_diffaero
```

**Why:** You wrote `generate_diffaero` back in Step 1
([solutions/step1_diffaero_obstacles_solution.py](solutions/step1_diffaero_obstacles_solution.py)).
It produces the DiffAero *training distribution* of obstacles (30 obstacles, 33% spheres + 67%
tall tilted pillars, scattered around the start→goal line). The point of evaluating a policy in
sim is to put it in scenes drawn from the *same distribution it trained on* — otherwise you're
testing out-of-distribution and a failure tells you nothing.

## Step 1.2 — Select the obstacle field by `--obstacles`

**Where:** in `__init__`, the line that currently always uses the DiffPhys field:

```python
self.field = generate_field(seed=seed, scale=scale)
```

**Replace with:**

```python
if self.obstacles == "diffaero":
    self.field = generate_diffaero(seed=seed, scale=scale)
else:
    self.field = generate_field(seed=seed, scale=scale)
print("[obstacle_field]", self.field.summary())
self._spawn_obstacles(self.field)
```

**Why:** `self.obstacles` is already parsed from `--obstacles` and stored in `__init__`, but
nothing read it yet — the flag was dead. Both generators return the **same** `ObstacleField`
dataclass (spheres/boxes/cylinders + start/goal), so `_spawn_obstacles` consumes either one
without knowing which generator produced it. That shared container is exactly why Step 1 was
worth building as its own abstraction: the spawner is reused untouched. (Keep `_spawn_obstacles`
on the line right after, as you have it.)

## Step 1.3 — Fix the camera dispatch name mismatch (bug)

**Where:** in `__init__`:

```python
if self.policy == "diffaero":
    self._setup_camera_diffaero_intrinsics()    # ← method doesn't exist
else:
    self._setup_camera_diffphys_intrinsics()    # ← method doesn't exist
```

**Replace with:**

```python
if self.policy == "diffaero":
    self._setup_camera_diffaero()
else:
    self._setup_camera_diffphys()
```

**Why:** Your methods are named `_setup_camera_diffaero` / `_setup_camera_diffphys` (no
`_intrinsics` suffix), so as written this raises `AttributeError` the instant you launch.
Make the call sites match the method names. (Either naming is fine — just pick one and be
consistent. The solution uses the bare names.)

## Step 1.4 — Finish `_setup_camera_diffaero()` (the Euclidean-scale table)

**Where:** your method stops partway, at:

```python
    u = np.arange(DA_RENDER_W, dtype=np.float32)
```

and the `_intrinsics` matrix just above it has a stray `]` (count the brackets — there's one
too many). Here is the **complete, correct method** to replace what you have from the `def`
line down:

```python
def _setup_camera_diffaero(self):
    """Forward-facing depth camera for DiffAero: 86° HFOV, no downward pitch,
    rendered at 36x64 (4x the 9x16 policy grid) so a 4x4 min-pool preserves
    thin pillars."""
    self._camera = MonocularCamera("depth_cam", config={
        "depth": True,
        "position": np.array([0.20, 0.0, 0.05]),     # body-frame mount offset (m)
        "orientation": np.array([0.0, -DA_CAM_ANGLE_DEG, 180.0]),  # 180° yaw = look forward
        "resolution": (DA_RENDER_W, DA_RENDER_H),     # (width, height) = (64, 36)
        "frequency": 30,
        "intrinsics": None,                           # fall back to FOV; we override below
    })

    # Pinhole intrinsics from the FOVs. fy != fx because the 16:9 grid is wide,
    # but the pixels are square, so the vertical FOV is derived, not independent.
    self._camera.fov = DA_FOV_X_DEG
    self._camera.fx = 0.5 * DA_RENDER_W / math.tan(0.5 * math.radians(DA_FOV_X_DEG))
    self._camera.fy = 0.5 * DA_RENDER_H / math.tan(0.5 * math.radians(DA_FOV_Y_DEG))
    self._camera.cx = 0.5 * DA_RENDER_W
    self._camera.cy = 0.5 * DA_RENDER_H
    self._camera._intrinsics = np.array([
        [self._camera.fx, 0.0, self._camera.cx],
        [0.0, self._camera.fy, self._camera.cy],
        [0.0, 0.0, 1.0],
    ])

    # --- Precompute the planar-Z → Euclidean-range scale table (once) ---
    u = np.arange(DA_RENDER_W, dtype=np.float32)
    v = np.arange(DA_RENDER_H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)                        # uu,vv shape (36, 64); row=v, col=u
    xn = (uu - self._camera.cx) / self._camera.fx     # normalized image-plane x per pixel
    yn = (vv - self._camera.cy) / self._camera.fy     # normalized image-plane y per pixel
    self._da_euclid_scale = np.sqrt(1.0 + xn*xn + yn*yn).astype(np.float32)
```

**Why the intrinsics:** `fx`/`fy` are the focal lengths in pixels; the standard pinhole relation
is `f = 0.5 * size / tan(0.5 * FOV)`. We compute `fy` from `DA_FOV_Y_DEG` (which is *derived*:
`86° × 9/16 = 48.375°`) — **not** from `fx` — because the grid is wide-but-not-tall while the
pixels are square. If you set `fy = fx` (as DiffPhys does, where its grid is closer to square)
the vertical scaling would be wrong.

**Why the scale table (the crux of Step 1):** Isaac's `get_depth()` returns **planar Z-depth** —
distance along the optical axis. DiffAero was trained on **Euclidean range** — true straight-line
distance. For a pixel at normalized coords `(xn, yn)`, the ray to that pixel makes an angle with
the optical axis, and:

```
range = z_depth × sqrt(1 + xn² + yn²)
```

On the optical axis (`xn=yn=0`) the factor is 1; toward the corners it grows. Because the camera
intrinsics never change, this factor is the **same every frame** — so you build the `(36, 64)`
table once here and just multiply each depth frame by it in Step 1.5. Cheap and exact, no
per-frame trig.

## Step 1.5 — Make `_publish_depth` a dispatcher and add the DiffAero publisher

**Where:** you currently have one `_publish_depth(self)` method (the DiffPhys one). Two edits:

**(a) Rename** your existing method from `def _publish_depth(self):` to
`def _publish_depth_diffphys(self):` — leave its body exactly as-is.

**(b) Add** a dispatcher and the new DiffAero publisher right above it:

```python
def _publish_depth(self):
    """Dispatch to the per-policy depth publisher (called every render tick)."""
    if self.policy == "diffaero":
        self._publish_depth_diffaero()
    else:
        self._publish_depth_diffphys()

def _publish_depth_diffaero(self):
    """Render → Euclidean range → 9x16 → publish over UDP for the DiffAero policy."""
    cam = getattr(self._camera, "_camera", None)
    if cam is None or not getattr(self._camera, "_camera_full_set", False):
        return                                  # camera not initialized yet this frame
    depth = cam.get_depth()                     # planar Z-depth, ~(36, 64)
    if depth is None:
        return

    # 1) Sanitize: sky / no-return pixels come back as inf/nan. Map them to a big
    #    "far" value so they encode as perception ≈ 0 (nothing there), not garbage.
    far = 1e3
    depth = np.nan_to_num(np.asarray(depth, np.float32), nan=far, posinf=far, neginf=far)

    # 2) Defensive resize to the exact render grid (Isaac usually already gives this).
    if depth.shape != (DA_RENDER_H, DA_RENDER_W):
        yi = np.linspace(0, depth.shape[0] - 1, DA_RENDER_H).astype(np.int64)
        xi = np.linspace(0, depth.shape[1] - 1, DA_RENDER_W).astype(np.int64)
        depth = depth[yi][:, xi]

    # 3) Planar Z-depth → Euclidean range, using the precomputed per-pixel table.
    euclid = depth * self._da_euclid_scale          # (36, 64)

    # 4) Min-pool 4x4 down to the 9x16 policy grid (keep the NEAREST surface per cell).
    euclid = euclid.reshape(DA_OUT_H, DA_POOL, DA_OUT_W, DA_POOL).min(axis=(1, 3))  # (9,16)

    self._depth_pub.send(np.ascontiguousarray(euclid, np.float32))
    self._dump_depth_debug(euclid)
```

**Why min-pool (not average or max)?** When you shrink 4×4 → 1, you want the **closest**
obstacle in that cell to survive — averaging would blend a thin pillar into the far background
and the drone would clip it. (Side note: DiffPhys `max_pool`s `3/d`, which is *also* a
min-over-distance because `3/d` shrinks as `d` grows — same intent, mirrored encoding.)

**Why render 36×64 then pool, instead of rendering 9×16 directly?** A raw 9×16 raster is so
coarse that a thin pillar can fall *between* pixel centers and disappear entirely. Rendering 4×
denser and min-pooling is a cheap supersample that guarantees thin obstacles register.

**Why ship Euclidean range here but raw Z-depth on the DiffPhys path?** Each policy is fed what
it saw in training (see the table). The transport doesn't care — it ships whatever array you
hand it.

## Step 1.6 — Make the debug colormap range policy-aware

**Where:** in `_dump_depth_debug`, the `imshow` call hardcodes the DiffPhys clamp:

```python
im = ax.imshow(depth, origin="upper", cmap="turbo", vmin=0.3, vmax=24.0)
```

**Replace with:**

```python
vmax = DA_MAX_DIST if self.policy == "diffaero" else 24.0
im = ax.imshow(depth, origin="upper", cmap="turbo", vmin=0.3, vmax=vmax)
```

**Why:** DiffAero range tops out at 5 m. With `vmax=24` every DiffAero frame washes out to dark
blue and the debug PNG is useless. Scaling the colormap to 5 m makes obstacle structure visible
so you can actually debug orientation/alignment in Part 3.

## Step 1.7 — Sanity-check the camera math (no Isaac needed)

```bash
python3 starling-deployment/tutorial/solutions/step2_diffaero_camera_solution.py
```

Expect: `hfov 86°`, `vfov 48.375°`, output shape `(9, 16)`, close obstacles → high perception,
far → ~0. This runs the *same math* you just wrote, in pure numpy, so a green check here means
your intrinsics + scale table + pooling are correct before you ever boot the simulator.

That's the entire sim side. **Commit it** before moving on — Part 2 is a separate process, and
you want a known-good checkpoint to fall back to.

---

# PART 2 — The offboard policy (`diffaero_offboard.py`)

This is where "how it all ties together" really lives. **Start from a copy** of
[diffdrone_offboard.py](../diffdrone_offboard.py):

```bash
cp starling-deployment/diffdrone_offboard.py starling-deployment/diffaero_offboard.py
```

The entire MAVLink scaffold — connect, arm, OFFBOARD handshake, receive thread, CLIMB→POLICY
phase machine, heartbeats — is **identical** and should be reused verbatim. You'll only rewrite
the policy-specific pieces. Below, "keep" = leave the copied code alone; "replace/add" = your
edits. I go in the order you'd touch the file top to bottom.

## Step 2.1 — Understand what you're keeping (don't rewrite these)

Read these in your copy and convince yourself they're policy-independent:

- **`DroneState` + `receive_loop`** — a background thread reads MAVLink messages, converts
  NED/FRD → ENU/FLU, and stores pose/velocity/rotation behind a lock. The control loop just
  calls `state.get()`. Same for both policies.
- **`quat_ENU_FLU_to_NED_FRD`** — converts an ENU/FLU rotation to the `[w,x,y,z]` quaternion
  PX4 wants. Both policies output ENU; PX4 wants NED; this is the bridge.
- **MAVLink senders** — `send_attitude_target` (msg 82), `send_position_target_ned` (msg 84),
  `arm`, `set_offboard_mode`, `set_param_float`, `send_heartbeat`.
- **The two flight phases** — `CLIMB` streams *position* setpoints (PX4 requires a setpoint
  stream before it will enter OFFBOARD) and climbs to altitude; once arrived + settled + in
  OFFBOARD it hands off to `POLICY`, which streams *attitude* setpoints from the network.

Now the parts that change.

## Step 2.2 — Replace the constants block

**Where:** the top-of-file constants. **Replace the DiffPhys-specific ones with:**

```python
CONTROL_HZ = 30.0          # DiffAero trained at dt = 0.0333 s; run inference at the same rate
HEARTBEAT_HZ = 2.0
G = 9.80665

PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# DiffAero perception grid (must match the sim publisher exactly): height x width
DEPTH_H, DEPTH_W = 9, 16
# Camera max range [m] (sensor.max_dist). The encoding is depth = 1 - clamp(r,0,5)/5.
CAM_MAX_DIST = 5.0

# Point-mass action limits (max_acc.xy / max_acc.z from the checkpoint config).
MAX_ACC_XY = 20.0
MAX_ACC_Z = 40.0
```

**Why:** Every one of these is read off `checkpoints/DiffAero/sha2c_pmc/hydra/config.yaml`. The
network has no idea what units you feed it — it only knows the distribution it trained on. If
`DEPTH_H/W` or `CAM_MAX_DIST` disagree with the sim side, the perception image is meaningless;
if `CONTROL_HZ` is wrong, the velocities the policy integrates against are off.

## Step 2.3 — Rewrite the attitude/thrust converter (the gravity gotcha)

**Where:** DiffPhys has `act_world_to_attitude_target(...)`. **Delete it and add:**

```python
def diffaero_quat_to_attitude_target(quat_xyzw_enu_flu, acc_norm, max_accel, verbose=False):
    """Convert the DiffAero attitude command (ENU/FLU quaternion, xyzw) + thrust-accel
    magnitude into (q_des_ned_frd [w,x,y,z], thrust_norm [0-1]) for SET_ATTITUDE_TARGET."""
    R_des_enu = Rotation.from_quat([
        quat_xyzw_enu_flu[0], quat_xyzw_enu_flu[1],
        quat_xyzw_enu_flu[2], quat_xyzw_enu_flu[3],
    ]).as_matrix()
    q_des = quat_ENU_FLU_to_NED_FRD(R_des_enu)           # → [w,x,y,z] NED/FRD
    thrust_norm = float(np.clip(acc_norm / max_accel, 0.0, 1.0))
    if verbose:
        rpy = Rotation.from_matrix(R_des_enu).as_euler("xyz", degrees=True)
        print(f"  acc_norm={acc_norm:.3f}  des RPY=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f})  "
              f"thrust_norm={thrust_norm:.3f}")
    return q_des, thrust_norm
```

**Why this is the single most important difference** (and the easiest way to flip the drone):

- **DiffPhysDrone** outputs *net* acceleration with gravity **already cancelled** inside its
  dynamics. So its offboard code has to **add** `[0,0,g]` back to recover the true thrust vector
  (`thrust_accel = act_world + [0,0,g]`), then builds a quaternion from that thrust direction.
- **DiffAero** is a point-mass model where gravity is handled separately, and its exported actor
  already **baked the attitude quaternion** for you. So you do **NOT** add gravity and you do
  **NOT** build a quaternion — you just convert the actor's quaternion to NED/FRD and normalize
  the thrust magnitude. If you copy the DiffPhys `+[0,0,g]`, the drone over-thrusts and flips.

**Thrust normalization:** `thrust_norm = acc_norm / max_accel` maps a thrust acceleration to
PX4's `[0,1]` throttle. So hover (`acc_norm ≈ g`) lands at `g/max_accel`. You must tell PX4 the
**same** hover point via `MPC_THR_HOVER = g/max_accel` (Step 2.6) or the drone sinks/rockets at
handoff.

## Step 2.4 — Rewrite the policy wrapper (`DiffAeroPolicy`)

**Where:** replace the `DiffDronePolicy` class. This is the heart of the port. Build it
incrementally — first the loader, then `normalize_depth`, then `step`.

### 2.4a — Loader

```python
class DiffAeroPolicy:
    def __init__(self, checkpoint_path, vel_ema_factor=0.1,
                 max_acc_xy=MAX_ACC_XY, max_acc_z=MAX_ACC_Z):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pt2_path = self._resolve_pt2(checkpoint_path)
        print(f"Loading DiffAero TorchScript actor from {pt2_path} ...")
        self.module = torch.jit.load(str(pt2_path), map_location=self.device)
        self.module.eval()

        # Action limits the actor rescales its tanh output into.
        self.min_action = torch.tensor([[-max_acc_xy, -max_acc_xy, 0.0]],
                                        dtype=torch.float32, device=self.device)
        self.max_action = torch.tensor([[max_acc_xy, max_acc_xy, max_acc_z]],
                                        dtype=torch.float32, device=self.device)

        self.vel_ema_factor = float(vel_ema_factor)
        self.vel_ema = None                                  # velocity EMA → commanded yaw
        self._up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

    @staticmethod
    def _resolve_pt2(checkpoint_path):
        p = Path(checkpoint_path)
        if p.is_file():
            return p
        for c in (p / "checkpoints" / "exported_actor.pt2", p / "exported_actor.pt2"):
            if c.exists():
                return c
        raise FileNotFoundError(f"No exported_actor.pt2 under {checkpoint_path}")

    def reset(self):
        self.vel_ema = None
```

**Why TorchScript and not a raw nn.Module?** DiffAero ships a **self-contained** exported actor
(`checkpoints/exported_actor.pt2`) that bakes the entire output chain — `tanh → rescale to
[min_action, max_action] → Rz @ action → point-mass attitude quaternion` — into the graph. You
load it and call it; you don't reimplement the network. That's why your Python does *less* math
than the DiffPhys wrapper.

### 2.4b — Perception encoding

```python
    def normalize_depth(self, range_m):
        """Metric Euclidean range (9,16) → trained perception encoding (1,9,16).
           close → ~1, far / no-return → 0."""
        d = torch.as_tensor(range_m, dtype=torch.float32, device=self.device)
        d = d.clamp(0.0, CAM_MAX_DIST)
        perception = 1.0 - d / CAM_MAX_DIST
        return perception.reshape(1, DEPTH_H, DEPTH_W)
```

**Why this exact formula:** it's the transform the network saw in training. Note the sign is the
**opposite** of DiffPhys (which used `3/d − 0.6`): here a *near* obstacle gives a value close to
1 and empty space gives 0. The sim already shipped Euclidean range in metres (Step 1.5), so the
wrapper only applies this one transform.

### 2.4c — The `step` method (building the observation)

```python
    @torch.no_grad()
    def step(self, position_enu, velocity_enu, R_enu, target_pos_enu,
             max_vel=5.0, depth_range=None):
        R = torch.tensor(R_enu, dtype=torch.float32, device=self.device)
        v_world = torch.tensor(velocity_enu, dtype=torch.float32, device=self.device)

        # --- yaw-only LOCAL frame (columns = local axes expressed in world) ---
        # This matches diffaero's axis_rotmat("Z", yaw): forward flattened to horizontal,
        # left = up × forward, up = world +Z.
        fwd = R[:, 0].clone()
        fwd[2] = 0.0
        fwd = torch.nn.functional.normalize(fwd, dim=0)
        left = torch.cross(self._up, fwd, dim=0)
        Rz = torch.stack([fwd, left, self._up], dim=1)       # (3,3)
        uz = R[:, 2]                                         # body up-axis in world

        # --- target velocity: head toward goal, capped at max_vel ---
        relpos = (torch.tensor(target_pos_enu, dtype=torch.float32, device=self.device)
                  - torch.tensor(position_enu, dtype=torch.float32, device=self.device))
        dist = relpos.norm()
        denom = torch.maximum(dist / max_vel, torch.ones((), device=self.device))
        target_vel_world = relpos / denom                    # |.| ≤ max_vel

        # --- project into the local frame; assemble the 9-D state ---
        target_vel_local = Rz.t() @ target_vel_world         # Rz.t() = world → local
        v_local = Rz.t() @ v_world
        state9 = torch.cat([target_vel_local, uz, v_local]).unsqueeze(0)   # (1,9)

        # --- commanded yaw from a velocity EMA (align_yaw_with_vel_ema) ---
        if self.vel_ema is None:
            self.vel_ema = v_world.clone()
        else:
            self.vel_ema = torch.lerp(self.vel_ema, v_world, self.vel_ema_factor)
        orientation = self.vel_ema.unsqueeze(0)
        if orientation.norm() < 1e-3:                        # near-stationary: yaw undefined
            orientation = fwd.unsqueeze(0)

        # --- perception ---
        if depth_range is not None:
            perception = self.normalize_depth(depth_range)
        else:
            perception = torch.zeros(1, DEPTH_H, DEPTH_W, device=self.device)

        # --- run the baked actor ---
        acc_cmd, quat_cmd, acc_norm = self.module(
            (state9, perception), orientation, Rz.unsqueeze(0),
            self.min_action, self.max_action)

        return (acc_cmd.squeeze(0).cpu().numpy(),
                quat_cmd.squeeze(0).cpu().numpy(),
                float(acc_norm.reshape(-1)[0].cpu()))
```

**Why the local frame:** DiffAero's observation is `obs_frame=local` — everything is expressed in
the **yaw-only** frame (heading rotated out, but not pitch/roll). `Rz` has the local axes as
columns, so `Rz.t() @ world_vector` rotates a world vector *into* that frame (transpose = inverse
for a rotation). `uz` (body up in world) is left in world frame on purpose — that's how the
network learned to sense its own tilt.

**Why the 9-D order is `[target_vel_local, uz, v_local]`:** this exact concatenation is what the
network was trained on. Swap any block and the policy will fly confidently into obstacles — a
mis-ordered observation is the #1 cause of "it ignores everything." Don't improvise the order.

**Why a velocity EMA for yaw:** DiffAero points the nose where it's *going*, not where it happens
to face (`align_yaw_with_vel_ema`). The EMA smooths velocity so heading doesn't jitter; the
exported actor turns `orientation` into the commanded yaw and folds it into `quat_cmd`. When
nearly stopped, velocity direction is meaningless, so fall back to current forward.

**Why zeros when there's no depth:** zero perception = "everything is far / clear," so before the
first UDP frame arrives the drone just cruises straight toward the goal instead of dodging
phantoms.

## Step 2.5 — Wire depth + goal into `main()`

**Where:** in `main()`. **Keep** the `--depth` subscriber block as-is (it's transport-agnostic):

```python
depth_sub = None
if args.depth:
    from depth_transport import DepthSubscriber
    depth_sub = DepthSubscriber()
```

**Replace** the policy construction and the `--max-accel`/`--max-vel` argparse defaults to
DiffAero values:

```python
parser.add_argument("--max-vel", type=float, default=5.0,
                    help="Target cruise speed [m/s]; training sampled [3, 6].")
parser.add_argument("--max-accel", type=float, default=30.0,
                    help="Thrust accel mapping to full throttle [m/s^2]; hover = g/max_accel.")
...
policy = DiffAeroPolicy(args.checkpoint)
```

## Step 2.6 — Set the hover throttle, then the phase loop

**Where:** the startup just before the main loop. **Keep** the existing `MPC_THR_HOVER` block but
make sure it uses the DiffAero mapping:

```python
hover_thrust = float(np.clip(G / args.max_accel, 0.0, 1.0))
set_param_float(mav, "MPC_THR_HOVER", hover_thrust)
```

**Why:** this is the matching half of Step 2.3. Your converter maps `acc_norm → acc_norm/max_accel`,
so hover (`g`) → `g/max_accel`; PX4 must agree or the handoff lurches.

**Where:** inside the `POLICY` branch of the loop. **Replace** the DiffPhys call with:

```python
acc_cmd, quat_cmd, acc_norm = policy.step(
    pos, vel, R_enu, goal_enu, max_vel=args.max_vel, depth_range=depth_range)
q_des, thrust_norm = diffaero_quat_to_attitude_target(
    quat_cmd, acc_norm, args.max_accel, verbose=verbose)
send_attitude_target(mav, q_des, thrust_norm)
```

**Keep** the `goal_enu` construction (XY from `--goal`, Z pinned to `--climb-alt`; hover at `pos`
if no goal) and the `CLIMB` branch exactly as in the DiffPhys file. The CLIMB→POLICY transition,
the `depth_range = depth_sub.latest() if depth_sub else None` line, and the heartbeat cadence are
all unchanged.

---

# PART 3 — Run it and verify

Two terminals, after PX4 SITL is up:

```bash
# Terminal 1 — sim + camera + obstacles
python run_px4_sim.py --policy diffaero --obstacles diffaero --seed 0 --scale 5.0

# Terminal 2 — the policy
python diffaero_offboard.py --checkpoint checkpoints/DiffAero/sha2c_pmc --depth \
    --goal <goal_x> <goal_y> --climb-alt 10
```

(Get `<goal_x> <goal_y>` from the `[obstacle_field] ... target=[...]` line the sim prints at
startup.)

**Verification ladder — fix in this order, because each rung depends on the one below it:**

1. **Depth sane?** Open `camera_debug.png`. With the `vmax=5` fix (Step 1.6) obstacles should
   read **near 0 m** and open space near **5 m** — you should see structure, not a flat field.
   The RGB pane is the literal drone view; an obstacle in front in RGB must show as a near patch
   in the depth pane.
2. **Perception aligned?** An obstacle on the drone's **left** in RGB must light up the **left**
   columns of the 9×16 grid. If it's mirrored or upside down, you have a row/col flip — fix it in
   `_publish_depth_diffaero` *before* `self._depth_pub.send(...)` (so debug == published). The
   commented flip lines in the DiffPhys `_publish_depth` show the pattern (`depth[::-1]` for rows,
   `depth[:, ::-1]` for cols).
3. **Handoff stable?** At CLIMB→POLICY the drone should hold altitude, not lurch. A jump up/down
   almost always means `MPC_THR_HOVER ≠ g/max_accel` (Steps 2.3/2.6), or you accidentally kept the
   DiffPhys `+[0,0,g]`.
4. **Avoidance?** It should bend around pillars toward the goal. If it flies *into* them with
   confidence, suspect the `state9` field order or the perception sign (Step 2.4) — a confidently
   wrong policy is nearly always a swapped or mis-signed observation, not a "bad" network.

---

## Mental model to keep

- **Two endpoints, one dumb pipe.** All policy differences live in the two endpoints; the UDP
  transport never changes (its header carries H/W per frame).
- **Frames everywhere.** PX4 = NED/FRD. Policies = ENU/FLU. DiffAero's observation = yaw-only
  local. Every boundary is a conversion; most bugs are a missing or wrong one.
- **The train-time contract is law.** Grid shape, FOV, depth encoding, the 9-D state order, action
  limits, dt — all come from the checkpoint config and must be reproduced exactly. The network
  only knows the distribution it saw.
- **DiffAero does more in the graph, less in Python.** The exported actor bakes the attitude math,
  so the offboard code is *thinner* than DiffPhys — except the gravity convention flips, which is
  the one thing you must not copy from the DiffPhys file.
```
