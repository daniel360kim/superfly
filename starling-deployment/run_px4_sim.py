#!/usr/bin/env python
"""
Launch Pegasus + PX4 SITL with a single Iris quadrotor, plus a forward-facing
depth camera matching the DiffPhysDrone single_agent training config:
    fov_x_half_tan = 0.82  ->  horizontal FOV = 2*atan(0.82) = 78.6 deg
    cam_angle      = 20    ->  camera pitched 20 deg DOWN about the body left-axis
The 48x64 metric (planar Z) depth is published over UDP each control tick for
the offboard policy process to consume (see depth_transport.py).

Run this AFTER PX4 SITL is up; then run diffdrone_offboard.py --depth.

Procedural obstacle field (default):
    python run_px4_sim.py --seed 0 --obstacles diffphys

Warehouse background, no procedural obstacles (scene geometry only):
    python run_px4_sim.py --environment Warehouse --obstacles none --spawn 0 0 1.0 --policy diffaero
    python diffaero_offboard.py --checkpoint <dir> --depth --goal 15 0 --climb-alt 2.0

    python run_px4_sim.py --environment Warehouse --obstacles none --spawn 0 0 1.0 --policy diffphys
    python diffdrone_offboard.py --checkpoint <pt> --depth --goal 15 0 2.0 --climb-alt 2.0

Warehouse with Shelves (more rack rows; small ~40 m building, aisles along -X):
    # Spawn in the main aisle (~Y=1 m); fly toward the open end at X≈-3 m.
    python run_px4_sim.py --environment "Warehouse with Shelves" --obstacles none \
        --spawn -12 1 0.1 --policy diffaero
    python diffaero_offboard.py --checkpoint <dir> --depth --goal -3 1 --climb-alt 1.5 --max-vel 3.0

    python run_px4_sim.py --environment "Warehouse with Shelves" --obstacles none \
        --spawn -12 1 0.1 --policy diffphys
    python diffdrone_offboard.py --checkpoint <pt> --depth --goal -3 1 1.5 --climb-alt 1.5 --max-speed 3.0

Custom USD stage (e.g. ConiferForest), centimetre-authored so scaled to metres
with --env-scale 0.01, no procedural obstacles (fly through the scene geometry):
    python run_px4_sim.py --obstacles none --policy diffaero --spawn 0 0 1.0 \
        --usd-environment omniverse://airlab-nucleus.andrew.cmu.edu/Library/Stages/ConiferForest/ConiferForest_stage.stage.usd \
        --env-scale 0.01
    python diffaero_offboard.py --checkpoint <dir> --depth --goal 15 0 --climb-alt 2.0

Training-distribution layout but with realistic geometry (--obstacle-assets swaps
each procedural primitive for a USD asset from OBSTACLE_ASSETS, scaled to fit):
    python run_px4_sim.py --obstacles diffaero --policy diffaero --obstacle-assets
    python diffaero_offboard.py --checkpoint <dir> --depth --goal 40 30 --climb-alt 2.0

Velocity-command DiffAero policy (PX4 velocity loop, no depth for env=pc checkpoints):
    python run_px4_sim.py --environment Warehouse --obstacles none --spawn 0 0 1.0 \
        --policy diffaero --auto-stop --no-debug-frames
    python diffaero_vel_offboard.py --checkpoint checkpoints/DiffAero/sha2c_vel_cmd \
        --goal 15 0 --climb-alt 2.0 --quiet
"""

import argparse
import math
import subprocess
import carb
from isaacsim import SimulationApp

# SimulationApp must boot before any other omni/isaacsim import below, so this
# flag is parsed separately from main()'s argparse (which runs much later).
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--headless", action="store_true",
                          help="Run Isaac Sim without the GUI viewport. The GUI's RTX "
                               "render is usually the actual frame-rate bottleneck, not "
                               "the policy/physics; use camera_debug.png/depth_debug.npy "
                               "to inspect the drone's view instead of watching live.")
_pre_args, _ = _pre_parser.parse_known_args()

simulation_app = SimulationApp({"headless": _pre_args.headless})

import time
from pathlib import Path
import omni.timeline
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless PNG backend, no GUI needed
import matplotlib.pyplot as plt
import cv2
from omni.isaac.core.world import World
from isaacsim.core.api.objects import FixedCuboid, FixedSphere
import isaacsim.core.utils.prims as prim_utils
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_transport import DepthPublisher, RENDER_H, RENDER_W
from rgb_transport import RGBPublisher
from agile_debug_transport import AgileDebugSubscriber
from obstacle_field import generate as generate_field


# --- Realistic asset catalog for --obstacle-assets (Stage 1) ---
# Each procedural obstacle primitive is replaced by a random USD asset from its
# category, scaled so its bounding box matches the primitive's extent (so the
# depth distribution the policy sees stays close to its training distribution).
#
# Categories:
#   "tall" -> slender, vertical obstacles (DiffAero pillars, cylinders): trees
#             (outdoor) and poles/buildings (urban) are mixed here.
#   "low"  -> low, blocky clutter (ground voxels): bushes, rocks, crates.
#   "rock" -> spheres: boulders/rocks (scaled uniformly to stay round).
#
# Entries are either an absolute "scheme://.../foo.usd" URL, or a path relative to
# the NVIDIA assets root (isaacsim.storage.native.get_assets_root_path(), the same
# root Pegasus uses for /Isaac/Environments). At spawn time each entry is verified
# to exist; anything that does not resolve is skipped with a warning and the
# obstacle falls back to its analytic primitive.
#
# The paths below are all verified present in the public Isaac 5.1 asset library
# (the default assets root). NOTE: that library has no street furniture (poles/
# lamps) and no rock meshes, so Stage 1 here is a vegetation scene: "tall" mixes
# broad and columnar trees (cypress/poplar approximate pole-like urban verticals)
# and spheres map to rounded shrubs. For genuine urban geometry, add asset paths
# from your own Nucleus (airlab-nucleus.andrew.cmu.edu/Library) or use a full
# city USD via --usd-environment (Stage 2).
OBSTACLE_ASSETS = {
    # Slender, tall obstacles (DiffAero pillars, vertical cylinders) -> trees.
    "tall": [
        "/NVIDIA/Assets/Vegetation/Trees/American_Beech.usd",
        "/NVIDIA/Assets/Vegetation/Trees/Red_Oak.usd",
        "/NVIDIA/Assets/Vegetation/Trees/Colorado_Spruce.usd",
        "/NVIDIA/Assets/Vegetation/Trees/Douglas_Fir.usd",
        "/NVIDIA/Assets/Vegetation/Trees/Italian_Cypress.usd",   # columnar, pole-like
        "/NVIDIA/Assets/Vegetation/Trees/Lombardy_Poplar.usd",   # columnar, pole-like
    ],
    # Low, blocky clutter (ground voxels) -> shrubs + a crate for variety.
    "low": [
        "/NVIDIA/Assets/Vegetation/Shrub/Boxwood.usd",
        "/NVIDIA/Assets/Vegetation/Shrub/Holly.usd",
        "/NVIDIA/Assets/Vegetation/Shrub/Yew.usd",
        "/Isaac/Props/Blocks/nvidia_cube.usd",
    ],
    # Spheres -> rounded shrubs (no rock meshes in the default library), uniform-scaled.
    "rock": [
        "/NVIDIA/Assets/Vegetation/Shrub/Boxwood.usd",
        "/NVIDIA/Assets/Vegetation/Shrub/Rhododendron.usd",
        "/NVIDIA/Assets/Vegetation/Shrub/Holly.usd",
    ],
}

# --- DiffPhysDrone single_agent camera params ---
FOV_X_HALF_TAN = 0.82
CAM_ANGLE_DEG = 20.0
FOV_X_DEG = 2.0 * math.degrees(math.atan(FOV_X_HALF_TAN))  # ~78.6 deg horizontal

# --- DiffAero (sha2c_pmc) camera params (diffaero/cfg/sensor/camera.yaml) ---
# 16(w) x 9(h), horizontal FOV 86 deg, max range 5 m, forward-facing (no pitch),
# mounted at body [0.2, 0, 0.05]. DiffAero's perception is the EUCLIDEAN ray range
# encoded as 1 - clamp(r,0,5)/5. DiffAero defines vfov = hfov * H/W (angle-linear,
# = 48.375 deg), so we set fx/fy independently to match both FOVs.
DA_OUT_W, DA_OUT_H = 16, 9             # network perception grid (cols, rows)
DA_POOL = 4                            # min-pool factor (nearest surface per cell)
DA_RENDER_W, DA_RENDER_H = DA_OUT_W * DA_POOL, DA_OUT_H * DA_POOL  # (64, 36)
DA_FOV_X_DEG = 86.0
DA_FOV_Y_DEG = DA_FOV_X_DEG * DA_OUT_H / DA_OUT_W  # 48.375 deg (DiffAero definition)
DA_CAM_ANGLE_DEG = 0.0                 # forward, no downward pitch
DA_MAX_DIST = 5.0

# --- DepthNav camera params (depthnav training: 72x128 depth, no tilt) ---
# 128(w) x 72(h), horizontal FOV ~89 deg (habitat's default, scene_manager.py;
# training set no hfov override), near 0.25 / far 20 m, forward-facing (no pitch).
# depthnav consumes RAW planar metric depth: the policy clamps to [near, far] and
# inverts (1/(d+eps)) + maxpools internally (depthnav_policy.py), so we publish the
# planar Z-depth in metres with no normalization here.
DN_RENDER_W, DN_RENDER_H = 128, 72
DN_FOV_X_DEG = 89.0
DN_CAM_ANGLE_DEG = 0.0                  # forward, no downward pitch
DN_NEAR, DN_FAR = 0.25, 20.0

# --- Agile Autonomy (Loquercio) camera params (wrapper/agile_core.py) ---
# Matches uzh-rpg/agile_autonomy flightmare.yaml: 640x480, 91 deg horizontal
# FOV, forward-facing (pitch 0), far 20 m. For RGB, match Flightmare's published
# left-camera body translation [0, baseline, 0.1] with baseline=0.1 m. The
# original Superfly depth adapter retains its prior [0.1, 0, 0] mount.
#
# RESOLUTION: the Loquercio net's MobileNet backbone takes a 224x224 input, and
# the original training loader (planner_learning data_loader.decode_depth_cv2)
# fed it VGA-class SGM depth (640x480) DOWNSAMPLED to 224 via cv2.resize
# (bilinear). We reproduce that: render at AG_RENDER_W/H, then bilinear-resize
# to AG_NET_SIZE (224) and ship THAT.
AG_RENDER_W, AG_RENDER_H = 640, 480     # VGA native render (flightmare.yaml)
AG_NET_SIZE = 224                       # net input; bilinear-downsampled, shipped
AG_FOV_X_DEG = 91.0                     # flightmare.yaml camera.fov
AG_CAM_ANGLE_DEG = 0.0                  # forward, no downward pitch
AG_FAR = 20.0

# --- RGB "drone_camera" used only for video logging (--record-rgb-video) ---
# The policy depth cameras render at 64x36..128x72, far too small to watch, so
# recording RGB gets its own camera at a watchable resolution. It is mounted at
# the same body pose (and matches the horizontal FOV) of the active policy's
# depth camera, so the RGB video shows the same viewpoint the policy sees.
RGB_W, RGB_H = 640, 360

# PegasusInterface.load_environment() only schedules its async loader and
# returns immediately. A standalone app can therefore reset the World and
# create render products while remote USD dependencies are still loading. In
# particular, RTX may try to upload Box.usd's DomeLight HDR before its HTTPS
# stream is ready and permanently fall back to an untextured sky for that run.
ASSET_LOAD_TIMEOUT_S = 120.0

# Sentinel file diffaero_offboard.py / diffdrone_offboard.py touch on exit
# (any reason: landed, Ctrl-C, crash). --auto-stop polls for it so this script
# can stop through its own normal loop exit instead of needing a manual
# Ctrl-C, which races Isaac's SIGINT teardown (see run()).
OFFBOARD_DONE_FILE = str(Path(__file__).resolve().parent / ".superfly_offboard_done")

# Phase sentinel the offboard scripts append to ("start <unix_ts>" at the
# climb/yaw -> policy handoff, "end <unix_ts>" at policy -> landing). Read at
# trajectory-save time so the .npz carries the policy window and metrics.py
# can clip clearance/speed to the policy flight exactly. Must match
# POLICY_PHASE_FILE in the offboard scripts / compare/run_comparison.py.
POLICY_PHASE_FILE = str(Path(__file__).resolve().parent / ".superfly_policy_phase")


class Mp4Writer:
    """cv2 MP4 writer paced by an external clock (headless-safe), lazily opened
    on the first frame so the size comes from the actual frame.

    write(frame, t) duplicates or skips frames so that video playback time
    tracks the supplied clock `t` (we pass SIM time): if the sim loop renders
    slower than fps, each frame is written multiple times instead of the video
    silently playing sped-up; if faster, extra frames are dropped. Never let a
    viz error kill the sim loop."""

    MAX_GAP_S = 5.0  # cap frozen-frame padding across long stalls, then resync

    def __init__(self, path, fps, label):
        self.path = Path(path)
        self._recording_path = self.path.with_name(
            f"{self.path.stem}._recording{self.path.suffix}"
        )
        self.fps = fps
        self.label = label
        self._writer = None
        self._t0 = None       # clock value at the first written frame
        self._frames = 0      # frames written so far
        self._failed = False

    def write(self, frame_bgr, t):
        """Append one BGR uint8 frame stamped at clock time t [s]."""
        if self._failed:
            return
        try:
            if self._writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._recording_path.unlink(missing_ok=True)
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self._recording_path), fourcc, self.fps, (w, h)
                )
                if not self._writer.isOpened():
                    raise RuntimeError(
                        f"cv2.VideoWriter failed to open {self._recording_path}"
                    )
                print(f"[record] {self.label} video -> {self.path} "
                      f"({w}x{h} @ {self.fps:.0f} fps)")
                self._t0 = t
            target = int((t - self._t0) * self.fps) + 1
            gap_cap = int(self.MAX_GAP_S * self.fps)
            if target - self._frames > gap_cap:
                for _ in range(gap_cap):
                    self._writer.write(frame_bgr)
            else:
                while self._frames < target:
                    self._writer.write(frame_bgr)
                    self._frames += 1
            self._frames = max(self._frames, target)
        except Exception as e:
            carb.log_warn(f"{self.label} video frame failed: {e}")
            self._failed = True

    def _reencode_h264(self):
        """OpenCV mp4v (MPEG-4 part 2) plays in QuickTime but not VS Code/Cursor
        (Chromium needs H.264). Re-encode once at close; no-op if ffmpeg missing."""
        source = self._recording_path
        if self._failed or not source.exists() or source.stat().st_size == 0:
            return
        tmp = self.path.with_name(f"{self.path.stem}._h264{self.path.suffix}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(tmp)],
                check=True,
            )
            tmp.replace(self.path)
            source.unlink(missing_ok=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            carb.log_warn(f"{self.label} video H.264 re-encode failed: {e}")
            if tmp.exists():
                tmp.unlink()

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._reencode_h264()
            print(f"[record] saved {self.label} video -> {self.path}")


class PegasusApp:

    SPAWN_YAW_DEG = 0.0  # EKF heading is mag-locked (~+Y); we rotate the field instead

    def __init__(self, seed: int = 0, scale: float = 5.0, spawn_yaw_deg: float = 0.0,
                 policy: str = "diffphys", obstacles: str = "diffphys",
                 environment: str = "Box Room", spawn_xyz: tuple = (0.0, 0.0, 1.0),
                 usd_environment: str = None, env_scale: float = 1.0,
                 obstacle_assets: bool = False, auto_stop: bool = False,
                 debug_frames: bool = True, log_traj: str = None,
                 goal_xyz: tuple = None, record_depth_video: str = None,
                 record_rgb_video: str = None,
                 record_video_fps: float = 15.0, record_video_scale: int = 4,
                 agile_overhead_debug_path: str = None,
                 agile_depth_flip: str = "none"):
        self.SPAWN_YAW_DEG = spawn_yaw_deg
        self.auto_stop = auto_stop
        self.debug_frames = debug_frames
        self.agile_overhead_debug_path = agile_overhead_debug_path
        self._agile_depth_flip = agile_depth_flip
        self._record_video_scale = max(1, record_video_scale)
        self._depth_video = (Mp4Writer(record_depth_video, record_video_fps, "depth")
                             if record_depth_video else None)
        self._rgb_video = (Mp4Writer(record_rgb_video, record_video_fps, "rgb")
                           if record_rgb_video else None)
        self._da_euclid_scale = None  # lazy per-pixel planar->Euclidean scale
        # Ground-truth trajectory logging (for the comparison harness). When
        # log_traj is set, each sim tick appends the drone's ENU pose+velocity
        # (Pegasus Vehicle.state, refreshed every physics step) and the obstacle
        # field is dumped alongside on exit so compare/metrics.py can score the
        # run offline (collision/clearance, time-to-goal, speed) without Isaac.
        self.log_traj = log_traj
        self._goal_xyz = goal_xyz  # explicit goal override for npz (None = use field p_target)
        self._traj = [] if log_traj else None
        self._traj_sim = [] if log_traj else None   # per-pose SIM time (see _log_pose)
        self._autostop_n = 0
        if self.auto_stop:
            # Clear any stale sentinel from a previous run so we don't stop
            # immediately on this one.
            Path(OFFBOARD_DONE_FILE).unlink(missing_ok=True)
        self.policy = policy
        # --obstacle-assets: replace procedural primitives with realistic USD
        # assets (OBSTACLE_ASSETS) scaled to each primitive's extent.
        self._seed = seed
        self.obstacle_assets = obstacle_assets
        self._assets_root = None          # lazily resolved NVIDIA assets root
        self._asset_warned = set()        # URLs already warned-about (dedupe)
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        if usd_environment:
            # Load an arbitrary USD stage as the background environment instead of a
            # named Pegasus environment. load_asset references the USD under
            # /World/layout (synchronous, so the prim exists before we scale it).
            # env_scale uniformly scales the stage: a centimetre-authored asset like
            # ConiferForest needs env_scale=0.01 to read correctly in this metre stage
            # (USD does NOT auto-convert metersPerUnit across references).
            self.pg.load_asset(usd_environment, "/World/layout")
            if env_scale != 1.0:
                from pxr import UsdGeom, Gf
                prim = self.world.stage.GetPrimAtPath("/World/layout")
                UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(env_scale, env_scale, env_scale))
            self._wait_for_stage_assets(usd_environment)
            print(f"[environment] loaded USD stage {usd_environment} (scale={env_scale})")
            # Custom USD stages typically carry no lighting, so the scene renders
            # black. Spawn a sun + ambient sky so it (and the depth camera) can see.
            self._spawn_lighting()
            # ...and usually no physics colliders either, so the drone falls through
            # the ground/trees. Add static triangle-mesh colliders to the geometry.
            self._add_colliders()
        else:
            # Do not use PegasusInterface.load_environment() here. It wraps
            # load_environment_async() in ensure_future() and returns before
            # /World/layout exists, which races the World reset/camera creation
            # below. Referencing the same official asset synchronously keeps
            # the pristine Pegasus USD while making its dependency ordering
            # deterministic in this standalone app.
            environment_asset = SIMULATION_ENVIRONMENTS[environment]
            self.pg.load_asset(environment_asset, "/World/layout")
            self._wait_for_stage_assets(environment_asset)
            print(f"[environment] loaded Pegasus scene {environment!r}: "
                  f"{environment_asset}")

        # Spawn the obstacle field (analytic primitives). The DiffPhysDrone-
        # distribution field is the default; --obstacles diffaero regenerates the
        # field to match DiffAero's training distribution (around the start->goal
        # line). --obstacles none skips procedural obstacles (use scene geometry only).
        _SPAWN_DEFAULT = (0.0, 0.0, 1.0)
        if obstacles == "none":
            spawn_pos = [float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])]
        elif obstacles == "diffaero":
            from obstacle_field import generate_diffaero
            self.field = generate_diffaero(seed=seed, scale=scale)
            print("[obstacle_field]", self.field.summary())
            self._spawn_obstacles(self.field)
            if tuple(spawn_xyz) != _SPAWN_DEFAULT:
                spawn_pos = [float(spawn_xyz[0]), float(spawn_xyz[1]),
                             float(spawn_xyz[2])]
            else:
                spawn_pos = [float(self.field.p_init[0]), float(self.field.p_init[1]), 0.1]
        else:
            self.field = generate_field(seed=seed, scale=scale)
            print("[obstacle_field]", self.field.summary())
            self._spawn_obstacles(self.field)
            if tuple(spawn_xyz) != _SPAWN_DEFAULT:
                spawn_pos = [float(spawn_xyz[0]), float(spawn_xyz[1]),
                             float(spawn_xyz[2])]
            else:
                spawn_pos = [float(self.field.p_init[0]), float(self.field.p_init[1]), 0.1]

        self._spawn_pos = spawn_pos  # kept for the trajectory npz 'start' field

        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": False,
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        if policy == "diffaero":
            self._setup_camera_diffaero()
        elif policy == "depthnav":
            self._setup_camera_depthnav()
        elif policy in ("agile", "agile_rgb"):
            self._setup_camera_agile()
        else:
            self._setup_camera_diffphys()
        config_multirotor.graphical_sensors = [self._camera]

        # Dedicated RGB camera for --record-rgb-video only: same body pose and
        # horizontal FOV as the policy's depth camera, but at a watchable
        # resolution. Only created when recording (an extra render product
        # costs real frame time).
        self._rgb_camera = None
        if self._rgb_video is not None and self.policy != "agile_rgb":
            self._rgb_camera = MonocularCamera("drone_camera", config={
                "depth": False,
                "position": np.array(self._camera._position),
                "orientation": np.array(self._camera._orientation),
                "resolution": (RGB_W, RGB_H),
                "frequency": 30,
                "intrinsics": None,
            })
            self._rgb_camera.fov = self._camera.fov
            self._rgb_camera.fx = 0.5 * RGB_W / math.tan(0.5 * math.radians(self._camera.fov))
            self._rgb_camera.fy = self._rgb_camera.fx
            self._rgb_camera.cx = 0.5 * RGB_W
            self._rgb_camera.cy = 0.5 * RGB_H
            self._rgb_camera._intrinsics = np.array([
                [self._rgb_camera.fx, 0.0, self._rgb_camera.cx],
                [0.0, self._rgb_camera.fy, self._rgb_camera.cy],
                [0.0, 0.0, 1.0]])
            config_multirotor.graphical_sensors.append(self._rgb_camera)

        # Spawn the drone at the field start (procedural mode) or --spawn (scene-only).
        # SPAWN_YAW_DEG cancels the sim's EKF heading offset: with spawn yaw 0 the
        # mag-driven EKF reported ENU yaw=90° (facing +Y), but the obstacle corridor
        # runs +X. Spawn rotated by -90° so the reconstructed heading reads ~0 (faces
        # +X, down the corridor). Flip sign if the log still shows yaw≈±90.
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            spawn_pos,
            Rotation.from_euler("XYZ", [0.0, 0.0, self.SPAWN_YAW_DEG], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()
        self._depth_pub = None if self.policy == "agile_rgb" else DepthPublisher()
        self._rgb_pub = RGBPublisher() if self.policy == "agile_rgb" else None
        self._rgb_frame_attached = False
        self._agile_debug_sub = (
            AgileDebugSubscriber() if self.policy in ("agile", "agile_rgb") else None)
        self._last_agile_depth = None
        self._last_agile_rgb = None
        self._dbg_n = 0           # frame counter for throttled debug dumps
        self._agile_dbg_n = 0     # independent from generic depth debug frames
        self._dbg_every = 1       # save a debug PNG every N published frames
        self.stop_sim = False
        self._camera_ready_logged = False  # prints once when the depth camera becomes ready

    def _wait_for_stage_assets(self, source: str,
                               timeout_s: float = ASSET_LOAD_TIMEOUT_S):
        """Drain USD dependency loading before physics and render setup.

        AddReference() composes the root layer synchronously, but referenced
        layers, materials, and remote textures can still be discovered during
        subsequent Kit updates. Require several idle updates so a transient
        zero observed before discovery cannot release initialization early.
        """
        import omni.usd

        layout = self.world.stage.GetPrimAtPath("/World/layout")
        if not layout or not layout.IsValid():
            raise RuntimeError(
                f"Environment {source!r} did not create a valid /World/layout prim")

        context = omni.usd.get_context()
        deadline = time.monotonic() + float(timeout_s)
        idle_updates = 0
        peak_pending = 0
        while idle_updates < 3:
            simulation_app.update()
            _, _, pending = context.get_stage_loading_status()
            pending = int(pending)
            peak_pending = max(peak_pending, pending)
            idle_updates = idle_updates + 1 if pending == 0 else 0
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s loading environment assets "
                    f"from {source!r} ({pending} file(s) still pending)")
        print(f"[environment] assets ready for {source} "
              f"(peak pending={peak_pending})")

    def _setup_camera_diffphys(self):
        """Forward-facing depth camera pitched CAM_ANGLE_DEG down (DiffPhysDrone).

        Pegasus MonocularCamera 'orientation' is Euler ZYX (deg) relative to the
        body frame; default [0,0,180] points the camera forward (+X body). A
        positive pitch about the camera's lateral axis tilts the view downward.
        """
        self._camera = MonocularCamera("depth_cam", config={
            "depth": True,
            "position": np.array([0.10, 0.0, 0.0]),
            # NOTE: with the 180° yaw, a positive Y-pitch tilts the view UP, so we
            # negate CAM_ANGLE_DEG to pitch the camera DOWN (matching training).
            "orientation": np.array([0.0, -CAM_ANGLE_DEG, 180.0]),
            "resolution": (RENDER_W, RENDER_H),   # (width, height) = (64, 48)
            "frequency": 30,
            "intrinsics": None,  # falls back to fov-based; we override fov below
        })
        # Force the horizontal FOV to match training exactly.
        self._camera.fov = FOV_X_DEG
        self._camera.fx = 0.5 * RENDER_W / math.tan(0.5 * math.radians(FOV_X_DEG))
        self._camera.fy = self._camera.fx
        self._camera.cx = 0.5 * RENDER_W
        self._camera.cy = 0.5 * RENDER_H
        self._camera._intrinsics = np.array([
            [self._camera.fx, 0.0, self._camera.cx],
            [0.0, self._camera.fy, self._camera.cy],
            [0.0, 0.0, 1.0]])

    def _setup_camera_diffaero(self):
        """Forward-facing depth camera matching DiffAero's sensor config.

        16x9 grid (rendered at DA_RENDER_W x DA_RENDER_H, then min-pooled),
        horizontal FOV 86 deg, vertical FOV = 86*9/16 deg (DiffAero's angle-linear
        definition), no downward pitch, mounted at body [0.2, 0, 0.05]. fx/fy are
        set independently so both FOVs match; the planar Z-depth is converted to
        EUCLIDEAN range with these intrinsics in _publish_depth.
        """
        self._camera = MonocularCamera("depth_cam", config={
            "depth": True,
            "position": np.array([0.20, 0.0, 0.05]),
            "orientation": np.array([0.0, -DA_CAM_ANGLE_DEG, 180.0]),
            "resolution": (DA_RENDER_W, DA_RENDER_H),  # (width, height) = (64, 36)
            "frequency": 30,
            "intrinsics": None,
        })
        self._camera.fov = DA_FOV_X_DEG
        self._camera.fx = 0.5 * DA_RENDER_W / math.tan(0.5 * math.radians(DA_FOV_X_DEG))
        self._camera.fy = 0.5 * DA_RENDER_H / math.tan(0.5 * math.radians(DA_FOV_Y_DEG))
        self._camera.cx = 0.5 * DA_RENDER_W
        self._camera.cy = 0.5 * DA_RENDER_H
        self._camera._intrinsics = np.array([
            [self._camera.fx, 0.0, self._camera.cx],
            [0.0, self._camera.fy, self._camera.cy],
            [0.0, 0.0, 1.0]])

    def _setup_camera_depthnav(self):
        """Forward-facing depth camera matching depthnav's training sensor.

        128x72, horizontal FOV ~89 deg (habitat default), near 0.25 / far 20 m,
        no downward pitch, mounted at body [0.1, 0, 0]. RAW planar Z-depth
        (distance_to_image_plane) is published in metres; depthnav_policy clamps
        to [near, far] and inverts/maxpools internally (no work needed here)."""
        self._camera = MonocularCamera("depth_cam", config={
            "depth": True,
            "position": np.array([0.10, 0.0, 0.0]),
            "orientation": np.array([0.0, -DN_CAM_ANGLE_DEG, 180.0]),
            "resolution": (DN_RENDER_W, DN_RENDER_H),  # (width, height) = (128, 72)
            "clipping_range": (DN_NEAR, DN_FAR),
            "frequency": 30,
            "intrinsics": None,
        })
        self._camera.fov = DN_FOV_X_DEG
        self._camera.fx = 0.5 * DN_RENDER_W / math.tan(0.5 * math.radians(DN_FOV_X_DEG))
        self._camera.fy = self._camera.fx
        self._camera.cx = 0.5 * DN_RENDER_W
        self._camera.cy = 0.5 * DN_RENDER_H
        self._camera._intrinsics = np.array([
            [self._camera.fx, 0.0, self._camera.cx],
            [0.0, self._camera.fy, self._camera.cy],
            [0.0, 0.0, 1.0]])

    def _setup_camera_agile(self):
        """Forward-facing depth camera for Agile Autonomy (Loquercio).

        VGA render (AG_RENDER_W x AG_RENDER_H) at 91 deg horizontal FOV
        (flightmare.yaml), square pixels (fy == fx), and no downward pitch.
        agile_rgb uses Flightmare's published left-camera body translation
        [0, 0.1, 0.1]; the original depth adapter retains its prior mount.
        _publish_depth_agile bilinear-downsamples planar depth to 224x224 and
        wrapper/agile_core.py applies the mm/80 encoding."""
        is_rgb = self.policy == "agile_rgb"
        camera_name = "rgb_cam" if is_rgb else "depth_cam"
        camera_position = np.array([0.0, 0.10, 0.10] if is_rgb else [0.10, 0.0, 0.0])
        self._camera = MonocularCamera(camera_name, config={
            "depth": not is_rgb,
            "position": camera_position,
            "orientation": np.array([0.0, -AG_CAM_ANGLE_DEG, 180.0]),
            "resolution": (AG_RENDER_W, AG_RENDER_H),  # VGA-class, downsampled to 224
            "frequency": 30,
            "intrinsics": None,
        })
        self._camera.fov = AG_FOV_X_DEG
        self._camera.fx = 0.5 * AG_RENDER_W / math.tan(0.5 * math.radians(AG_FOV_X_DEG))
        self._camera.fy = self._camera.fx
        self._camera.cx = 0.5 * AG_RENDER_W
        self._camera.cy = 0.5 * AG_RENDER_H
        self._camera._intrinsics = np.array([
            [self._camera.fx, 0.0, self._camera.cx],
            [0.0, self._camera.fy, self._camera.cy],
            [0.0, 0.0, 1.0]])

    def _spawn_lighting(self):
        """Add outdoor lighting (a directional 'sun' + an ambient dome) so a USD
        stage with no authored lights is actually visible. DistantLight angle is in
        degrees down from horizontal-ish; intensities are in the UsdLux nits scale."""
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

    def _add_colliders(self, root: str = "/World/layout", approximation: str = "none",
                       verbose: bool = True):
        """Give a loaded USD stage physics colliders so the drone collides with the
        ground/trees instead of falling through. Static environment geometry has no
        rigid body, so each Mesh just gets a CollisionAPI + MeshCollisionAPI with a
        triangle-mesh approximation ('none' = exact tris, correct for static scenes;
        thin trunks and ground stay solid). Other gprims get a plain CollisionAPI.
        The /World/layout scale flows into the colliders via the xform hierarchy."""
        from pxr import Usd, UsdGeom, UsdPhysics
        root_prim = self.world.stage.GetPrimAtPath(root)
        if not root_prim or not root_prim.IsValid():
            return
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

    def _resolve_asset(self, rel_or_url: str):
        """Resolve a catalog entry to a full URL and verify it exists on the
        Nucleus/assets server. Absolute 'scheme://...' URLs are used as-is;
        anything else is joined to the NVIDIA assets root. Returns the URL, or
        None (caller falls back to the analytic primitive)."""
        import omni.client
        if "://" in rel_or_url:
            url = rel_or_url
        else:
            if self._assets_root is None:
                from isaacsim.storage.native import get_assets_root_path
                self._assets_root = get_assets_root_path()
            if not self._assets_root:
                return None
            url = self._assets_root.rstrip("/") + "/" + rel_or_url.lstrip("/")
        try:
            result, _ = omni.client.stat(url)
            ok = (result == omni.client.Result.OK)
        except Exception:
            ok = False
        if not ok:
            if url not in self._asset_warned:
                carb.log_warn(f"[obstacle_assets] asset not found, skipping: {url}")
                self._asset_warned.add(url)
            return None
        return url

    def _spawn_asset(self, prim_path, category, rng, pos, full_size,
                     euler_deg=(0.0, 0.0, 0.0), uniform=False):
        """Reference a random realistic USD asset from OBSTACLE_ASSETS[category],
        scaled so its (unrotated) bounding box matches full_size [m] and centred
        at pos (ENU). euler_deg is an XYZ-euler tilt (degrees); uniform=True keeps
        aspect ratio (for round rocks). Returns True on success, else False so the
        caller spawns the analytic primitive instead."""
        candidates = list(OBSTACLE_ASSETS.get(category, []))
        if not candidates:
            return False
        rng.shuffle(candidates)
        from pxr import Usd, UsdGeom, Gf
        stage = self.world.stage
        for rel in candidates:
            url = self._resolve_asset(rel)
            if url is None:
                continue
            try:
                # Wrapper Xform we control + a child holding the reference, so our
                # transform ops never collide with the asset's own xformOps.
                parent = stage.DefinePrim(prim_path, "Xform")
                ref_prim = stage.DefinePrim(prim_path + "/ref", "Xform")
                if not ref_prim.GetReferences().AddReference(url):
                    stage.RemovePrim(prim_path)
                    continue
                # Natural (unscaled) extent of the referenced geometry.
                bbox = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                rng3 = bbox.ComputeUntransformedBound(parent).ComputeAlignedRange()
                size, mid = rng3.GetSize(), rng3.GetMidpoint()
                if rng3.IsEmpty() or min(size[0], size[1], size[2]) <= 1e-6:
                    carb.log_warn(f"[obstacle_assets] empty bounds, skipping: {url}")
                    stage.RemovePrim(prim_path)
                    continue
                sx = full_size[0] / size[0]
                sy = full_size[1] / size[1]
                sz = full_size[2] / size[2]
                if uniform:
                    sx = sy = sz = min(sx, sy, sz)
                # Place the asset's bbox centre at pos: world_centre = T + R@(S*mid).
                R = Rotation.from_euler("XYZ", euler_deg, degrees=True).as_matrix()
                mid_scaled = np.array([sx * mid[0], sy * mid[1], sz * mid[2]])
                T = np.array(pos, dtype=float) - R @ mid_scaled
                xf = UsdGeom.XformCommonAPI(parent)
                xf.SetScale(Gf.Vec3f(float(sx), float(sy), float(sz)))
                xf.SetRotate(Gf.Vec3f(*[float(e) for e in euler_deg]),
                             UsdGeom.XformCommonAPI.RotationOrderXYZ)
                xf.SetTranslate(Gf.Vec3d(float(T[0]), float(T[1]), float(T[2])))
                # Static colliders so the drone collides with the asset geometry.
                self._add_colliders(root=prim_path, verbose=False)
                return True
            except Exception as e:
                carb.log_warn(f"[obstacle_assets] spawn failed for {url}: {e}")
                try:
                    stage.RemovePrim(prim_path)
                except Exception:
                    pass
                continue
        return False

    def _spawn_obstacles(self, fld):
        """Spawn the analytic obstacle field as static Isaac prims (ENU, z up).

        With self.obstacle_assets set, each primitive is first attempted as a
        realistic USD asset (OBSTACLE_ASSETS) scaled to the primitive's extent;
        any obstacle whose asset doesn't resolve falls back to the primitive."""
        use_assets = self.obstacle_assets
        # Separate RNG (offset from the field seed) so asset variant choices stay
        # deterministic without perturbing the field sampling.
        arng = np.random.default_rng(self._seed + 12345) if use_assets else None
        n_asset = 0  # successfully spawned realistic assets

        # Spheres -> rocks (uniform scale so they stay round).
        for i, (cx, cy, cz, r) in enumerate(fld.spheres):
            if use_assets and self._spawn_asset(
                    f"/World/obstacles/sphere_{i}", "rock", arng,
                    pos=(cx, cy, cz), full_size=(2 * r, 2 * r, 2 * r), uniform=True):
                n_asset += 1
                continue
            FixedSphere(
                prim_path=f"/World/obstacles/sphere_{i}",
                position=np.array([cx, cy, cz]),
                radius=float(r),
                color=np.array([0.8, 0.3, 0.3]),
            )
        # Boxes (voxels): half-extents -> full-size scale. DiffPhys boxes are
        # axis-aligned 6-tuples; DiffAero boxes are 9-tuples carrying an XYZ-euler
        # rotation (radians) for tilted pillars. Slender+tall boxes become "tall"
        # assets (trees/poles/buildings); the rest become "low" clutter.
        for i, box in enumerate(fld.boxes):
            cx, cy, cz, hx, hy, hz = box[:6]
            roll = pitch = yaw = 0.0
            orientation = None
            if len(box) >= 9:
                roll, pitch, yaw = box[6], box[7], box[8]
                q = Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_quat()  # [x,y,z,w]
                orientation = np.array([q[3], q[0], q[1], q[2]])  # FixedCuboid wants [w,x,y,z]
            if use_assets:
                cat = "tall" if hz >= 1.5 * max(hx, hy) else "low"
                if self._spawn_asset(
                        f"/World/obstacles/box_{i}", cat, arng,
                        pos=(cx, cy, cz), full_size=(2 * hx, 2 * hy, 2 * hz),
                        euler_deg=(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))):
                    n_asset += 1
                    continue
            FixedCuboid(
                prim_path=f"/World/obstacles/box_{i}",
                position=np.array([cx, cy, cz]),
                scale=np.array([2 * hx, 2 * hy, 2 * hz]),
                orientation=orientation,
                color=np.array([0.3, 0.5, 0.8]),
            )
        # Vertical cylinders (axis = world Z), tall enough to span the flight
        # band -> "tall" assets (trees/poles), else a primitive cylinder.
        CYL_HEIGHT = 12.0
        for i, (cx, cy, r) in enumerate(fld.cyl_v):
            if use_assets and self._spawn_asset(
                    f"/World/obstacles/cylv_{i}", "tall", arng,
                    pos=(cx, cy, CYL_HEIGHT / 2 - 1.0),
                    full_size=(2 * r, 2 * r, CYL_HEIGHT)):
                n_asset += 1
                continue
            self._spawn_cylinder(f"/World/obstacles/cylv_{i}",
                                 pos=(cx, cy, CYL_HEIGHT / 2 - 1.0),
                                 radius=float(r), height=CYL_HEIGHT, axis="Z")
        # Horizontal cylinders (2 minor ground obstacles). Stored (cx,cy,cz,r)
        # after the field rotation; spawn lying along world X. Kept as primitives.
        CYLH_LEN = 6.0
        for i, (cx, cy, cz, r) in enumerate(fld.cyl_h):
            self._spawn_cylinder(f"/World/obstacles/cylh_{i}",
                                 pos=(cx, cy, cz),
                                 radius=float(r), height=CYLH_LEN, axis="X")

        if use_assets:
            total = len(fld.spheres) + len(fld.boxes) + len(fld.cyl_v)
            print(f"[obstacle_assets] spawned {n_asset}/{total} realistic assets "
                  f"({total - n_asset} fell back to primitives)")

    def _spawn_cylinder(self, path, pos, radius, height, axis="Z"):
        """Create a static USD Cylinder prim. UsdGeom.Cylinder is Z-axis by default;
        for a world-X horizontal cylinder, rotate 90° about Y."""
        orientation = None
        if axis == "X":
            q = Rotation.from_euler("XYZ", [0.0, 90.0, 0.0], degrees=True).as_quat()  # [x,y,z,w]
            orientation = np.array([q[3], q[0], q[1], q[2]])  # create_prim wants [w,x,y,z]
        prim_utils.create_prim(
            path, "Cylinder",
            position=np.array([float(pos[0]), float(pos[1]), float(pos[2])]),
            orientation=orientation,
            attributes={"radius": float(radius), "height": float(height), "axis": "Z"},
        )

    def _publish_depth(self):
        if self.policy == "agile_rgb":
            self._publish_rgb_agile()
        elif self.policy == "diffaero":
            self._publish_depth_diffaero()
        elif self.policy == "depthnav":
            self._publish_depth_depthnav()
        elif self.policy == "agile":
            self._publish_depth_agile()
        else:
            self._publish_depth_diffphys()

    def _camera_ready(self):
        """True once MonocularCamera.start() has run (sets _camera_full_set).
        Prints once on the False->True transition so you can see exactly when
        _publish_depth starts actually producing frames (vs. silently
        early-returning every tick before that)."""
        cam = getattr(self._camera, "_camera", None)
        full_set = cam is not None and getattr(self._camera, "_camera_full_set", False)
        if full_set and not self._camera_ready_logged:
            self._camera_ready_logged = True
            t = time.time()
            sensor = "RGB" if self.policy == "agile_rgb" else "depth"
            print(f"[capture] {sensor} camera ready at wall_clock="
                  f"{time.strftime('%H:%M:%S', time.localtime(t))}.{int(t % 1 * 1000):03d}")
        return cam, full_set

    def _publish_depth_diffphys(self):
        """Grab the camera's planar Z-depth, resize to 48x64, publish over UDP."""
        cam, full_set = self._camera_ready()
        if cam is None or not full_set:
            return
        # get_depth() returns 'distance_to_image_plane' = planar/optical-axis
        # Z-depth, matching the native render convention (NOT Euclidean range).
        depth = cam.get_depth()
        if depth is None:
            return
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size == 0:
            return
        # Replace inf / nan / no-return with far value (>= clamp max of 24 m).
        depth = np.nan_to_num(depth, nan=24.0, posinf=24.0, neginf=24.0)
        if depth.shape != (RENDER_H, RENDER_W):
            # Nearest-neighbour resize to the exact policy resolution.
            yi = (np.linspace(0, depth.shape[0] - 1, RENDER_H)).astype(np.int64)
            xi = (np.linspace(0, depth.shape[1] - 1, RENDER_W)).astype(np.int64)
            depth = depth[yi][:, xi]

        # --- orientation fix (apply BEFORE publishing so debug == published) ---
        # If the debug PNG shows rows/cols flipped vs the native convention
        # (row 0 = up, col 0 = left), uncomment the matching line:
        # depth = depth[::-1]        # flip rows (vertical)
        # depth = depth[:, ::-1]     # flip cols (horizontal/mirror)

        self._depth_pub.send(depth)
        self._dump_depth_debug(depth)
        self._record_depth_video_frame(depth)

    def _publish_depth_diffaero(self):
        """Publish the 9x16 EUCLIDEAN range image DiffAero expects.

        Steps: planar Z-depth (distance_to_image_plane) -> Euclidean range via the
        precomputed per-pixel scale -> min-pool DA_POOL x DA_POOL (nearest surface
        per output cell, matching DiffAero's one-ray-per-cell sampling). The policy
        process applies depth = 1 - clamp(r,0,5)/5. Convention: row 0 = up,
        col 0 = left (same as the DiffPhysDrone path)."""
        cam, full_set = self._camera_ready()
        if cam is None or not full_set:
            return
        depth = cam.get_depth()
        if depth is None:
            return
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size == 0:
            return
        depth = np.nan_to_num(depth, nan=DA_MAX_DIST, posinf=DA_MAX_DIST, neginf=DA_MAX_DIST)
        self._depth_pub.send(depth)
        self._dump_depth_debug(depth)
        self._record_depth_video_frame(depth)

    def _publish_depth_depthnav(self):
        """Publish the 72x128 RAW planar metric depth depthnav expects.

        Resize the camera's planar Z-depth (distance_to_image_plane) to exactly
        (72, 128) and send it in metres -- depthnav_policy clamps to [0.25, 20]
        and inverts internally, so we do NO normalization or pooling here.
        Convention: row 0 = up, col 0 = left (same as the other paths)."""
        cam, full_set = self._camera_ready()
        if cam is None or not full_set:
            return
        depth = cam.get_depth()
        if depth is None:
            return
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size == 0:
            return
        # Replace inf / nan / no-return with the far value.
        depth = np.nan_to_num(depth, nan=DN_FAR, posinf=DN_FAR, neginf=DN_FAR)
        if depth.shape != (DN_RENDER_H, DN_RENDER_W):
            # Nearest-neighbour resize to the exact policy resolution (72x128).
            yi = (np.linspace(0, depth.shape[0] - 1, DN_RENDER_H)).astype(np.int64)
            xi = (np.linspace(0, depth.shape[1] - 1, DN_RENDER_W)).astype(np.int64)
            depth = depth[yi][:, xi]
        self._depth_pub.send(depth)
        self._dump_depth_debug(depth)
        self._record_depth_video_frame(depth)

    def _publish_depth_agile(self):
        """Publish the 224x224 RAW planar metric depth Agile Autonomy expects.

        BILINEAR-downsample the VGA-class camera render (distance_to_image_plane)
        to exactly (AG_NET_SIZE, AG_NET_SIZE) = 224 and send it in metres --
        this reproduces Loquercio's training loader (cv2.resize of VGA SGM depth
        to 224), rather than rendering at 84 and letting the net upsample. The
        offboard's wrapper/agile_core.py does the mm/80 encoding, so NO
        normalization/pooling here. The frame is shipped zlib-compressed
        (compress=True) because a raw 224x224 float32 (200 KB) exceeds the UDP
        datagram cap. Convention: row 0 = up, col 0 = left (as the other paths)."""
        cam, full_set = self._camera_ready()
        if cam is None or not full_set:
            return
        depth = cam.get_depth()
        if depth is None:
            return
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size == 0:
            return
        # Replace inf / nan / no-return with the far value, then cap at AG_FAR
        # BEFORE resizing -- the training loader did np.minimum(depth, 20000)
        # ahead of cv2.resize, so far pixels don't bleed large values across an
        # obstacle edge during interpolation.
        depth = np.nan_to_num(depth, nan=AG_FAR, posinf=AG_FAR, neginf=AG_FAR)
        depth = np.clip(depth, 0.0, AG_FAR)
        # Bilinear downsample to the net's 224x224 input (matches training's
        # cv2.resize; INTER_LINEAR is cv2.resize's default, what the loader used).
        if depth.shape != (AG_NET_SIZE, AG_NET_SIZE):
            depth = cv2.resize(depth, (AG_NET_SIZE, AG_NET_SIZE),
                               interpolation=cv2.INTER_LINEAR)
        # Optional flip to match upstream agile_autonomy, whose depth callback does
        # cv2.flip(depth, -1) (both axes; the sensor was mounted inverted). The net
        # was trained on that convention. "both" == cv2.flip(-1); "v"/"h" isolate an
        # axis for A/B testing left-right vs up-down misregistration.
        if self._agile_depth_flip == "both":
            depth = depth[::-1, ::-1]
        elif self._agile_depth_flip == "v":
            depth = depth[::-1, :]
        elif self._agile_depth_flip == "h":
            depth = depth[:, ::-1]
        depth = np.ascontiguousarray(depth)
        self._last_agile_depth = depth
        self._depth_pub.send(depth, compress=True)
        self._dump_depth_debug(depth)
        self._record_depth_video_frame(depth)

    def _publish_rgb_agile(self):
        """Publish the exact RGB preprocessing used by CL4Nav Agile training.

        Isaac renders VGA RGB at the original Agile camera pose/FOV. OpenCV
        bilinear-resizes uint8 RGB to 224x224, matching data_loader.decode_img_cv2,
        then rgb_transport sends it losslessly (fragmented zlib).
        """
        cam, full_set = self._camera_ready()
        if cam is None or not full_set:
            return
        if not self._rgb_frame_attached:
            try:
                cam.add_rgb_to_frame()
                self._rgb_frame_attached = True
            except Exception as exc:
                carb.log_warn(f"RGB camera annotator attach failed: {exc}")
                return
        try:
            rgb = cam.get_rgb()
        except Exception as exc:
            carb.log_warn(f"RGB camera read failed: {exc}")
            return
        if rgb is None:
            return
        rgb = np.asarray(rgb)
        if rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] < 3:
            return
        rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
            if rgb.size and float(rgb.max()) <= 1.0:
                rgb *= 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        if rgb.shape[:2] != (AG_NET_SIZE, AG_NET_SIZE):
            rgb = cv2.resize(rgb, (AG_NET_SIZE, AG_NET_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(rgb)
        self._last_agile_rgb = rgb
        self._rgb_pub.send(rgb)
        self._dump_rgb_debug(rgb)
        self._record_rgb_array(rgb)

    def _dump_rgb_debug(self, rgb):
        if not self.debug_frames:
            return
        self._dbg_n += 1
        if self._dbg_n % self._dbg_every != 0:
            return
        try:
            np.save("rgb_debug.npy", rgb)
            cv2.imwrite("camera_debug.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            carb.log_warn(f"RGB camera debug dump failed: {exc}")

    def _dump_depth_debug(self, depth):
        """Save the EXACT published depth array + the drone's RGB view to
        camera_debug.png (+ depth .npy) every N frames. RGB comes from the SAME
        camera/render product as the depth, so it is the literal drone viewpoint.
        Check orientation vs native convention (row 0 = up, col 0 = left).
        Never let a viz error kill the sim loop."""
        if not self.debug_frames:
            return
        self._dbg_n += 1
        if self._dbg_n % self._dbg_every != 0:
            return
        try:
            np.save("depth_debug.npy", depth)

            # RGB from the same camera (annotator auto-attached on initialize()).
            rgb = None
            cam = getattr(self._camera, "_camera", None)
            if cam is not None:
                try:
                    rgb = cam.get_rgb()
                except Exception:
                    rgb = None

            ncols = 2 if rgb is not None else 1
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), squeeze=False)

            depth_vmax = {"diffaero": DA_MAX_DIST, "agile": AG_FAR}.get(self.policy, 24.0)
            ax = axes[0][0]
            im = ax.imshow(depth, origin="upper", cmap="turbo", vmin=0.3, vmax=depth_vmax)
            fig.colorbar(im, ax=ax, label="range [m]" if self.policy == "diffaero" else "depth [m]")
            ax.set_title(f"depth  frame={self._dbg_n}  "
                         f"min={depth.min():.2f} max={depth.max():.2f} m")
            ax.set_xlabel("col (0 = left)"); ax.set_ylabel("row (0 = up)")

            if rgb is not None:
                axrgb = axes[0][1]
                axrgb.imshow(np.asarray(rgb), origin="upper")
                axrgb.set_title("drone RGB view (same camera)")
                axrgb.set_xlabel("col (0 = left)"); axrgb.set_ylabel("row (0 = up)")

            fig.tight_layout()
            fig.savefig("camera_debug.png", dpi=90)
            plt.close(fig)
        except Exception as e:
            carb.log_warn(f"camera debug dump failed: {e}")

    def _local_to_world(self, pts_local):
        """PX4-local ENU (offboard) -> sim world ENU."""
        origin = np.asarray(self._spawn_pos, dtype=np.float64).reshape(3)
        pts = np.asarray(pts_local, dtype=np.float64)
        if pts.ndim == 1:
            return origin + pts.reshape(3)
        return origin + pts

    def _dump_agile_overhead_debug(self):
        """Overhead XY map: obstacles, goal, drone, and PlaNet candidate trajectories.

        Trajectories arrive over UDP from agile_offboard (local ENU); depth inset
        is the same 224x224 frame published to the policy."""
        if self.agile_overhead_debug_path is None or self._agile_debug_sub is None:
            return
        frame = self._agile_debug_sub.latest()
        if frame is None:
            return
        self._agile_dbg_n += 1
        if self._agile_dbg_n % self._dbg_every != 0:
            return
        try:
            st = self.drone.state
            pos_w = np.asarray(st.position, dtype=np.float64).reshape(3)
            vel_w = np.asarray(st.linear_velocity, dtype=np.float64).reshape(3)
            # Pegasus State exposes position/velocity only (no orientation quaternion).
            # Use the offboard-reported yaw for the camera FOV wedge; use velocity
            # heading for the sim-GT arrow when moving.
            yaw_policy = float(frame.yaw)
            speed = float(np.linalg.norm(vel_w[:2]))
            if speed > 0.2:
                yaw_gt = math.atan2(float(vel_w[1]), float(vel_w[0]))
            else:
                yaw_gt = yaw_policy

            fld = getattr(self, "field", None)
            goal_w = None
            if self._goal_xyz is not None:
                g = np.asarray(self._goal_xyz, dtype=np.float64).reshape(-1)
                if g.size == 2:
                    g = np.append(g, self._spawn_pos[2])
                goal_w = g
            elif fld is not None and fld.p_target is not None:
                goal_w = np.asarray(fld.p_target, dtype=np.float64).reshape(3)

            ncols = 2 if self._last_agile_depth is not None else 1
            fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6.5), squeeze=False)
            ax = axes[0][0]

            if fld is not None:
                for cx, cy, cz, r in getattr(fld, "spheres", []) or []:
                    ax.add_patch(plt.Circle((cx, cy), r, color="#cc4444", alpha=0.35, lw=0))
                for cx, cy, r in getattr(fld, "cyl_v", []) or []:
                    ax.add_patch(plt.Circle((cx, cy), r, color="#cc4444", alpha=0.45, lw=0))
                for box in getattr(fld, "boxes", []) or []:
                    cx, cy = float(box[0]), float(box[1])
                    hx, hy = float(box[3]), float(box[4])
                    ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                               color="#cc4444", alpha=0.25, lw=0))

            traj_w = self._local_to_world(frame.trajectories_local)
            for i in range(traj_w.shape[0]):
                xy = traj_w[i, :, :2]
                is_sel = i == frame.mode_idx
                ax.plot(xy[:, 0], xy[:, 1],
                        color="#ff00ff" if is_sel else "#996699",
                        lw=3.5 if is_sel else 1.5,
                        alpha=1.0 if is_sel else 0.55,
                        label=f"mode {i} a={frame.alphas[i]:.2f}" + (" *" if is_sel else ""))
                ax.scatter(xy[0, 0], xy[0, 1], s=20,
                           color="#ff00ff" if is_sel else "#996699", zorder=5)

            fov_half = math.radians(AG_FOV_X_DEG * 0.5)
            rng = 12.0
            arc_x = [pos_w[0]]
            arc_y = [pos_w[1]]
            for a in np.linspace(yaw_policy - fov_half, yaw_policy + fov_half, 24):
                arc_x.append(pos_w[0] + rng * math.cos(a))
                arc_y.append(pos_w[1] + rng * math.sin(a))
            arc_x.append(pos_w[0])
            arc_y.append(pos_w[1])
            ax.fill(arc_x, arc_y, color="#44aaff", alpha=0.12, lw=0)

            ax.scatter(pos_w[0], pos_w[1], s=80, c="#2266ff", marker="o", zorder=6, label="drone (sim GT)")
            hx = pos_w[0] + 3.0 * math.cos(yaw_gt)
            hy = pos_w[1] + 3.0 * math.sin(yaw_gt)
            ax.annotate("", xy=(hx, hy), xytext=(pos_w[0], pos_w[1]),
                        arrowprops=dict(arrowstyle="-|>", color="#2266ff", lw=2.0))

            pos_ob = self._local_to_world(frame.pos_local)
            ax.scatter(pos_ob[0], pos_ob[1], s=40, facecolors="none", edgecolors="#000000",
                       linewidths=1.2, zorder=6, label="drone (offboard)")

            if goal_w is not None:
                ax.scatter(goal_w[0], goal_w[1], s=120, marker="*", c="#22aa22", zorder=6, label="goal")

            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            ax.set_title(
                f"agile overhead  frame={self._agile_dbg_n}  trk={frame.tracker}  "
                f"sel=mode{frame.mode_idx}  speed={speed:.1f}m/s\n"
                f"alphas={np.round(frame.alphas, 3)}  "
                f"(magenta=selected trajectory, wedge=depth FOV)"
            )
            pts = [pos_w[:2]]
            if goal_w is not None:
                pts.append(goal_w[:2])
            pts.extend(traj_w.reshape(-1, 3)[:, :2])
            pts = np.asarray(pts, dtype=np.float64)
            cx, cy = float(pos_w[0]), float(pos_w[1])
            rad = max(15.0, float(np.max(np.linalg.norm(pts - pos_w[:2], axis=1)) + 5.0))
            ax.set_xlim(cx - rad, cx + rad)
            ax.set_ylim(cy - rad, cy + rad)

            if ncols == 2:
                axd = axes[0][1]
                d = np.asarray(self._last_agile_depth, dtype=np.float32)
                im = axd.imshow(d, origin="upper", cmap="turbo", vmin=0.3, vmax=AG_FAR)
                fig.colorbar(im, ax=axd, fraction=0.046, label="depth [m]")
                axd.set_title(f"policy depth ({d.shape[1]}x{d.shape[0]})\n"
                              "obstacle should appear here")
                axd.set_xlabel("col (0=left)")
                axd.set_ylabel("row (0=up)")

            fig.tight_layout()
            fig.savefig(self.agile_overhead_debug_path, dpi=100)
            plt.close(fig)
        except Exception as e:
            carb.log_warn(f"agile overhead debug dump failed: {e}")

    def _sim_time(self):
        """Clock used to pace the video recorders: Isaac's SIMULATED time, so
        recordings play back in real time relative to the drone's motion even
        when the render loop runs slower than wall clock (headless RTX etc.)."""
        try:
            return float(self.world.current_time)
        except Exception:
            return time.time()

    def _policy_depth_view(self, depth):
        """Mirror each offboard's preprocessing to produce the depth the policy
        NETWORK actually consumes (a distance grid [m] at the policy's input
        resolution). Returns (grid, vmax) for the colormap.

        diffaero:  PerceptionBuilder pipeline -- planar -> Euclidean range via
                   per-pixel ray scale, min-pool DA_POOL x DA_POOL -> 9x16
                   (crop is the whole image: the camera renders exactly the
                   training FOV). The net sees 1 - clamp(r,0,5)/5 of this grid.
        diffphys:  clamp(0.3, 24) then 4x4 pool of 48x64 -> 12x16 (the policy's
                   max-pool of 3/d is a min-pool of distance).
        depthnav:  the 72x128 planar clamped to [near, far] -- the network
                   inverts + maxpools INTERNALLY, so its input is this grid."""
        if self.policy == "diffaero":
            if self._da_euclid_scale is None:
                u = np.arange(DA_RENDER_W, dtype=np.float32)
                v = np.arange(DA_RENDER_H, dtype=np.float32)
                uu, vv = np.meshgrid(u, v)
                xn = (uu - self._camera.cx) / self._camera.fx
                yn = (vv - self._camera.cy) / self._camera.fy
                self._da_euclid_scale = np.sqrt(1.0 + xn * xn + yn * yn)
            d = np.where(depth <= 1e-3, DA_MAX_DIST, depth) * self._da_euclid_scale
            d = np.minimum(d, DA_MAX_DIST)
            d = d.reshape(DA_OUT_H, DA_POOL, DA_OUT_W, DA_POOL).min(axis=(1, 3))
            return d, DA_MAX_DIST
        if self.policy == "depthnav":
            return np.clip(depth, DN_NEAR, DN_FAR), DN_FAR
        if self.policy == "agile":
            # The net consumes the raw 224x224 planar depth (mm/80 encoding is
            # monotone in this grid), so this IS the policy's input view.
            return np.clip(depth, 0.0, AG_FAR), AG_FAR
        d = np.clip(depth, 0.3, 24.0)
        d = d.reshape(RENDER_H // 4, 4, RENDER_W // 4, 4).min(axis=(1, 3))
        return d, 24.0

    def _record_depth_video_frame(self, depth):
        """Append one turbo-colormap frame of the policy-input depth grid
        (see _policy_depth_view) to --record-depth-video."""
        if self._depth_video is None:
            return
        try:
            view, depth_vmax = self._policy_depth_view(depth)
            normed = np.clip((view - 0.3) / max(depth_vmax - 0.3, 1e-3), 0.0, 1.0)
            gray = (normed * 255).astype(np.uint8)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            h, w = frame.shape[:2]
            # Policy grids are tiny (16-128 px wide); upscale to at least
            # ~320 px wide, nearest-neighbour so cells stay crisp.
            scale = max(self._record_video_scale, -(-320 // w))
            if scale > 1:
                frame = cv2.resize(frame, (w * scale, h * scale),
                                   interpolation=cv2.INTER_NEAREST)
            self._depth_video.write(frame, self._sim_time())
        except Exception as e:
            carb.log_warn(f"depth video frame failed: {e}")

    def _record_rgb_array(self, rgb):
        """Record an already prepared RGB policy frame without another camera.

        The transport/model still consume ``rgb`` at its native resolution.
        Upscaling happens only on the BGR copy handed to the video writer.
        """
        if self._rgb_video is None:
            return
        try:
            frame = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            scale = self._record_video_scale
            if scale > 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
                )
            self._rgb_video.write(frame, self._sim_time())
        except Exception as exc:
            carb.log_warn(f"rgb video frame failed: {exc}")

    def _record_rgb_video_frame(self):
        """Append one frame from the RGB drone_camera to --record-rgb-video."""
        if self._rgb_video is None or self._rgb_camera is None:
            return
        cam = getattr(self._rgb_camera, "_camera", None)
        if cam is None or not getattr(self._rgb_camera, "_camera_full_set", False):
            return
        try:
            rgb = cam.get_rgb()
            if rgb is None:
                return
            rgb = np.asarray(rgb)
            if rgb.size == 0:
                return
            frame = cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            self._rgb_video.write(frame, self._sim_time())
        except Exception as e:
            carb.log_warn(f"rgb video frame failed: {e}")

    def _close_videos(self):
        for writer in (self._depth_video, self._rgb_video):
            if writer is not None:
                writer.close()

    def _log_pose(self, t):
        """Append (t, pos_enu(3), vel_enu(3)) from the drone's ground-truth state.
        Pegasus Vehicle.state is refreshed every physics step (ENU position +
        ENU linear_velocity); never let a logging hiccup kill the sim loop.

        t is WALL-clock (needed to map the offboard's phase-file unix
        timestamps to log indices); SIMULATED time is captured alongside into
        _traj_sim because durations must be scored in sim time -- the render
        loop runs slower than realtime (headless RTX), so wall-clock durations
        overstate flight times while the logged velocities are sim-frame m/s."""
        try:
            st = self.drone.state
            p = np.asarray(st.position, dtype=np.float64)
            v = np.asarray(st.linear_velocity, dtype=np.float64)
            self._traj.append((t, p[0], p[1], p[2], v[0], v[1], v[2]))
            self._traj_sim.append(self._sim_time())
        except Exception as e:
            carb.log_warn(f"trajectory log failed: {e}")

    def _save_trajectory(self):
        """Dump the logged trajectory + the obstacle field + goal/start to an .npz
        for compare/metrics.py. Field arrays are saved as-is (boxes are 6-tuples
        for diffphys, 9-tuples with XYZ-euler for diffaero). goal/start are saved
        even with --obstacles none (--goal override / spawn position), so
        reached/time-to-goal still get scored on scene-only scenario runs."""
        try:
            out = self.log_traj
            traj = np.asarray(self._traj, dtype=np.float64)
            data = dict(traj=traj, policy=self.policy, seed=self._seed)
            if self._traj_sim and len(self._traj_sim) == traj.shape[0]:
                data["t_sim"] = np.asarray(self._traj_sim, dtype=np.float64)
            # Wall-clock zero of the traj timestamps + the offboard's policy
            # phase handoffs (same machine clock), so metrics.py can clip
            # clearance/speed to the policy flight exactly.
            if getattr(self, "_t0", None) is not None:
                data["t_unix0"] = float(self._t0)
                try:
                    for line in Path(POLICY_PHASE_FILE).read_text().splitlines():
                        parts = line.split()
                        if len(parts) == 2 and parts[0] in ("start", "end"):
                            data[f"policy_{parts[0]}_unix"] = float(parts[1])
                except (FileNotFoundError, ValueError):
                    pass
            fld = getattr(self, "field", None)
            if fld is not None:
                data["spheres"] = (np.asarray(fld.spheres, dtype=np.float64).reshape(-1, 4)
                                   if fld.spheres else np.zeros((0, 4), dtype=np.float64))
                data["boxes"] = (np.asarray(fld.boxes, dtype=np.float64)
                                 if fld.boxes else np.zeros((0, 6), dtype=np.float64))
                data["cyl_v"] = (np.asarray(fld.cyl_v, dtype=np.float64).reshape(-1, 3)
                                 if fld.cyl_v else np.zeros((0, 3), dtype=np.float64))
                data["cyl_h"] = (np.asarray(fld.cyl_h, dtype=np.float64).reshape(-1, 4)
                                 if fld.cyl_h else np.zeros((0, 4), dtype=np.float64))
            if self._goal_xyz is not None:
                g = np.asarray(self._goal_xyz, dtype=np.float64).reshape(-1)
                if g.size == 2:  # --goal X Y: fill Z from the spawn altitude
                    g = np.append(g, self._spawn_pos[2])
                data["goal"] = g
            elif fld is not None:
                data["goal"] = np.asarray(fld.p_target, dtype=np.float64)
            start = fld.p_init if fld is not None else self._spawn_pos
            data["start"] = np.asarray(start, dtype=np.float64)
            np.savez(out, **data)
            print(f"[log-traj] saved {traj.shape[0]} poses -> {out}")
        except Exception as e:
            carb.log_warn(f"trajectory save failed: {e}")

    def run(self):
        t0 = time.time()
        self._t0 = t0   # wall-clock zero of the trajectory log timestamps
        print(f"[capture] SIM START wall_clock={time.strftime('%H:%M:%S', time.localtime(t0))}"
              f".{int(t0 % 1 * 1000):03d} -- start your screen recording now")
        if self.auto_stop:
            print(f"[auto-stop] watching {OFFBOARD_DONE_FILE} -- will stop "
                  "automatically once the offboard script exits (landed, "
                  "Ctrl-C, or crash), no manual Ctrl-C needed.")
        self.timeline.play()
        exit_reason = "unknown"
        try:
            while True:
                if not simulation_app.is_running():
                    exit_reason = "simulation_app.is_running() went False"
                    break
                if self.stop_sim:
                    exit_reason = "stop_sim (auto-stop done-file)"
                    break
                self.world.step(render=True)
                self._publish_depth()
                if self.policy in ("agile", "agile_rgb"):
                    self._dump_agile_overhead_debug()
                self._record_rgb_video_frame()
                if self._traj is not None:
                    self._log_pose(time.time() - t0)
                if self.auto_stop:
                    # Cheap stat() call; throttle slightly to avoid hammering
                    # the filesystem at 250 Hz physics rate.
                    self._autostop_n += 1
                    if self._autostop_n % 15 == 0 and Path(OFFBOARD_DONE_FILE).exists():
                        print("[auto-stop] offboard process finished -- "
                              "stopping sim loop normally.")
                        self.stop_sim = True
        except BaseException as e:
            exit_reason = f"exception {type(e).__name__}: {e}"
            raise
        finally:
            # carb.log_warn: Python prints are lost in Kit's fast shutdown,
            # carb messages reach the console reliably.
            carb.log_warn(f"sim loop exit after {time.time() - t0:.1f}s: {exit_reason}; "
                          f"done_file={Path(OFFBOARD_DONE_FILE).exists()}")
            if self._traj is not None:
                self._save_trajectory()
            self._close_videos()
            for cleanup in (lambda: carb.log_warn("PegasusApp closing."),
                            self.timeline.stop,
                            simulation_app.close):
                try:
                    cleanup()
                except Exception as e:
                    print(f"[cleanup] {cleanup} failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                         help="(Parsed at import time, before this runs.) Run without the "
                              "GUI viewport for much higher throughput; see camera_debug.png "
                              "/ depth_debug.npy to inspect the drone's view instead.")
    parser.add_argument("--environment", type=str, default="Box Room",
                        help="Pegasus/Isaac Sim background scene (key in SIMULATION_ENVIRONMENTS). "
                             "Examples: Box Room, Warehouse, Hospital. Ignored when "
                             "--usd-environment is set.")
    parser.add_argument("--usd-environment", type=str, default=None,
                        help="Path or omniverse:// URL of a USD stage to load as the background "
                             "environment instead of a named --environment. Example: "
                             "omniverse://airlab-nucleus.andrew.cmu.edu/Library/Stages/"
                             "ConiferForest/ConiferForest_stage.stage.usd")
    parser.add_argument("--env-scale", type=float, default=1.0,
                        help="Uniform scale applied to --usd-environment (e.g. 0.01 to convert a "
                             "centimetre-authored stage like ConiferForest to metres).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Obstacle-field RNG seed (eval suite: 0, 1, 2)")
    parser.add_argument("--scale", type=float, default=5.0,
                        help="World scale (corridor depth ~= 8*scale m)")
    parser.add_argument("--spawn", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                        metavar=("X", "Y", "Z"),
                        help="Drone spawn position (ENU metres). Required for "
                             "--obstacles none; optional override for procedural "
                             "fields (diffphys/diffaero) when not the default "
                             "(0,0,1).")
    parser.add_argument("--spawn-yaw", type=float, default=0.0,
                        help="Spawn yaw [deg]. EKF heading is mag-locked in sim, so this "
                             "mainly affects the initial facing; the field is rotated instead.")
    parser.add_argument("--policy", choices=["diffphys", "diffaero", "depthnav", "agile", "agile_rgb"],
                        default="diffphys",
                        help="Which policy's camera/depth pipeline to configure: "
                             "diffphys (12x16, 78.6 deg, 24 m, planar, pitched 20 deg down), "
                             "diffaero (9x16, 86 deg, 5 m, Euclidean, forward), "
                             "depthnav (72x128, 89 deg, 0.25-20 m, planar, forward), or "
                             "agile (depth) or agile_rgb (CL4Nav RGB), both 640x480 "
                             "render -> 224x224 at 91 deg, forward-facing.")
    parser.add_argument("--obstacles", choices=["diffphys", "diffaero", "none"], default="diffphys",
                        help="Obstacle-field distribution: diffphys, diffaero, or none "
                             "(scene geometry only, no procedural primitives).")
    parser.add_argument("--obstacle-assets", action="store_true",
                        help="Replace each procedural obstacle primitive with a realistic USD "
                             "asset (OBSTACLE_ASSETS), scaled to the primitive's extent. Keeps the "
                             "training-matched layout but with real geometry. Assets that don't "
                             "resolve on Nucleus fall back to the analytic primitive.")
    parser.add_argument("--auto-stop", action="store_true",
                        help="Watch for diffaero_offboard.py/diffdrone_offboard.py exiting "
                             "(via a sentinel file each writes on exit, any reason) and stop "
                             "this sim's loop automatically through its normal exit path. "
                             "Safer than Ctrl-C, which races Isaac's own SIGINT teardown.")
    parser.add_argument("--no-debug-frames", action="store_true",
                        help="Skip writing camera_debug.png and depth_debug.npy each frame "
                             "(matplotlib + disk I/O are a major sim bottleneck).")
    parser.add_argument("--agile-overhead-debug", type=str, default=None,
                        metavar="PATH",
                        help="Agile policy only: write the overhead trajectory/depth debug "
                             "PNG to PATH each sim frame. Can be used with --no-debug-frames "
                             "to avoid the heavier camera_debug/depth_debug outputs.")
    parser.add_argument("--agile-depth-flip", choices=["none", "both", "v", "h"],
                        default="none",
                        help="Agile policy only: flip the depth image before publishing. "
                             "Upstream agile_autonomy does cv2.flip(depth, -1) == 'both' "
                             "(the net was trained on inverted depth). 'v'/'h' isolate one "
                             "axis to A/B test up-down vs left-right misregistration.")
    parser.add_argument("--record-depth-video", type=str, default=None, metavar="PATH",
                        help="Encode the EXACT depth fed to the policy to an MP4 (turbo "
                             "colormap, headless-safe). Used by compare/run_comparison.py "
                             "--record-video.")
    parser.add_argument("--record-rgb-video", type=str, default=None, metavar="PATH",
                        help="Encode an onboard RGB view to an MP4 (headless-safe). Spawns a "
                             f"dedicated {RGB_W}x{RGB_H} 'drone_camera' at the policy depth "
                             "camera's body pose and horizontal FOV. Used by "
                             "compare/run_comparison.py --record-video.")
    parser.add_argument("--record-video-fps", type=float, default=15.0,
                        help="Target frame rate for --record-depth-video (default 15).")
    parser.add_argument("--record-video-scale", type=int, default=4,
                        help="Integer upscale applied to each depth frame before encoding "
                             "(default 4; nearest-neighbour).")
    parser.add_argument("--log-traj", type=str, default=None, metavar="PATH",
                        help="Log the drone's ground-truth ENU pose+velocity each tick and "
                             "dump it (with the obstacle field + goal/start) to this .npz on "
                             "exit, for compare/metrics.py to score the run. Used by "
                             "compare/run_comparison.py.")
    parser.add_argument("--goal", type=float, nargs="+", default=None, metavar="V",
                        help="Goal position (X Y [Z]) to save in the trajectory npz, overriding "
                             "the obstacle field's p_target. Used by compare/run_comparison.py "
                             "so metrics.py scores against the actual offboard goal.")
    # parse_known_args so Isaac Sim's own argv flags don't trip argparse
    args, _ = parser.parse_known_args()

    if args.usd_environment is None and args.environment not in SIMULATION_ENVIRONMENTS:
        available = ", ".join(sorted(SIMULATION_ENVIRONMENTS))
        parser.error(f"unknown environment {args.environment!r}; available: {available}")

    pg_app = PegasusApp(
        seed=args.seed,
        scale=args.scale,
        spawn_yaw_deg=args.spawn_yaw,
        policy=args.policy,
        obstacles=args.obstacles,
        environment=args.environment,
        spawn_xyz=tuple(args.spawn),
        usd_environment=args.usd_environment,
        env_scale=args.env_scale,
        obstacle_assets=args.obstacle_assets,
        auto_stop=args.auto_stop,
        debug_frames=not args.no_debug_frames,
        log_traj=args.log_traj,
        goal_xyz=tuple(args.goal) if args.goal is not None else None,
        record_depth_video=args.record_depth_video,
        record_rgb_video=args.record_rgb_video,
        record_video_fps=args.record_video_fps,
        record_video_scale=args.record_video_scale,
        agile_overhead_debug_path=(
            args.agile_overhead_debug
            if args.agile_overhead_debug is not None
            else (None if args.no_debug_frames else "agile_overhead_debug.png")
        ),
        agile_depth_flip=args.agile_depth_flip,
    )
    pg_app.run()


if __name__ == "__main__":
    main()
