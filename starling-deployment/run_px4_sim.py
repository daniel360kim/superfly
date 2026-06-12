#!/usr/bin/env python
"""
Launch Pegasus + PX4 SITL with a single Iris quadrotor, plus a forward-facing
depth camera matching the DiffPhysDrone single_agent training config:
    fov_x_half_tan = 0.82  ->  horizontal FOV = 2*atan(0.82) = 78.6 deg
    cam_angle      = 20    ->  camera pitched 20 deg DOWN about the body left-axis
The 48x64 metric (planar Z) depth is published over UDP each control tick for
the offboard policy process to consume (see depth_transport.py).

Run this AFTER PX4 SITL is up; then run diffdrone_offboard.py --depth.
"""

import argparse
import math
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless PNG backend, no GUI needed
import matplotlib.pyplot as plt
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
sys.path.insert(0, "/home/danielkim/superfly/starling-deployment")
from depth_transport import DepthPublisher, RENDER_H, RENDER_W
from obstacle_field import generate as generate_field

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


class PegasusApp:

    SPAWN_YAW_DEG = 0.0  # EKF heading is mag-locked (~+Y); we rotate the field instead

    def __init__(self, seed: int = 0, scale: float = 5.0, spawn_yaw_deg: float = 0.0,
                 policy: str = "diffphys", obstacles: str = "diffphys"):
        self.SPAWN_YAW_DEG = spawn_yaw_deg
        self.policy = policy
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        # Spawn the obstacle field (analytic primitives). The DiffPhysDrone-
        # distribution field is the default; --obstacles diffaero regenerates the
        # field to match DiffAero's training distribution (around the start->goal
        # line). Either way both policies can fly whichever course is selected.
        if obstacles == "diffaero":
            from obstacle_field import generate_diffaero
            self.field = generate_diffaero(seed=seed, scale=scale)
        else:
            self.field = generate_field(seed=seed, scale=scale)
        print("[obstacle_field]", self.field.summary())
        self._spawn_obstacles(self.field)

        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": False,
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        if policy == "diffaero":
            self._setup_camera_diffaero()
        else:
            self._setup_camera_diffphys()
        config_multirotor.graphical_sensors = [self._camera]

        # Spawn the drone on the ground at the field's start XY (it climbs from here).
        # SPAWN_YAW_DEG cancels the sim's EKF heading offset: with spawn yaw 0 the
        # mag-driven EKF reported ENU yaw=90° (facing +Y), but the obstacle corridor
        # runs +X. Spawn rotated by -90° so the reconstructed heading reads ~0 (faces
        # +X, down the corridor). Flip sign if the log still shows yaw≈±90.
        spawn_xy = [float(self.field.p_init[0]), float(self.field.p_init[1]), 0.1]
        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            spawn_xy,
            Rotation.from_euler("XYZ", [0.0, 0.0, self.SPAWN_YAW_DEG], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()
        self._depth_pub = DepthPublisher()
        self._dbg_n = 0           # frame counter for throttled debug dumps
        self._dbg_every = 30      # save a debug PNG every N published frames
        self.stop_sim = False

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

    
    def _spawn_obstacles(self, fld):
        """Spawn the analytic obstacle field as static Isaac prims (ENU, z up)."""
        # Spheres
        for i, (cx, cy, cz, r) in enumerate(fld.spheres):
            FixedSphere(
                prim_path=f"/World/obstacles/sphere_{i}",
                position=np.array([cx, cy, cz]),
                radius=float(r),
                color=np.array([0.8, 0.3, 0.3]),
            )
        # Boxes (voxels): half-extents -> full-size scale. DiffPhys boxes are
        # axis-aligned 6-tuples; DiffAero boxes are 9-tuples carrying an XYZ-euler
        # rotation (radians) for tilted pillars.
        for i, box in enumerate(fld.boxes):
            cx, cy, cz, hx, hy, hz = box[:6]
            orientation = None
            if len(box) >= 9:
                roll, pitch, yaw = box[6], box[7], box[8]
                q = Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_quat()  # [x,y,z,w]
                orientation = np.array([q[3], q[0], q[1], q[2]])  # FixedCuboid wants [w,x,y,z]
            FixedCuboid(
                prim_path=f"/World/obstacles/box_{i}",
                position=np.array([cx, cy, cz]),
                scale=np.array([2 * hx, 2 * hy, 2 * hz]),
                orientation=orientation,
                color=np.array([0.3, 0.5, 0.8]),
            )
        # Vertical cylinders (axis = world Z), tall enough to span the flight band.
        CYL_HEIGHT = 12.0
        for i, (cx, cy, r) in enumerate(fld.cyl_v):
            self._spawn_cylinder(f"/World/obstacles/cylv_{i}",
                                 pos=(cx, cy, CYL_HEIGHT / 2 - 1.0),
                                 radius=float(r), height=CYL_HEIGHT, axis="Z")
        # Horizontal cylinders (2 minor ground obstacles). Stored (cx,cy,cz,r)
        # after the field rotation; spawn lying along world X.
        CYLH_LEN = 6.0
        for i, (cx, cy, cz, r) in enumerate(fld.cyl_h):
            self._spawn_cylinder(f"/World/obstacles/cylh_{i}",
                                 pos=(cx, cy, cz),
                                 radius=float(r), height=CYLH_LEN, axis="X")

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
        if self.policy == "diffaero":
            self._publish_depth_diffaero()
        else:
            self._publish_depth_diffphys()

    def _publish_depth_diffphys(self):
        """Grab the camera's planar Z-depth, resize to 48x64, publish over UDP."""
        cam = getattr(self._camera, "_camera", None)
        if cam is None or not getattr(self._camera, "_camera_full_set", False):
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

    def _publish_depth_diffaero(self):
        """Publish the 9x16 EUCLIDEAN range image DiffAero expects.

        Steps: planar Z-depth (distance_to_image_plane) -> Euclidean range via the
        precomputed per-pixel scale -> min-pool DA_POOL x DA_POOL (nearest surface
        per output cell, matching DiffAero's one-ray-per-cell sampling). The policy
        process applies depth = 1 - clamp(r,0,5)/5. Convention: row 0 = up,
        col 0 = left (same as the DiffPhysDrone path)."""
        cam = getattr(self._camera, "_camera", None)
        if cam is None or not getattr(self._camera, "_camera_full_set", False):
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

    def _dump_depth_debug(self, depth):
        """Save the EXACT published depth array + the drone's RGB view to
        camera_debug.png (+ depth .npy) every N frames. RGB comes from the SAME
        camera/render product as the depth, so it is the literal drone viewpoint.
        Check orientation vs native convention (row 0 = up, col 0 = left).
        Never let a viz error kill the sim loop."""
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

            depth_vmax = DA_MAX_DIST if self.policy == "diffaero" else 24.0
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

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            self._publish_depth()
        carb.log_warn("PegasusApp closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Obstacle-field RNG seed (eval suite: 0, 1, 2)")
    parser.add_argument("--scale", type=float, default=5.0,
                        help="World scale (corridor depth ~= 8*scale m)")
    parser.add_argument("--spawn-yaw", type=float, default=0.0,
                        help="Spawn yaw [deg]. EKF heading is mag-locked in sim, so this "
                             "mainly affects the initial facing; the field is rotated instead.")
    parser.add_argument("--policy", choices=["diffphys", "diffaero"], default="diffphys",
                        help="Which policy's camera/depth pipeline to configure: "
                             "diffphys (12x16, 78.6 deg, 24 m, planar, pitched 20 deg down) "
                             "or diffaero (9x16, 86 deg, 5 m, Euclidean, forward).")
    parser.add_argument("--obstacles", choices=["diffphys", "diffaero"], default="diffphys",
                        help="Obstacle-field distribution: diffphys (existing field) or "
                             "diffaero (DiffAero training distribution around start->goal).")
    # parse_known_args so Isaac Sim's own argv flags don't trip argparse
    args, _ = parser.parse_known_args()
    pg_app = PegasusApp(seed=args.seed, scale=args.scale, spawn_yaw_deg=args.spawn_yaw,
                        policy=args.policy, obstacles=args.obstacles)
    pg_app.run()


if __name__ == "__main__":
    main()
