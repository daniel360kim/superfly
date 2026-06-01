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
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import sys
sys.path.insert(0, "/home/dtc-system/superfly/starling-deployment")
from depth_transport import DepthPublisher, RENDER_H, RENDER_W

# --- DiffPhysDrone single_agent camera params ---
FOV_X_HALF_TAN = 0.82
CAM_ANGLE_DEG = 20.0
FOV_X_DEG = 2.0 * math.degrees(math.atan(FOV_X_HALF_TAN))  # ~78.6 deg horizontal


class PegasusApp:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": False,
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        # Forward-facing depth camera, pitched CAM_ANGLE_DEG down.
        # Pegasus MonocularCamera 'orientation' is Euler ZYX (deg) relative to the
        # body frame; default [0,0,180] points the camera forward (+X body). A
        # positive pitch about the camera's lateral axis tilts the view downward.
        self._camera = MonocularCamera("depth_cam", config={
            "depth": True,
            "position": np.array([0.10, 0.0, 0.0]),
            "orientation": np.array([0.0, CAM_ANGLE_DEG, 180.0]),
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
        config_multirotor.graphical_sensors = [self._camera]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.1],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()
        self._depth_pub = DepthPublisher()
        self._dbg_n = 0           # frame counter for throttled debug dumps
        self._dbg_every = 30      # save a debug PNG every N published frames
        self.stop_sim = False

    def _publish_depth(self):
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

    def _dump_depth_debug(self, depth):
        """Save the EXACT published array to depth_debug.png + .npy every N frames,
        so the published orientation can be checked against the native convention
        (row 0 = up, col 0 = left). Never let a viz error kill the sim loop."""
        self._dbg_n += 1
        if self._dbg_n % self._dbg_every != 0:
            return
        try:
            np.save("depth_debug.npy", depth)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            im = ax.imshow(depth, origin="upper", cmap="turbo", vmin=0.3, vmax=24.0)
            fig.colorbar(im, ax=ax, label="depth [m]")
            ax.set_title(f"published depth  frame={self._dbg_n}  "
                         f"min={depth.min():.2f} max={depth.max():.2f} m")
            ax.set_xlabel("col  (0 = left)")
            ax.set_ylabel("row  (0 = up)")
            fig.tight_layout()
            fig.savefig("depth_debug.png", dpi=90)
            plt.close(fig)
        except Exception as e:
            carb.log_warn(f"depth debug dump failed: {e}")

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            self._publish_depth()
        carb.log_warn("PegasusApp closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
