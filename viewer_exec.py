# Scene-setup script run via `isaac-sim.streaming.sh --exec viewer_exec.py`
# on airstation03. Unlike the OSMO version (osmo/isaac_webrtc_viewer.yaml),
# this does NOT construct SimulationApp itself -- the
# `isaacsim.exp.full.streaming` app profile that isaac-sim.streaming.sh
# launches has already done that, using NVIDIA's own proven streaming setup
# (depends on omni.services.livestream.nvcf, not the omni.kit.livestream.app/
# .webrtc extensions we were hand-enabling in the OSMO attempt). This script
# only needs to spawn the scene into the already-running app.
#
# Same bare "Iris" quadrotor as the OSMO version -- no PX4, no policy, just
# validating that something visible actually streams.

import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

timeline = omni.timeline.get_timeline_interface()
pg = PegasusInterface()
pg._world = World(**pg._world_settings)
world = pg.world
pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

config = MultirotorConfig()
config.backends = []  # no PX4, no policy -- bare vehicle, physics only
Multirotor(
    "/World/quadrotor1",
    ROBOTS["Iris"],
    0,
    [0.0, 0.0, 0.3],
    [0.0, 0.0, 0.0, 1.0],  # identity quaternion (x, y, z, w)
    config=config,
)
world.reset()

from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.3])

print("[viewer_exec] scene ready, starting timeline", flush=True)
timeline.play()
