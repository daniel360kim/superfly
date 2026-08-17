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

# ROOT CAUSE (confirmed by reading simulation_context.py directly): its
# __init__ has a class-level singleton guard --
# `if SimulationContext._sim_context_initialized: return` -- and the "Full
# Streaming" app already constructs one during its own boot, before this
# --exec script ever runs. So World(**pg._world_settings) above returned
# instantly without ever calling _init_stage(), which is the method that
# actually creates _physics_context. Neither reset() nor reset_async() nor
# any amount of app.update() pumping fixes this (all three tried and
# confirmed failing) because they all assume _init_stage() already ran.
# Call it directly ourselves, bypassing the blocked constructor.
world._init_stage(**pg._world_settings)

import omni.kit.app
from omni.isaac.core.utils.stage import is_stage_loading
_app = omni.kit.app.get_app()

pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
while is_stage_loading():
    _app.update()

world.reset()
for _ in range(10):
    _app.update()
print(f"[viewer_exec] physics_context={world._physics_context!r}", flush=True)

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
