"""Per-instance namespacing for concurrent Isaac + PX4 SITL trials on one box.

The harness historically ran ONE trial per account at a time: PX4 instance 0
(simulator TCP 4560, GCS MAVLink to UDP 14550), the depth frames on UDP 15001,
the agile debug frames on UDP 15002 and fixed /tmp sentinels. Parallel
campaigns (superfly.compare.runner --isaac-slots K) give each concurrent trial
a slot i in 0..K-1 and export SUPERFLY_INSTANCE=i to the sim and offboard
processes; every shared resource below is then derived from i.

SUPERFLY_INSTANCE unset (the default) = the historical names and ports,
byte-for-byte: nothing changes for a legacy (exclusive) campaign.

Dependency-free on purpose (os only): imported by the sim (Isaac's python) and
by every offboard venv, like the rest of superfly.common.
"""

import os

ENV = "SUPERFLY_INSTANCE"

PORT_STRIDE = 10            # instance i shifts the UDP transport ports by 10*i
DEPTH_PORT_BASE = 15001     # sim -> offboard depth frames (legacy port)
AGILE_DEBUG_PORT_BASE = 15002   # agile offboard -> sim debug frames (legacy port)
SIM_TCP_BASE = 4560         # PX4 simulator link: Pegasus listens on 4560 + i
LEGACY_GCS_PORT = 14550     # PX4's stock GCS link remote port (all instances)
GCS_PORT_BASE = 14650       # parallel mode: instance i's GCS link -> 14650 + i


def instance():
    """This process's instance number, or None (legacy, exclusive campaign)."""
    v = os.environ.get(ENV, "").strip()
    return int(v) if v else None


def depth_port(i=None):
    i = instance() if i is None else i
    return DEPTH_PORT_BASE + PORT_STRIDE * (i or 0)


def rgb_shm_path(i=None):
    """Shared-memory file of the sim -> offboard policy RGB frames (RGB students,
    2026-10-01; superfly.common.transport.RgbPublisher): one per uid and instance,
    "x" for a legacy (exclusive) campaign."""
    i = instance() if i is None else i
    return f"/dev/shm/superfly_rgb_{os.getuid()}_{'x' if i is None else int(i)}"


def agile_debug_port(i=None):
    i = instance() if i is None else i
    return AGILE_DEBUG_PORT_BASE + PORT_STRIDE * (i or 0)


def gcs_port(i):
    """UDP port the offboard listens on for PX4 instance i in parallel mode.
    PX4's stock GCS link sends EVERY instance to 14550, so concurrent
    instances would cross-talk into one listener; the harness re-points the
    link per instance (px4-rc.mavlink override, superfly.compare.runner)."""
    return GCS_PORT_BASE + int(i)


def tmp_path(base, i=None):
    """base unchanged for a legacy process, else base_<uid>_<i>."""
    i = instance() if i is None else i
    return base if i is None else f"{base}_{os.getuid()}_{i}"
