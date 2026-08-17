"""Eval-sim vehicle configs: which airframe the Pegasus quadrotor actually is.

Default is Pegasus's stock Iris (the harness's historical vehicle,
byte-identical behavior). "starling2max" builds the vehicle from the AirLab
sys-ID campaign (configs/vehicles/starling2max.yaml -- the single source of
truth): mass/inertia/CoM come from the lab's starling2max.usd, the thrust
curve uses the measured rotor constant / rolling-moment coefficient, and a
first-order rotor lag reproduces the measured 55 ms spin-up / 85 ms
spin-down motor response (Pegasus's stock QuadraticThrustCurve is
deliberately instantaneous).

Import only under Isaac's interpreter (pegasus imports).

VALIDATION NOTE (airstation03, before any campaign): (1) the starling USD's
rotor prim order/spin directions must match Pegasus's rot_dir convention
[-1,-1,1,1] -- a mismatch shows up as an immediate yaw spin on takeoff;
(2) PX4's none_iris airframe gains were tuned for the heavier Iris -- watch
the first climb for oscillation.
"""

from pathlib import Path

import numpy as np
import yaml

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.thrusters.quadratic_thrust_curve import QuadraticThrustCurve

_REPO = Path(__file__).resolve().parents[3]
STARLING_SPEC = _REPO / "configs" / "vehicles" / "starling2max.yaml"

# The lab USD with the sys-ID mass/inertia/CoM applied (Slite dj1982T35dt0qG).
# Override with SUPERFLY_VEHICLE_USD (e.g. a staged local copy on OSMO, where
# omniverse:// cannot authenticate).
STARLING_USD = ("omniverse://airlab-nucleus.andrew.cmu.edu/Library/Assets/"
                "ModalAI/starling_2_max/starling2max.usd")


class FirstOrderQuadraticThrustCurve(QuadraticThrustCurve):
    """QuadraticThrustCurve + a first-order rotor-speed lag.

    The stock curve applies the commanded rotor velocity instantaneously
    ("no delay introduced" in its update()); the Starling 2 Max sys-ID
    measured tau ~55 ms when spinning up and ~85 ms when spinning down, and
    that asymmetric lag is exactly the plant behavior an agile policy fights.
    velocity -> reference through dv = (1 - exp(-dt/tau)) * (ref - v) each
    update, with tau chosen per-rotor by the sign of (ref - v)."""

    def __init__(self, config={}, tau_up: float = 0.055, tau_down: float = 0.085):
        super().__init__(config)
        self._tau_up = float(tau_up)
        self._tau_down = float(tau_down)
        self._lagged = [0.0 for _ in range(self._num_rotors)]

    def update(self, state, dt: float):
        refs = list(self._input_reference)
        for i in range(self._num_rotors):
            ref = np.maximum(self.min_rotor_velocity[i],
                             np.minimum(refs[i], self.max_rotor_velocity[i]))
            tau = self._tau_up if ref >= self._lagged[i] else self._tau_down
            alpha = 1.0 - np.exp(-dt / max(tau, 1e-6))
            self._lagged[i] = self._lagged[i] + alpha * (ref - self._lagged[i])
        # Run the stock (instantaneous) update against the LAGGED references:
        # it re-clips and applies the quadratic force + rolling moment.
        self._input_reference = list(self._lagged)
        out = super().update(state, dt)
        self._input_reference = refs   # keep the caller's reference visible
        return out


def load_starling_spec() -> dict:
    return yaml.safe_load(STARLING_SPEC.read_text())


def vehicle_usd_and_curve(name: str):
    """(usd_file, thrust_curve, label) for --vehicle <name>."""
    import os
    if name == "iris":
        # Historical default: stock Iris USD + stock curve (Pegasus defaults).
        return ROBOTS["Iris"], QuadraticThrustCurve(), "Iris (Pegasus stock)"
    if name == "starling2max":
        spec = load_starling_spec()
        k_t = float(spec["rotor"]["thrust_constant"])
        k_m = float(spec["rotor"]["rolling_moment_coefficient"])
        w_max = float(spec["assumed"]["max_rotor_velocity_rad_s"])
        curve = FirstOrderQuadraticThrustCurve(
            {
                "num_rotors": 4,
                "rotor_constant": [k_t] * 4,
                "rolling_moment_coefficient": [k_m] * 4,
                "rot_dir": [-1, -1, 1, 1],
                "min_rotor_velocity": [0.0] * 4,
                "max_rotor_velocity": [w_max] * 4,
            },
            tau_up=float(spec["rotor"]["time_constant_up_s"]),
            tau_down=float(spec["rotor"]["time_constant_down_s"]),
        )
        usd = os.environ.get("SUPERFLY_VEHICLE_USD", STARLING_USD)
        label = (f"Starling 2 Max (m=0.557 kg, kT={k_t:g}, "
                 f"w_max={w_max:g} rad/s, tau 55/85 ms)")
        return usd, curve, label
    raise ValueError(f"unknown vehicle {name!r}")
