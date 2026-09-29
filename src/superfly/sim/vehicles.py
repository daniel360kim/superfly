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


#: Rotor positions of the Starling 2 Max [m], body FLU, |x| / |y| -- ModalAI's
#: D0012 PX4 params (CA_ROTOR*_PX/PY 0.085 / 0.0625; PX4's FRD y is flipped),
#: as installed in airstation03's PX4 airframe 10099_starling.
STARLING_ARM_X = 0.085
STARLING_ARM_Y = 0.0625
#: Each Iris rotor is its own rigid body with no authored mass; give it a small
#: explicit one so the total is the spec's mass, not a PhysX default.
STARLING_ROTOR_MASS = 0.005


def apply_vehicle_overrides(stage, root: str, name: str) -> str | None:
    """Turn the spawned Iris at `root` into the Starling 2 Max: sys-ID mass,
    inertia and CoM on the body, the Starling's rotor positions (same Pegasus
    rotor order and spin directions as the Iris: 0 front-right, 1 back-left,
    2 front-left, 3 back-right, rot_dir [-1,-1,1,1] -- which is also PX4's
    quad-x order), rotor rigid bodies at a small explicit mass. Call after the
    Multirotor is created and before world.reset(). No-op (None) unless the
    Starling is being built on the Iris frame. Returns a one-line summary."""
    import os
    from pxr import Gf, UsdGeom, UsdPhysics
    if name != "starling2max" or os.environ.get("SUPERFLY_VEHICLE_USD"):
        return None
    spec = load_starling_spec()
    n_rot = 4
    body = stage.GetPrimAtPath(f"{root}/body")
    if not body.IsValid():
        raise RuntimeError(f"{root}/body not found -- not the Pegasus Iris layout")
    m = UsdPhysics.MassAPI.Apply(body)
    m.CreateMassAttr().Set(float(spec["mass_kg"]) - n_rot * STARLING_ROTOR_MASS)
    I = spec["inertia_flu"]
    m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(float(I["ixx_roll"]), float(I["iyy_pitch"]),
                                               float(I["izz_yaw"])))
    m.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in spec["com_offset_flu_m"]]))
    m.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    # Iris order: 0 front-right, 1 back-left, 2 front-left, 3 back-right
    signs = [(1, -1), (-1, 1), (1, 1), (-1, -1)]
    moved = []
    for i, (sx, sy) in enumerate(signs):
        rotor = stage.GetPrimAtPath(f"{root}/rotor{i}")
        joint = stage.GetPrimAtPath(f"{root}/rotor{i}/joint{i}")
        if not rotor.IsValid() or not joint.IsValid():
            raise RuntimeError(f"{root}/rotor{i} or its joint not found")
        x, y = sx * STARLING_ARM_X, sy * STARLING_ARM_Y
        xf = UsdGeom.Xformable(rotor)
        ops = [op for op in xf.GetOrderedXformOps()
               if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        z = 0.023
        if ops:
            old = ops[0].Get()
            z = float(old[2]) if old is not None else z
            ops[0].Set(type(old)(x, y, z) if old is not None else Gf.Vec3d(x, y, z))
        else:
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        # The joint's body0 frame is the Iris body mesh, rotated -90 deg about
        # z against the body (localPos0 = (y, -x, z), read off the stock USD).
        j = UsdPhysics.Joint(joint)
        j.GetLocalPos0Attr().Set(Gf.Vec3f(y, -x, z))
        # The Iris props (~0.12 m radius) overlap at the Starling's 0.125 m
        # rotor spacing; as colliders they would push each other apart.
        from pxr import Usd
        for c in Usd.PrimRange(rotor):
            if c.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr().Set(False)
        rm = UsdPhysics.MassAPI.Apply(rotor)
        rm.CreateMassAttr().Set(STARLING_ROTOR_MASS)
        rm.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-6, 1e-6, 2e-6))
        moved.append(f"r{i}({x:+.3f},{y:+.4f})")
    return (f"Starling overrides on {root}: body m={spec['mass_kg'] - n_rot * STARLING_ROTOR_MASS:.3f} "
            f"+ 4 x {STARLING_ROTOR_MASS} kg rotors, I={tuple(I.values())}, "
            f"rotors {' '.join(moved)}")


def load_starling_spec() -> dict:
    return yaml.safe_load(STARLING_SPEC.read_text())


#: Pegasus maps each PX4 motor output u in [0, 1] to a rotor-speed reference
#: omega = (u + offset) * scaling + zero_position_armed (PX4MavlinkBackend
#: ThrusterControl), stock scaling 1000 / zero 100 rad/s. That was sized for
#: the Iris (max_rotor_velocity 1100 = full scale at u = 1). The Starling's
#: rotor saturates at 830 rad/s, so with the stock scaling it clips at u =
#: 0.73 and PX4's allocator -- which believes it has headroom to 1.0 -- loses
#: differential authority silently above that. Scaling (830 - 100) makes u = 1
#: the real saturation, i.e. the same normalized throttle curve shape the Iris
#: has. SUPERFLY_STARLING_INPUT_SCALING overrides (1000 = the stock mapping).
PEGASUS_ZERO_POSITION_ARMED = 100.0


def vehicle_backend_overrides(name: str) -> dict:
    """Extra PX4MavlinkBackendConfig keys for --vehicle <name> ({} = stock)."""
    import os
    if name != "starling2max":
        return {}
    w_max = float(load_starling_spec()["assumed"]["max_rotor_velocity_rad_s"])
    scaling = float(os.environ.get("SUPERFLY_STARLING_INPUT_SCALING",
                                   w_max - PEGASUS_ZERO_POSITION_ARMED))
    return {"input_offset": [0.0] * 4,
            "input_scaling": [scaling] * 4,
            "zero_position_armed": [PEGASUS_ZERO_POSITION_ARMED] * 4}


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
        usd = os.environ.get("SUPERFLY_VEHICLE_USD", "")
        if usd:
            label = (f"Starling 2 Max from {usd} (kT={k_t:g}, "
                     f"w_max={w_max:g} rad/s, tau 55/85 ms)")
            return usd, curve, label
        # Default: the sys-ID body on the Pegasus Iris frame (see
        # apply_vehicle_overrides). The lab USD lives on Nucleus, whose login
        # on airstation03 has expired (ATTEMPTS 2026-08) and which OSMO cannot
        # reach at all -- a headless Kit then blocks on a browser login and the
        # vehicle never spawns (observed 2026-09-29: PX4 waits for a heartbeat
        # forever). SUPERFLY_VEHICLE_USD=<STARLING_USD> restores the lab USD
        # once auth works.
        spec = load_starling_spec()
        label = (f"Starling 2 Max sys-ID on the Iris frame (m={spec['mass_kg']} kg, "
                 f"I={[round(v, 5) for v in spec['inertia_flu'].values()]}, arms "
                 f"+-{STARLING_ARM_X}/+-{STARLING_ARM_Y} m, kT={k_t:g}, "
                 f"w_max={w_max:g} rad/s, tau 55/85 ms)")
        return ROBOTS["Iris"], curve, label
    raise ValueError(f"unknown vehicle {name!r}")
