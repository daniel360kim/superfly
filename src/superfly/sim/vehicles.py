"""Eval-sim vehicle configs: which airframe the Pegasus quadrotor actually is.

Default is Pegasus's stock Iris (the harness's historical vehicle,
byte-identical behavior). The Starling 2 Max comes from the AirLab sys-ID
campaign (configs/vehicles/starling2max.yaml -- the single source of truth):
the thrust curve uses the measured rotor constant / rolling-moment
coefficient, and a first-order rotor lag reproduces the measured 55 ms
spin-up / 85 ms spin-down motor response (Pegasus's stock
QuadraticThrustCurve is deliberately instantaneous). Two airframes carry it:

  starling2max       the lab's own starling2max.usd (Nucleus; the real
                     airframe: body, landing gear, 0.091 m props at
                     (+-0.095, +-0.13) m, prop tips 0.252 m from the centre),
                     sys-ID mass / inertia / CoM re-applied from the yaml, rotor
                     links given an explicit small mass. Since 2026-09-29.
  starling2max_iris  the same sys-ID body on the Pegasus IRIS frame: Iris body
                     mesh (arms reach 0.273 m) and Iris props moved to the
                     Starling's (+-0.085, +-0.0625) -- what every
                     `--vehicle starling2max` run before 2026-09-29 flew. Kept
                     for comparison.

Import only under Isaac's interpreter (pegasus imports).

Rotor order / spin: both follow Pegasus's (and PX4 quad-x's) order -- 0
front-right, 1 back-left, 2 front-left, 3 back-right -- with rot_dir
[-1,-1,1,1] set in the thrust curve (a USD carries no spin direction); a
mismatch shows up as an immediate yaw spin on takeoff. PX4 flies both with
the none_iris airframe (Iris gains).
"""

from pathlib import Path

import numpy as np
import yaml

from pegasus.simulator.params import ROBOTS
from pegasus.simulator.logic.thrusters.quadratic_thrust_curve import QuadraticThrustCurve

_REPO = Path(__file__).resolve().parents[3]
STARLING_SPEC = _REPO / "configs" / "vehicles" / "starling2max.yaml"

# The lab USD (compare.registry.STARLING_USD; $SUPERFLY_VEHICLE_USD overrides).
# Loading it from Nucleus needs the API token in the sim's environment
# (OMNI_API_TOKEN, or OMNI_USER='$omni-api-token' + OMNI_PASS): compare.runner
# loads it from ~/.omni_env and stats the USD before the first trial
# (superfly.common.nucleus).
from superfly.compare.registry import STARLING_USD, vehicle_usd_url  # noqa: E402,F401

#: --vehicle names that are the Starling 2 Max (sys-ID thrust curve + mass)
STARLING_VEHICLES = ("starling2max", "starling2max_iris")


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
    """Per-vehicle USD fix-ups after the Multirotor is created and before
    world.reset(): starling2max -> _starling_usd_overrides, starling2max_iris
    -> _iris_frame_overrides, anything else -> None. Returns a one-line
    summary (printed by the sim)."""
    if name == "starling2max":
        return _starling_usd_overrides(stage, root)
    if name == "starling2max_iris":
        return _iris_frame_overrides(stage, root)
    return None


#: Pegasus rotor order as (sign x, sign y) of each rotor axis, body FLU:
#: 0 front-right, 1 back-left, 2 front-left, 3 back-right.
PEGASUS_ROTOR_SIGNS = [(1, -1), (-1, 1), (1, 1), (-1, -1)]


def _set_body_mass(stage, root: str, spec: dict, n_rot: int = 4):
    from pxr import Gf, UsdPhysics
    body = stage.GetPrimAtPath(f"{root}/body")
    if not body.IsValid():
        raise RuntimeError(f"{root}/body not found -- not a Pegasus multirotor layout")
    m = UsdPhysics.MassAPI.Apply(body)
    m.CreateMassAttr().Set(float(spec["mass_kg"]) - n_rot * STARLING_ROTOR_MASS)
    I = spec["inertia_flu"]
    m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(float(I["ixx_roll"]), float(I["iyy_pitch"]),
                                               float(I["izz_yaw"])))
    m.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(v) for v in spec["com_offset_flu_m"]]))
    m.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return I


def _set_rotor_mass(rotor):
    from pxr import Gf, UsdPhysics
    rm = UsdPhysics.MassAPI.Apply(rotor)
    rm.CreateMassAttr().Set(STARLING_ROTOR_MASS)
    rm.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1e-6, 1e-6, 2e-6))


def _rotor_in_body(stage, root: str, i: int):
    """Rotor i's origin (= its spin axis) in the body frame, FLU [m]."""
    from pxr import Usd, UsdGeom
    tc = Usd.TimeCode.Default()
    rot = UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/rotor{i}")).ComputeLocalToWorldTransform(tc)
    body = UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/body")).ComputeLocalToWorldTransform(tc)
    return (rot * body.GetInverse()).ExtractTranslation()


def _starling_usd_overrides(stage, root: str) -> str:
    """The lab starling2max.usd as spawned at `root`: check it is the layout
    Pegasus drives (body + rotor0..3 rigid bodies with joint0..3, each rotor
    axis in its Pegasus / PX4 quad-x quadrant, so rot_dir [-1,-1,1,1] spins
    them the right way), re-apply the yaml's sys-ID mass/inertia/CoM to the
    body (the USD carries the same numbers, rounded), give each rotor link an
    explicit small mass (the USD authors none, so PhysX would derive one from
    the prop collider's volume at its default density: a heavier vehicle than
    the sys-ID), and deactivate the CAD export's own DistantLight (it would
    light the whole scene). Rotor positions, meshes and colliders are the
    USD's own, unchanged (props 0.091 m, 8 mm apart front-to-back: their
    colliders do not overlap, unlike the Iris props on the Iris frame)."""
    spec = load_starling_spec()
    rotors = []
    for i, (sx, sy) in enumerate(PEGASUS_ROTOR_SIGNS):
        rotor = stage.GetPrimAtPath(f"{root}/rotor{i}")
        joint = stage.GetPrimAtPath(f"{root}/rotor{i}/joint{i}")
        if not rotor.IsValid() or not joint.IsValid():
            raise RuntimeError(f"{root}/rotor{i} or its joint{i} not found -- not the "
                               "Pegasus rotor layout Multirotor drives")
        tb = _rotor_in_body(stage, root, i)
        if (tb[0] > 0) != (sx > 0) or (tb[1] > 0) != (sy > 0):
            raise RuntimeError(f"rotor{i} sits at body ({tb[0]:+.3f}, {tb[1]:+.3f}); Pegasus / "
                               f"PX4 quad-x expects the ({'+' if sx > 0 else '-'}x, "
                               f"{'+' if sy > 0 else '-'}y) quadrant -- its spin direction "
                               "and PX4's allocation would be wrong")
        _set_rotor_mass(rotor)
        rotors.append(f"r{i}({tb[0]:+.3f},{tb[1]:+.3f})")
    I = _set_body_mass(stage, root, spec)
    light = stage.GetPrimAtPath(f"{root}/defaultLight")
    if light.IsValid():
        light.SetActive(False)
    return (f"Starling 2 Max lab USD at {root}: body m={spec['mass_kg'] - 4 * STARLING_ROTOR_MASS:.3f} "
            f"+ 4 x {STARLING_ROTOR_MASS} kg rotors, I={tuple(I.values())}, rotors "
            f"{' '.join(rotors)} (the USD's), CAD defaultLight "
            f"{'deactivated' if light.IsValid() else 'absent'}")


def _iris_frame_overrides(stage, root: str) -> str:
    """Turn the spawned Iris at `root` into the Starling 2 Max
    (--vehicle starling2max_iris): sys-ID mass, inertia and CoM on the body,
    the Starling's rotor positions from ModalAI's PX4 params (same Pegasus
    rotor order and spin directions as the Iris: 0 front-right, 1 back-left,
    2 front-left, 3 back-right, rot_dir [-1,-1,1,1] -- which is also PX4's
    quad-x order), rotor rigid bodies at a small explicit mass. The Iris body
    mesh (arms to 0.273 m from the centre) and the Iris props (0.129 m) stay."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    spec = load_starling_spec()
    I = _set_body_mass(stage, root, spec)
    moved = []
    for i, (sx, sy) in enumerate(PEGASUS_ROTOR_SIGNS):
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
        # The Iris props (~0.13 m radius) overlap at the Starling's 0.125 m
        # rotor spacing; as colliders they would push each other apart.
        for c in Usd.PrimRange(rotor):
            if c.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr().Set(False)
        _set_rotor_mass(rotor)
        moved.append(f"r{i}({x:+.3f},{y:+.4f})")
    return (f"Starling overrides on the Iris frame at {root}: body m="
            f"{spec['mass_kg'] - 4 * STARLING_ROTOR_MASS:.3f} "
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
    if name not in STARLING_VEHICLES:
        return {}
    w_max = float(load_starling_spec()["assumed"]["max_rotor_velocity_rad_s"])
    scaling = float(os.environ.get("SUPERFLY_STARLING_INPUT_SCALING",
                                   w_max - PEGASUS_ZERO_POSITION_ARMED))
    return {"input_offset": [0.0] * 4,
            "input_scaling": [scaling] * 4,
            "zero_position_armed": [PEGASUS_ZERO_POSITION_ARMED] * 4}


def vehicle_usd_and_curve(name: str):
    """(usd_file, thrust_curve, label) for --vehicle <name>."""
    if name == "iris":
        # Historical default: stock Iris USD + stock curve (Pegasus defaults).
        return ROBOTS["Iris"], QuadraticThrustCurve(), "Iris (Pegasus stock)"
    if name in STARLING_VEHICLES:
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
        if name == "starling2max":
            # The lab USD (Nucleus; the sim needs the API token in its
            # environment -- compare.runner provides it, see
            # superfly.common.nucleus). SUPERFLY_VEHICLE_USD points at another
            # copy (e.g. staged locally on OSMO, which cannot reach Nucleus).
            usd = vehicle_usd_path(name)
            label = (f"Starling 2 Max lab USD {usd} (m={spec['mass_kg']} kg, "
                     f"I={[round(v, 5) for v in spec['inertia_flu'].values()]}, kT={k_t:g}, "
                     f"w_max={w_max:g} rad/s, tau 55/85 ms)")
            return usd, curve, label
        # starling2max_iris: the sys-ID body on the Pegasus Iris frame (see
        # _iris_frame_overrides) -- the pre-2026-09-29 `starling2max`, built
        # this way while the Nucleus login on airstation03 was broken.
        label = (f"Starling 2 Max sys-ID on the Iris frame (m={spec['mass_kg']} kg, "
                 f"I={[round(v, 5) for v in spec['inertia_flu'].values()]}, arms "
                 f"+-{STARLING_ARM_X}/+-{STARLING_ARM_Y} m, kT={k_t:g}, "
                 f"w_max={w_max:g} rad/s, tau 55/85 ms)")
        return ROBOTS["Iris"], curve, label
    raise ValueError(f"unknown vehicle {name!r}")


def vehicle_usd_path(name: str) -> str:
    """The USD --vehicle <name> spawns from (Pegasus's Iris for the Iris frames)."""
    return vehicle_usd_url(name) or ROBOTS["Iris"]
