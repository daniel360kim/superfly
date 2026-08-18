"""Method registry for the comparison harness: how to launch each offboard
script, which interpreter/venv it needs, and where its policy artifact lives.

One dict entry per method. `ckpt_kind` drives checkpoint_ready() so nothing
special-cases method names:
    file       a single .pth/.pt file
    hydra_dir  a DiffAero Hydra run dir holding checkpoints/exported_actor.pt2
    tf_prefix  a TF2 checkpoint PREFIX (<prefix>.index alongside; no
               `checkpoint` pointer file) -- never a plain file

Interpreters resolve from each method submodule's own venv
(methods/<repo>/.venv); agile goes through scripts/agile_python.sh, an exec
shim that exports ACADOS_SOURCE_DIR/LD_LIBRARY_PATH before Python starts
(acados' generated .so needs them in the process env at interpreter launch)
and then execs the repo-root .venv. Checkpoint/interpreter are overridable
per-method on the harness CLI (--<method>-python / --<method>-checkpoint).
"""

import os
import shlex
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = _REPO / "scripts"
CHECKPOINTS_DIR = _REPO / "checkpoints"
METHODS_DIR = _REPO / "methods"

DEFAULT_MAX_SPEED = 3.0

# Agile cruise/MPC tuning validated 2026-07-08 (after fixing the net's de-yaw
# bug in the agile policy core): a faster control loop and heavier attitude
# tracking than the acados-port defaults keep the streamed attitude setpoint
# achievable and avoid the stale-state limit cycle (see agile-oscillation-fix
# / agile-autonomy-integration memory notes for the full history).
DEFAULT_AGILE_MAX_SPEED = 4.0
AGILE_MAX_TILT_DEG = 30.0
AGILE_CONTROL_HZ = 100.0
AGILE_MPC_Q_ATT = 200.0


def effective_agile_max_speed(args) -> float:
    """Agile cruise cap passed to agile_offboard --max-vel: --max-speed when the
    user changes it from the default, else the validated agile cruise speed."""
    if float(args.max_speed) != DEFAULT_MAX_SPEED:
        return float(args.max_speed)
    return DEFAULT_AGILE_MAX_SPEED


def method_registry():
    return {
        "depthnav": dict(
            policy="depthnav",
            offboard="depthnav_offboard.py",
            control_hz=50.0,
            goal_argc=3,
            python=METHODS_DIR / "depthnav" / ".venv" / "bin" / "python",
            checkpoint=CHECKPOINTS_DIR / "DepthNav" / "level1_4"
            / "level1_4_iteration_13500.pth",
            ckpt_kind="file",
            speed_args=lambda a: ["--target-speed", str(a.max_speed)],
        ),
        # Velocity-command DepthNav: same network family and same sim depth
        # camera as "depthnav", but the policy emits a velocity setpoint that
        # goes to PX4's velocity loop (SET_POSITION_TARGET_LOCAL_NED) rather
        # than a thrust vector to the attitude loop. Trained for 0.7-1.5 m/s,
        # so --max-speed should sit in that band. Unlike "depthnav", this
        # offboard writes the "end" phase marker on goal-reach, so
        # policy_reported_reached behaves like the other methods. Its
        # checkpoint has never existed yet -- scripts/train_depthnav.py is
        # what finally produces it (checkpoint_ready gates it out until then).
        "depthnav_vel": dict(
            policy="depthnav",               # same sim camera + UDP transport
            offboard="depthnav_vel_offboard.py",
            control_hz=50.0,
            goal_argc=3,
            python=METHODS_DIR / "depthnav" / ".venv" / "bin" / "python",
            checkpoint=CHECKPOINTS_DIR / "DepthNav" / "level1_vel"
            / "level1_vel.pth",
            ckpt_kind="file",
            speed_args=lambda a: [
                "--target-speed", str(a.max_speed),
                # keep PX4's saturation aligned with VelocityBoundedYaw's bounds
                "--max-vel-xy", str(max(2.5, a.max_speed * 1.7)),
                "--max-vel-z", str(max(1.5, a.max_speed)),
            ],
        ),
        "diffaero": dict(
            policy="diffaero",
            offboard="diffaero_offboard.py",
            control_hz=30.0,
            goal_argc=2,                     # diffaero --goal takes X Y only
            python=METHODS_DIR / "diffaero" / ".venv" / "bin" / "python",
            checkpoint=CHECKPOINTS_DIR / "DiffAero" / "sha2c_pmc",
            ckpt_kind="hydra_dir",
            speed_args=lambda a: ["--max-vel", str(a.max_speed)],
        ),
        "diffaero_vel": dict(
            policy="diffaero",               # same sim camera + UDP transport
            offboard="diffaero_vel_offboard.py",
            control_hz=30.0,
            goal_argc=2,
            python=METHODS_DIR / "diffaero" / ".venv" / "bin" / "python",
            # velocity-command actor (action_is_velocity); the _oa run consumes depth
            checkpoint=CHECKPOINTS_DIR / "DiffAero" / "sha2c_vel_cmd_oa",
            ckpt_kind="hydra_dir",
            speed_args=lambda a: ["--max-vel", str(a.max_speed)],
        ),
        # Planar velocity-command DiffAero for the Starling 2 Max low-speed
        # deployment: the actor outputs horizontal [vx, vy] only (dynamics
        # pmv_planar, trained for 0.8-1.5 m/s cruise -- run comparisons with
        # --max-speed in that band); the offboard's altitude PID holds
        # climb_alt. Checkpoint produced by scripts/train_diffaero.py
        # --config "env=oa algo=sha2c dynamics=pmv_planar sensor=camera
        # network=mlp env.max_target_vel=1.5 env.min_target_vel=0.8
        # env.max_time=60" (the pre-reorg planar runs are unrecoverable,
        # see ATTEMPTS.md 2026-08-17).
        "diffaero_vel_planar": dict(
            policy="diffaero",
            offboard="diffaero_vel_offboard.py",
            control_hz=30.0,
            goal_argc=2,
            python=METHODS_DIR / "diffaero" / ".venv" / "bin" / "python",
            # v2: r_drone 0.3 (real Starling footprint) + n_obstacles 40
            # for clearance margin; v1 (r_drone 0.2) flew 0.10-0.14 m clearances.
            checkpoint=CHECKPOINTS_DIR / "DiffAero" / "pmv_planar_starling_v2",
            ckpt_kind="hydra_dir",
            speed_args=lambda a: ["--max-vel", str(a.max_speed)],
        ),
        "agile": dict(
            policy="agile",
            offboard="agile_offboard.py",
            control_hz=30.0,
            goal_argc=2,                     # agile --goal takes X Y only
            python=SCRIPTS_DIR / "agile_python.sh",
            checkpoint=CHECKPOINTS_DIR / "AgileAutonomy" / "ckpt-50" / "ckpt-50",
            ckpt_kind="tf_prefix",
            speed_args=lambda a: [
                "--max-vel", str(effective_agile_max_speed(a)),
                "--max-tilt-deg", str(AGILE_MAX_TILT_DEG),
                "--att-lp", "1.0",
                "--control-hz", str(AGILE_CONTROL_HZ),
                "--q-att", str(AGILE_MPC_Q_ATT),
                # 2026-07-30 margin-tuning campaign hook: per-leg agile knobs
                # (e.g. "--keepout --obs-r 0.25") injected via the environment
                # so campaign scripts never edit this file. Recorded in each
                # trial's metrics.json "commands" like every other arg.
            ] + shlex.split(os.environ.get("AGILE_EXTRA_ARGS", "")),
        ),
    }


def checkpoint_ready(cfg, ckpt: Path) -> bool:
    """Does the method's deployable artifact exist? Dispatch on ckpt_kind."""
    kind = cfg["ckpt_kind"]
    if kind == "hydra_dir":
        return (ckpt / "checkpoints" / "exported_actor.pt2").exists()
    if kind == "tf_prefix":
        return Path(str(ckpt) + ".index").exists()
    return ckpt.exists()


def resolve_python(method, default: Path, override: str) -> str:
    """Pick the interpreter: explicit override > method venv (if present) >
    the interpreter running the harness (with a warning)."""
    if override:
        return override
    if Path(default).exists():
        return str(default)
    print(f"[{method}] venv {default} not found; falling back to {sys.executable}. "
          f"Override with --{method}-python.", file=sys.stderr)
    return sys.executable
