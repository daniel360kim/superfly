#!/usr/bin/env python
"""
Comparison harness: fly DiffPhysDrone / DiffAero / DiffAero-vel / DepthNav / Agile Autonomy
through the SAME Isaac-Sim + PX4-SITL scenarios and score them on the same
metrics (success rate, collision/clearance, time & speed).

Every run is driven by a scenarios JSON file: a list of trials, each with its
own environment, start/goal, optional procedural obstacle field, and optional
per-scenario timeouts (environments differ in size, so a flight budget that
fits one may not fit another). Every method flies every scenario.

Scenario entry schema (only "goal" is always required; "start" too unless a
procedural obstacle field supplies the spawn):
    {
      "name": "forest_a",                  // label; default scenario<i>
      "environment": "Box Room",           // named Pegasus scene, OR:
      "usd_environment": "omniverse://.../stage.usd",
      "env_scale": 0.01,                   // scale for usd_environment
      "obstacles": "none",                 // none | diffphys | diffaero
      "seed": 0,                           // procedural-field RNG seed
      "scale": 5.0,                        // procedural-field size
      "start": [x, y, z],                  // spawn; optional for procedural
                                           // fields (default: field p_init XY,
                                           // z=0.1 on the ground)
      "goal": [x, y, z],                   // world-frame goal
      "climb_alt": 2.0,                    // climb height [m] ABOVE the spawn
                                           // altitude (PX4 local frame)
      "timeout": 180,                      // POLICY-phase budget [s]
      "pre_policy_timeout": 120,           // arm/climb/yaw budget [s]
      "landing_timeout": 90                // post-policy landing budget [s]
    }

It does NOT retrain or rewrite any method: per (method, scenario) it launches
the merged run_px4_sim.py with --log-traj (one launcher, all policies)
plus the method's own *_offboard.py, waits for the offboard sentinel
(--auto-stop), then scores the logged trajectory with compare/metrics.py.

Because each method needs its own Python (DiffAero/DepthNav venvs, a torch env
for DiffPhysDrone) and run_px4_sim needs Isaac's Python, interpreters are
configurable (per-method + --sim-python) with project-venv defaults. PX4 SITL
is (re)launched automatically per trial unless --no-px4-manage.

Examples:
    # see exactly what would run, no Isaac needed:
    python compare/run_comparison.py scenarios.json --dry-run

    # real headless run with per-trial videos, then aggregate:
    python compare/run_comparison.py scenarios.json --headless --record-video --report
cd$i
    # just re-aggregate existing results (each run gets its own timestamped
    # folder under compare/results/, e.g. compare/results/20260702_153000):
    python compare/run_comparison.py --report-only --results-dir results/20260702_153000
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DEPLOY = _HERE.parent                       # starling-deployment/
_REPO = _DEPLOY.parent                       # superfly/
sys.path.insert(0, str(_DEPLOY))             # import obstacle_field, metrics
import metrics                               # noqa: E402  (compare/metrics.py)

# Sentinel each *_offboard.py writes on exit; run_px4_sim --auto-stop watches it.
# Must match OFFBOARD_DONE_FILE in run_px4_sim.py / diffdrone_offboard.py.
OFFBOARD_DONE_FILE = "/tmp/superfly_offboard_done"

# Phase sentinel each *_offboard.py appends to: "start <ts>" at the CLIMB/YAW ->
# POLICY handoff, "end <ts>" at the POLICY -> LANDING handoff (diffaero /
# diffaero_vel; the others fly POLICY until they exit or we time them out). Must match
# POLICY_PHASE_FILE in the offboard scripts. Lets --timeout budget the policy
# flight only, with separate caps for the pre-policy (arm/climb/yaw) and
# post-policy (landing) phases.
POLICY_PHASE_FILE = "/tmp/superfly_policy_phase"

# Fixed run parameters that used to be CLI flags but never needed changing.
VIDEO_FPS = 15.0          # --record-video MP4 frame rate
VIDEO_SCALE = 4           # min upscale for the depth-grid frames
PX4_MODEL = "none_iris"   # PX4 SITL vehicle model
PX4_BOOT_TIMEOUT = 60.0   # seconds to wait for PX4 to reach a ready state
SIM_GRACE = 30.0          # seconds for the sim to auto-stop after offboard ends

# Scene-mesh clearance for USD environments (see extract_scene_mesh.py):
# extracted once per (usd, env_scale, params) into MESH_CACHE_DIR and reused
# across trials and runs. Extraction boots a headless Kit, so the timeout is
# generous; failures only cost the clearance columns, never the trial.
MESH_CACHE_DIR = _HERE / "mesh_cache"
SCENE_MESH_SAMPLE_H = 0.05      # surface sample spacing [m]
SCENE_MESH_GROUND_DEG = 30.0    # ground-like face tilt threshold [deg]
SCENE_MESH_TIMEOUT = 1200.0     # seconds for one extraction (incl. Kit boot)
# Extraction is cropped to the flight corridor (USD city scenes are km-scale;
# sampling the whole stage at 5 cm is hopeless): the start/goal AABB plus this
# margin. Clearance to geometry beyond the margin is unmeasured, so it must
# comfortably exceed both any plausible path deviation and the largest
# clearance worth reporting.
SCENE_MESH_XY_MARGIN = 50.0     # horizontal margin [m] around start/goal
SCENE_MESH_Z_BELOW = 20.0       # crop floor: this far under min(start_z, goal_z)
SCENE_MESH_Z_ABOVE = 50.0       # crop ceiling: this far over max(start_z, goal_z)

DEFAULT_MAX_SPEED = 3.0
DEFAULT_AGILE_MAX_SPEED = 7.0   # upstream agile_autonomy test_time_velocity


def effective_agile_max_speed(args) -> float:
    """Agile cruise cap passed to agile_offboard --max-vel.

    --agile-max-speed wins when set. Otherwise --max-speed applies to agile too
    when the user changes it from the default; with no speed flags agile stays at
    the upstream 7 m/s default while other methods use DEFAULT_MAX_SPEED."""
    if args.agile_max_speed is not None:
        return float(args.agile_max_speed)
    if float(args.max_speed) != DEFAULT_MAX_SPEED:
        return float(args.max_speed)
    return DEFAULT_AGILE_MAX_SPEED


# --------------------------------------------------------------------------- #
# Per-method registry: how to launch each offboard script + where its policy
# artifact lives. Interpreter/checkpoint are overridable on the CLI.
# --------------------------------------------------------------------------- #
def method_registry():
    return {
        "diffphys": dict(
            policy="diffphys",
            offboard="diffdrone_offboard.py",
            control_hz=15.0,
            goal_argc=3,
            # DiffPhysDrone offboard only needs torch + DiffPhysDrone/model.py.
            python=_REPO / "DiffPhysDrone" / ".venv" / "bin" / "python",
            checkpoint=_REPO / "checkpoints" / "DiffPhysDrone" / "checkpoint0004.pth",
            speed_args=lambda a: ["--margin", str(a.drone_radius),
                                  "--max-speed", str(a.max_speed)],
        ),
        "diffaero": dict(
            policy="diffaero",
            offboard="diffaero_offboard.py",
            control_hz=30.0,
            goal_argc=2,                     # diffaero --goal takes X Y only
            python=_REPO / "diffaero" / ".venv" / "bin" / "python",
            # diffaero --checkpoint is a DIRECTORY holding checkpoints/exported_actor.pt2
            checkpoint=_REPO / "checkpoints" / "DiffAero" / "sha2c_pmc",
            speed_args=lambda a: ["--max-vel", str(a.max_speed)],
        ),
        "diffaero_vel": dict(
            policy="diffaero",               # same sim depth camera + UDP transport
            offboard="diffaero_vel_offboard.py",
            control_hz=30.0,
            goal_argc=2,
            python=_REPO / "diffaero" / ".venv" / "bin" / "python",
            # velocity-command actor (action_is_velocity); sha2c_vel_cmd_oa uses depth
            checkpoint=_REPO / "checkpoints" / "DiffAero" / "sha2c_vel_cmd_oa",
            speed_args=lambda a: ["--max-vel", str(a.max_speed)],
        ),
        "diffaero_vel_planar": dict(
            policy="diffaero",
            offboard="diffaero_vel_offboard.py",
            control_hz=30.0,
            goal_argc=2,
            python=_REPO / "diffaero" / ".venv" / "bin" / "python",
            checkpoint=_REPO / "checkpoints" / "DiffAero" / "planar_cnn_sr0.97",
            speed_args=lambda a: [
                "--max-vel", "1.5",
                "--max-vel-xy", "1.5",
                "--max-vel-z", "1.5",
            ],
        ),
        "depthnav": dict(
            policy="depthnav",
            offboard="depthnav_offboard.py",
            control_hz=50.0,
            goal_argc=3,
            python=_REPO / "depthnav" / ".venv" / "bin" / "python",
            checkpoint=_REPO / "checkpoints" / "DepthNav" / "level1_4_iteration_13500.pth",
            speed_args=lambda a: ["--target-speed", str(a.max_speed)],
        ),
        "agile": dict(
            policy="agile",
            offboard="agile_offboard.py",
            control_hz=30.0,
            goal_argc=2,                     # agile --goal takes X Y only (like diffaero)
            # Not a venv python but an exec shim: acados' generated .so needs
            # ACADOS_SOURCE_DIR/LD_LIBRARY_PATH in the process env BEFORE python
            # starts; the shim sets them, then execs starling-deployment/.venv.
            python=_DEPLOY / "agile_python.sh",
            # agile --checkpoint is a TF2 checkpoint PREFIX (ckpt-50.index/.data-*
            # alongside; there is no `checkpoint` pointer file) -- never a plain file.
            checkpoint=_REPO / "checkpoints" / "AgileAutonomy" / "ckpt-50",
            speed_args=lambda a: ["--max-vel", str(effective_agile_max_speed(a)),
                                  "--max-tilt-deg", str(a.agile_max_tilt_deg)],
        ),
    }


def checkpoint_ready(method, ckpt: Path) -> bool:
    """diffaero's checkpoint is a directory (needs exported_actor.pt2); agile's is
    a TF2 checkpoint PREFIX (needs <prefix>.index); the others are .pth/.pt files."""
    if method in ("diffaero", "diffaero_vel", "diffaero_vel_planar"):
        return (ckpt / "checkpoints" / "exported_actor.pt2").exists()
    if method == "agile":
        return Path(str(ckpt) + ".index").exists()
    return ckpt.exists()


def resolve_python(method, default: Path, override: str) -> str:
    """Pick the interpreter: explicit override > project venv (if present) >
    the interpreter running this harness (with a warning)."""
    if override:
        return override
    if Path(default).exists():
        return str(default)
    print(f"[{method}] venv {default} not found; falling back to {sys.executable}. "
          f"Override with --{method}-python.", file=sys.stderr)
    return sys.executable


# --------------------------------------------------------------------------- #
# PX4 SITL process management: switching --policy mid-session (same PX4
# process across trials) was observed to leave PX4 unable to produce a
# heartbeat for the next trial's offboard (Preflight Fail / stuck failsafe,
# and a MAVLink in-place reboot didn't reliably fix it either). Killing and
# relaunching PX4 before every trial sidesteps the whole class of
# stale-session bugs (and resets EKF home / arming latches / the simulated
# battery) at the cost of a boot wait per trial.
#
# PX4 is launched as the bare px4 binary (what `make px4_sitl <model>`
# ultimately execs: env PX4_SYS_AUTOSTART=<id> build/px4_sitl_default/bin/px4
# with cwd = .../rootfs -- see PX4's simulator_mavlink/CMakeLists.txt) rather
# than through make: no cmake rebuild check per trial, and a one-process kill
# tree instead of make->cmake->sh->px4. `make px4_sitl` remains as a fallback
# for a missing first build.
# --------------------------------------------------------------------------- #
_px4_proc = None    # the PX4 SITL subprocess this harness currently owns, if any
_px4_log_pos = 0    # px4_sitl.log size when the current PX4 was launched; only
                    # content past this offset belongs to the current boot

# --px4-model -> PX4_SYS_AUTOSTART id for the direct-binary launch (the ids the
# corresponding `make px4_sitl <model>` helper targets pass).
PX4_MODEL_AUTOSTART = {
    "none_iris": "10016",
}

# Params pushed into every fresh PX4 via its pxh stdin right after launch, so a
# trial can never be cut short by the simulated battery: PX4 SITL's rcS
# defaults COM_LOW_BAT_ACT to 2 (auto-LAND on low battery), and "Battery
# unhealthy" failsafes were observed force-landing mid-trial. Warn-only + a
# battery that never drains removes that failure mode entirely.
PX4_BOOT_PARAMS = [
    ("COM_LOW_BAT_ACT", "0"),    # low battery -> warning only, never auto-land
    ("SIM_BAT_MIN_PCT", "100"),  # simulated battery never drains below 100%
]


def _sanitized_env():
    """os.environ minus the Isaac Sim library injections. The harness often runs
    from a shell with Isaac's LD_LIBRARY_PATH exported, whose bundled libssl
    breaks any child that links system libcurl (observed: `cmake: ... libssl.so.3:
    version OPENSSL_3.2.0 not found` killing every `make px4_sitl` instantly).
    PX4 must never see those paths."""
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"):
        env.pop(var, None)
    return env


def stop_px4():
    """Kill the currently-managed PX4 SITL process (and its process group --
    `make px4_sitl <model>` forks through cmake/sh down to the actual px4
    binary, so the whole tree needs killing together), PLUS any other px4
    binary still alive system-wide. The latter matters because PX4 refuses to
    start a second instance ("PX4 server already running for instance 0",
    guarded by an flock on /tmp/px4_lock-0) if ANY px4 process holds that lock
    -- including a stale one from an earlier terminal/session that this
    harness never launched itself (this is exactly what caused every
    "Waiting for heartbeat" hang: a leftover process from an earlier manual
    restart kept silently blocking every subsequent `make px4_sitl`)."""
    global _px4_proc
    if _px4_proc is not None and _px4_proc.poll() is None:
        try:
            os.killpg(os.getpgid(_px4_proc.pid), signal.SIGTERM)
            _px4_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(_px4_proc.pid), signal.SIGKILL)
            _px4_proc.wait(timeout=5)
        except ProcessLookupError:
            pass
    _px4_proc = None

    try:
        out = subprocess.run(["pgrep", "-f", "bin/px4"], capture_output=True, text=True)
        pids = [int(p) for p in out.stdout.split()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(2.0)
            subprocess.run(["pkill", "-9", "-f", "bin/px4"])
    except FileNotFoundError:
        pass  # pgrep/pkill unavailable; best effort only


def start_px4(px4_dir: Path, model: str, log_path: Path):
    """Launch a fresh PX4 SITL in its own process group (so stop_px4 can kill
    it and any children together), appending its output to log_path.

    Prefers the bare px4 binary (build/px4_sitl_default/bin/px4, cwd=rootfs,
    PX4_SYS_AUTOSTART from PX4_MODEL_AUTOSTART); falls back to
    `make px4_sitl <model>` when the binary hasn't been built yet. stdin is a
    pipe: push_px4_boot_params writes `param set` lines into it, which PX4's
    pxh shell consumes once the startup script finishes."""
    global _px4_proc, _px4_log_pos
    _px4_log_pos = log_path.stat().st_size if log_path.exists() else 0
    log_f = open(log_path, "a")

    px4_bin = px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"
    rootfs = px4_dir / "build" / "px4_sitl_default" / "rootfs"
    autostart = PX4_MODEL_AUTOSTART.get(model)
    env = _sanitized_env()
    if px4_bin.exists() and rootfs.is_dir() and autostart is not None:
        env["PX4_SYS_AUTOSTART"] = autostart
        cmd, cwd = [str(px4_bin)], rootfs
        log_f.write(f"\n=== starting `{px4_bin}` (PX4_SYS_AUTOSTART={autostart}) "
                    f"in {rootfs} ===\n")
    else:
        cmd, cwd = ["make", "px4_sitl", model], px4_dir
        log_f.write(f"\n=== starting `make px4_sitl {model}` in {px4_dir} ===\n")
    log_f.flush()
    _px4_proc = subprocess.Popen(cmd, cwd=str(cwd), env=env,
                                 stdin=subprocess.PIPE,
                                 stdout=log_f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    return _px4_proc


def push_px4_boot_params():
    """Queue PX4_BOOT_PARAMS on the px4 process's stdin. The pxh shell only
    reads stdin after the startup script returns (i.e. after Isaac connects on
    TCP 4560), so these sit in the pipe buffer until then and are applied well
    before the offboard script arms."""
    if _px4_proc is None or _px4_proc.stdin is None:
        return
    try:
        for name, value in PX4_BOOT_PARAMS:
            _px4_proc.stdin.write(f"param set {name} {value}\n".encode())
        _px4_proc.stdin.flush()
    except (BrokenPipeError, OSError) as e:
        print(f"  [px4] warning: could not push boot params ({e}); "
              f"battery failsafe params NOT set for this trial.", file=sys.stderr)


def _new_log_text(log_path: Path):
    """Log content produced by the CURRENT PX4 launch only. The log is opened
    in append mode, so scanning the whole file would match boot markers from
    previous (possibly successful) launches and declare a dead PX4 'booted' --
    exactly how the cmake/OpenSSL launch failures went undetected."""
    if not log_path.exists():
        return ""
    with open(log_path, "rb") as f:
        f.seek(_px4_log_pos)
        return f.read().decode(errors="ignore")


def _die_with_log_tail(log_path: Path, msg: str, n_lines: int = 15):
    tail = "\n".join(_new_log_text(log_path).splitlines()[-n_lines:])
    raise SystemExit(f"{msg}\n--- last {n_lines} lines of {log_path} ---\n{tail}")


def wait_px4_booted(log_path: Path, timeout: float):
    """Poll the PX4 log (current launch's portion only) until PX4 is ready for
    Isaac to attach.

    PX4's own startup script (etc/init.d-posix/rcS) BLOCKS on connecting to the
    external simulator (Isaac/Pegasus, the TCP *server* on port 4560) before it
    gets far enough to start its mavlink module -- so a MAVLink heartbeat
    literally cannot arrive until *after* Isaac connects. Waiting for a
    heartbeat here (before Isaac is even launched) is a deadlock. Instead this
    just confirms PX4 reached "Waiting for simulator to accept connection" (or
    already got past it); the real heartbeat gets confirmed later, naturally,
    by the offboard script's own wait_for_heartbeat() after --warmup."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = _new_log_text(log_path)
        if "PX4 server already running" in text:
            _die_with_log_tail(log_path, "PX4 refused to start -- instance 0 lock still "
                               "held by a stale process (try `pkill -9 -f bin/px4`).")
        if "Waiting for simulator to accept connection" in text or "INFO  [mavlink]" in text:
            return
        if _px4_proc is not None and _px4_proc.poll() is not None:
            _die_with_log_tail(log_path, f"PX4 exited during boot "
                               f"(code {_px4_proc.returncode}).")
        if "make: ***" in text or "gmake: ***" in text:
            _die_with_log_tail(log_path, "PX4 build/launch failed under make.")
        time.sleep(0.5)
    _die_with_log_tail(log_path, f"PX4 didn't reach a ready state within {timeout:.0f}s.")


def restart_px4(px4_dir: Path, model: str, boot_timeout: float, log_path: Path):
    """Kill whatever PX4 SITL is currently running (this harness's own, or any
    stale leftover) and launch + wait for a fresh one, so every trial starts
    from a guaranteed-clean flight-controller state (EKF home, arming latches,
    simulated battery all reset)."""
    print("  [px4] restarting PX4 SITL for a clean state ...")
    stop_px4()
    time.sleep(1.0)  # let the old process's lockfile fully release
    start_px4(px4_dir, model, log_path)
    print(f"  [px4] waiting up to {boot_timeout:.0f}s for PX4 to boot ...")
    wait_px4_booted(log_path, boot_timeout)
    push_px4_boot_params()
    print("  [px4] PX4 SITL up (failsafe params queued), waiting for Isaac "
          "to connect on TCP 4560.")


def stop_stale_sims():
    """Kill any leftover run_px4_sim.py (a manual run or a crashed trial that
    outlived its harness). A stale sim keeps the TCP 4560 simulator port and
    the MAVLink 4560/tcpin bind, so the next trial's PX4MavlinkBackend dies
    with 'OSError: [Errno 98] Address already in use' and the trial hangs."""
    try:
        out = subprocess.run(["pgrep", "-f", "run_px4_sim.py"],
                             capture_output=True, text=True)
        pids = [int(p) for p in out.stdout.split()]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if pids:
            print(f"  [cleanup] killed {len(pids)} stale run_px4_sim.py process(es) "
                  "holding the simulator ports.")
            time.sleep(2.0)
            subprocess.run(["pkill", "-9", "-f", "run_px4_sim.py"])
    except FileNotFoundError:
        pass  # pgrep/pkill unavailable; best effort only


# --------------------------------------------------------------------------- #
# Scene-mesh clearance for USD environments. run_px4_sim only dumps the
# analytic obstacle field into traj.npz, so USD-scene trials have nothing for
# metrics.py to measure clearance against. extract_scene_mesh.py (run under
# Isaac's python -- the omniverse:// resolver needs Kit) samples the scene's
# non-ground surfaces into a cached .npz that metrics.py then scores against.
# --------------------------------------------------------------------------- #
def scene_mesh_bounds(start, goal):
    """Corridor crop box for extraction: start/goal AABB + fixed margins."""
    s, g = np.asarray(start, float), np.asarray(goal, float)
    lo, hi = np.minimum(s, g), np.maximum(s, g)
    return [lo[0] - SCENE_MESH_XY_MARGIN, lo[1] - SCENE_MESH_XY_MARGIN,
            lo[2] - SCENE_MESH_Z_BELOW,
            hi[0] + SCENE_MESH_XY_MARGIN, hi[1] + SCENE_MESH_XY_MARGIN,
            hi[2] + SCENE_MESH_Z_ABOVE]


def scene_mesh_path(usd, env_scale, bounds):
    """Cache file for one (usd, env_scale, corridor, extraction-params) combo."""
    bkey = ",".join(f"{b:.1f}" for b in bounds)
    key = hashlib.sha1(f"{usd}|{env_scale}|{bkey}|{SCENE_MESH_SAMPLE_H}|"
                       f"{SCENE_MESH_GROUND_DEG}|v1".encode()).hexdigest()[:10]
    stem = Path(str(usd)).stem.replace(".stage", "") or "scene"
    return MESH_CACHE_DIR / f"{stem}_{key}.npz"


def ensure_scene_mesh(usd, env_scale, start, goal, sim_python):
    """Return the cached scene-mesh npz for this scene + flight corridor,
    extracting it now (headless Kit subprocess) if missing. Returns None on
    failure -- the trial is still scored, just without clearance/collision."""
    if not usd:
        return None
    bounds = scene_mesh_bounds(start, goal)
    path = scene_mesh_path(usd, env_scale, bounds)
    if path.exists():
        return path
    MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sim_python or "python", str(_HERE / "extract_scene_mesh.py"), str(usd),
           "--env-scale", str(env_scale), "--out", str(path),
           "--sample-h", str(SCENE_MESH_SAMPLE_H),
           "--ground-deg", str(SCENE_MESH_GROUND_DEG),
           "--bounds", *(f"{b:.1f}" for b in bounds)]
    print(f"  [scene-mesh] extracting scene geometry (one-time per scene, boots "
          f"a headless Kit):\n      {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(_HERE), timeout=SCENE_MESH_TIMEOUT, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"  [scene-mesh] extraction failed ({e}); clearance/collision will "
              f"not be scored for this scene.", file=sys.stderr)
        path.unlink(missing_ok=True)   # no half-written cache entries
        return None
    return path if path.exists() else None


def rescore_clearance(results_dir, sim_python):
    """Retrofit scene-mesh clearance into an existing results dir: every trial
    whose metrics.json has a usd_environment scenario but no clearance gets
    rescored from its traj.npz (used via --rescore-clearance; the per-run
    scenario/hyperparams recorded in metrics.json drive the rescore, so no
    scenarios file is needed)."""
    failed_scenes = set()   # don't re-boot Kit for a scene that already failed
    for mfile in sorted(Path(results_dir).rglob("metrics.json")):
        r = json.loads(mfile.read_text())
        scen = r.get("scenario") or {}
        if not scen.get("usd_environment") or r.get("clearance_source") == "scene_mesh":
            continue
        npz = mfile.parent / "traj.npz"
        if not npz.exists():
            print(f"[rescore] {mfile}: no traj.npz, skipping.")
            continue
        usd = scen["usd_environment"]
        if usd in failed_scenes:
            continue
        if not scen.get("start") or not scen.get("goal"):
            print(f"[rescore] {mfile}: scenario lacks start/goal, skipping.")
            continue
        mesh = ensure_scene_mesh(usd, scen.get("env_scale", 1.0),
                                 scen["start"], scen["goal"], sim_python)
        if mesh is None:
            failed_scenes.add(usd)
            continue
        hp = r.get("hyperparams", {})
        rescored = metrics.score_trajectory(
            str(npz), drone_radius=hp.get("drone_radius", 0.2),
            goal_radius=hp.get("goal_radius", 1.0), scene_mesh=str(mesh))
        r.update(rescored)
        r["scene_mesh"] = str(mesh)
        r["success"] = (bool(r.get("reached")) or bool(r.get("policy_reported_reached"))) \
            and not r.get("collided", False)
        mfile.write_text(json.dumps(r, indent=2, default=str))
        clr = r.get("min_clearance_m")
        print(f"[rescore] {mfile.parent.parent.name}/{mfile.parent.name}: "
              f"min_clearance={'n/a' if clr is None else f'{clr:.2f} m'}, "
              f"collided={r.get('collided')}, success={r['success']}")


def load_scenarios(path):
    """Load the scenarios JSON (see the module docstring for the entry schema)
    and resolve each entry's start/goal.

    Procedural fields ("obstacles": diffphys|diffaero) are deterministic in
    (seed, scale); spawn defaults to the field's p_init (XY) on the ground
    (z=0.1) but can be overridden with "start". "goal" defaults to the field's
    target but can be overridden. Scene-only entries ("obstacles": none, the
    must give both "start" and "goal". Timeout keys are optional per-scenario
    overrides of the CLI --timeout/--pre-policy-timeout/--landing-timeout."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path}: expected a non-empty JSON list of scenarios.")
    scenarios = []
    for i, entry in enumerate(data):
        s = dict(
            name=entry.get("name", f"scenario{i}"),
            environment=entry.get("environment", "Box Room"),
            usd_environment=entry.get("usd_environment"),
            env_scale=float(entry.get("env_scale", 1.0)),
            obstacles=entry.get("obstacles", "none"),
            seed=int(entry.get("seed", 0)),
            scale=float(entry.get("scale", 5.0)),
            climb_alt=entry.get("climb_alt"),
            timeout=entry.get("timeout"),
            pre_policy_timeout=entry.get("pre_policy_timeout"),
            landing_timeout=entry.get("landing_timeout"),
        )
        if s["obstacles"] not in ("none", "diffphys", "diffaero"):
            raise SystemExit(f"scenario {s['name']}: obstacles must be "
                             f"none|diffphys|diffaero, got {s['obstacles']!r}")
        if s["obstacles"] == "none":
            if "start" not in entry or "goal" not in entry:
                raise SystemExit(f"scenario {s['name']}: 'start' and 'goal' are "
                                 f"required when obstacles is none: {entry}")
            s["start"] = np.asarray(entry["start"], float)
            s["goal"] = np.asarray(entry["goal"], float)
        else:
            import obstacle_field
            gen = (obstacle_field.generate_diffaero if s["obstacles"] == "diffaero"
                   else obstacle_field.generate)
            fld = gen(seed=s["seed"], scale=s["scale"])
            s["start"] = (np.asarray(entry["start"], float) if "start" in entry
                          else np.asarray(fld.p_init, float))
            s["goal"] = (np.asarray(entry["goal"], float) if "goal" in entry
                         else np.asarray(fld.p_target, float))
        scenarios.append(s)
    names = [s["name"] for s in scenarios]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate scenario names {names} -- results "
                         "files are keyed by name, so they must be unique.")
    return scenarios


def build_commands(method, cfg, args, scenario, npz_path, video_dir=None):
    """Construct the (sim_cmd, offboard_cmd) argv lists for one trial."""
    start, goal = scenario["start"], scenario["goal"]

    sim_python = args.sim_python or "python"
    sim_cmd = [sim_python, "run_px4_sim.py",
               "--policy", cfg["policy"],
               "--obstacles", scenario["obstacles"],
               "--seed", str(scenario["seed"]),
               "--scale", str(scenario["scale"]),
               "--auto-stop", "--no-debug-frames",
               "--log-traj", str(npz_path)]
    if args.headless:
        sim_cmd.append("--headless")
    if video_dir is not None:
        # depth.mp4 = policy-input depth grid (turbo colormap); rgb.mp4 =
        # onboard RGB drone_camera at the same viewpoint.
        sim_cmd += ["--record-depth-video", str(video_dir / "depth.mp4"),
                    "--record-rgb-video", str(video_dir / "rgb.mp4"),
                    "--record-video-fps", str(VIDEO_FPS),
                    "--record-video-scale", str(VIDEO_SCALE)]
    if scenario["usd_environment"]:
        sim_cmd += ["--usd-environment", scenario["usd_environment"],
                    "--env-scale", str(scenario["env_scale"])]
    else:
        sim_cmd += ["--environment", scenario["environment"]]
    if scenario["obstacles"] == "none":
        sim_cmd += ["--spawn", *(f"{v}" for v in start)]
    else:
        # Procedural fields: obstacle layout follows the field corridor; spawn
        # XY comes from the resolved start (field default or JSON override).
        sim_cmd += ["--spawn", f"{start[0]:.4f}", f"{start[1]:.4f}", "0.1000"]
    sim_cmd += ["--goal", *(f"{v:.4f}" for v in goal)]

    off_python = resolve_python(method, cfg["python"], getattr(args, f"{method}_python"))
    ckpt = getattr(args, f"{method}_checkpoint") or str(cfg["checkpoint"])
    # The offboard scripts fly --goal in PX4's EKF local frame, whose origin is
    # the spawn/arming point -- NOT the world frame the obstacle field, sim
    # --goal, and metrics use. Shift to spawn-relative here, or every method
    # flies to spawn+goal and gets scored against the world goal (observed:
    # diffaero landed 8.3 m from the world goal after "reaching" its own).
    # Z: for scene-only scenarios `start` is the literal --spawn (may be an
    # elevated rooftop), so goal z shifts too; procedural fields spawn on the
    # ground (~0, the field's z is flight-band metadata), so z stays absolute.
    rel_goal = np.asarray(goal, float).copy()
    rel_goal[0] -= float(start[0])
    rel_goal[1] -= float(start[1])
    if scenario["obstacles"] == "none":
        rel_goal[2] -= float(start[2])
    goal_vals = [f"{v:.4f}" for v in (rel_goal[:cfg["goal_argc"]])]
    # climb_alt is in PX4's local frame (metres ABOVE the spawn altitude --
    # the offboards fly on LOCAL_POSITION_NED, origin at the arming point),
    # so it needs no world-frame shift even for elevated spawns.
    climb_alt = scenario["climb_alt"] if scenario["climb_alt"] is not None else args.climb_alt
    off_cmd = [off_python, cfg["offboard"],
               "--checkpoint", str(ckpt),
               "--connect", args.connect,
               "--depth",
               "--goal", *goal_vals,
               "--climb-alt", str(climb_alt)]
    off_cmd += cfg["speed_args"](args)
    return sim_cmd, off_cmd


def run_trial(method, cfg, args, scenario):
    """Launch sim + offboard for one (method, scenario), wait, score.
    Returns the metrics dict (or None on dry-run).

    Per-trial outputs live under <results-dir>/<scenario>/<method>/:
    traj.npz (ground-truth trajectory + field), metrics.json (scores + the
    exact sim/offboard commands), and with --record-video depth.mp4/rgb.mp4."""
    results = Path(args.results_dir)
    label = scenario["name"]
    goal = scenario["goal"]
    trial_dir = results / label / method
    trial_dir.mkdir(parents=True, exist_ok=True)

    npz_path = trial_dir / "traj.npz"
    sim_cmd, off_cmd = build_commands(method, cfg, args, scenario, npz_path,
                                      video_dir=trial_dir if args.record_video else None)

    print(f"\n=== {method}  {label}  goal={np.round(goal,2).tolist()} ===")
    print("  sim:      " + " ".join(sim_cmd))
    print("  offboard: " + " ".join(off_cmd))
    if args.dry_run:
        return None

    # A stale Isaac sim from an earlier manual run keeps TCP 4560 bound and
    # breaks this trial's MAVLink backend (Errno 98) -- clear it first.
    stop_stale_sims()

    if not args.no_px4_manage:
        restart_px4(Path(args.px4_dir), PX4_MODEL, PX4_BOOT_TIMEOUT,
                   results / "px4_sitl.log")

    # Fresh sentinels so a stale file doesn't stop the sim immediately (or
    # start the policy-phase timer from a previous trial's handoff).
    Path(OFFBOARD_DONE_FILE).unlink(missing_ok=True)
    Path(POLICY_PHASE_FILE).unlink(missing_ok=True)
    sim = subprocess.Popen(sim_cmd, cwd=str(_DEPLOY))
    try:
        print(f"  [warmup] giving Isaac {args.warmup:.0f}s to boot before offboard ...")
        _sleep_or_die(sim, args.warmup, "sim exited during warmup")
        off = subprocess.Popen(off_cmd, cwd=str(_DEPLOY))
        wait_offboard_phased(off, args, scenario)
        # Ensure the sim's --auto-stop trips even if offboard was killed (its
        # own on-exit sentinel write only runs on a clean/Ctrl-C exit).
        Path(OFFBOARD_DONE_FILE).write_text(str(time.time()))
        try:
            sim.wait(timeout=SIM_GRACE)
        except subprocess.TimeoutExpired:
            print("  [cleanup] sim still running after grace; terminating.")
            sim.terminate()
    finally:
        for p in (sim,):
            if p.poll() is None:
                p.kill()

    if not npz_path.exists():
        print(f"  [warn] no trajectory logged at {npz_path}; trial failed to run.")
        res = dict(method=method, n_poses=0, success=False,
                   reached=False, collided=False, error="no trajectory logged")
    else:
        # USD-scene trials carry no analytic obstacle field, so clearance/
        # collision are scored against the extracted scene geometry instead
        # (cached per scene; extraction runs AFTER the flight so its Kit boot
        # never competes with the trial's own Isaac instance).
        scene_mesh = ensure_scene_mesh(scenario["usd_environment"],
                                       scenario["env_scale"],
                                       scenario["start"], scenario["goal"],
                                       args.sim_python)
        res = metrics.score_trajectory(str(npz_path), drone_radius=args.drone_radius,
                                       goal_radius=args.goal_radius,
                                       scene_mesh=str(scene_mesh) if scene_mesh else None)
        if scene_mesh is not None:
            res["scene_mesh"] = str(scene_mesh)
    # The offboard hands off to LANDING only once inside its own goal threshold,
    # so a logged "end" event is the policy reporting goal-reached (in its EKF
    # frame). Count it toward success alongside the ground-truth radius check
    # (diffphys/depthnav have no landing phase, so for them this stays False and
    # only the ground-truth check applies).
    _, t_policy_end = _read_policy_phase()
    res["policy_reported_reached"] = t_policy_end is not None
    res["success"] = (bool(res.get("reached")) or res["policy_reported_reached"]) \
        and not res.get("collided", False)
    res["label"] = label
    # Hyperparameters + the scenario that produced this trial, so each
    # per-trial result file is self-describing without cross-referencing
    # run_manifest.json.
    res["hyperparams"] = dict(
        max_speed=args.max_speed, drone_radius=args.drone_radius, goal_radius=args.goal_radius,
        climb_alt=(scenario["climb_alt"] if scenario["climb_alt"] is not None
                   else args.climb_alt),
        connect=args.connect)
    if method == "agile":
        res["hyperparams"]["agile_max_speed"] = effective_agile_max_speed(args)
    res["scenario"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in scenario.items()}
    res["commands"] = dict(sim=" ".join(sim_cmd), offboard=" ".join(off_cmd))
    (trial_dir / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    print("  result: " + json.dumps({k: res.get(k) for k in
          ("success", "collided", "min_clearance_m", "time_to_goal_s", "peak_speed_mps")}))
    return res


def _read_policy_phase():
    """Parse POLICY_PHASE_FILE into (t_start, t_end) wall-clock timestamps
    (None where the event hasn't happened yet)."""
    t_start = t_end = None
    try:
        for line in Path(POLICY_PHASE_FILE).read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "start":
                t_start = float(parts[1])
            elif len(parts) == 2 and parts[0] == "end":
                t_end = float(parts[1])
    except (FileNotFoundError, ValueError):
        pass
    return t_start, t_end


def _kill_offboard(off, reason):
    print(f"  [timeout] {reason}; terminating offboard.")
    off.terminate()
    try:
        off.wait(timeout=10)
    except subprocess.TimeoutExpired:
        off.kill()


def wait_offboard_phased(off, args, scenario):
    """Wait for the offboard process to exit, budgeting each flight phase
    separately via POLICY_PHASE_FILE: pre_policy_timeout covers heartbeat/
    arm/climb/yaw (before "start"), timeout covers ONLY the policy flight
    ("start" until "end" or exit), and landing_timeout covers landing after
    "end". So a slow PX4 boot or a long final descent can't eat the policy's
    flight budget. Each budget is the scenario's value if set, else the CLI
    default (environments differ in size, so budgets are per-scenario)."""
    timeout = scenario["timeout"] or args.timeout
    pre_policy = scenario["pre_policy_timeout"] or args.pre_policy_timeout
    landing = scenario["landing_timeout"] or args.landing_timeout
    launched = time.time()
    while off.poll() is None:
        t_start, t_end = _read_policy_phase()
        now = time.time()
        if t_start is None:
            if now - launched > pre_policy:
                _kill_offboard(off, "offboard never reached the policy handoff "
                               f"within {pre_policy:.0f}s")
                return
        elif t_end is None:
            if now - t_start > timeout:
                _kill_offboard(off, f"policy phase exceeded {timeout:.0f}s")
                return
        else:
            if now - t_end > landing:
                _kill_offboard(off, f"landing exceeded {landing:.0f}s")
                return
        time.sleep(0.5)


def _sleep_or_die(proc, secs, msg):
    """Sleep up to `secs`, but bail early if the subprocess dies first."""
    end = time.time() + secs
    while time.time() < end:
        if proc.poll() is not None:
            raise SystemExit(f"[error] {msg} (exit code {proc.returncode}).")
        time.sleep(0.5)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(results_dir):
    """Pool every per-trial metrics JSON (<scenario>/<method>/metrics.json;
    old flat <method>_<label>.json layouts still match) into a per-method
    summary. Non-trial JSONs (manifest, scenarios copy) lack a "method" key
    and are skipped."""
    files = sorted(Path(results_dir).rglob("*.json"))
    by_method = {}
    for f in files:
        r = json.loads(f.read_text())
        if not isinstance(r, dict) or "method" not in r:
            continue
        by_method.setdefault(r["method"], []).append(r)

    rows = []
    for method, runs in sorted(by_method.items()):
        n = len(runs)
        succ = [r for r in runs if r.get("success")]
        reached = [r for r in runs if r.get("reached")]
        coll = [r for r in runs if r.get("collided")]

        def _mean(key, subset):
            vals = [r[key] for r in subset if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        rows.append(dict(
            method=method, n=n,
            success_rate=len(succ) / n if n else float("nan"),
            collision_rate=len(coll) / n if n else float("nan"),
            mean_min_clearance_m=_mean("min_clearance_m", runs),
            mean_time_to_goal_s=_mean("time_to_goal_s", reached),
            mean_speed_mps=_mean("mean_speed_mps", runs),
            peak_speed_mps=_mean("peak_speed_mps", runs),
        ))
    return rows


def print_report(rows, results_dir):
    if not rows:
        print("No results found to aggregate.")
        return
    # (dict key, header label, width, decimals). decimals=None => string/int.
    cols = [("method", "method", 10, None), ("n", "n", 4, None),
            ("success_rate", "success", 9, 2), ("collision_rate", "collide", 9, 2),
            ("mean_min_clearance_m", "clear[m]", 10, 2),
            ("mean_time_to_goal_s", "t_goal[s]", 11, 2),
            ("mean_speed_mps", "v_avg", 8, 2), ("peak_speed_mps", "v_pk", 8, 2)]
    header = "".join(f"{label:>{w}}" for _, label, w, _ in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        line = ""
        for key, _, w, dec in cols:
            v = r[key]
            if dec is None:
                cell = f"{v}"
            elif v is None or (isinstance(v, float) and v != v):  # NaN
                cell = "nan"
            else:
                cell = f"{v:.{dec}f}"
            line += f"{cell:>{w}}"
        print(line)

    out = Path(results_dir) / "summary.csv"
    keys = ["method", "n", "success_rate", "collision_rate", "mean_min_clearance_m",
            "mean_time_to_goal_s", "mean_speed_mps", "peak_speed_mps"]
    def _cell(v):
        if isinstance(v, float):
            return "" if v != v else f"{v:.3f}"  # NaN -> empty cell
        return f"{v}"

    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(_cell(r[k]) for k in keys))
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="?", default=None,
                    help="Path to the scenarios JSON file driving the run (see the "
                         "module docstring for the entry schema). Every method flies "
                         "every scenario. Not needed with --report-only.")
    ap.add_argument("--methods", nargs="+",
                    default=["diffphys", "diffaero", "depthnav", "agile"],
                    choices=["diffphys", "diffaero", "diffaero_vel", "diffaero_vel_planar",
                             "depthnav", "agile"],
                    help="Methods to compare (default: all four; add diffaero_vel "
                         "or diffaero_vel_planar for velocity-command DiffAero).")
    ap.add_argument("--climb-alt", type=float, default=2.0,
                    help="Default climb height [m] ABOVE the spawn altitude before the "
                         "policy takes over (PX4 local frame, so it works unchanged on "
                         "elevated spawns). Override per scenario with \"climb_alt\".")
    ap.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                    help="Cruise speed for diffphys/diffaero/depthnav offboards, and "
                         "for agile when --agile-max-speed is not set (agile default "
                         "stays 7 m/s if this is left at 3 m/s).")
    ap.add_argument("--agile-max-speed", type=float, default=None,
                    help="Cruise speed for agile only (overrides --max-speed for "
                         "agile; default 7 m/s upstream test_time_velocity).")
    ap.add_argument("--agile-max-tilt-deg", type=float, default=30.0,
                    help="Attitude tilt clamp [deg] passed to agile_offboard.py.")
    ap.add_argument("--drone-radius", type=float, default=0.2,
                    help="Collision radius [m] for clearance scoring (also diffphys --margin).")
    ap.add_argument("--goal-radius", type=float, default=1.0,
                    help="Distance [m] to goal that counts as reached.")
    ap.add_argument("--connect", default="udp:localhost:14550")
    ap.add_argument("--headless", action="store_true",
                    help="Run Isaac Sim without the GUI viewport (faster; recommended "
                         "for batch comparison runs).")
    ap.add_argument("--record-video", action="store_true",
                    help="Save per-trial MP4s to <results-dir>/<scenario>/<method>/"
                         "depth.mp4 (turbo colormap of the depth grid fed to the "
                         "policy) and rgb.mp4 (onboard RGB drone_camera at the same "
                         "viewpoint). Headless-safe.")
    ap.add_argument("--sim-python", default=os.environ.get("ISAACSIM_PYTHON"),
                    help="Interpreter for run_px4_sim.py (Isaac's python). "
                         "Default $ISAAC_PYTHON or 'python'.")
    for m in ("diffphys", "diffaero", "diffaero_vel", "diffaero_vel_planar",
              "depthnav", "agile"):
        ap.add_argument(f"--{m}-python", default=None, help=f"Interpreter for {m} offboard.")
        ap.add_argument(f"--{m}-checkpoint", default=None, help=f"Checkpoint override for {m}.")
    ap.add_argument("--warmup", type=float, default=45.0,
                    help="Seconds to let Isaac boot before launching offboard.")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="Default max POLICY-phase flight time [s] per trial (measured "
                         "from the climb/yaw -> policy handoff, excluding takeoff and "
                         "landing). Override per scenario with \"timeout\".")
    ap.add_argument("--pre-policy-timeout", type=float, default=120.0,
                    help="Default max seconds to reach the policy handoff (heartbeat "
                         "wait + arm + climb + yaw). Override per scenario with "
                         "\"pre_policy_timeout\".")
    ap.add_argument("--landing-timeout", type=float, default=90.0,
                    help="Default max seconds for the post-policy landing phase (only "
                         "diffaero / diffaero_vel report a landing handoff). Override "
                         "per scenario "
                         "with \"landing_timeout\".")
    ap.add_argument("--px4-dir", default=os.environ.get("PX4_DIR", str(Path.home() / "PX4-Autopilot")),
                    help="PX4-Autopilot checkout to launch SITL from. "
                         "Default $PX4_DIR or ~/PX4-Autopilot.")
    ap.add_argument("--no-px4-manage", action="store_true",
                    help="Don't launch/restart PX4 SITL -- assume you're managing it "
                         "yourself in another terminal (the old workflow).")
    ap.add_argument("--results-dir", default=None,
                    help="Directory for outputs. Default: a new folder named "
                         "results/<YYYYMMDD_HHMMSS> is created for every run so "
                         "runs never overwrite each other. Pass explicitly to "
                         "reuse a dir (required with --report-only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands for each trial without launching anything.")
    ap.add_argument("--report", action="store_true",
                    help="Aggregate into a summary table + summary.csv after running.")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip running; just aggregate existing results in --results-dir.")
    ap.add_argument("--rescore-clearance", action="store_true",
                    help="With --report-only: before aggregating, retrofit scene-mesh "
                         "clearance/collision into USD-environment trials that were "
                         "scored before the scene-mesh metric existed (extracts the "
                         "scene geometry via --sim-python if not already cached).")
    args = ap.parse_args()

    if args.results_dir is None:
        if args.report_only:
            raise SystemExit("--report-only requires --results-dir pointing at an "
                             "existing run folder (no default -- there's no run to "
                             "auto-name).")
        import datetime
        args.results_dir = str(_HERE / "results" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    if args.report_only:
        if args.rescore_clearance:
            rescore_clearance(args.results_dir, args.sim_python)
        print_report(aggregate(args.results_dir), args.results_dir)
        return

    if args.scenarios is None:
        raise SystemExit("a scenarios JSON file is required (see the module "
                         "docstring for the entry schema).")
    scenarios = load_scenarios(args.scenarios)

    # Results layout: <results-dir>/command.txt (exact harness invocation),
    # scenarios.json (verbatim copy of the config), run_manifest.json (parsed
    # args + resolved scenarios), px4_sitl.log, summary.csv, and one
    # <scenario>/<method>/ folder per trial (traj.npz, metrics.json, videos).
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable] + sys.argv)
    (results_dir / "command.txt").write_text(command + "\n")
    shutil.copy2(args.scenarios, results_dir / "scenarios.json")
    manifest = dict(
        command=command,
        args={k: v for k, v in vars(args).items() if k != "scenarios"},
        scenarios_file=args.scenarios,
        scenarios=[{**s, "start": s["start"].tolist(), "goal": s["goal"].tolist()}
                   for s in scenarios],
    )
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if not args.dry_run and not args.no_px4_manage and not Path(args.px4_dir).is_dir():
        raise SystemExit(f"--px4-dir {args.px4_dir} not found -- pass --px4-dir or "
                         f"set $PX4_DIR, or use --no-px4-manage to manage PX4 yourself.")

    registry = method_registry()
    ran_any = False
    try:
        for method in args.methods:
            cfg = registry[method]
            ckpt = Path(getattr(args, f"{method}_checkpoint") or cfg["checkpoint"])
            if not args.dry_run and not checkpoint_ready(method, ckpt):
                print(f"\n[skip] {method}: checkpoint not found at {ckpt} "
                      f"-- skipping (provide one or --{method}-checkpoint).")
                continue
            for scenario in scenarios:
                run_trial(method, cfg, args, scenario)
                ran_any = True
    finally:
        if not args.no_px4_manage:
            stop_px4()

    if args.dry_run:
        print("\n[dry-run] no processes launched.")
    if args.report and ran_any:
        print_report(aggregate(args.results_dir), args.results_dir)


if __name__ == "__main__":
    main()
