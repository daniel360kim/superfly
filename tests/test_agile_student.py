"""agile_student integration: the ONNX student flown by the comparison harness.

Every test here pins this deployment to the *authoritative* reference, the
test-5 evaluation harness
(``superfly_expert_sampler.sim_episode``): the 22-dim state encoder, the
depth veto, the (M, N) decode, and the three committed ONNX runs. A silent
divergence in frames, units or mode selection between the two is the whole
risk of this integration, so it is asserted rather than documented.

CPU only, no acados, no Isaac. Run with the sampler venv (it has onnxruntime
and the reference module on the path):

    PYTHONPATH=src /home/ubuntu/anyanything/superfly_expert_sampler/.venv/bin/python \
        -m pytest tests/test_agile_student.py -q
"""
from pathlib import Path
import math
import sys

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from superfly.policies.agile.core import (            # noqa: E402
    AgilePolicy, depth_veto_blocked,
    STUDENT_VETO_LOOK_M, STUDENT_VETO_STEP_M, STUDENT_VETO_MARGIN_M,
    STUDENT_VETO_RADIUS_M,
)
from superfly.policies.agile.model import (           # noqa: E402
    OnnxStudentBackend, decode_student_output, is_onnx_checkpoint,
)

CHECKPOINTS = _REPO / "checkpoints" / "Student"
RUNS = {"t5_m2_r2": (2, 10), "t5_h25_local": (3, 5), "t5fix_s_r1": (3, 10)}

# The reference lives in a sibling sub-project, not in this repo.
_SAMPLER = Path.home() / "anyanything" / "superfly_expert_sampler_mppi" / "src"


def _reference():
    """sim_episode + render_depth from superfly_expert_sampler_mppi, or skip."""
    if str(_SAMPLER) not in sys.path:
        sys.path.insert(0, str(_SAMPLER))
    try:
        from superfly_expert_sampler import sim_episode, render_depth
    except Exception as exc:                            # pragma: no cover
        pytest.skip(f"reference sampler not importable: {exc}")
    return sim_episode, render_depth


def _random_state(rng):
    from scipy.spatial.transform import Rotation
    pos = rng.uniform(-20, 20, 3)
    # a real attitude, not a near-identity one: R must actually matter here
    R = Rotation.from_euler("xyz", [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                                    rng.uniform(-np.pi, np.pi)]).as_matrix()
    vel = rng.uniform(-4, 4, 3)
    omega_world = rng.uniform(-2, 2, 3)
    goal = rng.uniform(-40, 40, 3)
    return pos, R, vel, omega_world, goal


class _NoNet(AgilePolicy):
    """AgilePolicy's encoders without loading a net or building acados."""

    def __init__(self, goal_speed=0.0, modes=3, out_seq_len=10):
        from superfly.policies.agile.model import StudentModelConfig
        self.is_student = True
        self.config = StudentModelConfig(modes=modes, out_seq_len=out_seq_len)
        self.goal_speed = float(goal_speed)


# --------------------------------------------------------------------------- #
# 1. the 22-dim state encoder == sim_episode.OnnxPolicy.encode_state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_student_state_encoder_matches_reference(seed):
    sim_episode, _ = _reference()
    rng = np.random.default_rng(seed)
    pos, R, vel, omega_world, goal = _random_state(rng)
    goal_speed = float(rng.choice([0.0, 1.0, 2.0, 3.0]))

    # omega_body is what BOTH sides are handed: MAVLink gives a body rate, and
    # sim_episode.run_episode's Obs.omega is already a body rate (integrated
    # from dR = R.T @ R_new). The reference then applies R.T to it anyway, so
    # the same body rate goes in unrotated on both sides of this comparison --
    # no rotation is applied to the test input to make them agree.
    omega_body = omega_world
    obs = sim_episode.Obs(t=0.0, p=pos, v=vel, R=R, omega=omega_body,
                          goal_p=goal, goal_speed=goal_speed, depth=None)
    expected = sim_episode.OnnxPolicy.encode_state(obs)     # (1, 1, 22)

    got = _NoNet(goal_speed)._student_state_to_model_input(
        pos, R, vel, omega_body, goal)

    assert got.shape == (1, 1, 22) == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)


def test_student_state_encoder_clamps_goal_and_uses_raw_R():
    """The two deliberate departures from the legacy 21-dim encoding."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("z", 1.3).as_matrix()          # 74 deg of yaw
    pos = np.zeros(3)
    goal = np.array([100.0, 0.0, 0.0])                     # far away: clamps
    v = _NoNet(0.0)._student_state_to_model_input(
        pos, R, np.zeros(3), np.zeros(3), goal)[0, 0]
    np.testing.assert_allclose(v[3:12], R.reshape(-1), atol=1e-6)   # raw R, not de-yawed
    assert np.isclose(np.linalg.norm(v[18:21]), 10.0)              # metric, clamped
    assert v[21] == 0.0


# --------------------------------------------------------------------------- #
# 2. the depth veto == sim_episode.DepthVetoPolicy.blocked, on a real render
# --------------------------------------------------------------------------- #
def _wall_depth(render_depth, x_face=3.0):
    """A synthetic wall 3 m ahead, rendered with the students' own camera and
    resized exactly as the training loader does."""
    prims = [dict(kind="box", c=[x_face, 0.0, 1.75], half=[0.25, 6.0, 3.0])]
    frame = render_depth.render(prims, np.array([0.0, 0.0, 1.75]), np.eye(3))
    return render_depth.resize_bilinear(frame, 224)


def test_depth_veto_matches_reference_on_a_wall():
    sim_episode, render_depth = _reference()
    depth = _wall_depth(render_depth)
    assert depth.shape == (224, 224)
    assert depth.min() < 3.5                     # the wall really is in frame

    t = 0.5 * np.arange(1, 11)
    modes = np.stack([
        np.stack([3.0 * t, np.zeros_like(t), np.zeros_like(t)], 1),    # straight in
        np.stack([3.0 * t, 1.2 * t, np.zeros_like(t)], 1),             # veer left
        np.stack([3.0 * t, -2.5 * t, np.zeros_like(t)], 1),            # hard right
        np.stack([0.5 * t, np.zeros_like(t), np.zeros_like(t)], 1),    # creep, never reaches
    ])

    ref = sim_episode.DepthVetoPolicy.__new__(sim_episode.DepthVetoPolicy)
    ref.look_m, ref.step_m = STUDENT_VETO_LOOK_M, STUDENT_VETO_STEP_M
    ref.margin, ref.radius_m = STUDENT_VETO_MARGIN_M, STUDENT_VETO_RADIUS_M
    ref.last_scores = None
    expected = ref.blocked(depth.astype(np.float64), modes)

    got, scores = depth_veto_blocked(depth, modes)

    assert expected[0], "a path straight into a wall 3 m ahead must be vetoed"
    assert not expected.all(), "the synthetic case must leave a survivor"
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_allclose(scores, ref.last_scores, rtol=0, atol=1e-9)


def test_depth_veto_passes_everything_on_an_empty_frame():
    depth = np.full((224, 224), 20.0, np.float32)
    t = 0.5 * np.arange(1, 11)
    modes = np.stack([np.stack([3.0 * t, s * t, np.zeros_like(t)], 1)
                      for s in (0.0, 1.0, -1.0)])
    blocked, _ = depth_veto_blocked(depth, modes)
    assert not blocked.any()


# --------------------------------------------------------------------------- #
# 3. decode handles every shipped (M, N)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(1, 2, 31), (1, 3, 16), (1, 1, 31)])
def test_decode_handles_every_mode_waypoint_shape(shape):
    sim_episode, _ = _reference()
    rng = np.random.default_rng(shape[1] * 100 + shape[2])
    out = rng.normal(size=shape).astype(np.float32)

    alphas, flat = decode_student_output(out)
    modes, n = shape[1], (shape[2] - 1) // 3

    assert alphas.shape == (modes,)
    assert flat.shape == (modes, 3 * n)
    assert np.all(np.diff(alphas) >= -1e-7), "modes must come out sorted by |alpha|"

    # Same numbers the reference decoder produces, up to its mode ordering.
    ref_wps, ref_alpha = sim_episode.OnnxPolicy.decode_output(out)
    order = np.argsort(ref_alpha, kind="stable")
    np.testing.assert_allclose(alphas, ref_alpha[order], rtol=0, atol=1e-6)
    # flat is [x_1..n | y_1..n | z_1..n]; agile_core reshapes it to (3, n) and
    # transposes, which must land back on the reference's (n, 3) waypoints.
    got_wps = flat.reshape(modes, 3, n).transpose(0, 2, 1)
    np.testing.assert_allclose(got_wps, ref_wps[order], rtol=0, atol=1e-6)


def test_decode_accepts_a_two_tensor_graph():
    rng = np.random.default_rng(0)
    wps = rng.normal(size=(1, 3, 10, 3))
    costs = np.array([[0.9, 0.2, 0.5]])
    alphas, flat = decode_student_output([wps, costs])
    np.testing.assert_allclose(alphas, [0.2, 0.5, 0.9], atol=1e-6)
    np.testing.assert_allclose(flat.reshape(3, 3, 10).transpose(0, 2, 1),
                               wps[0][[1, 2, 0]], atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. the three real models load and produce finite waypoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("run", sorted(RUNS))
def test_committed_students_run_on_a_zero_depth_frame(run):
    pytest.importorskip("onnxruntime")
    path = CHECKPOINTS / run / "student.onnx"
    assert path.exists(), f"{path} is committed per checkpoints/README.md"
    assert is_onnx_checkpoint(path)

    backend = OnnxStudentBackend(path)
    modes, n = RUNS[run]
    assert (backend.modes, backend.out_seq_len) == (modes, n)
    assert backend.config.raw_state_dim == 22

    alphas, trajectories = backend.infer(
        np.zeros((1, 1, 224, 224, 3), np.float32), np.zeros((1, 1, 22), np.float32))
    assert alphas.shape == (modes,)
    assert trajectories.shape == (modes, 3 * n)
    assert np.isfinite(alphas).all() and np.isfinite(trajectories).all()
    assert np.all(np.diff(alphas) >= -1e-7)
    # absolute metres, not a normalised direction: a 5 s plan cannot be 0.01 m
    # long nor 500 m long.
    reach = np.abs(trajectories.reshape(modes, 3, n)).max()
    assert 0.05 < reach < 100.0, reach


def test_run_meta_present_for_every_committed_student():
    import json
    for run in RUNS:
        meta = json.loads((CHECKPOINTS / run / "run_meta.json").read_text())
        assert meta["schema"] == "superfly-run-meta-v1"
        assert meta["method"] == "agile_student"
        assert meta["artifact"] == "student.onnx"


# --------------------------------------------------------------------------- #
# 5. registry wiring
# --------------------------------------------------------------------------- #
def test_registry_entry_is_wired_and_ready():
    from superfly.compare.registry import method_registry, checkpoint_ready
    cfg = method_registry()["agile_student"]
    assert cfg["policy"] == "agile"            # same sim camera + depth transport
    assert cfg["goal_argc"] == 2
    assert cfg["control_hz"] == method_registry()["agile"]["control_hz"]
    assert cfg["ckpt_kind"] == "onnx"
    assert checkpoint_ready(cfg, cfg["checkpoint"])

    class _Args:
        max_speed = 3.0
    speed_args = cfg["speed_args"](_Args())
    assert "--mode-select" in speed_args
    assert speed_args[speed_args.index("--mode-select") + 1] == "veto"
    assert "--goal-speed" in speed_args
    assert "--max-vel" in speed_args


# --------------------------------------------------------------------------- #
# 6. the plan pipeline end to end, with the acados MPC stubbed out
# --------------------------------------------------------------------------- #
class _StubMPC:
    """Records what agile_core hands the MPC. acados is not installed on gs2."""
    def __init__(self):
        self.calls = []
        self._warmed = False

    def compute(self, x0, world_pts, cruise_alt, yaw_des, **kw):
        self.calls.append((np.asarray(world_pts).copy(), kw))
        raise RuntimeError("stub: force the PD fallback")


@pytest.fixture
def stub_mpc(monkeypatch):
    import superfly.policies.agile.core as core
    holder = {}

    def _factory():
        holder["mpc"] = _StubMPC()
        return holder["mpc"]

    monkeypatch.setattr(core, "MPC", _factory)
    return holder


@pytest.mark.parametrize("run", sorted(RUNS))
def test_plan_pipeline_shapes_and_mpc_time_base(run, stub_mpc):
    pytest.importorskip("onnxruntime")
    from scipy.spatial.transform import Rotation
    from superfly.policies.agile.core import (
        AgileObs, STUDENT_MPC_WAYPOINTS, STUDENT_WAYPOINT_DT)

    modes, n = RUNS[run]
    policy = AgilePolicy(str(CHECKPOINTS / run / "student.onnx"),
                         max_vel=3.0, control_hz=30.0, net_every=1,
                         goal_speed=0.0, mode_select="veto")
    assert policy.is_student
    assert policy.waypoint_dt == STUDENT_WAYPOINT_DT
    assert policy.mode_select == "veto"

    _, render_depth = _reference()
    depth = _wall_depth(render_depth)
    pos = np.array([0.0, 0.0, 1.75])
    R = Rotation.from_euler("z", 0.9).as_matrix()      # a yaw the legacy path de-yaws
    cmd = policy.compute(AgileObs(position_enu=pos, velocity_enu=np.zeros(3),
                                  R_enu=R, angular_rate_body=np.zeros(3),
                                  goal_enu=np.array([20.0, 5.0, 1.75]),
                                  depth=depth))

    # a plan per mode, each with the implicit t=0 waypoint prepended
    assert policy._world_points_per_mode.shape == (modes, n + 1, 3)
    np.testing.assert_allclose(policy._world_points_per_mode[:, 0],
                               np.tile(pos, (modes, 1)), atol=1e-9)
    assert 0 <= policy._mode_idx < modes
    assert cmd.veto_scores is None or cmd.veto_scores.shape == (modes,)

    # the MPC is fitted through t = 0, 0.5, 1.0, 1.5 s -- four points at the
    # student's true spacing, NOT the legacy 0.1 s one
    pts, kw = stub_mpc["mpc"].calls[0]
    assert kw["dt_wp"] == STUDENT_WAYPOINT_DT
    assert pts.shape == (min(STUDENT_MPC_WAYPOINTS, n + 1), 3)
    np.testing.assert_allclose(pts[0], pos, atol=1e-9)
    assert np.isfinite(cmd.attitude_ned_frd_wxyz).all()
    assert 0.0 < cmd.thrust_norm < 1.0


def test_student_plan_is_not_velocity_rescaled(stub_mpc):
    """_scale_body_plan must be identity for a student at any --max-vel."""
    pytest.importorskip("onnxruntime")
    ckpt = str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx")
    plan = np.arange(30, dtype=np.float64).reshape(3, 10)
    for max_vel in (1.0, 3.0, 7.0):
        policy = AgilePolicy(ckpt, max_vel=max_vel, control_hz=30.0)
        np.testing.assert_array_equal(policy._scale_body_plan(plan), plan)


def test_omega_channel_is_rotated_like_the_reference():
    """Guard against silently reverting to feeding omega_body raw: the two
    conventions differ (the whole point of finding 4), so the test must fail if
    the R.T is dropped."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [0.3, -0.25, -2.0]).as_matrix()
    omega_body = np.array([0.4, -0.2, 0.7])
    v = _NoNet(0.0)._student_state_to_model_input(
        np.zeros(3), R, np.zeros(3), omega_body, np.array([5.0, 0.0, 0.0]))[0, 0]
    np.testing.assert_allclose(v[15:18], R.T @ omega_body, atol=1e-6)
    assert not np.allclose(v[15:18], omega_body, atol=1e-3), \
        "R.T must actually change this vector, or the test proves nothing"


# --------------------------------------------------------------------------- #
# 7. the MPC reference IS sim_episode's fit_cubic / eval_cubic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("from_rest", [True, False])
def test_mpc_reference_equals_reference_cubic(from_rest):
    sim_episode, _ = _reference()
    from superfly.policies.agile.mpc import (
        fit_cubic_state, build_reference_cubic, DT, N)

    rng = np.random.default_rng(3)
    p = np.array([1.0, -2.0, 1.8])
    v = np.zeros(3) if from_rest else np.array([2.6, 0.4, 0.0])
    wps = p[None] + np.cumsum(rng.normal([1.4, 0.0, 0.0], 0.3, size=(10, 3)), axis=0)
    wps[:, 2] = np.clip(wps[:, 2], 0.6, 3.9)          # inside Z_REF: no clipping here

    sim_episode.set_waypoint_dt(0.5)
    ref_c = sim_episode.fit_cubic(p, v, wps)
    got_c = fit_cubic_state(p, v, wps, 0.5)
    np.testing.assert_allclose(got_c, ref_c, rtol=0, atol=1e-9)

    # and the reference the MPC is actually handed, sampled at its own nodes
    yref, yterm, _ = build_reference_cubic(got_c, yaw_des=0.0)
    for i in range(N + 1):
        p_ref, v_ref, _ = sim_episode.eval_cubic(ref_c, i * DT)
        row = yref[i] if i < N else yterm
        np.testing.assert_allclose(row[:3], p_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(row[7:10], v_ref, rtol=0, atol=1e-9)

    # the property that motivated the fix: from rest the reference commands zero
    if from_rest:
        np.testing.assert_allclose(yref[0][7:10], 0.0, atol=1e-12)


def _turn_cubic(turn_deg, v0=3.0, cruise=3.0, dt_wp=0.5):
    """A plan whose waypoints are a uniform `cruise` m/s straight line at
    `turn_deg` off the vehicle's heading, pinned to a vehicle doing v0 m/s
    along +x -- the geometry of every mid-flight avoidance decision."""
    from superfly.policies.agile.mpc import fit_cubic_state
    th = math.radians(turn_deg)
    d = np.array([math.cos(th), math.sin(th), 0.0])
    p = np.array([0.0, 0.0, 2.0])
    wps = np.array([p + cruise * dt_wp * (j + 1) * d for j in range(3)])
    return p, fit_cubic_state(p, np.array([v0, 0.0, 0.0]), wps, dt_wp)


@pytest.mark.parametrize("turn_deg", [0, 45, 90, 135, 180])
def test_reference_speed_is_capped_at_v_cap(turn_deg):
    """The 2026-09-16 overspeed: a cubic pinned to (p, v) and forced through
    waypoints 1-3 bulges in the MIDDLE of the MPC's 1.0 s horizon, well above
    the 3.0 m/s its own waypoints are spaced at. sim_episode never sees that
    (it reads the cubic only within one 1/15 s decision) and clamps the command
    at V_CAP anyway; this port reads it across the whole horizon, so the cap
    has to be on the reference."""
    from superfly.policies.agile.mpc import build_reference_cubic, N, DT

    _, c = _turn_cubic(turn_deg)
    raw, raw_t, _ = build_reference_cubic(c, 0.0)
    cap, cap_t, _ = build_reference_cubic(c, 0.0, max_vel=3.5)
    raw_sp = [float(np.linalg.norm((raw[i] if i < N else raw_t)[7:10]))
              for i in range(N + 1)]
    cap_sp = [float(np.linalg.norm((cap[i] if i < N else cap_t)[7:10]))
              for i in range(N + 1)]
    assert max(cap_sp) <= 3.5 + 1e-9, f"capped reference still at {max(cap_sp):.2f} m/s"
    if turn_deg >= 90:
        # the regression itself: uncapped, this reference demands >= 4.5 m/s
        assert max(raw_sp) > 4.4, max(raw_sp)
    # a straight plan the vehicle is already flying is left alone
    if turn_deg == 0:
        np.testing.assert_allclose(cap_sp, raw_sp, atol=1e-9)
    # position and velocity references still agree: no node advances further
    # than the capped speed allows (the slack is the trapezoidal integration
    # error of the cubic's own velocity over one 0.1 s node, < 1 cm)
    for i in range(N):
        step = np.linalg.norm((cap[i + 1] if i + 1 < N else cap_t)[:3] - cap[i][:3])
        assert step <= 3.5 * DT + 0.01, f"node {i} advances {step / DT:.2f} m/s"


def test_reference_cap_drops_along_track_acceleration():
    """clamp_command's rule: while the cap binds the reference must not ask for
    any more speed along its own direction of travel. The reference
    acceleration is recovered from the flatness pair it is encoded as,
    a = T * z_body - g."""
    from scipy.spatial.transform import Rotation
    from superfly.policies.agile.mpc import (build_reference_cubic, eval_cubic,
                                             G, N, DT)

    _, c = _turn_cubic(180)
    cap, _, _ = build_reference_cubic(c, 0.0, max_vel=3.5)
    capped_any = False
    for i in range(N):
        _, v_raw, _ = eval_cubic(c, i * DT)
        sp = float(np.linalg.norm(v_raw))
        if sp <= 3.5:
            continue
        capped_any = True
        q = cap[i][3:7]
        z_b = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 2]
        a = float(cap[i][10]) * z_b - np.array([0.0, 0.0, G])
        assert float(a @ (v_raw / sp)) <= 1e-6, \
            f"node {i}: reference still accelerates along track at {sp:.2f} m/s"
    assert capped_any, "this fixture must actually exercise the cap"


def _q_from_tilt(tilt_deg, heading_deg):
    """A commanded attitude tilted `tilt_deg` from vertical towards `heading_deg`."""
    from superfly.policies.agile.mpc import flatness_attitude, G
    th, hd = math.radians(tilt_deg), math.radians(heading_deg)
    sf = G * np.array([math.sin(th) * math.cos(hd), math.sin(th) * math.sin(hd),
                       math.cos(th)])
    q, _ = flatness_attitude(sf, hd)
    return q


def _tilt_dir(q):
    """(tilt deg, horizontal unit direction) of a commanded attitude's body-z."""
    from scipy.spatial.transform import Rotation
    z = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 2]
    h = float(np.hypot(z[0], z[1]))
    d = np.array([z[0], z[1]]) / h if h > 1e-9 else np.zeros(2)
    return math.degrees(math.atan2(h, z[2])), d


def test_command_clamp_is_inert_at_cruise():
    """It must not touch a student flying its labelled 3 m/s cruise: at 3.0 m/s
    against a 3.5 cap it still allows (3.5-3.0)/0.15 = 3.3 m/s^2 along track,
    more than these plans ask for."""
    from superfly.policies.agile.mpc import clamp_speed_command, G
    q = _q_from_tilt(15.0, 0.0)                  # ~2.6 m/s^2 of forward push
    for v in ([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 2.0, 0.0],
              [0.0, 0.0, -5.0], [2.0, 0.0, -1.0]):
        out = clamp_speed_command(q, np.array(v), 3.5, 0.0, thrust_cmd=G)
        np.testing.assert_allclose(out, q, atol=1e-12), v


def test_command_clamp_counts_vertical_speed_too():
    """V_CAP is a 3-D cap upstream, and these runs climb at up to 2.3 m/s: a
    horizontal-only trigger would let that ride for free."""
    from superfly.policies.agile.mpc import clamp_speed_command, G
    q = _q_from_tilt(25.0, 0.0)
    flat = clamp_speed_command(q, np.array([3.2, 0.0, 0.0]), 3.5, 0.0, thrust_cmd=G)
    climb = clamp_speed_command(q, np.array([3.2, 0.0, 2.0]), 3.5, 0.0, thrust_cmd=G)
    t_flat, _ = _tilt_dir(flat)
    t_climb, _ = _tilt_dir(climb)
    assert t_climb < t_flat, "the climb must eat into the same budget"


def test_command_clamp_removes_only_the_along_track_push():
    """Above the cap: no forward acceleration (it brakes), but turning and
    braking commands survive untouched."""
    from superfly.policies.agile.mpc import clamp_speed_command, G
    v = np.array([5.0, 0.0, 0.0])                       # over the cap

    # a forward tilt is turned into a braking tilt, never left pushing
    t, d = _tilt_dir(clamp_speed_command(_q_from_tilt(25.0, 0.0), v, 3.5, 0.0,
                                         thrust_cmd=G))
    assert float(d @ np.array([1.0, 0.0])) < 0.0, "must brake above the cap"

    # a pure left turn keeps its cross-track component (and gains a brake)
    q = _q_from_tilt(25.0, 90.0)
    t_out, d_out = _tilt_dir(clamp_speed_command(q, v, 3.5, math.pi / 2,
                                                 thrust_cmd=G))
    assert float(d_out @ np.array([0.0, 1.0])) > 0.2, "the turn must survive"
    assert float(d_out @ np.array([1.0, 0.0])) < 0.0, "and it brakes too"

    # a braking command is never weakened
    q = _q_from_tilt(25.0, 180.0)
    t_in, _ = _tilt_dir(q)
    t_out, d_out = _tilt_dir(clamp_speed_command(q, v, 3.5, math.pi, thrust_cmd=G))
    assert t_out >= t_in - 1e-9
    assert float(d_out @ np.array([1.0, 0.0])) < 0.0


def test_command_clamp_brake_is_still_tilt_limited():
    """The brake is an attitude PX4 has to fly: a big excess must not command a
    90 deg flip. agile_core re-applies the tracker's own tilt limit after it."""
    from superfly.policies.agile.mpc import clamp_speed_command, clamp_attitude_tilt, G
    q = clamp_speed_command(_q_from_tilt(25.0, 0.0), np.array([9.0, 0.0, 0.0]),
                            3.5, 0.0, thrust_cmd=G)
    assert _tilt_dir(q)[0] > 30.0, "this fixture must exercise the limiter"
    q = clamp_attitude_tilt(q, 30.0, 0.0)
    assert _tilt_dir(q)[0] == pytest.approx(30.0, abs=1e-6)
    assert float(_tilt_dir(q)[1] @ np.array([1.0, 0.0])) < 0.0, "still braking"


def test_tilt_is_rescaled_to_the_mpc_thrust():
    """The measured climb: thrust 0.748 against a 0.577 hover is 1.30x, so the
    same tilt buys 1.30x the horizontal acceleration the MPC solved for."""
    from superfly.policies.agile.mpc import scale_tilt_to_thrust, G
    from scipy.spatial.transform import Rotation
    q = _q_from_tilt(30.0, 0.0)
    f_cmd = 1.30 * G
    out = scale_tilt_to_thrust(q, G, f_cmd, 0.0)
    z_in = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 2]
    z_out = Rotation.from_quat([out[1], out[2], out[3], out[0]]).as_matrix()[:, 2]
    # the PRODUCT thrust * sin(tilt) is what the MPC planned, to 1 %
    assert f_cmd * float(np.hypot(z_out[0], z_out[1])) == pytest.approx(
        G * float(np.hypot(z_in[0], z_in[1])), rel=0.01)
    # never amplifies: an UNDER-thrusting loop is left alone
    np.testing.assert_allclose(scale_tilt_to_thrust(q, G, 0.8 * G, 0.0), q,
                               atol=1e-12)


@pytest.mark.parametrize("lag_s,thrust_ratio", [(0.05, 1.0), (0.15, 1.0),
                                               (0.30, 1.0), (0.30, 1.30)])
def test_command_clamp_bounds_the_vehicle_through_the_attitude_lag(lag_s, thrust_ratio):
    """The 2026-09-17 failure mode, reproduced and fixed. A tracker that always
    wants full 30 deg of forward tilt, a plant that follows the attitude
    setpoint with a first-order lag, and (in the last case) the measured 1.30x
    climb thrust. A clamp that merely DELETES the along-track push at the cap
    settles above it -- 3.78 / 4.46 / 5.34 m/s at these lags -- because the
    attitude the vehicle holds still carries the push it was given a moment
    ago. Limiting the along-track acceleration to (cap - |v|)/lag brakes
    instead, and bounds it."""
    from scipy.spatial.transform import Rotation
    from superfly.policies.agile.mpc import (clamp_speed_command,
                                             scale_tilt_to_thrust, G)
    dt, v = 0.01, np.zeros(3)
    z_act = np.array([0.0, 0.0, 1.0])
    peak = 0.0
    for _ in range(2500):                                    # 25 s
        f = thrust_ratio * G
        q = scale_tilt_to_thrust(_q_from_tilt(30.0, 0.0), G, f, 0.0)
        q = clamp_speed_command(q, v, 3.5, 0.0, thrust_cmd=f, lag_s=0.15)
        z_cmd = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 2]
        z_act = z_act + (z_cmd - z_act) * (dt / lag_s)       # PX4 tracking lag
        z_act = z_act / np.linalg.norm(z_act)
        a = f * z_act - np.array([0.0, 0.0, G])
        a[2] = 0.0                                           # altitude loop's job
        v = v + a * dt
        peak = max(peak, float(np.linalg.norm(v)))
    assert peak <= 4.0, f"peaked at {peak:.2f} m/s"
    assert float(np.linalg.norm(v)) == pytest.approx(3.5, abs=0.05)


def test_alt_setpoint_cannot_ratchet_away_from_the_vehicle(monkeypatch):
    """(b) The measured ratchet: the plan's vertical intent is RELATIVE, so a
    net that keeps asking to be 0.6 m higher raises an integrated setpoint for
    ever. Setpoint 1.6 -> 4.0 m in three seconds against a vehicle that had not
    moved. The lead cap bounds it; the integrator no longer winds up on it."""
    pytest.importorskip("onnxruntime")
    import superfly.policies.agile.core as core
    from superfly.policies.agile.core import STUDENT_ALT_LEAD
    monkeypatch.setattr(core, "MPC", _RefMPC)
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=100.0, net_every=1,
                         alt_follow=True, mode_select="cost",
                         hover_thrust=HOVER_ASSUMED)
    policy.net = _LevelNet(2, 10, sink=+0.6)     # "be 0.6 m higher every plan"
    policy.config = policy.net.config
    depth = np.full((224, 224), 20.0, np.float32)
    p = np.array([0.0, 0.0, 1.70])               # a vehicle that never climbs
    v = np.array([3.0, 0.0, 0.0])
    for _ in range(300):                         # 3 s, the measured window
        cmd = policy.compute(core.AgileObs(p, v, np.eye(3), np.zeros(3),
                                           np.array([0.0, 60.0, 1.96]), depth))
    assert policy._alt_sp <= 1.70 + STUDENT_ALT_LEAD + 1e-6, (
        f"setpoint ratcheted to {policy._alt_sp:.2f} m above a vehicle at 1.70")
    assert policy._alt_sp > 1.70, "but it must still ask for the climb"
    assert abs(policy._alt_i) < policy.alt_i_limit - 1e-6, \
        "the integrator must not wind up on a ramp the vehicle cannot follow"


def test_command_clamp_keeps_headroom_for_the_state_signal(stub_mpc):
    """The clamp can only bound the speed it is shown, and the offboard's own
    speed under-reads its excursions on the diffaero field. The COMMAND cap
    therefore sits below V_CAP -- while the REFERENCE cap does not, so the plan
    the student flies is unchanged."""
    pytest.importorskip("onnxruntime")
    from superfly.policies.agile.core import STUDENT_CMD_CAP_MARGIN
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=100.0, net_every=1,
                         mode_select="cost")
    assert policy.max_vel == 3.5, "the reference cap is V_CAP"
    assert policy.cmd_cap == pytest.approx(3.5 - STUDENT_CMD_CAP_MARGIN)
    assert 0.0 < STUDENT_CMD_CAP_MARGIN <= 0.5, "headroom, not a speed limit"


def test_cap_margin_is_overridable(monkeypatch, stub_mpc):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("AGILE_STUDENT_CAP_MARGIN", "0")
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=100.0, net_every=1,
                         mode_select="cost")
    assert policy.cmd_cap == pytest.approx(3.5)


def test_state_log_row_is_parseable_and_aligned(tmp_path):
    """The instrumentation that this audit needed and did not have: a per-tick
    row with an ABSOLUTE timestamp, so the control state lines up with
    traj.npz's t_unix0 without guessing an anchor from 1 Hz stdout."""
    from superfly.common.state_log import StateLog
    path = tmp_path / "state.csv"
    log = StateLog(path)
    log.write(1789674458.69, 10.07, "POLICY", "mpc", [1.0, 2.0, 3.0],
              [3.0, 0.0, 0.5], 12.3, 11.0, 0.577, 1.8, 14.2, 0)
    log.write(1789674458.70, 10.08, "POLICY", "pd", [1.0, 2.0, 3.0],
              [0.0, 0.0, 0.0], None, 1.0, 0.5, None, 0.0, None)
    log.close()
    lines = path.read_text().strip().split("\n")
    assert lines[0].split(",") == list(StateLog.COLUMNS)
    assert all(len(r.split(",")) == len(StateLog.COLUMNS) for r in lines[1:])
    cols = {k: i for i, k in enumerate(StateLog.COLUMNS)}
    first = lines[1].split(",")
    assert float(first[cols["t_unix"]]) == pytest.approx(1789674458.69, abs=1e-3)
    assert float(first[cols["speed"]]) == pytest.approx(np.hypot(3.0, 0.5), abs=1e-3)
    # missing values stay empty rather than becoming a lying 0.0
    assert lines[2].split(",")[cols["tilt_cmd_deg"]] == ""
    assert lines[2].split(",")[cols["alt_sp"]] == ""


def test_state_log_is_off_without_the_env_var(monkeypatch, tmp_path):
    from superfly.common.state_log import StateLog
    monkeypatch.delenv("SUPERFLY_STATE_LOG", raising=False)
    assert StateLog.from_env() is None
    monkeypatch.setenv("SUPERFLY_STATE_LOG", str(tmp_path / "s.csv"))
    log = StateLog.from_env()
    assert log is not None
    log.close()
    # an unwritable path must not kill the flight
    monkeypatch.setenv("SUPERFLY_STATE_LOG", str(tmp_path / "nope" / "s.csv"))
    assert StateLog.from_env() is None


def test_runner_points_the_state_log_at_the_trial_dir():
    import inspect
    from superfly.compare import runner
    src = inspect.getsource(runner.run_trial)
    assert 'SUPERFLY_STATE_LOG=str(trial_dir / "state.csv")' in src
    assert "env=off_env" in src and "env=dict(off_env" in src


def test_plan_age_offsets_the_reference():
    from superfly.policies.agile.mpc import build_reference_cubic, DT
    c = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    a, _, _ = build_reference_cubic(c, 0.0, t_offset=0.0)
    b, _, _ = build_reference_cubic(c, 0.0, t_offset=2 * DT)
    np.testing.assert_allclose(b[0][:3], a[2][:3], atol=1e-9)


# --------------------------------------------------------------------------- #
# 8. the student flies its own vertical plan
# --------------------------------------------------------------------------- #
def test_world_plan_z_is_clipped_to_the_reference_band(stub_mpc):
    pytest.importorskip("onnxruntime")
    from superfly.policies.agile.core import AgileObs, STUDENT_Z_REF
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=30.0, net_every=1,
                         alt_follow=True, mode_select="veto")
    # a plan that would leave the band in both directions
    traj = np.zeros((3, 30), np.float32)
    traj[:, 0:10] = np.arange(1, 11) * 1.5              # x
    traj[:, 20:30] = np.linspace(-6.0, 6.0, 10)         # z, way outside Z_REF
    policy._adopt_plan(np.array([0.1, 0.2, 0.3], np.float32), traj,
                       np.array([0.0, 0.0, 2.0]), np.eye(3), None, np.zeros(3))
    z = policy._world_points_per_mode[:, 1:, 2]
    assert z.min() >= STUDENT_Z_REF[0] - 1e-9
    assert z.max() <= STUDENT_Z_REF[1] + 1e-9
    assert policy._cubic is not None


def test_registry_forces_alt_follow_and_net_thread():
    from superfly.compare.registry import method_registry, STUDENT_MAX_VEL

    class _Args:
        max_speed = 3.0
    args = method_registry()["agile_student"]["speed_args"](_Args())
    assert "--alt-follow" in args, "the student must fly its own vertical plan"
    assert "--net-thread" in args, "inference must not block the control loop"
    assert float(args[args.index("--max-vel") + 1]) == STUDENT_MAX_VEL == 3.5


def test_non_finite_plan_keeps_the_previous_one(stub_mpc):
    pytest.importorskip("onnxruntime")
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=30.0, alt_follow=True)
    good = np.tile(np.arange(1, 11, dtype=np.float32), (3, 3))[:3, :30].copy()
    policy._adopt_plan(np.float32([0.1, 0.2, 0.3]), good,
                       np.zeros(3), np.eye(3), None, np.zeros(3))
    kept = policy._world_points.copy()
    bad = good.copy(); bad[1, 4] = np.nan
    policy._adopt_plan(np.float32([0.1, 0.2, 0.3]), bad,
                       np.ones(3), np.eye(3), None, np.zeros(3))
    np.testing.assert_array_equal(policy._world_points, kept)
    assert np.isfinite(policy._world_points).all()


def test_onnx_session_threads_are_pinned():
    pytest.importorskip("onnxruntime")
    b = OnnxStudentBackend(CHECKPOINTS / "t5fix_s_r1" / "student.onnx")
    assert b.intra_op_threads == OnnxStudentBackend.DEFAULT_INTRA_OP_THREADS == 8


# --------------------------------------------------------------------------- #
# 9. re-review regressions R1/R2/R3
# --------------------------------------------------------------------------- #
def test_z_band_always_contains_the_handover_altitude():
    """R1: 12 shipped scenarios climb to 5 m and 4 to 3 m. A band that stops at
    4.0 m would pull a level plan below the vehicle and command a dive."""
    from superfly.policies.agile.core import (
        student_z_band, STUDENT_Z_REF, STUDENT_Z_HEADROOM)
    assert student_z_band(None) == STUDENT_Z_REF
    assert student_z_band(2.0) == STUDENT_Z_REF          # inside: untouched
    for alt in (3.0, 5.0):
        lo, hi = student_z_band(alt)
        assert lo == STUDENT_Z_REF[0]
        assert hi >= alt + STUDENT_Z_HEADROOM
        assert lo < alt < hi, "the band must contain the vehicle"


def test_level_plan_at_high_climb_alt_is_not_pulled_into_a_dive(stub_mpc):
    pytest.importorskip("onnxruntime")
    from superfly.policies.agile.core import AgileObs
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=30.0, alt_follow=True)
    policy._cruise_alt = 5.0                              # a climb_alt-5 scenario
    traj = np.zeros((3, 30), np.float32)
    traj[:, 0:10] = np.arange(1, 11) * 1.5                # level, straight ahead
    pos = np.array([0.0, 0.0, 5.0])
    policy._adopt_plan(np.float32([0.1, 0.2, 0.3]), traj, pos, np.eye(3), None,
                       np.zeros(3))
    z = policy._world_points_per_mode[..., 2]
    np.testing.assert_allclose(z, 5.0, atol=1e-9)         # nothing clipped
    assert abs(float(policy._cubic[1][2])) < 1e-9         # and no vz demanded


def test_registry_pins_the_student_handover_altitude():
    from superfly.compare.registry import (
        method_registry, DEFAULT_STUDENT_CLIMB_ALT, STUDENT_Z_REF_NOTE)
    cfg = method_registry()["agile_student"]
    assert cfg["climb_alt"] == DEFAULT_STUDENT_CLIMB_ALT == 2.0
    assert STUDENT_Z_REF_NOTE            # the choice is documented in the module
    for m, c in method_registry().items():
        if m != "agile_student":
            assert c.get("climb_alt") is None, f"{m} must keep the scenario's"


def test_plan_clock_starts_at_the_snapshot_not_at_adoption(stub_mpc):
    """R2: in threaded mode adoption is one forward pass (~140 ms) after the
    state the cubic is pinned to; at 3 m/s that is 0.4 m of built-in lag."""
    pytest.importorskip("onnxruntime")
    import time
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=30.0, alt_follow=True)
    traj = np.zeros((3, 30), np.float32)
    traj[:, 0:10] = np.arange(1, 11) * 1.5
    t_submit = time.time() - 0.14                          # a 140 ms old snapshot
    policy._adopt_plan(np.float32([0.1, 0.2, 0.3]), traj, np.zeros(3), np.eye(3),
                       None, np.zeros(3), t_submit)
    assert policy._plan_time == pytest.approx(t_submit, abs=1e-9)
    assert time.time() - policy._plan_time >= 0.13


def test_reference_vz_is_clipped_with_z():
    """R3: at the band edge the position and velocity references must agree."""
    from superfly.policies.agile.mpc import build_reference_cubic, N
    # a cubic climbing hard through the ceiling
    c = np.array([[0.0, 0.0, 3.9], [3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    yref, yterm, _ = build_reference_cubic(c, 0.0, min_alt=0.5, max_alt=4.0)
    for i in range(N + 1):
        row = yref[i] if i < N else yterm
        pz, vz = float(row[2]), float(row[9])
        assert pz <= 4.0 + 1e-9
        if pz >= 4.0 - 1e-9:
            assert vz <= 1e-9, "no upward vz demanded at the ceiling"
    # and the mirror case at the floor
    c = np.array([[0.0, 0.0, 0.6], [3.0, 0.0, -4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    yref, yterm, _ = build_reference_cubic(c, 0.0, min_alt=0.5, max_alt=4.0)
    for i in range(N + 1):
        row = yref[i] if i < N else yterm
        if float(row[2]) <= 0.5 + 1e-9:
            assert float(row[9]) >= -1e-9


def test_backend_reports_a_measured_forward_time():
    pytest.importorskip("onnxruntime")
    b = OnnxStudentBackend(CHECKPOINTS / "t5fix_s_r1" / "student.onnx")
    assert b.forward_ms > 0.0


# --------------------------------------------------------------------------- #
# 10. the Isaac-only steady descent (reviewer's descent.py plant)
# --------------------------------------------------------------------------- #
# The evaluation simulator applies f = a_cmd + g exactly, so it has no
# throttle-mapping error. PX4 does: a_z = G*(thrust*cos_tilt/hover_true) - G.
# With --alt-follow and an alt_target re-pinned to the vehicle at every replan,
# alt_err is ~0 by construction and the only term that can cover a hover-throttle
# error is -kd_alt*vz -- a PERMANENT sink. These tests are that plant.
HOVER_TRUE = 0.577                 # measured on the 2026-09-16 airstation03 Iris
HOVER_ASSUMED = 9.8066 / 20.0      # the legacy G/MAX_ACCEL assumption = 0.490


class _LevelNet:
    """A plan with a prescribed constant vertical velocity (0 = perfectly level),
    so nothing the real network does can explain a descent."""

    def __init__(self, modes, out_seq_len, sink=0.0, climb=0.0):
        from superfly.policies.agile.model import StudentModelConfig
        self.config = StudentModelConfig(modes=modes, out_seq_len=out_seq_len)
        self.sink, self.climb = sink, climb
        self.forward_ms = 1.0

    def infer(self, depth, imu):
        n = self.config.out_seq_len
        t = 0.5 * np.arange(1, n + 1)
        z = self.sink * t + self.climb * np.minimum(t, 2.0)
        flat = np.concatenate([3.0 * t, 0.0 * t, z])
        return (np.arange(1, self.config.modes + 1, dtype=np.float32) * 0.1,
                np.tile(flat, (self.config.modes, 1)).astype(np.float32))


class _RefMPC:
    """The real build_reference_cubic; attitude level so the plant is 1-D."""
    _warmed = False

    def compute(self, x0, world_pts, cruise_alt, yaw_des, dt_wp=0.1, max_vel=7.0,
                obstacles_xy_r=None, alt_hold=True, att_lookahead_s=0.1,
                cubic=None, t_offset=0.0, min_alt=0.15, max_alt=None):
        from superfly.policies.agile.mpc import build_reference_cubic, N
        ys, yt, q0 = build_reference_cubic(cubic, yaw_des, t_offset=t_offset,
                                           min_alt=min_alt, max_alt=max_alt)
        info = {"q_pred": np.array([1.0, 0.0, 0.0, 0.0]),
                "p_ref1": ys[min(1, N - 1)][:3],
                "v_ref1": ys[min(1, N - 1)][7:10],
                "q_ref0": q0, "T_ref0": float(ys[0][10]), "status": 0}
        return np.zeros(4), 0, info


def _fly_vertical(monkeypatch, seconds=20.0, z0=1.96, sink=0.0, climb=0.0,
                  hover_thrust=None, hover_true=HOVER_TRUE, control_hz=100.0,
                  estimate=True):
    """Drive the REAL AgilePolicy.compute against the PX4 vertical plant."""
    pytest.importorskip("onnxruntime")
    import superfly.policies.agile.core as core
    monkeypatch.setattr(core, "MPC", _RefMPC)
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=control_hz, net_every=7,
                         alt_follow=True, mode_select="cost",
                         hover_thrust=hover_thrust)
    policy.net = _LevelNet(2, 10, sink=sink, climb=climb)
    policy.config = policy.net.config
    policy.hover_estimate_enabled = estimate

    dt = 1.0 / control_hz
    p = np.array([0.0, 0.0, float(z0)])
    v = np.zeros(3)
    depth = np.full((224, 224), 20.0, np.float32)
    trace = []
    for _ in range(int(seconds * control_hz)):
        cmd = policy.compute(core.AgileObs(p.copy(), v.copy(), np.eye(3),
                                           np.zeros(3),
                                           np.array([0.0, 60.0, z0]), depth))
        a_z = 9.8066 * (cmd.thrust_norm / hover_true) - 9.8066
        v[2] += a_z * dt
        p[2] += v[2] * dt
        p[1] += 3.0 * dt
        trace.append((p[2], v[2], cmd.thrust_norm, cmd.alt_sp))
    return policy, np.array(trace)


def test_level_plan_holds_altitude_against_a_wrong_hover_throttle(monkeypatch):
    """The headline case: shipped hover assumption (0.490) vs a 0.577 airframe.
    Before the fix this sank at -0.45 m/s to the band floor."""
    policy, tr = _fly_vertical(monkeypatch, seconds=20.0, z0=1.96,
                               hover_thrust=HOVER_ASSUMED)
    z, vz = tr[-1][0], tr[-1][1]
    assert abs(z - 1.96) < 0.05, f"held {z:.3f} m, want 1.96 +- 0.05"
    assert abs(vz) < 0.05, f"steady vz {vz:+.3f} m/s"
    assert tr[int(2 * 100):, 0].min() > 1.5, "never dips far below the setpoint"


def test_level_plan_holds_with_the_measured_hover_throttle(monkeypatch):
    _policy, tr = _fly_vertical(monkeypatch, seconds=20.0, z0=1.96,
                                hover_thrust=HOVER_TRUE)
    assert abs(tr[-1][0] - 1.96) < 0.05


def test_the_nets_own_sink_bias_is_held_within_a_tenth(monkeypatch):
    """A plan that really does descend at 0.11 m/s: the setpoint follows it, so
    the vehicle tracks the plan rather than running away from it."""
    _policy, tr = _fly_vertical(monkeypatch, seconds=20.0, z0=1.96, sink=-0.11,
                                hover_thrust=HOVER_ASSUMED)
    z, sp = tr[-1][0], tr[-1][3]
    assert abs(z - sp) < 0.1, f"z {z:.3f} vs setpoint {sp:.3f}"
    assert sp < 1.96, "a descending plan must lower the setpoint"


def test_a_genuine_climb_over_is_still_followed(monkeypatch):
    """The reason --alt-follow exists: a mode that goes OVER an obstacle must
    actually climb. The absolute setpoint follows the reference's vz."""
    _policy, tr = _fly_vertical(monkeypatch, seconds=6.0, z0=1.96, climb=+0.8,
                                hover_thrust=HOVER_ASSUMED)
    z, sp = tr[-1][0], tr[-1][3]
    assert sp > 2.4, f"setpoint only reached {sp:.2f} m"
    assert z > 2.3, f"vehicle only reached {z:.2f} m"
    assert abs(z - sp) < 0.25


def test_hover_throttle_is_estimated_online_and_adopted(monkeypatch):
    policy, tr = _fly_vertical(monkeypatch, seconds=20.0, z0=1.96,
                               hover_thrust=HOVER_ASSUMED)
    assert policy._hover_adopted, "the estimator never converged"
    assert abs(policy.hover_thrust - HOVER_TRUE) < 0.02, policy.hover_thrust
    assert policy.hover_thrust_param == pytest.approx(HOVER_ASSUMED)
    # and adopting it did not bump the vehicle
    assert abs(tr[-1][0] - 1.96) < 0.05


def test_estimator_can_be_disabled_and_the_parameter_still_holds(monkeypatch):
    """Half (1) + a correct parameter must hold on its own, with no estimator."""
    policy, tr = _fly_vertical(monkeypatch, seconds=20.0, z0=1.96,
                               hover_thrust=HOVER_TRUE, estimate=False)
    assert not policy._hover_adopted
    assert abs(tr[-1][0] - 1.96) < 0.05


def test_student_defaults_to_the_measured_hover_throttle_and_fast_integrator():
    pytest.importorskip("onnxruntime")
    from superfly.policies.agile.core import (
        STUDENT_HOVER_THRUST, STUDENT_KI_ALT, AgilePolicy as _P)
    import superfly.policies.agile.core as core

    class _NullMPC:
        _warmed = False
    orig, core.MPC = core.MPC, lambda: _NullMPC()
    try:
        p = _P(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"), alt_follow=True)
    finally:
        core.MPC = orig
    assert p.hover_thrust == STUDENT_HOVER_THRUST == 0.577
    assert p.ki_alt == STUDENT_KI_ALT == 1.5


def test_registry_passes_the_measured_hover_throttle():
    from superfly.compare.registry import method_registry

    class _Args:
        max_speed = 3.0
    args = method_registry()["agile_student"]["speed_args"](_Args())
    assert float(args[args.index("--hover-thrust") + 1]) == 0.577


def test_integrator_has_authority_for_a_worse_airframe(monkeypatch):
    """Half (1) only works while the integrator can actually supply
    G*(h_true/h_assumed - 1). The legacy +-2 m/s^2 clamp saturates at a ratio of
    1.20 and the trial's own airframe is already at 1.18."""
    _policy, tr = _fly_vertical(monkeypatch, seconds=20.0, z0=1.96,
                                hover_thrust=HOVER_ASSUMED, hover_true=0.63,
                                estimate=False)
    assert abs(tr[-1][0] - 1.96) < 0.05, f"held {tr[-1][0]:.3f} m"


def test_alt_setpoint_is_not_dragged_by_the_vehicles_own_velocity(monkeypatch):
    """The subtle half of the bug: the cubic is pinned to p'(0) = v_current, so
    advancing the setpoint with the CUBIC's vz makes it chase the vehicle again.
    It must be advanced by the plan's waypoint-z profile instead."""
    pytest.importorskip("onnxruntime")
    import superfly.policies.agile.core as core
    monkeypatch.setattr(core, "MPC", _RefMPC)
    policy = AgilePolicy(str(CHECKPOINTS / "t5fix_s_r1" / "student.onnx"),
                         max_vel=3.5, control_hz=100.0, net_every=1,
                         alt_follow=True, mode_select="cost",
                         hover_thrust=HOVER_ASSUMED)
    policy.net = _LevelNet(2, 10, sink=0.0)
    policy.config = policy.net.config
    depth = np.full((224, 224), 20.0, np.float32)
    # a vehicle that is sagging fast, with a perfectly LEVEL plan
    p = np.array([0.0, 0.0, 1.70])
    v = np.array([3.0, 0.0, -0.8])
    policy.compute(core.AgileObs(p, v, np.eye(3), np.zeros(3),
                                 np.array([0.0, 60.0, 1.96]), depth))
    sp0 = policy._alt_sp
    for _ in range(100):                      # 1 s of the same sagging state
        policy.compute(core.AgileObs(p, v, np.eye(3), np.zeros(3),
                                     np.array([0.0, 60.0, 1.96]), depth))
    assert abs(policy._alt_sp - sp0) < 0.02, (
        f"a level plan moved the setpoint {policy._alt_sp - sp0:+.3f} m while the "
        "vehicle sagged at -0.8 m/s")
