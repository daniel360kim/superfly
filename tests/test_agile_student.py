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
    R = Rotation.from_euler("xyz", rng.uniform(-0.6, 0.6, 3)).as_matrix()
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

    obs = sim_episode.Obs(t=0.0, p=pos, v=vel, R=R, omega=omega_world,
                          goal_p=goal, goal_speed=goal_speed, depth=None)
    expected = sim_episode.OnnxPolicy.encode_state(obs)     # (1, 1, 22)

    # The deployment is handed a BODY-frame angular rate (MAVLink gives body
    # rates directly); the reference stores a world one and rotates it. Same
    # quantity, so this is the honest mapping between the two call sites.
    got = _NoNet(goal_speed)._student_state_to_model_input(
        pos, R, vel, R.T @ omega_world, goal)

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
