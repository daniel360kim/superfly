"""RPG quadrotor MPC (acados) tracking the agile_autonomy network trajectory.

Ported from SAFE_Benchmark's integrations/aerial_nav/agents/mpc_acados.py, a
faithful acados re-implementation of uzh-rpg/rpg_mpc:
  state  x (10) = [px,py,pz, qw,qx,qy,qz, vx,vy,vz]   (world ENU pos/vel, body->world
                   quaternion w-first)
  input  u (4)  = [T, wx,wy,wz]   (T = mass-normalized collective thrust [m/s^2] along
                   body z; w = body rates [rad/s])
  dynamics:
     pdot = v ; qdot = 0.5 * q (x) [0, w] ; vdot = R(q)[:,2] * T - [0,0,G]
  cost  : LINEAR_LS ; bounds: T in [1,40] m/s^2, wx,wy in [-8,8], wz in [-5,5]
  horizon: N=10, dt=0.1 s ; solver: SQP-RTI, ERK(RK4), Gauss-Newton, qpOASES

The reference is the SELECTED network trajectory through the differential-
flatness map. When build_reference speed-caps the reference velocity to max_vel,
only the velocity is scaled (upstream / rpg style); acceleration follows the
cubic fit of the net waypoints. build_reference_cubic (the student's reference)
caps the same way and additionally drops the along-track reference acceleration
while the cap binds, which is sim_episode.clamp_command's rule.

Obstacle keep-outs: the K_OBS nearest (x, y, r) cylinders are soft (slacked)
path constraints, set per solve via compute(obstacles_xy_r=...). agile_core
feeds these from its depth-derived obstacle memory so obstacles keep existing
after leaving the camera FOV.
"""

from __future__ import annotations

import math
import os
import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

G = 9.8066
NX, NU, NY, NY_E, N = 10, 4, 14, 10, 10
DT = 0.1
# Cap the reference horizontal acceleration so the flatness attitude/thrust reference
# stays dynamically feasible (a jerky cubic fit through the 10 net waypoints can spike
# the implied thrust past the bound and demand an over-tilted attitude reference).
MAX_REF_ACCEL_XY = 6.0
# Obstacle avoidance as MPC constraints: the K nearest pillars are online parameters; the
# drone must stay outside (radius + OBS_MARGIN) of each. Soft (slacked) so the QP never
# goes infeasible. OBS_MARGIN covers the drone's rotor span + a safety gap.
# Env-overridable (AGILE_MPC_OBS_MARGIN / agile_offboard --obs-margin) for the
# 2026-07-30 margin-tuning campaign; read at solver-build time inside _model()
# (it is baked into the generated constraint C code -> ~2 s codegen rebuild).
K_OBS = 6
OBS_MARGIN_DEFAULT = 0.7


def _obs_margin() -> float:
    return float(os.environ.get("AGILE_MPC_OBS_MARGIN", str(OBS_MARGIN_DEFAULT)))


# ---------------------------------------------------------------------------
# acados model + solver
# ---------------------------------------------------------------------------

def _model():
    import casadi as ca
    from acados_template import AcadosModel
    px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
    qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), ca.SX.sym('qz')
    vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
    x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz)
    T, wx, wy, wz = ca.SX.sym('T'), ca.SX.sym('wx'), ca.SX.sym('wy'), ca.SX.sym('wz')
    u = ca.vertcat(T, wx, wy, wz)
    qdot = 0.5 * ca.vertcat(-wx * qx - wy * qy - wz * qz,
                            wx * qw + wz * qy - wy * qz,
                            wy * qw - wz * qx + wx * qz,
                            wz * qw + wy * qx - wx * qy)
    vdot = ca.vertcat(2 * (qw * qy + qx * qz) * T,
                      2 * (qy * qz - qw * qx) * T,
                      (1 - 2 * qx ** 2 - 2 * qy ** 2) * T - G)
    f = ca.vertcat(vx, vy, vz, qdot, vdot)
    m = AcadosModel()
    m.name = 'quadrotor_agile'
    m.x = x
    m.u = u
    m.xdot = ca.SX.sym('xdot', NX, 1)
    m.f_expl_expr = f
    m.f_impl_expr = m.xdot - f
    # Obstacle-avoidance path constraints: p = [ox,oy,r] x K_OBS (online params).
    # h_k = (px-ox)^2 + (py-oy)^2 - (r+margin)^2 >= 0 -> drone stays clear of each pillar.
    p = ca.SX.sym('p', 3 * K_OBS)
    margin = _obs_margin()
    h = ca.vertcat(*[(px - p[3 * k]) ** 2 + (py - p[3 * k + 1]) ** 2
                     - (p[3 * k + 2] + margin) ** 2 for k in range(K_OBS)])
    m.p = p
    m.con_h_expr = h
    return m


def make_solver(json_file, qp_solver='FULL_CONDENSING_QPOASES'):
    from acados_template import AcadosOcp, AcadosOcpSolver
    ocp = AcadosOcp()
    ocp.model = _model()
    try:
        ocp.solver_options.N_horizon = N
    except Exception:
        ocp.dims.N = N
    ocp.solver_options.tf = N * DT

    # Upstream mpc_params.yaml: Q_pos_xy/z=100, Q_attitude=200, Q_velocity=10.
    # Attitude weight 200 is applied per quaternion component in the LINEAR_LS cost.
    qpos = float(os.environ.get("AGILE_MPC_Q_POS", "100"))
    qatt = float(os.environ.get("AGILE_MPC_Q_ATT", "50"))
    Q = np.diag([qpos, qpos, qpos, qatt, qatt, qatt, qatt, 10., 10., 10.])
    # Input weight R = diag([thrust, wx, wy, wz]). Port default 0.1; upstream
    # mpc_params.yaml uses R_thrust=R_pitchroll=R_yaw=1.0. Raising R relative to
    # Q makes the solver smoother / less aggressive on attitude+thrust inputs.
    rin = float(os.environ.get("AGILE_MPC_R", "0.1"))
    R = np.diag([rin, rin, rin, rin])
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q
    Vx = np.zeros((NY, NX)); Vx[:NX, :NX] = np.eye(NX); ocp.cost.Vx = Vx
    Vu = np.zeros((NY, NU)); Vu[NX:, :] = np.eye(NU); ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(NX)
    ocp.cost.yref = np.zeros(NY)
    ocp.cost.yref_e = np.zeros(NY_E)

    # Input bounds from upstream mpc_params.yaml: max_bodyrate_xy=6, max_bodyrate_z=2,
    # min_thrust=5, max_thrust=20 (mass-normalized, i.e. m/s^2: ~0.5 g .. 2 g about the
    # g=9.81 hover). The port previously used a much wider [1, 40] thrust band, which lets
    # the MPC plan near-zero / very high collective thrust and produce more extreme
    # attitude solutions. Thrust bounds are env-overridable to A/B the old band.
    t_min = float(os.environ.get("AGILE_MPC_T_MIN", "5.0"))
    t_max = float(os.environ.get("AGILE_MPC_T_MAX", "20.0"))
    wxy = float(os.environ.get("AGILE_MPC_MAX_BODYRATE_XY", "6.0"))
    wz = float(os.environ.get("AGILE_MPC_MAX_BODYRATE_Z", "2.0"))
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array([t_min, -wxy, -wxy, -wz])
    ocp.constraints.ubu = np.array([t_max, wxy, wxy, wz])
    x0 = np.zeros(NX); x0[3] = 1.0
    ocp.constraints.x0 = x0

    # Obstacle-avoidance soft constraints h_k >= 0 (slacked so the QP can't go infeasible).
    ocp.constraints.lh = np.zeros(K_OBS)
    ocp.constraints.uh = 1e9 * np.ones(K_OBS)
    ocp.constraints.idxsh = np.arange(K_OBS)
    ocp.cost.zl = 1e3 * np.ones(K_OBS)      # linear slack penalty (push out of obstacles)
    ocp.cost.zu = np.zeros(K_OBS)
    ocp.cost.Zl = 1e3 * np.ones(K_OBS)      # quadratic slack penalty
    ocp.cost.Zu = np.zeros(K_OBS)
    # default params = far-away dummy obstacles (overwritten each solve)
    ocp.parameter_values = np.tile([1e3, 1e3, 0.01], K_OBS)

    ocp.solver_options.qp_solver = qp_solver
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_iter_max = 50
    # Generated C goes to a per-user scratch dir (never the repo). The json bakes
    # in this path, so both must be set together and be writable.
    ocp.code_export_directory = os.environ.get(
        "AGILE_MPC_CODE_EXPORT_DIR", "/tmp/acados_agile_c_generated_code")
    try:
        return AcadosOcpSolver(ocp, json_file=json_file, build=True, generate=True)
    except TypeError:
        # Older acados without build/generate kwargs.
        return AcadosOcpSolver(ocp, json_file=json_file)


# ---------------------------------------------------------------------------
# Differential-flatness reference (net trajectory -> MPC reference over the horizon)
# ---------------------------------------------------------------------------

def flatness_attitude(specific_force, yaw_des, prev_q_wxyz=None):
    """specific_force = a - gravity (= a + [0,0,G]) -> (q_wxyz, thrust).
    z_body = sf/||sf||; complete the frame with the desired yaw heading. Returns a
    unit body->world ENU quaternion (w-first), hemisphere-aligned to prev_q_wxyz."""
    sf = np.asarray(specific_force, dtype=np.float64)
    n = float(np.linalg.norm(sf))
    if n < 1e-9:
        sf = np.array([0.0, 0.0, G]); n = G
    z_b = sf / n
    x_h = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    y_b = np.cross(z_b, x_h)
    if np.linalg.norm(y_b) < 1e-6:
        y_b = np.cross(z_b, np.array([0.0, 1.0, 0.0]))
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b); x_b /= np.linalg.norm(x_b)
    R = np.column_stack([x_b, y_b, z_b])
    q_xyzw = Rotation.from_matrix(R).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # -> wxyz
    if prev_q_wxyz is not None and float(np.dot(q, prev_q_wxyz)) < 0.0:
        q = -q                                                  # hemisphere align
    return q, n


def clamp_attitude_tilt(q_pred_wxyz, max_tilt_deg, yaw_des):
    """Clamp the predicted attitude's tilt (angle of body-z from vertical) to
    max_tilt_deg and rebuild with the desired heading, so the setpoint streamed to
    PX4 stays trackable. Returns a wxyz quaternion; the MPC's thrust is used
    separately. Default max_tilt_deg should be 15: >=20 deg re-enters the
    attitude limit cycle at 3 m/s cruise."""
    q = np.asarray(q_pred_wxyz, dtype=np.float64)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    z = R[:, 2].copy()
    max_h = float(np.sin(np.radians(max_tilt_deg)))
    horiz = float(np.hypot(z[0], z[1]))
    if horiz > max_h and horiz > 1e-9:
        z[0] *= max_h / horiz
        z[1] *= max_h / horiz
        z[2] = float(np.sqrt(max(1e-6, 1.0 - max_h * max_h)))
        z /= np.linalg.norm(z)
    q_clamped, _ = flatness_attitude(z, yaw_des)   # rebuild R from clamped body-z + heading
    return q_clamped


def scale_tilt_to_thrust(q_cmd_wxyz, thrust_mpc, thrust_cmd, yaw_des):
    """Make the HORIZONTAL acceleration the one the MPC actually solved for.

    The MPC solves a (tilt, collective thrust) pair; this port streams the tilt
    but replaces the thrust with the altitude loop's, so the horizontal
    acceleration the vehicle gets is thrust_cmd * sin(tilt) where the MPC
    planned thrust_mpc * sin(tilt). In the 2026-09-17 climbs thrust_cmd reached
    0.748 against a 0.577 hover -- 1.30x -- and every speed excursion in those
    runs is inside a climb. Scale the horizontal part of the commanded body-z by
    thrust_mpc / thrust_cmd and the product is exact again.

    Reduce only (k clamped to <= 1). The symmetric case -- the altitude loop
    UNDER-thrusting in a descent, where the MPC's horizontal intent would need
    MORE tilt -- is deliberately not applied: adding tilt in a descent is the
    one direction that can make an overspeed worse, and the descents in these
    runs are exactly where the vehicle is already fast. Both thrusts are
    mass-normalised specific forces [m/s^2].

    Returns a wxyz quaternion (unchanged when the loop is not over-thrusting)."""
    q = np.asarray(q_cmd_wxyz, dtype=np.float64)
    if thrust_mpc is None or thrust_cmd is None or thrust_cmd <= 1e-6:
        return q
    k = float(thrust_mpc) / float(thrust_cmd)
    if not np.isfinite(k) or k >= 1.0 - 1e-9:
        return q
    k = max(k, 0.0)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    z = R[:, 2].copy()
    z[0] *= k
    z[1] *= k
    z[2] = float(np.sqrt(max(1e-6, 1.0 - float(z[0] ** 2 + z[1] ** 2))))
    z /= np.linalg.norm(z)
    q_scaled, _ = flatness_attitude(z, yaw_des)
    return q_scaled


def clamp_speed_command(q_cmd_wxyz, vel_enu, max_vel, yaw_des, thrust_cmd=G,
                        lag_s=0.15):
    """sim_episode.clamp_command's V_CAP rule, applied to the ATTITUDE command
    this port actually streams, with the plant's attitude lag compensated.

    The evaluation harness commands an acceleration and, whenever the vehicle is
    above V_CAP, deletes the positive along-track part of it. Its plant applies
    that command through a 0.1 s lag, so "delete" is enough there. Here the
    command is an attitude that PX4 tracks with its own dynamics, and deleting
    the along-track push only at the cap is NOT enough: the attitude the vehicle
    is actually holding still carries the push it was given a moment ago, so the
    loop chatters around an equilibrium ABOVE the cap. Simulated on a point mass
    with a first-order attitude lag and a tracker that always wants full tilt,
    a pure delete settles at 3.78 / 4.46 / 5.34 m/s for a 0.05 / 0.15 / 0.30 s
    lag -- which is the shape of the 2026-09-17 re-runs (median 3.1, excursions
    to 4.1-6.2, all in climbs, where the thrust excess raises the equilibrium
    further).

    So limit the along-track acceleration to what can still be stopped:

        a_along <= (max_vel - |v|) / lag_s

    One constant with a physical meaning (the attitude loop's response time).
    Far below the cap it is inert -- at 3.0 m/s against a 3.5 cap it still
    allows 3.3 m/s^2, more than these plans ever ask for. At the cap it is
    zero. Above it, it is negative: the clamp brakes, rather than merely
    declining to push, which is what recovers from the overshoot the lag
    creates. Same simulation, lag 0.30 s and a 1.30x thrust excess: 5.34 ->
    4.10 m/s peak and an exact 3.50 m/s settle.

    |v| is the FULL 3-D speed (sim_episode.V_CAP caps the 3-D speed, and these
    runs climb at up to 2.3 m/s -- a horizontal-only trigger lets that ride for
    free), but only the HORIZONTAL command is touched: the vertical axis belongs
    to the altitude loop, which recomputes thrust from this attitude's own
    cos(tilt) immediately afterwards.

    `thrust_cmd` is the specific force the vehicle will be given [m/s^2], i.e.
    what converts body-z tilt into acceleration. Returns a wxyz quaternion."""
    v = np.asarray(vel_enu, dtype=np.float64).reshape(3)
    q = np.asarray(q_cmd_wxyz, dtype=np.float64)
    if max_vel is None or max_vel <= 0.0 or lag_s <= 0.0:
        return q
    n_xy = float(np.hypot(v[0], v[1]))
    if n_xy < 1e-9:
        return q
    f = float(thrust_cmd)
    if not np.isfinite(f) or f <= 1e-6:
        return q
    vh = np.array([v[0], v[1]]) / n_xy
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    z = R[:, 2].copy()
    along = float(z[0] * vh[0] + z[1] * vh[1]) * f          # m/s^2 along track
    a_max = (float(max_vel) - float(np.linalg.norm(v))) / float(lag_s)
    if along <= a_max:
        return q
    z[0] += (a_max - along) / f * vh[0]
    z[1] += (a_max - along) / f * vh[1]
    h2 = float(z[0] ** 2 + z[1] ** 2)
    z[2] = float(np.sqrt(max(1e-6, 1.0 - h2))) if h2 < 1.0 else 0.0
    z /= np.linalg.norm(z)
    q_clamped, _ = flatness_attitude(z, yaw_des)
    return q_clamped


def fit_cubic_state(p_cur, v_cur, wps, dt_wp):
    """Port of superfly_expert_sampler.sim_episode.fit_cubic: the cubic
    p(t) = c0 + c1 t + c2 t^2 + c3 t^3 that is PINNED to the current state --
    c0 = p, c1 = v -- with c2/c3 least-squares through waypoints 1..3 at
    dt_wp, 2 dt_wp, 3 dt_wp. Returns (4, 3).

    This is the difference that matters against np.polyfit: polyfit is
    unconstrained, so from rest it hands the tracker a non-zero velocity at
    t = 0 (measured: 1.36 m/s where the evaluation harness commands 0) -- a
    step demand every time the vehicle is slower than the plan's mean speed."""
    p_cur = np.asarray(p_cur, dtype=np.float64).reshape(3)
    v_cur = np.asarray(v_cur, dtype=np.float64).reshape(3)
    wps = np.asarray(wps, dtype=np.float64)
    ts = dt_wp * np.arange(1, 4)
    rhs = wps[:3] - p_cur[None] - np.outer(ts, v_cur)
    A = np.stack([ts ** 2, ts ** 3], 1)
    c23 = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.vstack([p_cur, v_cur, c23])


def eval_cubic(c, t):
    """sim_episode.eval_cubic: (p, v, a) of the cubic at time t."""
    p = c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3
    v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2
    a = 2 * c[2] + 6 * c[3] * t
    return p, v, a


def build_reference_cubic(cubic, yaw_des, t_offset=0.0, prev_q0=None,
                          min_alt=0.15, max_alt=None, max_vel=None):
    """The STUDENT's MPC reference: sample `cubic` (from fit_cubic_state) at the
    solver's own node times, exactly as sim_episode's tracker evaluates it at
    tau = t - t_decision.

    Unlike build_reference this does NOT re-integrate position from an
    independently capped velocity: the position reference IS the cubic's, so
    the manoeuvre is tracked as the evaluation harness tracks it.

    `max_vel` (= sim_episode.V_CAP) is the ONE thing that must still be
    enforced here, and the reason is a difference in WHERE the two trackers
    read the cubic. sim_episode evaluates it only at tau <= one decision period
    (1/15 s) and then limits the COMMAND -- clamp_command strips the positive
    along-track acceleration whenever the VEHICLE is above V_CAP, so the
    vehicle can never exceed it. This port has no command-level clamp (the MPC
    thrust is discarded and only the attitude is streamed), and it samples the
    SAME cubic over the solver's whole 1.0 s horizon, i.e. straight through the
    mid-horizon velocity bulge that a cubic pinned to (p, v) and forced through
    waypoints 1-3 always has. Measured on a plan whose waypoints are a uniform
    3.0 m/s: |v_ref| peaks at 3.9 m/s (straight, from 1 m/s), 4.5 m/s (90 deg
    turn) and 5.7 m/s (180 deg turn) around t = 0.75 s -- the plan's own speed
    is nowhere near it, the cubic's shape is. Handing that to the MPC as a
    velocity AND position reference is what flew the 2026-09-16 Isaac trials at
    5.6-6.0 m/s on labels capped at 3.0.

    So: cap the reference speed at max_vel, remove the along-track reference
    acceleration while the cap binds (clamp_command's rule), and pull the
    position reference back by the distance the cap removed, so the
    position and velocity references keep agreeing -- the same consistency the
    z clip below maintains. max_vel=None leaves the raw cubic (the legacy
    behaviour, and what the reference-equivalence test asserts).

    `t_offset` is the plan's age [s]: between net updates the manoeuvre is
    consumed rather than dragged along with the vehicle.

    Returns (yref_stages [N,14], yref_term [10], q0)."""
    yref_stages = np.zeros((N, NY))
    yref_term = np.zeros(NY_E)
    prev_q = prev_q0
    q0 = None
    c = np.asarray(cubic, dtype=np.float64)
    nodes = [eval_cubic(c, float(t_offset) + i * DT) for i in range(N + 1)]
    pos = [np.asarray(n[0], np.float64).copy() for n in nodes]
    vel = [np.asarray(n[1], np.float64).copy() for n in nodes]
    acc = [np.asarray(n[2], np.float64).copy() for n in nodes]
    if max_vel is not None and max_vel > 0.0:
        # Cap the speed, drop the along-track acceleration while the cap binds
        # (sim_episode.clamp_command), and pull every later node's position back
        # by the distance the cap removed -- trapezoidal, so the position and
        # velocity references keep agreeing to the integration error of the
        # cubic itself (< 1 cm per node).
        rate = []
        for i in range(N + 1):
            sp = float(np.linalg.norm(vel[i]))
            if sp > max_vel:
                vh = vel[i] / sp
                along = float(acc[i] @ vh)
                if along > 0.0:
                    acc[i] = acc[i] - along * vh
                v_cap = vel[i] * (max_vel / sp)
                rate.append(vel[i] - v_cap)
                vel[i] = v_cap
            else:
                rate.append(np.zeros(3))
        excess = np.zeros(3)
        for i in range(1, N + 1):
            excess = excess + 0.5 * (rate[i - 1] + rate[i]) * DT
            pos[i] = pos[i] - excess
    for i in range(N + 1):
        p, v, a = pos[i], vel[i], acc[i]
        a_xy = float(np.hypot(a[0], a[1]))
        if a_xy > MAX_REF_ACCEL_XY:
            a[:2] *= MAX_REF_ACCEL_XY / a_xy
        a[2] = float(np.clip(a[2], -MAX_REF_ACCEL_XY, MAX_REF_ACCEL_XY))
        # Clip z, and zero vz at the boundary: leaving vz unclipped makes the
        # position and velocity references disagree exactly where the clip
        # bites, and the MPC then chases a climb/descent the position reference
        # forbids.
        if p[2] < min_alt:
            p[2] = min_alt
            v[2] = max(v[2], 0.0)
        if max_alt is not None and p[2] > max_alt:
            p[2] = max_alt
            v[2] = min(v[2], 0.0)
        q, T = flatness_attitude(a + np.array([0.0, 0.0, G]), yaw_des, prev_q)
        prev_q = q
        if i == 0:
            q0 = q
        if i < N:
            yref_stages[i] = np.concatenate([p, q, v, [T], [0.0, 0.0, 0.0]])
        else:
            yref_term = np.concatenate([p, q, v])
    return yref_stages, yref_term, q0


def build_reference(world_pts, pos_current, cruise_alt, yaw_des, dt_wp=0.1,
                    max_vel=7.0, prev_q0=None, alt_hold=True, min_alt=0.15):
    """Net's 10 world waypoints -> per-node MPC reference over the horizon, made
    DYNAMICALLY FEASIBLE (rpg-style): velocity direction from a cubic fit of the
    net trajectory, speed-capped to max_vel, position integrated forward from the
    drone's CURRENT position (so the reference never demands an instant jump to
    cruise speed from rest).

    alt_hold=True (default): z is held at cruise_alt (the net's z output is
    discarded); alt_hold=False follows the net's z (clamped, floored at min_alt).
    Returns (yref_stages [N,14], yref_term [10], q0)."""
    wp = np.asarray(world_pts, dtype=np.float64)
    nwp = len(wp)
    t = np.arange(nwp) * dt_wp
    cx = np.polyfit(t, wp[:, 0], 3)
    cy = np.polyfit(t, wp[:, 1], 3)
    dcx, dcy = np.polyder(cx, 1), np.polyder(cy, 1)
    ddcx, ddcy = np.polyder(cx, 2), np.polyder(cy, 2)
    if not alt_hold:
        cz = np.polyfit(t, wp[:, 2], 3)
        dcz, ddcz = np.polyder(cz, 1), np.polyder(cz, 2)
    t_max = t[-1]
    yref_stages = np.zeros((N, NY))
    yref_term = np.zeros(NY_E)
    prev_q = prev_q0
    q0 = None
    px, py = float(pos_current[0]), float(pos_current[1])     # anchor at current pos
    pz = float(pos_current[2])
    for i in range(N + 1):
        ti = min(i * DT, t_max)
        vz = float(np.clip(np.polyval(dcz, ti), -max_vel, max_vel)) if not alt_hold else 0.0
        v = np.array([np.polyval(dcx, ti), np.polyval(dcy, ti), vz])
        az = float(np.clip(np.polyval(ddcz, ti), -MAX_REF_ACCEL_XY, MAX_REF_ACCEL_XY)) if not alt_hold else 0.0
        a = np.array([np.polyval(ddcx, ti), np.polyval(ddcy, ti), az])
        s = float(np.hypot(v[0], v[1]))                       # speed-cap (feasible ref, horizontal)
        if s > max_vel and s > 1e-9:
            ratio = max_vel / s
            v[:2] *= ratio
        a_xy = float(np.hypot(a[0], a[1]))
        if a_xy > MAX_REF_ACCEL_XY:
            a[:2] *= MAX_REF_ACCEL_XY / a_xy
        if i > 0:                                             # integrate feasible path
            px += v[0] * DT
            py += v[1] * DT
            pz += v[2] * DT
        p = np.array([px, py, cruise_alt if alt_hold else max(pz, min_alt)])
        q, T = flatness_attitude(a + np.array([0.0, 0.0, G]), yaw_des, prev_q)
        prev_q = q
        if i == 0:
            q0 = q
        if i < N:
            yref_stages[i] = np.concatenate([p, q, v, [T], [0.0, 0.0, 0.0]])
        else:
            yref_term = np.concatenate([p, q, v])
    return yref_stages, yref_term, q0


# ---------------------------------------------------------------------------
# MPC wrapper
# ---------------------------------------------------------------------------

class MPC:
    def __init__(self, json_file='/tmp/acados_quad_agile_upstream.json'):
        try:
            self.solver = make_solver(json_file, 'FULL_CONDENSING_QPOASES')
        except Exception as exc:
            print(f"[agile_mpc] qpOASES unavailable ({exc}); falling back to HPIPM", flush=True)
            self.solver = make_solver(json_file, 'PARTIAL_CONDENSING_HPIPM')
        self._prev_q0 = None
        self._warmed = False
        self._warm_start()

    def _warm_start(self):
        """acados initializes the state/control guess to ZEROS, including a degenerate
        zero-quaternion -> the first real solve is garbage (saturated rates). Seed a
        valid hover guess and run a few RTI solves so the warm-started trajectory is
        sane before the first real command."""
        hover_x = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0])
        hover_u = np.array([G, 0.0, 0.0, 0.0])
        for i in range(N + 1):
            self.solver.set(i, 'x', hover_x)
        for i in range(N):
            self.solver.set(i, 'u', hover_u)
        ys = np.tile(np.concatenate([hover_x, [G, 0.0, 0.0, 0.0]]), (N, 1))
        for _ in range(10):
            self.solve(hover_x, ys, hover_x)

    def solve(self, x0, yref_stages, yref_term):
        """x0 (10), yref_stages (N,14), yref_term (10) -> (u0 [T,wx,wy,wz], status)."""
        s = self.solver
        s.set(0, 'lbx', np.asarray(x0, dtype=np.float64))
        s.set(0, 'ubx', np.asarray(x0, dtype=np.float64))
        for i in range(N):
            s.set(i, 'yref', np.asarray(yref_stages[i], dtype=np.float64))
        s.set(N, 'yref', np.asarray(yref_term, dtype=np.float64))
        status = s.solve()
        return s.get(0, 'u'), int(status)

    def _set_obstacles(self, obstacles_xy_r, pos_xy):
        """Set the K nearest obstacles (to pos) as the avoidance-constraint params on
        every node. Static obstacles -> same params across the horizon. Pads with
        far dummies."""
        params = np.tile([1e3, 1e3, 0.01], K_OBS)
        if obstacles_xy_r:
            obs = sorted(obstacles_xy_r,
                         key=lambda o: (o[0] - pos_xy[0]) ** 2 + (o[1] - pos_xy[1]) ** 2)
            for k, (ox, oy, r) in enumerate(obs[:K_OBS]):
                params[3 * k:3 * k + 3] = [ox, oy, r]
        for i in range(N + 1):
            self.solver.set(i, 'p', params)

    def _attitude_at(self, t_ahead):
        """Predicted body->world attitude (wxyz) `t_ahead` seconds into the MPC
        horizon, SLERP-interpolated between the discretization nodes (spaced DT).

        Node 0's attitude is x0 (the CURRENT measured attitude), so sampling at
        the control period (~1/control_hz) instead of a whole node (DT=0.1 s)
        gives PX4 a setpoint just ahead of the current pose rather than the 0.1 s
        prediction. Sending the 0.1 s-ahead node at a 30 Hz control rate is a
        3-step over-anticipation that drives the attitude limit cycle."""
        node = max(0.0, float(t_ahead)) / DT
        i0 = int(math.floor(node))
        if i0 >= N:
            return np.asarray(self.solver.get(N, 'x')[3:7], dtype=np.float64)
        q0 = np.asarray(self.solver.get(i0, 'x')[3:7], dtype=np.float64)
        frac = node - i0
        if frac <= 1e-6:
            return q0
        q1 = np.asarray(self.solver.get(i0 + 1, 'x')[3:7], dtype=np.float64)
        return _slerp_wxyz(q0, q1, frac)

    def compute(self, x0, world_pts, cruise_alt, yaw_des, dt_wp=0.1, max_vel=7.0,
                obstacles_xy_r=None, alt_hold=True, att_lookahead_s=DT,
                cubic=None, t_offset=0.0, min_alt=0.15, max_alt=None):
        """High-level: build the feasible flatness reference from the selected net
        trajectory (anchored at x0's position), set the obstacle-avoidance
        constraints, and solve. Returns (u0, status, info).

        att_lookahead_s: how far into the predicted horizon to sample the attitude
        setpoint streamed to PX4 (default DT = the legacy stage-1 0.1 s node). Set
        to the control period to stop over-anticipating at low control rates."""
        self._set_obstacles(obstacles_xy_r, x0[:2])
        # Anchor the reference quaternion to the CURRENT attitude hemisphere, else the
        # LINEAR_LS residual ||q - q_ref|| can blow up when q_ref lands on the opposite
        # sign of the same attitude -> the MPC commands a violent rate to "flip" it.
        q_anchor = np.asarray(x0[3:7], dtype=np.float64)
        if cubic is not None:
            # Student: track the state-pinned cubic itself (see
            # build_reference_cubic); world_pts is then only a diagnostic.
            yref_stages, yref_term, q0 = build_reference_cubic(
                cubic, yaw_des, t_offset=t_offset, prev_q0=q_anchor,
                min_alt=min_alt, max_alt=max_alt, max_vel=max_vel)
        else:
            yref_stages, yref_term, q0 = build_reference(
                world_pts, x0[:3], cruise_alt, yaw_des, dt_wp, max_vel,
                prev_q0=q_anchor, alt_hold=alt_hold)
        self._prev_q0 = q0
        # SQP-RTI is one iteration per call; the VERY FIRST command would otherwise be
        # a cold/half-converged transient (saturated rates) at the CLIMB->POLICY
        # handoff. Converge it with a few solves the first time, then 1/tick after.
        n_iter = 1 if self._warmed else 5
        self._warmed = True
        for _ in range(n_iter):
            u0, status = self.solve(x0, yref_stages, yref_term)
        # Feed PX4 the predicted attitude `att_lookahead_s` ahead + thrust and let
        # PX4's fast attitude loop track it. Sampling at the control period (not the
        # 0.1 s stage-1 node) avoids the over-anticipation limit cycle at 30 Hz.
        q_pred = self._attitude_at(att_lookahead_s)  # wxyz, ENU body->world
        info = {"q_ref0": q0, "T_ref0": float(yref_stages[0][10]), "status": status,
                "q_pred": q_pred, "u0": np.asarray(u0, dtype=np.float64),
                # stage-1 position reference (used by agile_core's --alt-follow to
                # slave the altitude-hold thrust PD to the reference z).
                "p_ref1": np.asarray(yref_stages[min(1, N - 1)][:3], dtype=np.float64),
                # stage-1 reference VELOCITY: agile_core advances its absolute
                # altitude setpoint at this rate rather than snapping it to
                # p_ref1, which is pinned to the vehicle every replan.
                "v_ref1": np.asarray(yref_stages[min(1, N - 1)][7:10], dtype=np.float64)}
        return np.asarray(u0, dtype=np.float64), status, info


def _slerp_wxyz(q0, q1, frac):
    """Spherical linear interpolation between two wxyz quaternions (hemisphere
    aligned). frac in [0,1]: 0 -> q0, 1 -> q1."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    d = float(np.dot(q0, q1))
    if d < 0.0:                      # take the shorter arc
        q1 = -q1
        d = -d
    if d > 0.9995:                   # nearly parallel -> linear + renormalize
        q = q0 + frac * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta0 = math.acos(d)
    sin0 = math.sin(theta0)
    s0 = math.sin((1.0 - frac) * theta0) / sin0
    s1 = math.sin(frac * theta0) / sin0
    return s0 * q0 + s1 * q1


def state_x0(pos_enu, R_enu, vel_enu):
    """Build the MPC state x0 = [pos(3), quat_wxyz(4), vel(3)] from the drone
    state (R_enu = body FLU -> world ENU)."""
    q_xyzw = Rotation.from_matrix(np.asarray(R_enu, dtype=np.float64)).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return np.concatenate([np.asarray(pos_enu, np.float64), q, np.asarray(vel_enu, np.float64)])
