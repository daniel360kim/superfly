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
cubic fit of the net waypoints.

Obstacle keep-outs: the K_OBS nearest (x, y, r) cylinders are soft (slacked)
path constraints, set per solve via compute(obstacles_xy_r=...). agile_core
feeds these from its depth-derived obstacle memory so obstacles keep existing
after leaving the camera FOV.
"""

from __future__ import annotations

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
K_OBS = 6
OBS_MARGIN = 0.7


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
    h = ca.vertcat(*[(px - p[3 * k]) ** 2 + (py - p[3 * k + 1]) ** 2
                     - (p[3 * k + 2] + OBS_MARGIN) ** 2 for k in range(K_OBS)])
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
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q
    Vx = np.zeros((NY, NX)); Vx[:NX, :NX] = np.eye(NX); ocp.cost.Vx = Vx
    Vu = np.zeros((NY, NU)); Vu[NX:, :] = np.eye(NU); ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(NX)
    ocp.cost.yref = np.zeros(NY)
    ocp.cost.yref_e = np.zeros(NY_E)

    # Input bounds from upstream mpc_params.yaml: max_bodyrate_xy=6, max_bodyrate_z=2.
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.lbu = np.array([1.0, -6.0, -6.0, -2.0])
    ocp.constraints.ubu = np.array([40.0, 6.0, 6.0, 2.0])
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

    def compute(self, x0, world_pts, cruise_alt, yaw_des, dt_wp=0.1, max_vel=7.0,
                obstacles_xy_r=None, alt_hold=True):
        """High-level: build the feasible flatness reference from the selected net
        trajectory (anchored at x0's position), set the obstacle-avoidance
        constraints, and solve. Returns (u0, status, info)."""
        self._set_obstacles(obstacles_xy_r, x0[:2])
        # Anchor the reference quaternion to the CURRENT attitude hemisphere, else the
        # LINEAR_LS residual ||q - q_ref|| can blow up when q_ref lands on the opposite
        # sign of the same attitude -> the MPC commands a violent rate to "flip" it.
        q_anchor = np.asarray(x0[3:7], dtype=np.float64)
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
        # Feed PX4 the predicted NEXT attitude (stage 1, dt=0.1 s ahead) + thrust and
        # let PX4's fast attitude loop track it (robust at a 30 Hz update; raw
        # body-rate setpoints would need a faster outer loop).
        q_pred = np.asarray(self.solver.get(1, 'x')[3:7], dtype=np.float64)  # wxyz, ENU body->world
        info = {"q_ref0": q0, "T_ref0": float(yref_stages[0][10]), "status": status,
                "q_pred": q_pred, "u0": np.asarray(u0, dtype=np.float64)}
        return np.asarray(u0, dtype=np.float64), status, info


def state_x0(pos_enu, R_enu, vel_enu):
    """Build the MPC state x0 = [pos(3), quat_wxyz(4), vel(3)] from the drone
    state (R_enu = body FLU -> world ENU)."""
    q_xyzw = Rotation.from_matrix(np.asarray(R_enu, dtype=np.float64)).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return np.concatenate([np.asarray(pos_enu, np.float64), q, np.asarray(vel_enu, np.float64)])
