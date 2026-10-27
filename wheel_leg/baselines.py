"""
Classical baseline: LQR balance controller for the wheel-legged robot.

The robot is approximated as a wheeled inverted pendulum (WIP) in the sagittal plane:
    state  s = [x, theta, x_dot, theta_dot]   (theta = body lean, + = forward)
    input  u = total wheel torque (split equally between both wheels)

Model parameters (body mass, CoM height, pitch inertia, wheel inertia) are extracted
from the MuJoCo model at the current leg configuration, so the same code can design an
LQR for any leg height. The RL policy is later compared against this controller.
"""
import numpy as np
import scipy.linalg as sl
import mujoco

WHEEL_RADIUS = 0.08
G = 9.81


def quat_to_pitch(q):
    """Pitch (rotation about world y) of a wxyz quaternion, + = leaning forward."""
    w, x, y, z = q
    return np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))


def extract_wip_params(m: mujoco.MjModel, d: mujoco.MjData):
    """Lumped WIP parameters at the robot's current configuration (needs mj_forward)."""
    wl, wr = m.body("l_wheel").id, m.body("r_wheel").id
    mw = m.body_mass[wl] + m.body_mass[wr]
    Iw = m.body_inertia[wl][1] + m.body_inertia[wr][1]
    M = m.body_subtreemass[1] - mw

    cw = 0.5 * (d.xipos[wl] + d.xipos[wr])
    cb = (d.subtree_com[1] * m.body_subtreemass[1] - cw * mw) / M
    rel = cb - cw
    l = np.hypot(rel[0], rel[2])

    Ib = 0.0
    for b in range(1, m.nbody):
        if b in (wl, wr):
            continue
        r = d.xipos[b] - cb
        Rb = d.ximat[b].reshape(3, 3)
        I_world = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
        Ib += I_world[1, 1] + m.body_mass[b] * (r[0] ** 2 + r[2] ** 2)

    # lean angle of the CoM relative to the pelvis pitch (CoM not exactly above the axle)
    com_lean_offset = np.arctan2(rel[0], rel[2])
    return dict(M=M, mw=mw, Iw=Iw, l=l, Ib=Ib, com_lean_offset=com_lean_offset)


def wip_linear_model(p, R=WHEEL_RADIUS):
    """Continuous-time linearized WIP: s_dot = A s + B u."""
    M, mw, Iw, l, Ib = p["M"], p["mw"], p["Iw"], p["l"], p["Ib"]
    # Mass matrix of [x_ddot, theta_ddot] at theta = 0
    Mm = np.array([[M + mw + Iw / R**2, M * l],
                   [M * l,              Ib + M * l**2]])
    gvec = np.array([0.0, M * G * l])       # d(generalized force)/d(theta)
    bvec = np.array([1.0 / R, -1.0])        # effect of wheel torque u
    Minv = np.linalg.inv(Mm)
    A = np.zeros((4, 4))
    A[0, 2] = A[1, 3] = 1.0
    A[2:, 1] = Minv @ gvec
    B = np.zeros((4, 1))
    B[2:, 0] = Minv @ bvec
    return A, B


class LQRBalancer:
    """Discrete LQR on the WIP model. Call reset() whenever leg height changes if you
    want gain scheduling; by default gains are designed once at the nominal pose."""

    def __init__(self, m, d, dt, Q=None, R=None, torque_limit=16.0):
        self.dt = dt
        self.torque_limit = torque_limit
        self.Q = np.diag([5.0, 100.0, 5.0, 1.0]) if Q is None else Q
        self.R = np.array([[0.5]]) if R is None else R
        self.design(m, d)

    def design(self, m, d):
        # extract parameters with the pelvis upright and at the origin, keeping the
        # current leg configuration (design must not depend on the current tilt)
        d0 = mujoco.MjData(m)
        d0.qpos[:] = d.qpos
        d0.qpos[:3] = [0.0, 0.0, d.qpos[2]]
        d0.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d0)
        p = extract_wip_params(m, d0)
        A, B = wip_linear_model(p)
        # zero-order-hold discretization
        n = A.shape[0]
        Md = sl.expm(np.block([[A, B], [np.zeros((1, n + 1))]]) * self.dt)
        Ad, Bd = Md[:n, :n], Md[:n, n:]
        P = sl.solve_discrete_are(Ad, Bd, self.Q, self.R)
        self.K = np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        self.params = p
        return self.K

    def state(self, m, d, x_ref=0.0, v_ref=0.0):
        """Sagittal WIP state from the simulator (pitch from the pelvis IMU)."""
        # lean of the CoM over the axle from forward kinematics (joint encoders + IMU),
        # so leg sag / height changes don't bias the balance point
        wl, wr = m.body("l_wheel").id, m.body("r_wheel").id
        axle = 0.5 * (d.xpos[wl] + d.xpos[wr])
        rel = d.subtree_com[1] - axle
        theta = np.arctan2(rel[0], rel[2])
        pitch_rate = d.qvel[4]
        # wheel joint speed is relative to the shank, which rotates with the body,
        # so absolute wheel spin = joint speed + body pitch rate (legs held fixed)
        w_joint = 0.5 * (d.qvel[m.jnt_dofadr[m.joint("l_wheel").id]] +
                         d.qvel[m.jnt_dofadr[m.joint("r_wheel").id]])
        x_dot = (w_joint + pitch_rate) * WHEEL_RADIUS
        x_axle = 0.5 * (d.xpos[m.body("l_wheel").id][0] + d.xpos[m.body("r_wheel").id][0])
        return np.array([x_axle - x_ref, theta, x_dot - v_ref, pitch_rate])

    def __call__(self, m, d, x_ref=0.0, v_ref=0.0):
        s = self.state(m, d, x_ref, v_ref)
        u = float(-(self.K @ s)[0])
        u = np.clip(u, -self.torque_limit, self.torque_limit)
        return 0.5 * u, 0.5 * u     # left, right wheel torque
