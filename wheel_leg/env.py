"""
Gymnasium environment: wheel-legged humanoid lower body (MuJoCo).

Task: balance on two wheels while tracking a command
      [forward velocity vx (m/s), yaw rate wz (rad/s), pelvis height h (m)].

Policy runs at 50 Hz (frame_skip=10 x 2 ms physics).

Observation (25), all from sensors a real robot has (IMU + joint encoders):
    projected gravity in pelvis frame (3)   -> tilt
    pelvis angular velocity (gyro) (3)
    leg joint positions (4)  [l_hip, l_knee, r_hip, r_knee]
    leg joint velocities (4)
    wheel joint velocities (2)
    command [vx, wz, h] (3)
    previous action (6)

Action (6) in [-1, 1]:
    [0:4] leg joint position targets (PD servos in the sim)  hip = 0.5 + 0.8 a, knee = -1.0 + 1.4 a
    [4:6] wheel commands: velocity targets +-30 rad/s (wheel_mode="velocity", default; a 500 Hz
          motor velocity loop turns them into torque) or raw torques +-8 N m (wheel_mode="torque")

Domain randomization (randomize=True), resampled every episode:
    payload (upper body) mass 2-8 kg and height, ground friction, wheel motor strength,
    leg servo stiffness, joint damping, actuation latency (0-10 ms), sensor noise.
External pushes (pushes=True): random horizontal shoves on the pelvis every 2-5 s.

Wheel motors follow a linear torque-speed curve (no torque at WHEEL_OMEGA_MAX), like a real
DC motor, so the policy can't "solve" balance by accelerating indefinitely.

Curriculum: set_difficulty(d in [0,1]) scales command ranges and push strength; the
training script ramps it from 0.2 to 1 over the first part of training.
"""
import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "wheel_leg.xml")

LEG_JOINTS = ["l_hip", "l_knee", "r_hip", "r_knee"]
WHEEL_JOINTS = ["l_wheel", "r_wheel"]
WHEEL_RADIUS = 0.08
HIP_C, HIP_S = 0.5, 0.8        # hip target = HIP_C + HIP_S * a
KNEE_C, KNEE_S = -1.0, 1.4     # knee target = KNEE_C + KNEE_S * a
WHEEL_TAU = 8.0
WHEEL_OMEGA_MAX = 30.0         # rad/s, no-load speed of the wheel motor (≈2.4 m/s)
WHEEL_KD = 0.8                 # N m per rad/s, wheel velocity loop gain (velocity mode)

# command ranges
VX_RANGE = (-0.8, 0.8)
WZ_RANGE = (-1.5, 1.5)
H_RANGE = (0.42, 0.64)


def crouch_to_height(q):
    """Approximate pelvis height for crouch parameter q (hip=q, knee=-2q)."""
    return 0.6 * np.cos(q) + WHEEL_RADIUS


class WheelLegEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, randomize=True, pushes=True, command_mode="random",
                 frame_skip=10, max_episode_steps=1000, render_mode=None, wheel_mode="velocity"):
        super().__init__()
        assert wheel_mode in ("velocity", "torque")
        self.wheel_mode = wheel_mode
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.randomize = randomize
        self.pushes = pushes
        self.command_mode = command_mode          # "random" | "fixed" (set via set_command)
        self.frame_skip = frame_skip
        self.dt = frame_skip * self.model.opt.timestep
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None

        m = self.model
        self.pelvis_id = m.body("pelvis").id
        self.payload_id = m.body("upper_body").id
        self.leg_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in LEG_JOINTS])
        self.leg_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in LEG_JOINTS])
        self.wheel_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in WHEEL_JOINTS])
        self.leg_dofs_all = np.array([m.jnt_dofadr[m.joint(j).id] for j in LEG_JOINTS + WHEEL_JOINTS])
        self.leg_act = np.array([m.actuator(a).id for a in
                                 ["l_hip_act", "l_knee_act", "r_hip_act", "r_knee_act"]])
        self.wheel_act = np.array([m.actuator(a).id for a in ["l_wheel_act", "r_wheel_act"]])
        self.ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[:, 1].copy()

        # nominal values for domain randomization
        self._nom = dict(
            payload_mass=m.body_mass[self.payload_id],
            payload_inertia=m.body_inertia[self.payload_id].copy(),
            payload_pos=m.body_pos[self.payload_id].copy(),
            friction=m.geom_friction.copy(),
            gainprm=m.actuator_gainprm.copy(),
            biasprm=m.actuator_biasprm.copy(),
            gear=m.actuator_gear.copy(),
            damping=m.dof_damping.copy(),
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)

        self.command = np.array([0.0, 0.0, 0.55])
        self.prev_action = np.zeros(6)
        self.steps = 0
        self.delay_substeps = 0
        self.push_force = np.zeros(3)
        self.next_push_t = np.inf
        self.push_end_t = -1.0
        self.next_cmd_t = np.inf
        self.ep_info = {}
        self.difficulty = 1.0
        self.inner_controller = None   # torque mode only: callable(model, data) -> (tau_l, tau_r)

    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))

    # ------------------------------------------------------------------ helpers
    def action_to_ctrl(self, a):
        ctrl = np.empty(6)
        ctrl[[0, 2]] = HIP_C + HIP_S * a[[0, 2]]
        ctrl[[1, 3]] = KNEE_C + KNEE_S * a[[1, 3]]
        ctrl[4:6] = (WHEEL_OMEGA_MAX if self.wheel_mode == "velocity" else WHEEL_TAU) * a[4:6]
        ctrl[:4] = np.clip(ctrl[:4], self.ctrl_lo[:4], self.ctrl_hi[:4])
        return ctrl

    @staticmethod
    def crouch_action(q):
        """Action that holds crouch q with zero wheel torque."""
        ah, ak = (q - HIP_C) / HIP_S, (-2 * q - KNEE_C) / KNEE_S
        return np.array([ah, ak, ah, ak, 0.0, 0.0])

    def set_command(self, vx=None, wz=None, h=None):
        if vx is not None: self.command[0] = np.clip(vx, *VX_RANGE)
        if wz is not None: self.command[1] = np.clip(wz, *WZ_RANGE)
        if h is not None: self.command[2] = np.clip(h, *H_RANGE)

    def _sample_command(self):
        r, s = self.np_random, self.difficulty
        h_mid, h_half = np.mean(H_RANGE), 0.5 * (H_RANGE[1] - H_RANGE[0])
        h = r.uniform(h_mid - s * h_half, h_mid + s * h_half)
        if r.random() < 0.2:                       # stand still
            self.command[:] = [0.0, 0.0, h]
        else:
            vx = s * r.uniform(*VX_RANGE) if r.random() < 0.8 else 0.0
            wz = s * r.uniform(*WZ_RANGE) if r.random() < 0.6 else 0.0
            self.command[:] = [vx, wz, h]
        self.next_cmd_t = self.data.time + r.uniform(3.0, 6.0)

    def _randomize_model(self):
        m, n, r = self.model, self._nom, self.np_random
        if not self.randomize:
            m.body_mass[self.payload_id] = n["payload_mass"]
            m.body_inertia[self.payload_id] = n["payload_inertia"]
            m.body_pos[self.payload_id] = n["payload_pos"]
            m.geom_friction[:] = n["friction"]
            m.actuator_gainprm[:] = n["gainprm"]
            m.actuator_biasprm[:] = n["biasprm"]
            m.actuator_gear[:] = n["gear"]
            m.dof_damping[:] = n["damping"]
            self.delay_substeps = 0
        else:
            mass = r.uniform(2.0, 8.0)
            m.body_mass[self.payload_id] = mass
            m.body_inertia[self.payload_id] = n["payload_inertia"] * mass / n["payload_mass"]
            m.body_pos[self.payload_id] = n["payload_pos"] + [r.uniform(-0.03, 0.03), 0, r.uniform(-0.05, 0.1)]
            mu = r.uniform(0.5, 1.2)
            m.geom_friction[:, 0] = mu
            kp_scale = r.uniform(0.7, 1.3)
            m.actuator_gainprm[:] = n["gainprm"]
            m.actuator_biasprm[:] = n["biasprm"]
            m.actuator_gainprm[self.leg_act, 0] *= kp_scale
            m.actuator_biasprm[self.leg_act, 1] *= kp_scale
            m.actuator_gear[:] = n["gear"]
            m.actuator_gear[self.wheel_act, 0] *= r.uniform(0.8, 1.2, size=2)
            m.dof_damping[:] = n["damping"] * r.uniform(0.5, 2.0)
            self.delay_substeps = int(r.integers(0, self.frame_skip // 2 + 1))   # 0-10 ms
        mujoco.mj_setConst(m, self.data)

    def _schedule_push(self):
        if self.pushes:
            self.next_push_t = self.data.time + self.np_random.uniform(2.0, 5.0)
        else:
            self.next_push_t = np.inf

    def push(self, force_xy, duration=0.1):
        """Apply a horizontal push (world frame, N) to the pelvis. Used by play/eval too."""
        self.push_force = np.array([force_xy[0], force_xy[1], 0.0])
        self.push_end_t = self.data.time + duration

    # ------------------------------------------------------------------ state
    def _rot(self):
        return self.data.xmat[self.pelvis_id].reshape(3, 3)

    def _base_state(self):
        R = self._rot()
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = self.data.qvel[3:6].copy()                 # local frame (freejoint)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        v = self.data.qvel[0:3]
        vx = np.cos(yaw) * v[0] + np.sin(yaw) * v[1]      # forward velocity, heading frame
        wz = (R @ gyro)[2]                                # world yaw rate
        return grav, gyro, vx, wz

    def _get_obs(self):
        grav, gyro, _, _ = self._base_state()
        jp = self.data.qpos[self.leg_qadr].copy()
        jv = self.data.qvel[self.leg_dadr].copy()
        wv = self.data.qvel[self.wheel_dadr].copy()
        if self.randomize:
            r = self.np_random
            grav += r.normal(0, 0.02, 3)
            gyro += r.normal(0, 0.05, 3)
            jp += r.normal(0, 0.01, 4)
            jv += r.normal(0, 0.1, 4)
            wv += r.normal(0, 0.2, 2)
        return np.concatenate([grav, gyro, jp, jv, wv, self.command, self.prev_action]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        """options (evaluation only, all optional): q (crouch), pitch, yaw, vx,
        payload_mass (kg), friction -- override the random/nominal values."""
        super().reset(seed=seed)
        m, d, r = self.model, self.data, self.np_random
        o = options or {}
        mujoco.mj_resetData(m, d)
        self._randomize_model()
        if "payload_mass" in o:
            pm = float(o["payload_mass"])
            m.body_mass[self.payload_id] = pm
            m.body_inertia[self.payload_id] = self._nom["payload_inertia"] * pm / self._nom["payload_mass"]
        if "friction" in o:
            m.geom_friction[:, 0] = float(o["friction"])
        mujoco.mj_setConst(m, d)

        q = o.get("q", r.uniform(0.3, 0.9))
        pitch0 = o.get("pitch", r.uniform(-0.05, 0.05))
        yaw0 = o.get("yaw", r.uniform(-np.pi, np.pi))
        # quaternion = yaw(z) * pitch(y)
        cy, sy, cp, sp = np.cos(yaw0 / 2), np.sin(yaw0 / 2), np.cos(pitch0 / 2), np.sin(pitch0 / 2)
        d.qpos[3:7] = [cy * cp, -sy * sp, cy * sp, sy * cp]
        d.qpos[self.leg_qadr] = [q, -2 * q, q, -2 * q]
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        wheel_z = min(d.xpos[m.body("l_wheel").id][2], d.xpos[m.body("r_wheel").id][2])
        d.qpos[2] -= wheel_z - WHEEL_RADIUS - 0.001
        d.qvel[0] = o.get("vx", r.uniform(-0.1, 0.1))
        mujoco.mj_forward(m, d)

        self.prev_action = self.crouch_action(q)
        d.ctrl[:] = 0.0
        d.ctrl[:4] = self.action_to_ctrl(self.prev_action)[:4]
        self._wheel_cmd = np.zeros(2)
        self.steps = 0
        self.push_end_t = -1.0
        self._schedule_push()
        if self.command_mode == "random":
            self._sample_command()
        else:
            self.next_cmd_t = np.inf
        self.ep_info = dict(err_vx=0.0, err_wz=0.0, err_h=0.0)
        return self._get_obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ctrl = self.action_to_ctrl(a)

        # random pushes
        if d.time >= self.next_push_t:
            ang = self.np_random.uniform(0, 2 * np.pi)
            mag = self.difficulty * self.np_random.uniform(20.0, 70.0)
            self.push([mag * np.cos(ang), mag * np.sin(ang)], duration=0.1)
            self._schedule_push()

        tau_sum = np.zeros(2)
        for i in range(self.frame_skip):
            if i == self.delay_substeps:
                d.ctrl[:4] = ctrl[:4]                  # actuation latency
                self._wheel_cmd = ctrl[4:6].copy()
            w = d.qvel[self.wheel_dadr]
            if self.wheel_mode == "velocity":          # motor driver velocity loop (500 Hz)
                tau = WHEEL_KD * (self._wheel_cmd - w)
            elif self.inner_controller is not None:    # e.g. LQR running at 500 Hz (evaluation)
                tau = np.asarray(self.inner_controller(m, d))
            else:
                tau = self._wheel_cmd
            # DC-motor torque-speed curve: available torque falls linearly with speed
            hi = WHEEL_TAU * np.clip(1.0 - w / WHEEL_OMEGA_MAX, 0.0, 1.0)
            lo = -WHEEL_TAU * np.clip(1.0 + w / WHEEL_OMEGA_MAX, 0.0, 1.0)
            d.ctrl[self.wheel_act] = np.clip(tau, lo, hi)
            tau_sum += d.ctrl[self.wheel_act]
            d.xfrc_applied[self.pelvis_id, :3] = self.push_force if d.time < self.push_end_t else 0.0
            mujoco.mj_step(m, d)
        self.steps += 1

        if self.command_mode == "random" and d.time >= self.next_cmd_t:
            self._sample_command()

        # ---------------- reward
        grav, gyro, vx, wz = self._base_state()
        h = d.qpos[2]
        e_vx, e_wz, e_h = vx - self.command[0], wz - self.command[1], h - self.command[2]
        jp = d.qpos[self.leg_qadr]
        tau = tau_sum / (self.frame_skip * WHEEL_TAU)

        terms = dict(
            track_vx=1.0 * np.exp(-e_vx ** 2 / 0.05),
            track_wz=0.5 * np.exp(-e_wz ** 2 / 0.25),
            track_h=0.5 * np.exp(-e_h ** 2 / 0.002),
            upright=0.5 * np.exp(-(grav[0] ** 2 + grav[1] ** 2) / 0.02),
            roll=-2.0 * grav[1] ** 2,
            pitch=-0.5 * grav[0] ** 2,
            ang_vel=-0.02 * (gyro[0] ** 2 + gyro[1] ** 2),
            action_rate=-0.02 * np.sum((a - self.prev_action) ** 2),
            torque=-0.01 * np.sum(tau ** 2),
            leg_sym=-0.5 * ((jp[0] - jp[2]) ** 2 + (jp[1] - jp[3]) ** 2),
        )
        reward = float(sum(terms.values()))

        tilt = np.arccos(np.clip(-grav[2], -1, 1))
        terminated = bool(tilt > 0.8 or h < 0.25)
        if terminated:
            reward -= 10.0
        truncated = self.steps >= self.max_episode_steps

        self.ep_info["err_vx"] += abs(e_vx)
        self.ep_info["err_wz"] += abs(e_wz)
        self.ep_info["err_h"] += abs(e_h)
        info = {"reward_terms": terms}
        if terminated or truncated:
            n = max(self.steps, 1)
            info["episode_metrics"] = {k: v / n for k, v in self.ep_info.items()} | {"fell": terminated}

        self.prev_action = a
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._cam.trackbodyid = self.pelvis_id
                self._cam.distance, self._cam.elevation, self._cam.azimuth = 2.5, -15, 120
            self._renderer.update_scene(self.data, self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
