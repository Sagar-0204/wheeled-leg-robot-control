"""
Drive the trained policy interactively in the MuJoCo viewer.

    python scripts/play.py                                   # runs/ppo_main/final_model.zip
    python scripts/play.py --run ppo_main --ckpt 10000000    # a specific checkpoint
    python scripts/play.py --lqr                             # classical LQR baseline instead

Keys (click the viewer window first):
    Up / Down       forward velocity +/- 0.1 m/s
    Left / Right    yaw rate +/- 0.3 rad/s
    Page Up / Down  height +/- 0.03 m
    P               push the robot (60 N, random direction)
    Space           zero velocity commands
    R               reset
"""
import argparse
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from wheel_leg.env import WheelLegEnv  # noqa: E402

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PGUP, KEY_PGDN, KEY_SPACE = 266, 267, 32


class Policy:
    """Trained PPO policy with the VecNormalize observation statistics baked in
    (no SB3 VecEnv needed -> same code path you'd use on a real robot / ROS 2 node)."""

    def __init__(self, model_path, vecnorm_path):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        self.mean, self.var = vn.obs_rms.mean, vn.obs_rms.var
        self.clip, self.eps = vn.clip_obs, vn.epsilon

    def __call__(self, obs):
        o = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        a, _ = self.model.predict(o.astype(np.float32), deterministic=True)
        return a


def resolve_paths(run, ckpt):
    run_dir = os.path.join(ROOT, "runs", run)
    if ckpt is None:
        return os.path.join(run_dir, "final_model.zip"), os.path.join(run_dir, "vecnormalize.pkl")
    c = os.path.join(run_dir, "checkpoints")
    return os.path.join(c, f"ppo_{ckpt}_steps.zip"), os.path.join(c, f"vecnormalize_{ckpt}.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_main")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lqr", action="store_true", help="run the LQR baseline (balance + velocity only)")
    ap.add_argument("--randomize", action="store_true", help="enable domain randomization")
    args = ap.parse_args()

    env = WheelLegEnv(randomize=args.randomize, pushes=False, command_mode="fixed",
                      wheel_mode="torque" if args.lqr else "velocity")
    obs, _ = env.reset(seed=0)
    env.set_command(0.0, 0.0, 0.55)

    if args.lqr:
        from wheel_leg.baselines import LQRBalancer
        lqr = LQRBalancer(env.model, env.data, env.model.opt.timestep)
        x_ref = [env.data.qpos[0]]

        def act(_obs):
            # hold the commanded height with the legs, balance + track vx with LQR
            from wheel_leg.env import H_RANGE  # noqa
            q = np.arccos(np.clip((env.command[2] - 0.08) / 0.6, -1, 1))
            a = env.crouch_action(q)
            x_ref[0] += env.command[0] * env.dt
            tl, tr = lqr(env.model, env.data, x_ref=x_ref[0], v_ref=env.command[0])
            a[4], a[5] = tl / 8.0, tr / 8.0
            return a
    else:
        policy = Policy(*resolve_paths(args.run, args.ckpt))
        act = policy

    pending = []

    def on_key(k):
        pending.append(k)

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.pelvis_id
        viewer.cam.distance = 2.5
        last_print = 0.0
        while viewer.is_running():
            t0 = time.time()
            while pending:
                k = pending.pop(0)
                vx, wz, h = env.command
                if k == KEY_UP: env.set_command(vx=vx + 0.1)
                elif k == KEY_DOWN: env.set_command(vx=vx - 0.1)
                elif k == KEY_LEFT: env.set_command(wz=wz + 0.3)
                elif k == KEY_RIGHT: env.set_command(wz=wz - 0.3)
                elif k == KEY_PGUP: env.set_command(h=h + 0.03)
                elif k == KEY_PGDN: env.set_command(h=h - 0.03)
                elif k == KEY_SPACE: env.set_command(vx=0.0, wz=0.0)
                elif k == ord("P"):
                    ang = np.random.uniform(0, 2 * np.pi)
                    env.push([60 * np.cos(ang), 60 * np.sin(ang)])
                elif k == ord("R"):
                    cmd = env.command.copy()
                    obs, _ = env.reset()
                    env.command[:] = cmd
                    if args.lqr:
                        lqr.design(env.model, env.data); x_ref[0] = env.data.qpos[0]

            obs = env._get_obs()
            obs, _, term, _, _ = env.step(act(obs))
            if term:
                print("fell -- press R to reset")
            viewer.sync()

            if time.time() - last_print > 0.5:
                _, _, vx, wz = env._base_state()
                c = env.command
                print(f"cmd vx={c[0]:+.2f} wz={c[1]:+.2f} h={c[2]:.2f} | "
                      f"actual vx={vx:+.2f} wz={wz:+.2f} h={env.data.qpos[2]:.2f}", end="\r")
                last_print = time.time()
            time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
