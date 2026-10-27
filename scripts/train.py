"""
Train a PPO policy for the wheel-legged lower body.

    python scripts/train.py                          # full run, 8 parallel envs, 30M steps
    python scripts/train.py --steps 2e6 --name test  # quick test
    python scripts/train.py --resume runs/ppo_main/checkpoints/ppo_5000000_steps.zip
    tensorboard --logdir runs                        # watch training curves

Outputs in runs/<name>/:  final_model.zip, vecnormalize.pkl, checkpoints/, tb/
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize  # noqa: E402

from wheel_leg.env import WheelLegEnv  # noqa: E402


def make_env(rank, seed, randomize, pushes):
    def _init():
        env = WheelLegEnv(randomize=randomize, pushes=pushes)
        env.reset(seed=seed + rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Ramps env difficulty (command ranges, push strength) from `start` to 1."""

    def __init__(self, ramp_steps, start=0.2):
        super().__init__()
        self.ramp_steps, self.start = ramp_steps, start

    def _difficulty(self):
        return self.start + (1 - self.start) * min(1.0, self.num_timesteps / self.ramp_steps)

    def _on_training_start(self):
        self.training_env.env_method("set_difficulty", self._difficulty())

    def _on_rollout_end(self):
        d = self._difficulty()
        self.training_env.env_method("set_difficulty", d)
        self.logger.record("metrics/difficulty", d)

    def _on_step(self):
        return True


class MetricsCallback(BaseCallback):
    """Logs tracking errors and fall rate to TensorBoard; saves VecNormalize with checkpoints."""

    def __init__(self, save_dir, save_freq):
        super().__init__()
        self.buf = []
        self.save_dir, self.save_freq = save_dir, save_freq

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode_metrics" in info:
                self.buf.append(info["episode_metrics"])
        if len(self.buf) >= 20:
            for k in self.buf[0]:
                self.logger.record(f"metrics/{k}", float(np.mean([b[k] for b in self.buf])))
            self.buf = []
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(os.path.join(self.save_dir, f"vecnormalize_{self.num_timesteps}.pkl"))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=30e6)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--name", default="ppo_main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-randomize", action="store_true", help="disable domain randomization")
    ap.add_argument("--no-pushes", action="store_true")
    ap.add_argument("--resume", default=None, help="path to a checkpoint .zip")
    ap.add_argument("--curriculum-steps", type=float, default=10e6,
                    help="steps over which difficulty ramps from 0.2 to 1")
    args = ap.parse_args()

    run_dir = os.path.join(ROOT, "runs", args.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.set_num_threads(1)   # MLP policy: CPU is faster than GPU, and threads fight the env workers

    fns = [make_env(i, args.seed, not args.no_randomize, not args.no_pushes) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv)

    if args.resume:
        vn_path = args.resume.replace("ppo_", "vecnormalize_").replace("_steps.zip", ".pkl")
        venv = VecNormalize.load(vn_path, venv)
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=os.path.join(run_dir, "tb"))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", venv,
            n_steps=1024, batch_size=2048, n_epochs=5,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=torch.nn.ELU, log_std_init=-2.0),
            tensorboard_log=os.path.join(run_dir, "tb"),
            seed=args.seed, device="cpu", verbose=1,
        )

    save_freq = max(500_000 // args.n_envs, 1)
    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir, name_prefix="ppo"),
        MetricsCallback(ckpt_dir, save_freq),
        CurriculumCallback(args.curriculum_steps),
    ]
    try:
        model.learn(total_timesteps=int(args.steps), callback=callbacks,
                    reset_num_timesteps=args.resume is None, progress_bar=False)
    finally:
        model.save(os.path.join(run_dir, "final_model"))
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
        print(f"saved to {run_dir}")
        venv.close()


if __name__ == "__main__":
    main()
