"""
Record a scripted demo of a trained policy to MP4/GIF (for the README / interviews).

    python scripts/record.py --run ppo_main                  # -> results/demo_ppo_main.mp4
    python scripts/record.py --run ppo_main --gif            # also a GIF
    python scripts/record.py --run ppo_main --ckpt 5000000

Scripted scenario (20 s): stand -> drive forward -> turn -> crouch -> push -> stand tall -> reverse
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from wheel_leg.env import WheelLegEnv  # noqa: E402
from play import Policy, resolve_paths  # noqa: E402

# (time, vx, wz, h, push_N)
SCENARIO = [
    (0.0, 0.0, 0.0, 0.55, 0),
    (2.0, 0.5, 0.0, 0.55, 0),
    (5.0, 0.5, 1.0, 0.55, 0),
    (8.0, 0.0, 0.0, 0.44, 0),
    (10.0, 0.0, 0.0, 0.44, 60),
    (12.0, 0.0, 0.0, 0.63, 0),
    (14.0, -0.4, 0.0, 0.63, 0),
    (17.0, 0.0, 0.0, 0.55, 60),
]


def run(policy, T=20.0, seed=0, randomize=False, frames=True):
    env = WheelLegEnv(randomize=randomize, pushes=False, command_mode="fixed",
                      render_mode="rgb_array" if frames else None)
    obs, _ = env.reset(seed=seed)
    imgs, log, k = [], [], 0
    fell = False
    while env.data.time < T:
        while k < len(SCENARIO) and env.data.time >= SCENARIO[k][0]:
            _, vx, wz, h, push = SCENARIO[k]
            env.set_command(vx, wz, h)
            if push:
                env.push([push, 0.0])
            k += 1
        obs, _, term, _, _ = env.step(policy(obs))
        _, _, vx, wz = env._base_state()
        log.append([env.data.time, *env.command, vx, wz, env.data.qpos[2]])
        if frames:
            imgs.append(env.render())
        if term:
            fell = True
            break
    env.close()
    return np.array(log), imgs, fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_main")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    import imageio.v2 as imageio
    policy = Policy(*resolve_paths(args.run, args.ckpt))
    log, imgs, fell = run(policy)
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    tag = args.run + (f"_{args.ckpt}" if args.ckpt else "")
    imageio.mimsave(os.path.join(out, f"demo_{tag}.mp4"), imgs, fps=50)
    if args.gif:
        imageio.mimsave(os.path.join(out, f"demo_{tag}.gif"), imgs[::3], duration=0.06, loop=0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    t = log[:, 0]
    for a, (ci, ai, name) in zip(ax, [(1, 4, "forward velocity [m/s]"),
                                      (2, 5, "yaw rate [rad/s]"),
                                      (3, 6, "pelvis height [m]")]):
        a.plot(t, log[:, ci], "--", color="gray", label="command")
        a.plot(t, log[:, ai], color="#2a6fdb", label="actual")
        a.set_ylabel(name); a.grid(alpha=0.3)
    for ts, *_, p in SCENARIO:
        if p:
            for a in ax:
                a.axvline(ts, color="#d9822b", lw=1, alpha=0.6)
    ax[0].legend(); ax[-1].set_xlabel("time [s]  (orange lines = 60 N push)")
    fig.suptitle(f"Command tracking: {tag}" + ("  (FELL)" if fell else ""))
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"tracking_{tag}.png"), dpi=120)
    print(f"{'FELL' if fell else 'completed'}; saved results/demo_{tag}.mp4 and tracking_{tag}.png")


if __name__ == "__main__":
    main()
