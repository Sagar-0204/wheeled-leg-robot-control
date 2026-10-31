"""
Look inside the trained policy -- companion to the study guide.

    python scripts/explore.py obs        # what the policy sees and does, live, with reward terms
    python scripts/explore.py reward     # which reward terms dominate over the 20 s demo
    python scripts/explore.py saliency   # which observations the policy's actions depend on most
    python scripts/explore.py ablate     # break one thing at a time and measure the fall rate

Options: --ckpt 21500000 (default) --run ppo_main --episodes 10
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from wheel_leg.env import WheelLegEnv, WHEEL_OMEGA_MAX  # noqa: E402
from play import Policy, resolve_paths  # noqa: E402

OBS_NAMES = (["grav_x", "grav_y", "grav_z", "gyro_x(roll)", "gyro_y(pitch)", "gyro_z(yaw)",
              "q_hip_L", "q_knee_L", "q_hip_R", "q_knee_R",
              "dq_hip_L", "dq_knee_L", "dq_hip_R", "dq_knee_R",
              "w_wheel_L", "w_wheel_R", "cmd_vx", "cmd_wz", "cmd_h"]
             + [f"prev_a{i}" for i in range(6)])
OBS_GROUPS = {"gravity (tilt)": range(0, 3), "gyro (angular rate)": range(3, 6),
              "leg joint angles": range(6, 10), "leg joint speeds": range(10, 14),
              "wheel speeds": range(14, 16), "command": range(16, 19), "previous action": range(19, 25)}
ACT_NAMES = ["hip_L target", "knee_L target", "hip_R target", "knee_R target",
             "wheel_L vel target", "wheel_R vel target"]


def scenario_events():
    from record import SCENARIO
    return SCENARIO


# ------------------------------------------------------------------ obs
def cmd_obs(policy, seconds=6.0):
    """Stand 2 s, drive forward, push at 4 s; print obs / action / reward every 0.5 s."""
    env = WheelLegEnv(randomize=False, pushes=False, command_mode="fixed")
    obs, _ = env.reset(seed=0, options=dict(q=0.6, pitch=0.0, yaw=0.0, vx=0.0))
    env.set_command(0.0, 0.0, 0.55)
    next_print = 0.0
    while env.data.time < seconds:
        if abs(env.data.time - 2.0) < 1e-6 or (env.data.time >= 2.0 and env.command[0] == 0.0 and env.data.time < 2.02):
            env.set_command(0.4, 0.0, 0.55)
        if 4.0 <= env.data.time < 4.02:
            env.push([80.0, 0.0])
        a = policy(obs)
        ctrl = env.action_to_ctrl(a)
        obs, r, term, _, info = env.step(a)
        if env.data.time >= next_print - 1e-9:
            next_print += 0.5
            _, _, vx, wz = env._base_state()
            print(f"\n=== t = {env.data.time:4.2f} s   cmd vx={env.command[0]:+.2f}  actual vx={vx:+.2f} m/s   "
                  f"height={env.data.qpos[2]:.3f} m   reward={r:+.3f}"
                  + ("   <-- 80 N push" if 4.0 <= env.data.time < 4.5 else ""))
            print("  observation: " + "  ".join(f"{n}={v:+.2f}" for n, v in zip(OBS_NAMES[:19], obs[:19])))
            print("  action -> motor targets: " + "  ".join(
                f"{n}={c:+.2f}{' rad' if i < 4 else ' rad/s'}" for i, (n, c) in enumerate(zip(ACT_NAMES, ctrl))))
            print("  reward terms: " + "  ".join(f"{k}={v:+.3f}" for k, v in info["reward_terms"].items()))
        if term:
            print("FELL"); break


# ------------------------------------------------------------------ reward
def cmd_reward(policy):
    env = WheelLegEnv(randomize=False, pushes=False, command_mode="fixed")
    obs, _ = env.reset(seed=0, options=dict(q=0.6, pitch=0.0, yaw=0.0, vx=0.0))
    ev, k, tot, n = scenario_events(), 0, {}, 0
    while env.data.time < 20.0:
        while k < len(ev) and env.data.time >= ev[k][0]:
            _, vx, wz, h, push = ev[k]
            env.set_command(vx, wz, h)
            if push:
                env.push([push, 0.0])
            k += 1
        obs, r, term, _, info = env.step(policy(obs))
        for key, v in info["reward_terms"].items():
            tot[key] = tot.get(key, 0.0) + v
        n += 1
        if term:
            print("FELL"); break
    total = sum(tot.values())
    print(f"20 s scripted demo, {n} policy steps, total reward {total:.1f} (max possible per step = 2.5)\n")
    print(f"{'term':12s} {'sum':>9s} {'per step':>9s}")
    for key, v in sorted(tot.items(), key=lambda kv: -abs(kv[1])):
        print(f"{key:12s} {v:9.2f} {v / n:9.4f}")


# ------------------------------------------------------------------ saliency
def collect_states(policy, n_eps=5, seconds=10.0):
    """States the policy actually visits: random commands, randomization, pushes."""
    states = []
    for s in range(n_eps):
        env = WheelLegEnv(randomize=True, pushes=True, command_mode="random")
        obs, _ = env.reset(seed=300 + s)
        while env.data.time < seconds:
            states.append(obs.copy())
            obs, _, term, _, _ = env.step(policy(obs))
            if term:
                break
    return np.array(states)


def cmd_saliency(policy):
    """Gradient of each action w.r.t. each NORMALIZED observation (= effect of a 1-std change),
    averaged over visited states. Answers: which inputs does the policy actually use?"""
    import torch
    S = collect_states(policy)
    o = np.clip((S - policy.mean) / np.sqrt(policy.var + policy.eps), -policy.clip, policy.clip)
    x = torch.tensor(o, dtype=torch.float32, requires_grad=True)
    pol = policy.model.policy
    mean_action = pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(x, pol.pi_features_extractor)))
    G = np.zeros((6, o.shape[1]))
    for j in range(6):
        g, = torch.autograd.grad(mean_action[:, j].sum(), x, retain_graph=True)
        G[j] = g.abs().mean(0).numpy()
    print(f"{len(S)} visited states. Mean |d action / d obs| per 1-std change of each input group, "
          f"as % of that action's total sensitivity:\n")
    head = f"{'input group':22s}" + "".join(f"{a.split()[0]:>10s}" for a in ACT_NAMES)
    print(head + "\n" + "-" * len(head))
    tot = G.sum(1, keepdims=True)
    for name, idx in OBS_GROUPS.items():
        row = G[:, list(idx)].sum(1, keepdims=True) / tot
        print(f"{name:22s}" + "".join(f"{100 * v:9.0f}%" for v in row[:, 0]))
    print("\nTop 8 single inputs for the wheel commands (what drives balance):")
    w = G[4:6].mean(0)
    for i in np.argsort(-w)[:8]:
        print(f"  {OBS_NAMES[i]:14s} {100 * w[i] / w.sum():5.1f}%")


# ------------------------------------------------------------------ ablations
class Wrapped:
    """Wraps the policy to break one thing: legs frozen, a sensor removed, extra noise, extra delay."""

    def __init__(self, policy, mode, env):
        self.p, self.mode, self.env, self.prev = policy, mode, env, None

    def __call__(self, obs):
        o = obs.copy()
        if self.mode == "no_gyro":
            o[3:6] = 0.0
        if self.mode == "no_wheel_speed":
            o[14:16] = 0.0
        if self.mode == "noise_x5":
            r = self.env.np_random
            o[0:3] += r.normal(0, 0.08, 3); o[3:6] += r.normal(0, 0.2, 3)
            o[6:10] += r.normal(0, 0.04, 4); o[10:14] += r.normal(0, 0.4, 4); o[14:16] += r.normal(0, 0.8, 2)
        a = self.p(o)
        if self.mode == "legs_frozen":
            q = np.arccos(np.clip((self.env.command[2] - 0.08) / 0.6, -1, 1))
            a = a.copy(); a[:4] = self.env.crouch_action(q)[:4]
        if self.mode == "delay_20ms":
            out = a if self.prev is None else self.prev
            self.prev = a
            return out
        return a


ABLATIONS = [
    ("full policy", "none", {}),
    ("legs frozen at nominal crouch", "legs_frozen", {}),
    ("gyro removed (zeros)", "no_gyro", {}),
    ("wheel speeds removed (zeros)", "no_wheel_speed", {}),
    ("sensor noise x5", "noise_x5", {}),
    ("extra 20 ms action delay", "delay_20ms", {}),
    ("ice: friction 0.3 (trained 0.5-1.2)", "none", {"friction": 0.3}),
    ("12 kg payload (trained 2-8 kg)", "none", {"payload_mass": 12.0}),
]


def cmd_ablate(policy, n_eps):
    from evaluate import random_push_events
    print(f"Standing at a random height with full randomization; random 80-120 N pushes from any\n"
          f"direction every 2-3 s; {n_eps} episodes x 20 s each. Same seeds for every row.\n")
    print(f"{'experiment':40s} {'falls':>7s}")
    for label, mode, opts in ABLATIONS:
        falls = 0
        for s in range(n_eps):
            rng = np.random.default_rng(1000 + s)
            h = float(rng.uniform(0.44, 0.63))
            ev = random_push_events(rng, 20.0, 80, 120)
            env = WheelLegEnv(randomize=True, pushes=False, command_mode="fixed")
            obs, _ = env.reset(seed=1000 + s, options=dict(q=float(np.arccos((h - 0.08) / 0.6)), pitch=0.0,
                                                           yaw=0.0, vx=0.0, **opts))
            env.set_command(0.0, 0.0, h)
            pol = Wrapped(policy, mode, env)
            k = 0
            while env.data.time < 20.0:
                while k < len(ev) and env.data.time >= ev[k][0]:
                    env.push(ev[k][2]); k += 1
                obs, _, term, _, _ = env.step(pol(obs))
                if term:
                    falls += 1
                    break
        print(f"{label:40s} {falls:>3d}/{n_eps}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["obs", "reward", "saliency", "ablate"])
    ap.add_argument("--run", default="ppo_main")
    ap.add_argument("--ckpt", default="21500000")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    policy = Policy(*resolve_paths(args.run, args.ckpt))
    {"obs": lambda: cmd_obs(policy), "reward": lambda: cmd_reward(policy),
     "saliency": lambda: cmd_saliency(policy), "ablate": lambda: cmd_ablate(policy, args.episodes)}[args.mode]()


if __name__ == "__main__":
    main()
