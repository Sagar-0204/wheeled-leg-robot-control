"""
Evaluate the trained RL policy and compare it with classical LQR balance controllers.

    python scripts/evaluate.py --select                 # pick the best checkpoint (fall rate, tracking)
    python scripts/evaluate.py --ckpt 21500000          # full RL-vs-LQR evaluation of one checkpoint
    python scripts/evaluate.py                          # same, using final_model

Controllers
    RL            PPO policy, 50 Hz, wheel velocity commands, sees noisy IMU/encoder observations
    LQR-nominal   LQR on the wheeled-inverted-pendulum model, designed once for the nominal robot
                  (4 kg payload, 0.55 m height). Runs at 500 Hz with PERFECT state (no noise).
    LQR-oracle    same, but redesigned for the TRUE payload and re-scheduled with leg height
                  every 50 ms -- a best-case classical controller that knows the plant exactly.
Both baselines use the same wheel motors (torque limit + torque-speed curve). Legs are held at
the commanded height by the leg servos, along a rate-limited (1 rad/s) crouch trajectory. The LQR controllers only balance and hold position;
they cannot turn or track height-dependent velocity commands, so tracking tests are RL-only.

Tests
    1 push x payload  : max recoverable push (0.1 s, forward and backward) for payloads 1-12 kg
                        (training range was 2-8 kg -> 1, 10, 12 kg test extrapolation)
    2 height + pushes : crouch/extend while being pushed (gain-scheduling stress test)
    3 randomized      : fall rate under full domain randomization + random pushes from any
                        direction every 2-3 s, for push bands 40-80 ... 160-200 N (all controllers)
    4 tracking        : 50 episodes, random velocity / yaw-rate / height commands (RL only)

Outputs: results/eval_summary.json, results/eval_results.md, results/eval_*.png
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

from wheel_leg.env import WheelLegEnv, WHEEL_RADIUS  # noqa: E402
from wheel_leg.baselines import LQRBalancer, quat_to_pitch  # noqa: E402
from play import Policy, resolve_paths  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
NOMINAL_PAYLOAD = 4.0
NOMINAL_H = 0.55


def height_to_q(h):
    """Crouch parameter for a pelvis height (hip=q, knee=-2q, thigh+shank = 0.6 m)."""
    return float(np.arccos(np.clip((h - WHEEL_RADIUS) / 0.6, -1.0, 1.0)))


# =============================================================================== controllers
class RLController:
    name = "RL"
    wheel_mode = "velocity"

    def __init__(self, policy):
        self.policy = policy

    def setup(self, env):
        pass

    def act(self, env, obs):
        return self.policy(obs)


class LQRController:
    """LQR balance + position hold on the wheels (500 Hz inner loop), legs hold commanded height."""
    wheel_mode = "torque"

    Q_RATE = 1.0   # rad/s, rate limit on the leg crouch trajectory (classical design would ramp)

    def __init__(self, oracle=False):
        self.oracle = oracle
        self.name = "LQR-oracle" if oracle else "LQR-nominal"

    def _design(self, env):
        m, d = env.model, env.data
        if self.oracle:
            self.lqr.design(m, d)
            return
        # design for the nominal plant: nominal payload, nominal height
        pm, pi = m.body_mass[env.payload_id], m.body_inertia[env.payload_id].copy()
        m.body_mass[env.payload_id] = NOMINAL_PAYLOAD
        m.body_inertia[env.payload_id] = env._nom["payload_inertia"] * NOMINAL_PAYLOAD / env._nom["payload_mass"]
        mujoco.mj_setConst(m, mujoco.MjData(m))   # scratch data: setConst overwrites qpos
        d0 = mujoco.MjData(m)
        d0.qpos[:] = d.qpos
        q = height_to_q(NOMINAL_H)
        d0.qpos[env.leg_qadr] = [q, -2 * q, q, -2 * q]
        self.lqr.design(m, d0)
        m.body_mass[env.payload_id], m.body_inertia[env.payload_id] = pm, pi
        mujoco.mj_setConst(m, mujoco.MjData(m))   # scratch data: setConst overwrites qpos

    def setup(self, env):
        self.lqr = LQRBalancer(env.model, env.data, env.model.opt.timestep)
        self._design(env)
        wl, wr = env.model.body("l_wheel").id, env.model.body("r_wheel").id
        self.x_ref = 0.5 * (env.data.xpos[wl][0] + env.data.xpos[wr][0])
        self.last_design = env.data.time
        self.q = height_to_q(env.command[2])
        env.inner_controller = lambda m, d: self.lqr(m, d, x_ref=self.x_ref, v_ref=env.command[0])

    def act(self, env, obs):
        self.x_ref += env.command[0] * env.dt
        if self.oracle and env.data.time - self.last_design >= 0.05:
            self.lqr.design(env.model, env.data)          # gain scheduling with leg height
            self.last_design = env.data.time
        dq = height_to_q(env.command[2]) - self.q
        self.q += float(np.clip(dq, -self.Q_RATE * env.dt, self.Q_RATE * env.dt))
        return env.crouch_action(self.q)


# =============================================================================== episode runner
def run_episode(ctrl, T, events=(), options=None, randomize=False, pushes=False,
                command_mode="fixed", seed=0, h0=NOMINAL_H, log=False, difficulty=1.0):
    """events: list of (t, kind, value). kind 'push' -> (Fx, Fy) world N for 0.1 s,
    kind 'cmd' -> (vx, wz, h)."""
    env = WheelLegEnv(randomize=randomize, pushes=pushes, command_mode=command_mode,
                      wheel_mode=ctrl.wheel_mode, max_episode_steps=10 ** 9)
    env.set_difficulty(difficulty)
    opts = dict(q=height_to_q(h0), pitch=0.0, yaw=0.0, vx=0.0)
    opts.update(options or {})
    if command_mode == "random":
        for k in ("q", "pitch", "yaw", "vx"):
            opts.pop(k)
    obs, _ = env.reset(seed=seed, options=opts)
    if command_mode == "fixed":
        env.set_command(0.0, 0.0, h0)
    ctrl.setup(env)

    events = sorted(events)
    k, fell, rec = 0, False, []
    x0 = env.data.qpos[0]
    err = np.zeros(3)
    n = 0
    while env.data.time < T - 1e-9:
        while k < len(events) and env.data.time >= events[k][0] - 1e-9:
            _, kind, val = events[k]
            if kind == "push":
                env.push(val, duration=0.1)
            else:
                env.set_command(*val)
            k += 1
        obs, _, term, _, _ = env.step(ctrl.act(env, obs))
        _, _, vx, wz = env._base_state()
        e = np.abs([vx - env.command[0], wz - env.command[1], env.data.qpos[2] - env.command[2]])
        err += e
        n += 1
        if log:
            rec.append([env.data.time, quat_to_pitch(env.data.qpos[3:7]), env.data.qpos[0] - x0,
                        env.data.qpos[2], vx, *env.command])
        if term:
            fell = True
            break
    env.close()
    out = dict(fell=fell, t_end=env.data.time, err_vx=err[0] / max(n, 1),
               err_wz=err[1] / max(n, 1), err_h=err[2] / max(n, 1))
    if log:
        out["log"] = np.array(rec)
    return out


# =============================================================================== tests
FORCES = list(range(40, 601, 40))
PAYLOADS = [1, 2, 4, 6, 8, 10, 12]


def max_push(ctrl, payload):
    """Largest push F (N, 0.1 s) such that every push <= F, forward AND backward, is survived."""
    best = 0
    for F in FORCES:
        ok = all(not run_episode(ctrl, 4.0, [(1.0, "push", (s * F, 0.0))],
                                 options=dict(payload_mass=payload))["fell"] for s in (1, -1))
        if not ok:
            break
        best = F
    return best


def test_push_payload(ctrls):
    res = {}
    for c in ctrls:
        res[c.name] = [max_push(c, p) for p in PAYLOADS]
        print(f"  {c.name:12s} max push [N] for payload {PAYLOADS} kg: {res[c.name]}")
    return res


HEIGHT_EVENTS = [(1.0, "cmd", (0.0, 0.0, 0.44)), (1.3, "push", (60.0, 0.0)),
                 (4.0, "cmd", (0.0, 0.0, 0.63)), (4.3, "push", (-60.0, 0.0)),
                 (7.0, "cmd", (0.0, 0.0, 0.50)), (7.3, "push", (80.0, 0.0))]


def test_height(ctrls):
    res = {}
    for c in ctrls:
        r = run_episode(c, 10.0, HEIGHT_EVENTS, options=dict(payload_mass=6.0), log=True)
        L = r["log"]
        res[c.name] = dict(fell=r["fell"], t_end=r["t_end"],
                           max_pitch_dev=float(np.abs(L[:, 1] - L[0, 1]).max()),
                           max_disp=float(np.abs(L[:, 2]).max()), log=L)
        print(f"  {c.name:12s} fell={r['fell']} max pitch dev={res[c.name]['max_pitch_dev']:.3f} rad "
              f"max disp={res[c.name]['max_disp']:.2f} m")
    return res


def random_push_events(rng, T, f_lo=80.0, f_hi=220.0):
    ev, t = [], float(rng.uniform(1.0, 2.0))
    while t < T - 1.0:
        F, ang = rng.uniform(f_lo, f_hi), rng.uniform(0, 2 * np.pi)
        ev.append((t, "push", (F * np.cos(ang), F * np.sin(ang))))
        t += float(rng.uniform(2.0, 3.0))
    return ev


PUSH_BANDS = [(40, 80), (80, 120), (120, 160), (160, 200)]


def test_randomized(ctrls, n_eps=30):
    """Full domain randomization (payload, friction, motors, latency, noise) while standing at a
    random height; random pushes from any direction every 2-3 s. Fall rate per push-strength band
    (identical randomization and push sequence for every controller)."""
    res = {c.name: [] for c in ctrls}
    for lo, hi in PUSH_BANDS:
        for c in ctrls:
            falls = 0
            for s in range(n_eps):
                rng = np.random.default_rng(1000 + s)
                h = float(rng.uniform(0.44, 0.63))
                ev = random_push_events(rng, 20.0, lo, hi)
                r = run_episode(c, 20.0, ev, randomize=True, pushes=False, seed=1000 + s, h0=h,
                                options=dict(yaw=0.0))
                falls += r["fell"]
            res[c.name].append(falls / n_eps)
        print(f"  pushes {lo}-{hi} N: " + "  ".join(f"{n} {100 * v[-1]:.0f}%" for n, v in res.items()), flush=True)
    return res


def test_tracking(policy, n_eps=50, seed0=2000):
    c = RLController(policy)
    rs = [run_episode(c, 20.0, randomize=True, pushes=True, command_mode="random", seed=seed0 + s)
          for s in range(n_eps)]
    out = dict(fall_rate=float(np.mean([r["fell"] for r in rs])),
               err_vx=float(np.mean([r["err_vx"] for r in rs])),
               err_wz=float(np.mean([r["err_wz"] for r in rs])),
               err_h=float(np.mean([r["err_h"] for r in rs])))
    return out


# =============================================================================== checkpoint selection
def select(run, n_eps):
    cdir = os.path.join(ROOT, "runs", run, "checkpoints")
    steps = sorted(int(f[4:-10]) for f in os.listdir(cdir) if f.startswith("ppo_") and f.endswith("_steps.zip"))
    steps = [s for s in steps if s >= 10_000_000]
    rows = []
    for s in steps:
        r = test_tracking(Policy(*resolve_paths(run, s)), n_eps=n_eps, seed0=5000)
        score = r["fall_rate"] * 10 + r["err_vx"] + 0.5 * r["err_wz"] + 10 * r["err_h"]
        rows.append((score, s, r))
        print(f"  {s:>9d}: falls {r['fall_rate']:.2f}  err vx {r['err_vx']:.3f} m/s  "
              f"wz {r['err_wz']:.3f} rad/s  h {100 * r['err_h']:.2f} cm  score {score:.3f}")
    best = min(rows)
    print(f"\nbest checkpoint: {best[1]}  (lower score = fewer falls, better tracking)")
    return best[1]


# =============================================================================== plots / report
def plot_push_payload(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"RL": "#2a6fdb", "LQR-nominal": "#d9822b", "LQR-oracle": "#7a7a7a"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axvspan(2, 8, color="#2a6fdb", alpha=0.07, label="RL training range")
    for name, v in res.items():
        ax.plot(PAYLOADS, v, "o-", color=colors[name], label=name, lw=2,
                ls="--" if name == "LQR-oracle" else "-")
    ax.set_xlabel("upper-body payload [kg]")
    ax.set_ylabel("max recoverable push [N, 0.1 s]")
    ax.set_title("Push recovery vs payload (forward and backward)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_height(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"RL": "#2a6fdb", "LQR-nominal": "#d9822b", "LQR-oracle": "#7a7a7a"}
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for name, r in res.items():
        L = r["log"]
        lab = name + (" (FELL)" if r["fell"] else "")
        ls = "--" if name == "LQR-oracle" else "-"
        ax[0].plot(L[:, 0], L[:, 3], color=colors[name], ls=ls, label=lab)
        ax[1].plot(L[:, 0], L[:, 1], color=colors[name], ls=ls)
        ax[2].plot(L[:, 0], L[:, 2], color=colors[name], ls=ls)
    L = next(iter(res.values()))["log"]
    ax[0].plot(L[:, 0], L[:, 7], ":", color="black", lw=1, label="height command")
    for t, kind, _ in HEIGHT_EVENTS:
        if kind == "push":
            for a in ax:
                a.axvline(t, color="#c03030", lw=1, alpha=0.5)
    ax[0].set_ylabel("pelvis height [m]"); ax[1].set_ylabel("pitch [rad]")
    ax[2].set_ylabel("displacement [m]"); ax[2].set_xlabel("time [s]  (red = 60-80 N push)")
    ax[0].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Height changes under pushes, 6 kg payload")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_randomized(rr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"RL": "#2a6fdb", "LQR-nominal": "#d9822b", "LQR-oracle": "#7a7a7a"}
    x = np.arange(len(PUSH_BANDS)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (n, v) in enumerate(rr.items()):
        ax.bar(x + (i - 1) * w, 100 * np.array(v), w, color=colors[n], label=n)
    ax.set_xticks(x, [f"{lo}-{hi} N" for lo, hi in PUSH_BANDS])
    ax.set_ylabel("episodes ending in a fall [%]"); ax.set_ylim(0, 100)
    ax.set_title("Fall rate under full randomization + random pushes (lower is better)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_report(summary, path):
    pp, hh, rr, tr = summary["push_payload"], summary["height"], summary["randomized"], summary["tracking"]
    names = list(pp)
    L = [f"# Evaluation: {summary['model']}", "",
         "## 1. Max recoverable push (N, 0.1 s, forward and backward)", "",
         "| payload [kg] | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i, p in enumerate(PAYLOADS):
        tag = "" if 2 <= p <= 8 else " *"
        L.append(f"| {p}{tag} | " + " | ".join(str(pp[n][i]) for n in names) + " |")
    L += ["", "\\* outside the RL training range (2-8 kg)", "",
          "## 2. Height changes under pushes (6 kg payload)", "",
          "| controller | fell | max pitch deviation [rad] | max displacement [m] |", "|---|---|---|---|"]
    for n, r in hh.items():
        L.append(f"| {n} | {'yes' if r['fell'] else 'no'} | {r['max_pitch_dev']:.3f} | {r['max_disp']:.2f} |")
    L += ["", "## 3. Fall rate: full domain randomization + random pushes from any direction every 2-3 s",
          "(standing at a random height, 30 episodes x 20 s per band)", "",
          "| push strength | " + " | ".join(rr) + " |", "|---|" + "---|" * len(rr)]
    for i, (lo, hi) in enumerate(PUSH_BANDS):
        L.append(f"| {lo}-{hi} N | " + " | ".join(f"{100 * v[i]:.0f}%" for v in rr.values()) + " |")
    L += ["", "## 4. RL command tracking (50 episodes x 20 s, full randomization + pushes)", "",
          f"- fall rate: {100 * tr['fall_rate']:.0f}%",
          f"- mean |velocity error|: {tr['err_vx']:.3f} m/s",
          f"- mean |yaw-rate error|: {tr['err_wz']:.3f} rad/s",
          f"- mean |height error|: {100 * tr['err_h']:.2f} cm", ""]
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_main")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--select", action="store_true", help="rank checkpoints >= 10M and exit")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.select:
        select(args.run, n_eps=min(args.episodes, 20))
        return

    t0 = time.time()
    policy = Policy(*resolve_paths(args.run, args.ckpt))
    model_name = f"{args.run}" + (f" @ {args.ckpt} steps" if args.ckpt else " (final)")
    ctrls = [RLController(policy), LQRController(oracle=False), LQRController(oracle=True)]

    print("1/4 push recovery vs payload");  pp = test_push_payload(ctrls)
    print("2/4 height changes under pushes"); hh = test_height(ctrls)
    print("3/4 randomized robustness");      rr = test_randomized(ctrls, 30)
    plot_randomized(rr, os.path.join(RESULTS, "eval_fall_rate.png"))
    print("4/4 RL command tracking");        tr = test_tracking(policy, args.episodes)
    print(f"  {tr}")

    plot_push_payload(pp, os.path.join(RESULTS, "eval_push_payload.png"))
    plot_height(hh, os.path.join(RESULTS, "eval_height_push.png"))
    summary = dict(model=model_name, push_payload=pp, payloads=PAYLOADS,
                   height={n: {k: v for k, v in r.items() if k != "log"} for n, r in hh.items()},
                   randomized=rr, tracking=tr)
    with open(os.path.join(RESULTS, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    write_report(summary, os.path.join(RESULTS, "eval_results.md"))
    print(f"\ndone in {time.time() - t0:.0f} s -> results/eval_results.md, eval_*.png")


if __name__ == "__main__":
    main()
