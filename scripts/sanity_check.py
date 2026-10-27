"""
Physics sanity checks for the wheel-legged lower body -- run BEFORE any RL.

If these fail, RL will fail too (and you won't know why). Checks:
  1. Model loads, mass and nominal height are as designed
  2. Passive fall: with wheels unpowered the robot must tip over (it's an inverted pendulum)
  3. LQR balance: starting tilted 0.1 rad, the classical controller must recover and hold
  4. Push recovery: 40 N shove for 0.1 s while balancing
  5. Height change: legs crouch/extend while LQR (designed at nominal height) balances
     -> tells us how robust the classical baseline is; RL should beat it here later

Usage:
    python scripts/sanity_check.py            # headless, saves plots to results/
    python scripts/sanity_check.py --view 3   # watch test 3 in the MuJoCo viewer
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wheel_leg.baselines import LQRBalancer, quat_to_pitch  # noqa: E402

MODEL = os.path.join(ROOT, "models", "wheel_leg.xml")
RESULTS = os.path.join(ROOT, "results")
FALL_PITCH = 0.8  # rad; beyond this we call it a fall


def leg_ctrl(q):
    """Leg servo targets for crouch parameter q (hip=q, knee=-2q keeps wheel under hip)."""
    return np.array([q, -2 * q, q, -2 * q])


def make_sim(tilt=0.0):
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("crouch").id)
    d.qpos[3:7] = [np.cos(tilt / 2), 0, np.sin(tilt / 2), 0]
    mujoco.mj_forward(m, d)
    return m, d


def simulate(m, d, controller, T, push=None, leg_schedule=None, viewer=None):
    """Run for T seconds. controller(m, d) -> (tau_l, tau_r) or None for unpowered wheels."""
    log = {"t": [], "pitch": [], "x": [], "z": [], "tau": []}
    pelvis = m.body("pelvis").id
    while d.time < T:
        if leg_schedule is not None:
            d.ctrl[:4] = leg_ctrl(leg_schedule(d.time))
        tau = (0.0, 0.0) if controller is None else controller(m, d)
        d.ctrl[4], d.ctrl[5] = tau

        d.xfrc_applied[pelvis, :] = 0
        if push is not None and push[0] <= d.time < push[0] + push[1]:
            d.xfrc_applied[pelvis, 0] = push[2]

        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()
            time.sleep(m.opt.timestep)

        p = quat_to_pitch(d.qpos[3:7])
        log["t"].append(d.time); log["pitch"].append(p)
        log["x"].append(d.qpos[0]); log["z"].append(d.qpos[2]); log["tau"].append(tau[0] + tau[1])
        if abs(p) > FALL_PITCH and viewer is None:
            break
    return {k: np.array(v) for k, v in log.items()}


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    return ok


def run_all(view=None):
    results, logs = [], {}

    # ---- 1. model
    m, d = make_sim()
    mass = m.body_subtreemass[m.body("pelvis").id]
    results.append(report("1 model", m.nu == 6 and abs(mass - 16.0) < 1e-6,
                          f"nq={m.nq} nv={m.nv} nu={m.nu} mass={mass:.1f} kg, "
                          f"pelvis z={d.qpos[2]:.3f} m"))

    def viewer_for(i):
        if view != i:
            return None
        import mujoco.viewer
        return mujoco.viewer.launch_passive(*_md)

    # ---- 2. passive fall
    m, d = make_sim(tilt=0.02); _md = (m, d)
    log = simulate(m, d, None, 3.0, viewer=viewer_for(2))
    fell = np.abs(log["pitch"]).max() > FALL_PITCH
    t_fall = log["t"][-1]
    results.append(report("2 passive fall", fell, f"tipped over after {t_fall:.2f} s (expected: falls)"))
    logs["passive"] = log

    # ---- 3. LQR balance from 0.1 rad
    m, d = make_sim(tilt=0.1); _md = (m, d)
    lqr = LQRBalancer(m, d, m.opt.timestep)
    log = simulate(m, d, lqr, 10.0, viewer=viewer_for(3))
    tail = log["t"] > 8
    ok = log["t"][-1] >= 9.99 and np.abs(log["pitch"][tail] - log["pitch"][tail].mean()).max() < 0.01
    results.append(report("3 LQR balance", ok,
                          f"survived {log['t'][-1]:.1f} s, final |x|={abs(log['x'][-1]):.3f} m, "
                          f"steady pitch={log['pitch'][tail].mean():+.3f} rad"))
    logs["lqr"] = log
    print("    WIP params:", {k: round(float(v), 4) for k, v in lqr.params.items()})
    print("    LQR gains K:", np.round(lqr.K, 2))

    # ---- 4. push recovery
    m, d = make_sim(); _md = (m, d)
    lqr = LQRBalancer(m, d, m.opt.timestep)
    log = simulate(m, d, lqr, 8.0, push=(3.0, 0.1, 40.0), viewer=viewer_for(4))
    ok = log["t"][-1] >= 7.99
    results.append(report("4 push recovery (40 N, 0.1 s)", ok,
                          f"max pitch {np.abs(log['pitch']).max():.3f} rad, "
                          f"max displacement {np.abs(log['x']).max():.3f} m"))
    logs["push"] = log

    # ---- 5. height change with fixed (nominal) LQR gains
    m, d = make_sim(); _md = (m, d)
    lqr = LQRBalancer(m, d, m.opt.timestep)
    # 0.5 -> 1.0 (deep crouch) -> 0.2 (tall) -> 0.5, smooth ramps
    sched = lambda t: float(np.interp(t, [0, 2, 4, 6, 8, 10, 12], [0.5, 0.5, 1.0, 1.0, 0.2, 0.2, 0.5]))
    log = simulate(m, d, lqr, 12.0, leg_schedule=sched, viewer=viewer_for(5))
    ok = log["t"][-1] >= 11.99
    results.append(report("5 height change (nominal LQR)", ok,
                          f"pelvis z range {log['z'].min():.2f}-{log['z'].max():.2f} m, "
                          f"max pitch {np.abs(log['pitch']).max():.3f} rad"))
    logs["height"] = log

    plot(logs)
    print(f"\n{sum(results)}/{len(results)} checks passed. Plots: {RESULTS}/sanity_check.png")
    return all(results)


def plot(logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(RESULTS, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    titles = {"passive": "2. Passive (no wheel torque)", "lqr": "3. LQR balance from 0.1 rad",
              "push": "4. Push recovery (40 N at t=3 s)", "height": "5. Height change, nominal LQR"}
    for a, (k, title) in zip(ax.flat, titles.items()):
        L = logs[k]
        a.plot(L["t"], L["pitch"], label="pitch [rad]", color="#2a6fdb")
        a.plot(L["t"], L["x"], label="x [m]", color="#d9822b")
        if k == "height":
            a.plot(L["t"], L["z"], label="pelvis z [m]", color="#3a9d5d")
        a.axhline(0, color="gray", lw=0.5)
        a.set_title(title, fontsize=10); a.set_xlabel("time [s]"); a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sanity_check.png"), dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=int, default=None, help="open the viewer for test N (2-5)")
    args = ap.parse_args()
    sys.exit(0 if run_all(args.view) else 1)
