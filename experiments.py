"""Improved-experiment driver for the SCI submission.

Extends the single-run reconstruction with everything a reviewer expects:

  multiseed    N-seed replication: every metric as mean +/- std, per-seed OPE,
               exact paired permutation tests (Proposed AI vs each baseline).
  sensitivity  success-rate vs the (implicit in the paper) permissible-deviation
               threshold: recomputed from the same deviation sequences, so the
               ranking's dependence on the hidden parameter is explicit.
  ablation     one-at-a-time sweeps of the AI controller's own hyper-parameters
               (LSTM hidden size / prediction horizon / fusion weight beta).
  wind         robustness under mean wind + turbulence (discrete Dryden-lite).
  cost         per-step control-loop latency of every method (embedded budget).

Run:  py experiments.py multiseed 0 1 2 3 4 5 6 7 8 9
      py experiments.py sensitivity 0 1 2 3 4
      py experiments.py ablation
      py experiments.py wind
      py experiments.py cost
Outputs land in out/ (csv + markdown tables). No file is written by importing.
"""
from __future__ import annotations

import sys, os, time, itertools
import numpy as np
import pandas as pd

from drone_sim.lstm import LSTM, make_delta_windows
from drone_sim.methods import ALL_METHODS, PurePursuit
from drone_sim.simulate import simulate, batch_ope
from drone_sim.trajectories import Path
from drone_sim.avoidance import Obstacle
from reproduce import build_paths, build_obstacles

OUT = os.path.join(os.path.dirname(__file__), "out")
METRICS = ("mean_dev", "adaptation", "n_evade", "min_dist", "mean_energy", "success")


# ------------------------------------------------------------------ shared
def make_ai(lstm, **over):
    """ProposedAI with the baseline configuration overridable for ablation."""
    from drone_sim.methods import ProposedAI
    cfg = dict(window=10, horizon=4, lookahead=25.0, beta=0.35)
    cfg.update(over)
    return ProposedAI(lstm, **cfg)


def instantiate(M, lstm):
    if M.__name__ == "ProposedAI":
        return make_ai(lstm)
    if M.__name__ == "PurePursuit":
        return M(30.0)
    if M.__name__ == "LineOfSight":
        return M(30.0, 0.5)
    if M.__name__ == "VectorField":
        return M(0.5, 6.0)
    return M(0.06)


def train_for(path, obstacles, seed, n_hidden=16):
    """Seeded LSTM training (identical protocol to reproduce.py)."""
    demo = simulate(PurePursuit(30.0), path, obstacles, steps=240, dt=0.5,
                    v=6.0, seed=seed, warmup=0, sigma_gps=1.5)
    meas, truth = demo["meas"], demo["traj"]
    psi = np.concatenate([[0.0], np.arctan2(np.diff(meas[:, 1]), np.diff(meas[:, 0]))])
    Xs, Ys = make_delta_windows(meas, truth, np.full(len(meas), 6.0), psi, window=10)
    Xs, Ys = Xs[::2], Ys[::2]
    lstm = LSTM(n_input=5, n_hidden=n_hidden, n_out=2, scale=1.0, seed=7)
    lstm.train_smoother(Xs, Ys, epochs=45, lr=0.05)
    return lstm


def run_all_methods(shapes, obstacles_of, seed, lstm_of, sim_kw=None):
    """One full pass: every shape x every method. Yields flat run dicts."""
    sim_kw = sim_kw or {}
    for shape, path in shapes.items():
        obstacles = obstacles_of(shape)
        lstm = lstm_of(shape)
        for M in ALL_METHODS:
            r = simulate(instantiate(M, lstm), path, obstacles,
                         steps=300, dt=0.5, v=6.0, sigma_gps=1.5,
                         wdot_max=0.35, seed=seed, **sim_kw)
            r["shape"], r["method"] = shape, M.name
            yield r


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over shapes per method (the paper's aggregation level)."""
    return df.groupby("method")[list(METRICS)].mean().reset_index()


def perm_test_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Exact paired permutation test on the mean difference (sign-flip).

    H0: the paired differences are exchangeable around zero. With n<=20 the
    2^n sign flips enumerate the full exact distribution (2^10 = 1024 flips).
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    obs = abs(d.mean())
    rng = np.random.default_rng(0) if n > 20 else None
    if rng is None:
        flips = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        means = np.abs(flips @ d) / n
        p = float(np.mean(means >= obs - 1e-15))
    else:
        sims = np.abs(rng.choice([-1.0, 1.0], size=(20000, n)) @ d) / n
        p = float(np.mean(sims >= obs - 1e-15))
    return p


# ----------------------------------------------------------------- commands
def cmd_multiseed(seeds):
    shapes = build_paths()
    lstm_cache, rows, devs_all = {}, [], []
    for seed in seeds:
        for shape, path in shapes.items():
            obstacles = build_obstacles(shape)
            lstm_cache[shape] = train_for(path, obstacles, seed)
        for r in run_all_methods(shapes, build_obstacles, seed,
                                 lambda s: lstm_cache[s]):
            r["seed"] = seed
            rows.append(r)
            path = shapes[r["shape"]]
            devs_all.append({
                "seed": seed, "shape": r["shape"], "method": r["method"],
                "devs": np.array([abs(path.closest(p)[1]) for p in r["traj"]])})
    df = pd.DataFrame(rows)
    df[["seed", "shape", "method"] + list(METRICS)].to_csv(
        os.path.join(OUT, "multiseed.csv"), index=False)
    np.save(os.path.join(OUT, "multiseed_devs.npy"),
            np.array([d["devs"] for d in devs_all], dtype=object),
            allow_pickle=True)

    # per-seed aggregation -> mean+/-std + per-seed OPE
    per_seed = df.groupby(["seed", "method"])[list(METRICS)].mean().reset_index()
    ope_rows = []
    for s in seeds:
        sub = per_seed[per_seed.seed == s].set_index("method").to_dict("index")
        o = batch_ope(sub)
        ope_rows.append({"seed": s, **o})
    ope_df = pd.DataFrame(ope_rows)
    ope_df.to_csv(os.path.join(OUT, "multiseed_ope.csv"), index=False)
    write_multiseed_summary(per_seed, ope_df, seeds)


def write_multiseed_summary(per_seed: pd.DataFrame, ope_df: pd.DataFrame, seeds):
    """Render (and write) the mean+/-std / permutation-test summary."""
    lines = ["# Multi-seed replication (mean +/- std over "
             f"{len(seeds)} seeds: {[int(s) for s in seeds]})\n",
             "| Method | mean dev (m) | adaptation (s) | #evade | min dist (m) "
             "| energy | success (%) | OPE |",
             "|---|---|---|---|---|---|---|---|"]
    order = [M.name for M in ALL_METHODS]
    std = per_seed.groupby("method")[list(METRICS)].std()
    mean = per_seed.groupby("method")[list(METRICS)].mean()
    ope_m, ope_s = ope_df.mean(numeric_only=True), ope_df.std(numeric_only=True)
    for m in order:
        r, s = mean.loc[m], std.loc[m]
        lines.append(
            f"| {m} | {r.mean_dev:.2f}+/-{s.mean_dev:.2f} | "
            f"{r.adaptation:.1f}+/-{s.adaptation:.1f} | "
            f"{r.n_evade:.1f}+/-{s.n_evade:.1f} | "
            f"{r.min_dist:.2f}+/-{s.min_dist:.2f} | "
            f"{r.mean_energy:.3f}+/-{s.mean_energy:.3f} | "
            f"{r.success:.2f}+/-{s.success:.2f} | "
            f"{ope_m[m]:.3f}+/-{ope_s[m]:.3f} |")
    lines += ["", "## Exact paired permutation tests (Proposed AI vs baseline, "
              "per-seed paired OPE, H0: equal means)", "",
              "| Baseline | dOPE (AI - base) | p (exact) | verdict (a=0.05) |",
              "|---|---|---|---|"]
    for m in order:
        if m == "Proposed AI":
            continue
        d = ope_df["Proposed AI"] - ope_df[m]
        p = perm_test_paired(ope_df["Proposed AI"], ope_df[m])
        verdict = "significant" if p < 0.05 else "not significant"
        lines.append(f"| {m} | {d.mean():+.3f} | {p:.4f} | {verdict} |")
    lines += ["", "## Same test applied to every metric (AI - baseline, "
              "10-seed paired means)", "",
              "| Metric | " + " | ".join(m for m in order if m != "Proposed AI") + " |",
              "|---" * len(order) + "|"]
    ai = per_seed[per_seed.method == "Proposed AI"].set_index("seed")
    for key in METRICS:
        cells = []
        for m in order:
            if m == "Proposed AI":
                continue
            b = per_seed[per_seed.method == m].set_index("seed")
            p = perm_test_paired(ai[key].values, b[key].values)
            d = ai[key].mean() - b[key].mean()
            cells.append(f"{d:+.3f} (p={p:.4f})")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    rank_counts = {}
    for _, row in ope_df.iterrows():
        vals = row.drop("seed").astype(float)
        ranked = list(vals.sort_values(ascending=False).index)
        rk = ranked.index("Proposed AI") + 1
        rank_counts[rk] = rank_counts.get(rk, 0) + 1
    lines.append("Proposed AI OPE rank per seed: " + ", ".join(
        f"#{k}: {v}/{len(seeds)}" for k, v in sorted(rank_counts.items())))
    with open(os.path.join(OUT, "summary_multiseed.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def cmd_sensitivity(seeds):
    thresholds = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    shapes = build_paths()
    rows = []
    for seed in seeds:
        lstm_cache = {}
        for shape, path in shapes.items():
            lstm_cache[shape] = train_for(path, build_obstacles(shape), seed)
        for r in run_all_methods(shapes, build_obstacles, seed,
                                 lambda s: lstm_cache[s]):
            path = shapes[r["shape"]]
            devs = np.array([abs(path.closest(p)[1]) for p in r["traj"]])
            for th in thresholds:
                rows.append({"seed": seed, "shape": r["shape"],
                             "method": r["method"], "threshold": th,
                             "success": float(np.mean(devs <= th) * 100.0)})
    df = pd.DataFrame(rows)
    agg = df.groupby(["method", "threshold"])["success"].agg(["mean", "std"])
    df.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    lines = ["# Success-rate sensitivity to the permissible-deviation threshold\n",
             f"({len(seeds)} seeds, mean over shapes; the paper's implicit "
             "default is 12 m)\n",
             "| Method | " + " | ".join(f"{t:g} m" for t in thresholds) + " |",
             "|---" * (len(thresholds) + 1) + "|"]
    for M in ALL_METHODS:
        vals = [agg.loc[(M.name, t)] for t in thresholds]
        lines.append(f"| {M.name} | " +
                     " | ".join(f"{v['mean']:.1f}+/-{v['std']:.1f}"
                                for v in vals) + " |")
    # rank stability: how often AI is top across thresholds/seeds
    top = df.groupby(["seed", "threshold"]).apply(
        lambda g: g.groupby("method").success.mean().idxmax())
    lines += ["", "Most-successful method by (seed, threshold): " +
              top.value_counts().to_dict().__str__()]
    with open(os.path.join(OUT, "summary_sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def cmd_ablation(seeds=(0, 1, 2)):
    shapes = build_paths()
    sweeps = {
        "hidden":  [{"hidden": h} for h in (8, 16, 32)],
        "horizon": [{"horizon": h} for h in (2, 4, 8)],
        "beta":    [{"beta": b} for b in (0.15, 0.35, 0.55)],
    }
    rows = []
    for seed in seeds:
        # one training per hidden size (horizon/beta reuse the same nets)
        nets = {}
        for shape, path in shapes.items():
            obstacles = build_obstacles(shape)
            for cfg in sweeps["hidden"]:
                nets[(shape, cfg["hidden"])] = train_for(
                    path, obstacles, seed, n_hidden=cfg["hidden"])
        for factor, cfgs in sweeps.items():
            for cfg in cfgs:
                # 'hidden' only selects which trained net to use; the other
                # keys are real ProposedAI constructor arguments
                sim_cfg = {k: v for k, v in cfg.items() if k != "hidden"}
                for shape, path in shapes.items():
                    obstacles = build_obstacles(shape)
                    h = cfg.get("hidden", 16)
                    lstm = nets[(shape, h)]
                    r = simulate(make_ai(lstm, **sim_cfg), path, obstacles,
                                 steps=300, dt=0.5, v=6.0, sigma_gps=1.5,
                                 wdot_max=0.35, seed=seed)
                    rows.append({"seed": seed, "shape": shape, "factor": factor,
                                 **{k: v for k, v in cfg.items()},
                                 "mean_dev": r["mean_dev"],
                                 "mean_energy": r["mean_energy"],
                                 "success": r["success"],
                                 "n_evade": r["n_evade"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    agg = df.groupby(["factor", "hidden", "horizon", "beta"], dropna=False)[
        ["mean_dev", "mean_energy", "success", "n_evade"]].agg(["mean", "std"])
    with open(os.path.join(OUT, "summary_ablation.md"), "w", encoding="utf-8") as f:
        f.write("# One-at-a-time ablation of the Proposed AI controller\n\n"
                + agg.to_string())
    print(agg)


def cmd_wind(seeds=(0, 1, 2)):
    shapes = build_paths()
    levels = [(0.0, 0.0), (1.5, 0.5), (3.0, 1.0)]   # (crosswind, gust_sigma)
    rows = []
    for seed in seeds:
        lstm_cache = {}
        for shape, path in shapes.items():
            lstm_cache[shape] = train_for(path, build_obstacles(shape), seed)
        for w, g in levels:
            for r in run_all_methods(
                    shapes, build_obstacles, seed, lambda s: lstm_cache[s],
                    sim_kw={"wind": (0.0, w), "gust_sigma": g}):
                r["seed"], r["wind"], r["gust"] = seed, w, g
                rows.append({k: r[k] for k in
                             ("seed", "wind", "shape", "method") + METRICS})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "wind.csv"), index=False)
    agg = df.groupby(["wind", "method"])[list(METRICS)].mean().reset_index()
    with open(os.path.join(OUT, "summary_wind.md"), "w", encoding="utf-8") as f:
        f.write("# Robustness under lateral wind + turbulence\n\n"
                + agg.to_string())
    print(agg)


def cmd_cost():
    """Per-step latency of step() on a representative mid-mission state."""
    shapes = build_paths()
    path, obstacles = shapes["straight"], build_obstacles("straight")
    lstm = train_for(path, obstacles, seed=0)
    state = {"pos": np.array([100.0, 3.0]), "vel": np.array([6.0, 0.0]),
             "psi": 0.0, "path": path, "obstacles": obstacles, "dt": 0.5}
    rows = []
    for M in ALL_METHODS:
        m = instantiate(M, lstm)
        m.step(state)                       # warm-up (JIT-ish caches, allocs)
        t0 = time.perf_counter()
        n = 2000
        for _ in range(n):
            m.step(state)
        dt_ms = (time.perf_counter() - t0) / n * 1e3
        rows.append({"method": M.name, "us_per_step": dt_ms * 1e3})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cost.csv"), index=False)
    print(df.to_string(index=False))


def cmd_summaries():
    """Rebuild the markdown summaries from saved csvs (no re-simulation).

    Useful after a summary-writing crash when the heavy computation already
    finished: multiseed.csv / multiseed_ope.csv / wind.csv are re-read.
    """
    df = pd.read_csv(os.path.join(OUT, "multiseed.csv"))
    seeds = sorted(df.seed.unique())
    per_seed = df.groupby(["seed", "method"])[list(METRICS)].mean().reset_index()
    ope_df = pd.read_csv(os.path.join(OUT, "multiseed_ope.csv"))
    write_multiseed_summary(per_seed, ope_df, seeds)

    w = pd.read_csv(os.path.join(OUT, "wind.csv"))
    agg = w.groupby(["wind", "method"])[list(METRICS)].mean().reset_index()
    with open(os.path.join(OUT, "summary_wind.md"), "w", encoding="utf-8") as f:
        f.write("# Robustness under lateral wind + turbulence\n\n"
                + agg.to_string())
    print(agg)


COMMANDS = {"multiseed": cmd_multiseed, "sensitivity": cmd_sensitivity,
             "ablation": cmd_ablation, "wind": cmd_wind, "cost": cmd_cost,
             "summaries": cmd_summaries}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "multiseed"
    args = [int(a) for a in sys.argv[2:] if a.lstrip("-").isdigit()]
    if cmd == "multiseed":
        cmd_multiseed(args or list(range(10)))
    elif cmd == "sensitivity":
        cmd_sensitivity(args or list(range(5)))
    else:
        COMMANDS[cmd]()
