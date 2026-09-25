# -*- coding: utf-8 -*-
"""Additional evidence experiments for the JCIE revision.

  arch     architecture ablation of the proposed controller: which component
           carries the result? Variants (3 seeds, paper protocol):
             full            window 10, horizon 4, beta 0.35, clip 1.5
             no-LSTM         use_lstm=False  -> dz = dz_kin (kinematic prior only)
             unbounded       clip_bound=1e6  -> no bounded correction (Eq. 2 off)
             no-anchor       beta=1.0        -> no measurement anchoring (Eq. 3 off)
             no-horizon      horizon=0       -> no proactive risk check (Eq. 4 off)
  noise    GPS-noise sweep sigma_gps in {0.5, 1.0, 1.5, 2.0, 3.0} m, 5 seeds,
           all five methods; the predictor stays the one trained at the paper's
           sigma = 1.5 m (tests deployment robustness to noise mismatch).
  pareto   energy-vs-deviation Pareto scatter data from the existing
           multiseed.csv (no re-simulation).
  plot     renders arch.png / noise.png / pareto.png into out/figs/.

Run:  py experiments_extra.py arch
      py experiments_extra.py noise 0 1 2 3 4
      py experiments_extra.py pareto
      py experiments_extra.py plot
"""
from __future__ import annotations

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_sim.methods import ALL_METHODS, ProposedAI
from drone_sim.simulate import simulate
from experiments import OUT, METRICS, train_for, make_ai, instantiate
from reproduce import build_paths, build_obstacles

ORDER = [M.name for M in ALL_METHODS]


# ------------------------------------------------------------------- arch
def cmd_arch(seeds=(0, 1, 2)):
    shapes = build_paths()
    variants = {
        "full": {},
        "no-LSTM": {"use_lstm": False},
        "unbounded": {"clip_bound": 1e6},
        "no-anchor": {"beta": 1.0},
        "no-horizon": {"horizon": 0},
    }
    rows = []
    for seed in seeds:
        nets = {}
        for shape, path in shapes.items():
            nets[shape] = train_for(path, build_obstacles(shape), seed)
        for vname, over in variants.items():
            for shape, path in shapes.items():
                r = simulate(make_ai(nets[shape], **over), path,
                             build_obstacles(shape),
                             steps=300, dt=0.5, v=6.0, sigma_gps=1.5,
                             wdot_max=0.35, seed=seed)
                rows.append({"seed": seed, "shape": shape, "variant": vname,
                             **{k: r[k] for k in
                                ("mean_dev", "mean_energy", "success", "n_evade")}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "arch.csv"), index=False)
    agg = df.groupby("variant")[["mean_dev", "mean_energy", "success",
                                 "n_evade"]].agg(["mean", "std"])
    with open(os.path.join(OUT, "summary_arch.md"), "w", encoding="utf-8") as f:
        f.write("# Architecture ablation (3 seeds, mean over shapes)\n\n"
                + agg.to_string())
    print(agg)
    return df


# ------------------------------------------------------------------ noise
def cmd_noise(seeds):
    levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    shapes = build_paths()
    rows = []
    for seed in seeds:
        nets = {}
        for shape, path in shapes.items():
            nets[shape] = train_for(path, build_obstacles(shape), seed)
        for level in levels:
            for shape, path in shapes.items():
                obstacles = build_obstacles(shape)
                for M in ALL_METHODS:
                    r = simulate(instantiate(M, nets[shape]), path, obstacles,
                                 steps=300, dt=0.5, v=6.0, sigma_gps=level,
                                 wdot_max=0.35, seed=seed)
                    rows.append({"seed": seed, "sigma": level, "shape": shape,
                                 "method": M.name,
                                 **{k: r[k] for k in METRICS}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "noise.csv"), index=False)
    agg = df.groupby(["sigma", "method"])[list(METRICS)].mean().reset_index()
    with open(os.path.join(OUT, "summary_noise.md"), "w", encoding="utf-8") as f:
        f.write("# GPS-noise sweep (predictor trained at sigma=1.5 m)\n\n"
                + agg.to_string())
    print(agg)
    return df


# ----------------------------------------------------------------- pareto
def cmd_pareto():
    """Per-seed (energy, deviation) points per method -> Pareto summary."""
    df = pd.read_csv(os.path.join(OUT, "multiseed.csv"))
    per_seed = df.groupby(["seed", "method"])[
        ["mean_dev", "mean_energy"]].mean().reset_index()
    dominated = {}
    for m in per_seed.method.unique():
        pts = per_seed[per_seed.method == m][["mean_dev", "mean_energy"]].values
        other = per_seed[per_seed.method != m][["mean_dev", "mean_energy"]].values
        dom = 0
        for d, e in pts:
            if np.any(np.all(other <= [d, e], axis=1) &
                      np.any(other < [d, e], axis=1)):
                dom += 1
        dominated[m] = f"{len(pts) - dom}/{len(pts)}"
    lines = ["# Energy-deviation Pareto summary (10 seeds, per-seed shape-mean)\n",
             "| Method | non-dominated points | mean dev | mean energy |",
             "|---|---|---|---|"]
    means = per_seed.groupby("method")[["mean_dev", "mean_energy"]].mean()
    for m in ORDER:
        lines.append(f"| {m} | {dominated[m]} | {means.loc[m, 'mean_dev']:.2f} "
                     f"| {means.loc[m, 'mean_energy']:.3f} |")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary_pareto.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ------------------------------------------------------------------- plot
def cmd_plot():
    figs = os.path.join(OUT, "figs")
    os.makedirs(figs, exist_ok=True)

    # noise sweep: two panels (energy, deviation)
    n = pd.read_csv(os.path.join(OUT, "noise.csv"))
    agg = n.groupby(["sigma", "method"])[
        ["mean_energy", "mean_dev"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in zip(axes, ("mean_energy", "mean_dev"),
                            ("mean energy", "mean deviation (m)")):
        for m in ORDER:
            sub = agg.xs(m, level="method")[key]
            ax.errorbar(sub.index, sub["mean"], yerr=sub["std"],
                        marker="o", ms=3, lw=1.4, capsize=2, label=m)
        ax.set_xlabel(r"GPS noise $\sigma_{gps}$ (m)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "noise.png"), dpi=300)
    plt.close(fig)

    # architecture ablation: grouped bars
    a = pd.read_csv(os.path.join(OUT, "arch.csv"))
    g = a.groupby("variant")[["mean_dev", "mean_energy", "success"]].agg(["mean", "std"])
    variants = ["full", "no-LSTM", "unbounded", "no-anchor", "no-horizon"]
    metrics = [("mean_dev", "mean deviation (m)"),
               ("mean_energy", "mean energy"),
               ("success", "success (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (key, lab) in zip(axes, metrics):
        mu = [g.loc[v, (key, "mean")] for v in variants]
        sd = [g.loc[v, (key, "std")] for v in variants]
        ax.bar(range(len(variants)), mu, yerr=sd, capsize=3,
               color="#4477aa", edgecolor="black", lw=0.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "arch.png"), dpi=300)
    plt.close(fig)

    # Pareto scatter
    ms = pd.read_csv(os.path.join(OUT, "multiseed.csv"))
    ps = ms.groupby(["seed", "method"])[["mean_dev", "mean_energy"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    colors = {"Pure Pursuit": "#999999", "Line of Sight": "#1f77b4",
              "Vector Field": "#2ca02c", "Nonlinear Stabilization": "#ff7f0e",
              "Proposed AI": "#d62728"}
    for m in ORDER:
        sub = ps[ps.method == m]
        ax.scatter(sub.mean_dev, sub.mean_energy, s=28, alpha=0.75,
                   label=m, color=colors[m], edgecolors="black", linewidths=0.4)
    pts = ps[["mean_dev", "mean_energy"]].values
    front = []
    for d, e in pts:
        if not np.any(np.all(pts <= [d, e], axis=1) &
                      np.any(pts < [d, e], axis=1)):
            front.append((d, e))
    front = np.array(sorted(front))
    ax.plot(front[:, 0], front[:, 1], "k--", lw=1, alpha=0.6, label="Pareto front")
    ax.set_xlabel("mean deviation (m)")
    ax.set_ylabel("mean energy")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "pareto.png"), dpi=300)
    plt.close(fig)
    print("written noise.png / arch.png / pareto.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "arch"
    args = [int(a) for a in sys.argv[2:] if a.lstrip("-").isdigit()]
    if cmd == "arch":
        cmd_arch(tuple(args) or (0, 1, 2))
    elif cmd == "noise":
        cmd_noise(args or [0, 1, 2, 3, 4])
    elif cmd == "pareto":
        cmd_pareto()
    elif cmd == "plot":
        cmd_plot()
