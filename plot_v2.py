"""Extra figures for the improved (SCI) manuscript: error-bar and robustness plots.

Reads the csv files written by experiments.py and renders:
  multiseed_bars.png  six metrics with mean+/-std error bars (10 seeds)
  sensitivity.png    success rate vs permissible-deviation threshold
  ablation.png        one-at-a-time sweeps (hidden / horizon / beta)
  wind.png            mean energy & deviation vs wind level, per method

Run AFTER experiments.py:  py plot_v2.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from drone_sim.methods import ALL_METHODS

FIG = os.path.join(os.path.dirname(__file__), "out", "figs")
OUT = os.path.join(os.path.dirname(__file__), "out")
METRICS = ["mean_dev", "adaptation", "n_evade", "min_dist", "mean_energy", "success"]
TITLES = {
    "mean_dev": ("mean deviation (m)", "lower better"),
    "adaptation": ("adaptation time (s)", "lower better"),
    "n_evade": ("# evasion maneuvers", "lower better"),
    "min_dist": ("min obstacle distance (m)", "higher better"),
    "mean_energy": ("mean energy (unit)", "lower better"),
    "success": ("success rate (%)", "higher better"),
}


def _colors(n):
    return plt.cm.tab10(np.linspace(0, 1, n))


# ------------------------------------------------------------- multiseed bars
def fig_multiseed_bars():
    df = pd.read_csv(os.path.join(OUT, "multiseed.csv"))
    ope = pd.read_csv(os.path.join(OUT, "multiseed_ope.csv"))
    methods = [M.name for M in ALL_METHODS]
    agg = df.groupby("method")[METRICS].agg(["mean", "std"])
    # append OPE as the 7th panel
    fig, axes = plt.subplots(1, 7, figsize=(22, 4.2))
    colors = _colors(len(methods))
    panels = METRICS + ["ope"]
    for ax, key in zip(axes, panels):
        if key == "ope":
            mean = ope[methods].mean()
            std = ope[methods].std()
        else:
            mean = agg.loc[methods, (key, "mean")]
            std = agg.loc[methods, (key, "std")]
        bars = ax.bar(methods, mean.values, yerr=std.values, capsize=3,
                      color=colors, error_kw={"lw": 1.1})
        title, direction = TITLES[key] if key in TITLES else \
            ("OPE (higher better)", "")
        best = mean.idxmin() if "lower better" in direction else mean.idxmax()
        bars[methods.index(best)].set_edgecolor("#222")
        bars[methods.index(best)].set_linewidth(1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Multi-seed replication: mean +/- std over 10 seeds", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "multiseed_bars.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- sensitivity
def fig_sensitivity():
    df = pd.read_csv(os.path.join(OUT, "sensitivity.csv"))
    agg = df.groupby(["method", "threshold"])["success"].mean().reset_index()
    methods = [M.name for M in ALL_METHODS]
    colors = _colors(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, m in enumerate(methods):
        sub = agg[agg.method == m]
        ax.plot(sub.threshold, sub.success, "-o", ms=4, lw=1.6,
                color=colors[i], label=m)
    ax.axvline(12.0, color="#888", ls=":", lw=1.2)
    ax.text(12.2, 4, "default\n12 m", fontsize=8, color="#555")
    ax.set_xlabel("permissible-deviation threshold (m)")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Success-rate sensitivity to the implicit threshold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sensitivity.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------- ablation
def fig_ablation():
    df = pd.read_csv(os.path.join(OUT, "ablation.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    factors = [("hidden", "LSTM hidden size", "horizon"), ("horizon", "prediction horizon", "beta"),
               ("beta", "fusion weight beta", None)]
    ykeys = [("mean_dev", "mean deviation (m)"), ("mean_energy", "mean energy"),
             ("success", "success (%)")]
    # one panel per factor: 3 lines (dev/energy/success) normalised? keep dev+OPE-ish:
    for ax, (factor, xlabel, _) in zip(axes, factors):
        sub = df[df.factor == factor].groupby(factor)[
            ["mean_dev", "mean_energy", "success", "n_evade"]].mean()
        x = sub.index.values.astype(float)
        for key, label, color in [("mean_dev", "mean deviation (m)", "#c0392b"),
                                  ("mean_energy", "mean energy", "#2b6cb0"),
                                  ("n_evade", "# evasion", "#27ae60")]:
            y = sub[key].values
            ax.plot(x, y / y.max(), "-o", ms=5, lw=1.8, color=color,
                    label=f"{label} (normalised)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("value / max (per metric)")
        ax.set_title(f"Ablation: {xlabel}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("One-at-a-time ablation of the Proposed AI controller "
                 "(normalised to the sweep max)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ wind
def fig_wind():
    df = pd.read_csv(os.path.join(OUT, "wind.csv"))
    methods = [M.name for M in ALL_METHODS]
    colors = _colors(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (key, title) in zip(axes, [("mean_dev", "mean deviation (m)"),
                                        ("mean_energy", "mean energy")]):
        agg = df.groupby(["wind", "method"])[key].mean().reset_index()
        for i, m in enumerate(methods):
            sub = agg[agg.method == m]
            ax.plot(sub.wind, sub[key], "-o", ms=5, lw=1.7, color=colors[i],
                    label=m)
        ax.set_xlabel("lateral wind speed (m/s, gust std = wind/2)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Robustness under lateral wind + turbulence", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "wind.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_multiseed_bars()
    print("multiseed_bars.png done")
    fig_sensitivity()
    print("sensitivity.png done")
    fig_ablation()
    print("ablation.png done")
    fig_wind()
    print("wind.png done")
