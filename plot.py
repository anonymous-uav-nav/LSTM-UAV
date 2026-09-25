"""可视化：为答辩把复现结果渲染成图。

图组（输出到 out/figs/，全部使用 Agg 无头后端，不弹窗）：
  1. trajectories.png    三种形状 × 五方法轨迹俯视叠加（参考虚线 + 安全区圆）
  2. deviation_s.png     横向偏差 |e| 随前进里程 s 的曲线（每方法一条线）
  3. metrics_bars.png    OPE / 均值偏差 / 能量 / 最近距 四联柱状
  4. energy_curve.png    Proposed AI 的逐航向转向能量曲线 + Table-1 恒等式标注

与 reproduce.py 共享同一套路径/障碍/LSTM/控制器工厂，保证数值一致。
运行： py plot.py [seed]   （默认 seed=0）
"""
from __future__ import annotations

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from reproduce import build_paths, build_obstacles, train_predictor
from drone_sim.methods import ALL_METHODS

FIG = os.path.join(os.path.dirname(__file__), "out", "figs")
os.makedirs(FIG, exist_ok=True)

# 与 reproduce.main 完全一致的方法实例化
def instantiate(M, lstm):
    if M.__name__ == "ProposedAI":
        return M(lstm, window=10, horizon=4, lookahead=25.0, beta=0.35)
    if M.__name__ == "PurePursuit":
        return M(30.0)
    if M.__name__ == "LineOfSight":
        return M(30.0, 0.5)
    if M.__name__ == "VectorField":
        return M(0.5, 6.0)
    return M(0.06)


def collect(seed):
    """对每个形状×每方法跑一遍闭环，返回所有轨迹、偏差序列与单 run 指标。"""
    from drone_sim.simulate import simulate
    data = []                                   # per-shape -> per-method run
    for shape, path in build_paths().items():
        obstacles = build_obstacles(shape)
        lstm, _loss = train_predictor(path, obstacles, seed)
        for M in ALL_METHODS:
            m = instantiate(M, lstm)
            r = simulate(m, path, obstacles, steps=300, dt=0.5, v=6.0,
                         sigma_gps=1.5, wdot_max=0.35, seed=seed)
            devs = np.array([abs(path.closest(p)[1]) for p in r["traj"]])
            data.append({"shape": shape, "method": M.name,
                         "traj": r["traj"], "proj_s": r["proj_s"],
                         "dev": devs, "r": r,
                         "path": path, "obstacles": obstacles})
    return data


# ------------------------------------------------------------------- 图 1
def fig_trajectories(data, shapes=("straight", "circle", "polyline")):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_METHODS)))
    for ax, shape in zip(axes, shapes):
        runs = [d for d in data if d["shape"] == shape]
        path = runs[0]["path"]
        # 参考轨迹
        ax.plot(path.pts[:, 0], path.pts[:, 1], "--", color="#444", lw=1.2,
                label="reference")
        # 安全区
        for o in runs[0]["obstacles"]:
            c = plt.Circle(o.centre, o.radius, color="#d9534f", alpha=0.15, ec="#d9534f", lw=0.7)
            ax.add_patch(c)
        ax.scatter([o.centre[0] for o in runs[0]["obstacles"]],
                   [o.centre[1] for o in runs[0]["obstacles"]],
                   marker="x", color="#d9534f", s=50, zorder=5)
        for i, run in enumerate(runs):
            ax.plot(run["traj"][:, 0], run["traj"][:, 1], color=colors[i],
                    lw=1.5, alpha=0.9, label=run["method"])
        ax.set_title(f"{shape} mission", fontsize=12)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.suptitle("Trajectories (truth) vs reference, by controller", y=1.0, fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(FIG, "trajectories.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------- 图 2
def fig_deviation(data, shapes=("straight", "circle", "polyline")):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALL_METHODS)))
    for ax, shape in zip(axes, shapes):
        runs = [d for d in data if d["shape"] == shape]
        for i, run in enumerate(runs):
            ax.plot(run["proj_s"], run["dev"], color=colors[i], lw=1.4,
                    label=run["method"])
        ax.set_title(f"{shape}: |lateral deviation| vs distance", fontsize=11)
        ax.set_xlabel("along-track distance s (m)")
        ax.set_ylabel("|deviation| (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "deviation_s.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------- 图 3
def fig_metrics(data, shapes=("straight", "circle", "polyline")):
    methods = [M.name for M in ALL_METHODS]
    agg = {m: {"mean_dev": [], "mean_energy": [], "min_dist": [], "n_evade": []}
           for m in methods}
    for d in data:
        a = agg[d["method"]]
        a["mean_dev"].append(d["r"]["mean_dev"])
        a["mean_energy"].append(d["r"]["mean_energy"])
        a["min_dist"].append(d["r"]["min_dist"])
        a["n_evade"].append(d["r"]["n_evade"])
    for m in methods:                       # 平均：agg[.][:] over shapes
        for k in agg[m]:
            agg[m][k] = float(np.mean(agg[m][k]))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    titles = {"mean_dev": "mean deviation (m, lower better)",
              "mean_energy": "mean energy (unit, lower better)",
              "min_dist": "min obstacle distance (m, higher better)",
              "n_evade": "# avoidance (lower better)"}
    for ax, key in zip(axes, titles):
        vals = [agg[m][key] for m in methods]
        bars = ax.bar(methods, vals, color=colors)
        best = min(vals) if key not in ("min_dist",) else max(vals)
        bars[vals.index(best)].set_edgecolor("#222"); bars[vals.index(best)].set_linewidth(1.6)
        ax.set_title(titles[key], fontsize=11)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Aggregated metrics (mean over 3 shapes)", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "metrics_bars.png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------- 图 4
def fig_energy(data, shape="straight"):
    run = next(d for d in data if d["shape"] == shape
               and d["method"] == "Proposed AI")
    from drone_sim.energy import step_energy
    E = step_energy(run["traj"])
    s = run["proj_s"][: E.shape[0]]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(s, E, lw=1.0, color="#2b6cb0", label="E_t per step")
    ax.axhline(run["r"]["mean_energy"], color="#e07b00", ls="--", lw=1.6,
               label=f"E_mean={run['r']['mean_energy']:.4f}")
    ax.axhline(run["r"]["energy_sigma"] + run["r"]["mean_energy"],
               color="#c0392b", ls=":", lw=1.4, label="E_mean + sigma")
    ax.set_title(f"Proposed AI — heading turn-energy field (Table 1), {shape} mission")
    ax.set_xlabel("along-track distance s (m)"); ax.set_ylabel("energy (E_t)")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    # 恒等式标注
    ident = f"E_total={run['r']['energy_total']:.3f}  =  E_mean×n_steps=" \
            f"{run['r']['mean_energy']}×{run['r']['n_steps']}"
    ax.text(0.02, 0.93, ident, transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="#f5f5f5", ec="#999"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "energy_curve.png"), dpi=300)
    plt.close(fig)


def main(seed=0):
    data = collect(seed)
    fig_trajectories(data)
    fig_deviation(data)
    fig_metrics(data)
    fig_energy(data)
    print("figs written to:", FIG)
    for f in sorted(os.listdir(FIG)):
        p = os.path.join(FIG, f)
        print(f"  {f:24s} {os.path.getsize(p)//1024:5d} KB")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)