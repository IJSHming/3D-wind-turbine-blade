"""出版级专业可视化绘图脚本 (plot_results.py):
从 results/benchmark_results.json 中读取 10 随机种子基准测试数据，
面向 3 种核心方法 (TuRBO, LLM-Meta-TuRBO, MALIBO) 及基准 (Baseline) 绘制 4 张高清学术图表：
1. 3D 叶片几何构型与气动展向分布对比 (figures/blade_shapes_comparison.png / .pdf)
2. 10 次实验柱状图性能对比 (figures/blade_bar_comparison.png / .pdf)
3. 箱线图统计分布对比 (figures/blade_boxplot_comparison.png / .pdf)
4. 平均收敛曲线对比与误差带 (figures/blade_convergence_curves.png / .pdf)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

from src.blade_geometry import (
    get_baseline_blade_params,
    interpolate_blade_span,
    loft_blade_3d,
    unpack_blade_params,
)

# 出版级学术排版配置
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.color": "#e5e7eb",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#d1d5db",
})

# 专属学术配色 (对齐顶级顶会规范)
COLOR_BASELINE = "#6B7280"   # 沉稳灰 (Baseline NREL 5MW)
COLOR_TURBO = "#2563EB"      # 科技蓝 (TuRBO)
COLOR_LLM_META = "#059669"   # 翡翠绿 (LLM-Meta-TuRBO)
COLOR_MALIBO = "#DC2626"     # 活力红 (MALIBO)

COLORS = {
    "Baseline": COLOR_BASELINE,
    "TuRBO": COLOR_TURBO,
    "LLM-Meta-TuRBO": COLOR_LLM_META,
    "MALIBO": COLOR_MALIBO,
}

LABELS = {
    "Baseline": "Baseline (NREL 5MW)",
    "TuRBO": "TuRBO-1",
    "LLM-Meta-TuRBO": "LLM-Meta-TuRBO",
    "MALIBO": "MALIBO",
}


def load_results(path: str = "results/benchmark_results.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到结果文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 1. 收敛曲线对比图 (Convergence Curves)
# ==============================================================================

def plot_convergence_curves(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    baseline_p = float(meta.get("baseline_power_kW", 2885.4))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # 绘制 Baseline 参考基线
    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p:.1f} kW)",
        zorder=2,
    )

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    for method in methods:
        runs = results.get(method, [])
        if not runs:
            continue
        histories = np.array([r["history"] for r in runs])  # (n_seeds, max_evals)
        mean_curve = np.mean(histories, axis=0)
        std_curve = np.std(histories, axis=0)
        sem_curve = std_curve / np.sqrt(len(runs))  # 标准误差带 (Standard Error)

        steps = np.arange(1, len(mean_curve) + 1)
        color = COLORS[method]
        label = f"{LABELS[method]} (Final: {mean_curve[-1]:.1f} ± {std_curve[-1]:.1f} kW)"

        ax.plot(steps, mean_curve, color=color, linewidth=2.4, label=label, zorder=4)
        ax.fill_between(
            steps,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=color,
            alpha=0.18,
            zorder=3,
        )

    ax.set_xlim(1, 300)
    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Aerodynamic Output Power (kW)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Mean Optimization Convergence Trajectories (10 Independent Seeds)\n"
        f"3D Wind Turbine Blade ($V_{{\\infty}} = {meta.get('wind_speed', 9.0):.1f}$ m/s, $\\Omega = {meta.get('rpm', 10.0):.1f}$ rpm)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_convergence_curves.png")
    pdf_path = os.path.join(output_dir, "blade_convergence_curves.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表生成] 收敛曲线已保存至: {png_path} & {pdf_path}")


# ==============================================================================
# 2. 10 随机种子柱状图对比 (Bar Comparison)
# ==============================================================================

def plot_bar_comparison(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    seeds = meta.get("seeds", [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999])
    n_seeds = len(seeds)
    baseline_p = float(meta.get("baseline_power_kW", 2885.4))

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    bar_width = 0.25
    x_indices = np.arange(n_seeds)

    for i, method in enumerate(methods):
        runs = results.get(method, [])
        if not runs:
            continue
        scores = [r["best_score"] for r in runs]
        offset = (i - 1) * bar_width
        rects = ax.bar(
            x_indices + offset,
            scores,
            width=bar_width,
            color=COLORS[method],
            label=LABELS[method],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.90,
            zorder=3,
        )

    # 参考线
    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p:.1f} kW)",
        zorder=4,
    )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Optimized Rotor Power (kW)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Individual Seed Performance Comparison across 10 Random Seeds\n"
        "3D Wind Turbine Blade Rotor Power Maximization (300 NFE Budget)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    y_min = baseline_p * 0.96
    ax.set_ylim(bottom=y_min)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.legend(loc="upper left", ncol=4, fontsize=10.5, framealpha=0.95)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_bar_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_bar_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表生成] 柱状图对比已保存至: {png_path} & {pdf_path}")


# ==============================================================================
# 3. 箱线图统计分布对比 (Boxplot Comparison)
# ==============================================================================

def plot_boxplot_comparison(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    baseline_p = float(meta.get("baseline_power_kW", 2885.4))

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    box_data = []
    labels = []
    colors_list = []

    for m in methods:
        runs = results.get(m, [])
        if runs:
            box_data.append([r["best_score"] for r in runs])
            labels.append(LABELS[m])
            colors_list.append(COLORS[m])

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # 参考线
    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p:.1f} kW)",
        zorder=2,
    )

    bp = ax.boxplot(
        box_data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.45,
        medianprops=dict(color="#111827", linewidth=2.0),
        meanprops=dict(color="#1e1b4b", linewidth=2.0, linestyle=":"),
        flierprops=dict(marker="o", color="#ef4444", markersize=6),
        zorder=3,
    )

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor("#111827")
        patch.set_linewidth(1.2)

    # 叠加抖动散点 (Jitter Scatter Points)
    rng = np.random.RandomState(42)
    for i, points in enumerate(box_data):
        x_jitter = rng.normal(i + 1, 0.04, size=len(points))
        ax.scatter(
            x_jitter,
            points,
            color=colors_list[i],
            edgecolor="#111827",
            s=45,
            alpha=0.85,
            zorder=5,
        )

    ax.set_ylabel("Optimized Rotor Power (kW)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Statistical Distribution & Stability Comparison (10 Independent Seeds)\n"
        "Solid line: Median | Dotted line: Mean | Dots: Individual Seeds",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_boxplot_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_boxplot_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表生成] 箱线图已保存至: {png_path} & {pdf_path}")


# ==============================================================================
# 4. 3D 叶片几何剖面与展向气动参数对比 (Blade Shapes & Aerodynamics)
# ==============================================================================

def plot_blade_shapes(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]

    # 提取基准叶片与各方法中位数 (代表性) 解
    x0 = get_baseline_blade_params()
    base_span = interpolate_blade_span(x0, n_stations=25)
    r_span = base_span["r"]

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    rep_solutions: Dict[str, np.ndarray] = {"Baseline": x0}

    for method in methods:
        runs = results.get(method, [])
        if not runs:
            continue
        scores = [r["best_score"] for r in runs]
        med_idx = int(np.argsort(scores)[len(scores) // 2])
        rep_solutions[method] = np.array(runs[med_idx]["best_x"], dtype=np.float64)

    # 4 子图综合工程看板:
    # (a) 弦长分布 c(r) 沿展向对比
    # (b) 扭角分布 beta(r) 沿展向对比
    # (c) Station 2 (中展段 r ~ 35m) 核心功率翼型剖面对比
    # (d) Station 3 (外展尖部 r ~ 54m) 高速减阻翼型剖面对比
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=300)

    # ---------------- 子图 1: 弦长展向分布 ----------------
    ax1 = axes[0, 0]
    for m_key, x_sol in rep_solutions.items():
        sp = interpolate_blade_span(x_sol, n_stations=25)
        color = COLORS[m_key]
        lw = 2.4 if m_key == "MALIBO" else (2.2 if m_key == "LLM-Meta-TuRBO" else 1.8)
        ls = "-" if m_key != "Baseline" else "--"
        ax1.plot(sp["r"], sp["chord"], color=color, linewidth=lw, linestyle=ls, label=LABELS[m_key])

    ax1.set_xlabel("Rotor Radial Position $r$ (m)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Chord Length $c(r)$ (m)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Spanwise Chord Distribution $c(r)$", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9.5, loc="upper right")

    # ---------------- 子图 2: 扭角展向分布 ----------------
    ax2 = axes[0, 1]
    for m_key, x_sol in rep_solutions.items():
        sp = interpolate_blade_span(x_sol, n_stations=25)
        color = COLORS[m_key]
        lw = 2.4 if m_key == "MALIBO" else (2.2 if m_key == "LLM-Meta-TuRBO" else 1.8)
        ls = "-" if m_key != "Baseline" else "--"
        ax2.plot(sp["r"], sp["twist_deg"], color=color, linewidth=lw, linestyle=ls, label=LABELS[m_key])

    ax2.set_xlabel("Rotor Radial Position $r$ (m)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Aerodynamic Twist $\\beta(r)$ (deg)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Spanwise Aerodynamic Twist Wash-out $\\beta(r)$", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9.5, loc="upper right")

    # ---------------- 子图 3 & 4: Station 2 与 Station 3 翼型几何剖面 ----------------
    import aerosandbox as asb

    def extract_airfoil_coords(th_18, n_pts=100):
        k_af = asb.KulfanAirfoil(
            upper_weights=th_18[0:8],
            lower_weights=th_18[8:16],
            leading_edge_weight=float(th_18[16]),
            TE_thickness=float(th_18[17]),
        )
        xc = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_pts)))
        yu = k_af.upper_coordinates(xc)[:, 1]
        yl = k_af.lower_coordinates(xc)[:, 1]
        x_cl = np.concatenate([xc[::-1], xc[1:]])
        y_cl = np.concatenate([yu[::-1], yl[1:]])
        return x_cl, y_cl

    for station_idx, ax, title_prefix, sub_title in [
        (1, axes[1, 0], "(c)", "Mid-Span Station 2 ($r = 35.3$ m) Airfoil Profile"),
        (2, axes[1, 1], "(d)", "Tip Region Station 3 ($r = 53.8$ m) Airfoil Profile"),
    ]:
        for m_key, x_sol in rep_solutions.items():
            stations = unpack_blade_params(x_sol)
            st = stations[station_idx]
            x_cl, y_cl = extract_airfoil_coords(st["theta"])
            color = COLORS[m_key]
            lw = 2.2 if m_key == "MALIBO" else (2.0 if m_key == "LLM-Meta-TuRBO" else 1.6)
            ls = "-" if m_key != "Baseline" else ":"
            ax.plot(x_cl, y_cl, color=color, linewidth=lw, linestyle=ls, label=f"{LABELS[m_key]} (c={st['chord']:.2f}m)")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.10, 0.22 if station_idx == 1 else 0.16)
        ax.set_xlabel("Normalized Section Coordinate $x/c$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Thickness $y/c$", fontsize=11, fontweight="bold")
        ax.set_title(f"{title_prefix} {sub_title}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9.0, loc="upper right")

    plt.suptitle(
        "3D Wind Turbine Blade Structural & Aerodynamic Lofting Comparison\n"
        "(Representative Median Configurations after 300 Function Evaluations)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_shapes_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_shapes_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"[图表生成] 3D 叶片几何对比图已保存至: {png_path} & {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制 3D 风机叶片基准测试图表")
    parser.add_argument(
        "--data",
        type=str,
        default="results/benchmark_results.json",
        help="基准测试结果文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="图表输出目录",
    )
    args = parser.parse_args()

    data = load_results(args.data)
    plot_convergence_curves(data, args.output_dir)
    plot_bar_comparison(data, args.output_dir)
    plot_boxplot_comparison(data, args.output_dir)
    plot_blade_shapes(data, args.output_dir)
    print("\n所有出版级学术高清图表已成功生成并导出！")


if __name__ == "__main__":
    main()
