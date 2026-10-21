"""出版级专业可视化绘图脚本 (plot_results.py):
从 results/benchmark_results.json 中读取 10 随机种子基准测试数据，
面向 3 种核心方法 (TuRBO, LLM-Meta-TuRBO, MALIBO) 及基准 (Baseline IEA-22-280-RWT) 绘制全套学术图表：
1. 完整 300 NFE 收敛曲线与标准误差带 (figures/blade_convergence_curves.png / .pdf)
2. 100 NFE 高分辨率聚焦收敛曲线 (figures/blade_convergence_curves_100nfe.png / .pdf)
3. 柱状图性能提升与标准差对比 (figures/blade_bar_comparison.png / .pdf)
4. 箱线图统计分布对比 (figures/blade_boxplot_comparison.png / .pdf)
5. 叶片展向弦长与扭角分布对比 (figures/blade_chord_twist_distributions.png / .pdf)
6. 6 个核心截面翼型轮廓多子图学术对比 (figures/blade_station_airfoils.png / .pdf)
7. 3D 叶片空间透视渲染图 (figures/blade_3d_rendering.png / .pdf)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import aerosandbox as asb

from src.blade_geometry import (
    DIM,
    HUB_RADIUS,
    NUM_STATIONS,
    ROTOR_RADIUS,
    STATION_RADII,
    STATION_RADIANS_FRAC,
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

# 学术专属配色
COLOR_BASELINE = "#6B7280"   # 沉稳灰 (Baseline IEA 22MW)
COLOR_TURBO = "#2563EB"      # 科技蓝 (TuRBO-1)
COLOR_LLM_META = "#059669"   # 翡翠绿 (LLM-Meta-TuRBO)
COLOR_MALIBO = "#DC2626"     # 活力红 (MALIBO)

COLORS = {
    "Baseline": COLOR_BASELINE,
    "TuRBO": COLOR_TURBO,
    "LLM-Meta-TuRBO": COLOR_LLM_META,
    "MALIBO": COLOR_MALIBO,
}

LABELS = {
    "Baseline": "Baseline (IEA 22MW)",
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
# 1 & 2. 收敛曲线对比图 (300 NFE 完整与 100 NFE 聚焦)
# ==============================================================================

def plot_convergence_curves(
    data: Dict[str, Any],
    output_dir: str = "figures",
    max_nfe: int = 300,
    filename_prefix: str = "blade_convergence_curves",
):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    baseline_p = float(meta.get("baseline_power_kW", 25593.2))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    # 绘制 Baseline 参考线
    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p/1e3:.2f} MW)",
        zorder=2,
    )

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    for method in methods:
        runs = results.get(method, [])
        if not runs:
            continue
        histories = np.array([r["history"] for r in runs])
        histories = histories[:, :max_nfe]

        mean_curve = np.mean(histories, axis=0)
        std_curve = np.std(histories, axis=0)
        sem_curve = std_curve / np.sqrt(len(runs))

        steps = np.arange(1, len(mean_curve) + 1)
        color = COLORS[method]
        label = LABELS[method]

        ax.plot(
            steps,
            mean_curve,
            color=color,
            linewidth=2.2,
            label=f"{label} (Final: {mean_curve[-1]/1e3:.2f} MW)",
            zorder=3,
        )
        ax.fill_between(
            steps,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=color,
            alpha=0.18,
            zorder=2,
        )

    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Penalized Aerodynamic Power (kW)", fontsize=12, fontweight="bold", labelpad=8)
    title_suffix = " (Full 300 NFE)" if max_nfe > 100 else " (Early Phase: First 100 NFE Focus)"
    ax.set_title(
        f"Optimization Convergence on IEA-22-280-RWT Blade{title_suffix}\n"
        f"(Rated Condition: $V_\\infty = 11.0\\,\\mathrm{{m/s}}$, $\\Omega = 7.061\\,\\mathrm{{rpm}}$, 10 Seeds Mean $\\pm$ SEM)",
        fontsize=12,
        pad=12,
    )
    ax.set_xlim(1, max_nfe)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 规范 legend 放置，严格避免遮挡曲线
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        fontsize=10,
        facecolor="#ffffff",
        edgecolor="#d1d5db",
    )

    plt.tight_layout()
    suffix = f"_{max_nfe}nfe" if max_nfe <= 100 else ""
    png_path = os.path.join(output_dir, f"{filename_prefix}{suffix}.png")
    pdf_path = os.path.join(output_dir, f"{filename_prefix}{suffix}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"已导出收敛图: {png_path} & {pdf_path}")


# ==============================================================================
# 3. 柱状图性能对比 (Bar Comparison)
# ==============================================================================

def plot_bar_comparison(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    baseline_p = float(meta.get("baseline_power_kW", 25593.2))

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    means = []
    stds = []
    labels = []
    colors = []

    for method in methods:
        runs = results.get(method, [])
        if not runs:
            continue
        scores = np.array([r["best_score"] for r in runs])
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        labels.append(LABELS[method])
        colors.append(COLORS[method])

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    x_pos = np.arange(len(methods))

    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p/1e3:.2f} MW)",
        zorder=2,
    )

    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        alpha=0.88,
        edgecolor="#333333",
        linewidth=1.2,
        width=0.48,
        zorder=3,
    )

    # 标注数值与相对基准提升百分比
    for bar, m, s in zip(bars, means, stds):
        h = bar.get_height()
        rel_boost = (m - baseline_p) / baseline_p * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + s + 180,
            f"{m/1e3:.2f} MW\n(+{rel_boost:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#1f2937",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Optimized Aerodynamic Power (kW)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Final Optimized Rotor Power Comparison on IEA 22MW Blade\n"
        "(Mean $\\pm$ Std over 10 Independent Seeds, NFE = 300)",
        fontsize=12,
        pad=12,
    )

    y_min = baseline_p * 0.95
    y_max = max(means) * 1.15
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10.5, facecolor="#ffffff", edgecolor="#d1d5db")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_bar_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_bar_comparison.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"已导出柱状图: {png_path} & {pdf_path}")


# ==============================================================================
# 4. 箱线图统计对比 (Boxplot Comparison)
# ==============================================================================

def plot_boxplot_comparison(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    baseline_p = float(meta.get("baseline_power_kW", 25593.2))

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    box_data = []
    labels = []

    for method in methods:
        runs = results.get(method, [])
        if not runs:
            continue
        scores = [r["best_score"] for r in runs]
        box_data.append(scores)
        labels.append(LABELS[method])

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)

    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p/1e3:.2f} MW)",
        zorder=2,
    )

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.45,
        medianprops=dict(color="#111827", linewidth=2.0),
        flierprops=dict(marker="o", markersize=6, markerfacecolor="red", alpha=0.8),
        zorder=3,
    )

    palette = [COLOR_TURBO, COLOR_LLM_META, COLOR_MALIBO]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.3)

    # 散点抖动展示全部 10 个独立种子样本点
    rng = np.random.RandomState(42)
    for i, scores in enumerate(box_data):
        jitter = rng.uniform(-0.08, 0.08, size=len(scores))
        ax.scatter(
            np.full(len(scores), i + 1) + jitter,
            scores,
            color=palette[i],
            edgecolor="#1f2937",
            s=42,
            alpha=0.92,
            zorder=4,
        )

    ax.set_ylabel("Final Optimized Power (kW)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Robustness and Stability Distribution Across 10 Seeds\n"
        "(IEA-22-280-RWT 22MW Blade, NFE = 300)",
        fontsize=12,
        pad=12,
    )
    y_min = baseline_p * 0.95
    y_max = max(max(s) for s in box_data) * 1.05
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10.5, facecolor="#ffffff", edgecolor="#d1d5db")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_boxplot_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_boxplot_comparison.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"已导出箱线图: {png_path} & {pdf_path}")


# ==============================================================================
# 5. 叶片展向弦长与扭角分布对比 (Chord & Twist Distributions)
# ==============================================================================

def plot_chord_twist_distributions(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    x0 = get_baseline_blade_params()

    best_blades = {"Baseline": x0}
    for m in ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]:
        runs = results.get(m, [])
        if runs:
            best_r = max(runs, key=lambda r: r["best_score"])
            best_blades[m] = np.array(best_r["best_x"])

    fig, (ax_c, ax_tw) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    for method, x_vec in best_blades.items():
        span_d = interpolate_blade_span(x_vec, n_stations=30)
        r_frac = span_d["r"] / ROTOR_RADIUS
        chord = span_d["chord"]
        twist = span_d["twist_deg"]
        color = COLORS[method]
        label = LABELS[method]

        ls = "--" if method == "Baseline" else "-"
        lw = 2.0 if method != "MALIBO" else 2.6
        zorder = 2 if method == "Baseline" else 3

        ax_c.plot(r_frac, chord, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_tw.plot(r_frac, twist, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)

    # 标注 6 个核心站位位置
    for rf in STATION_RADIANS_FRAC:
        ax_c.axvline(rf, color="#9ca3af", linestyle=":", alpha=0.6, zorder=1)
        ax_tw.axvline(rf, color="#9ca3af", linestyle=":", alpha=0.6, zorder=1)

    ax_c.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Chord Length $c$ (m)", fontsize=11, fontweight="bold")
    ax_c.set_title("Spanwise Chord Distribution $c(r/R)$", fontsize=12, pad=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    ax_tw.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_tw.set_ylabel("Twist Angle $\\beta$ (deg)", fontsize=11, fontweight="bold")
    ax_tw.set_title("Spanwise Twist Angle Distribution $\\beta(r/R)$", fontsize=12, pad=10)
    ax_tw.grid(True, linestyle="--", alpha=0.5)
    ax_tw.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    plt.suptitle("IEA-22-280-RWT Blade Aerodynamic Planform Parameter Distributions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_chord_twist_distributions.png")
    pdf_path = os.path.join(output_dir, "blade_chord_twist_distributions.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已导出弦长/扭角分布图: {png_path} & {pdf_path}")


# ==============================================================================
# 6. 6 截面翼型轮廓多子图学术对比 (Station Airfoil Profiles)
# ==============================================================================

def plot_station_airfoils(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    x0 = get_baseline_blade_params()

    best_blades = {"Baseline": x0}
    for m in ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]:
        runs = results.get(m, [])
        if runs:
            best_r = max(runs, key=lambda r: r["best_score"])
            best_blades[m] = np.array(best_r["best_x"])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=300)
    axes = axes.flatten()

    x_c = np.linspace(0.0, 1.0, 100)
    station_names = [
        "Station 1 (Root Transition, $r=20.8\\mathrm{m}$)",
        "Station 2 (Shoulder Max Chord, $r=30.3\\mathrm{m}$)",
        "Station 3 (Inboard Region, $r=56.5\\mathrm{m}$)",
        "Station 4 (Mid-Span Power, $r=82.6\\mathrm{m}$)",
        "Station 5 (Outboard High Mach, $r=108.7\\mathrm{m}$)",
        "Station 6 (Tip Washout, $r=132.5\\mathrm{m}$)",
    ]

    for k in range(NUM_STATIONS):
        ax = axes[k]
        for method, x_vec in best_blades.items():
            st_list = unpack_blade_params(x_vec)
            st_k = st_list[k]
            th = st_k["theta"]
            c_val = st_k["chord"]
            tw_val = st_k["twist"]

            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8],
                lower_weights=th[8:16],
                leading_edge_weight=float(th[16]),
                TE_thickness=float(th[17]),
            )
            try:
                yu = k_af.upper_coordinates(x_c)[:, 1]
                yl = k_af.lower_coordinates(x_c)[:, 1]
            except Exception:
                yu = 0.1 * np.sin(x_c * np.pi)
                yl = -0.1 * np.sin(x_c * np.pi)

            color = COLORS[method]
            label = LABELS[method] if k == 0 else None
            ls = "--" if method == "Baseline" else "-"
            lw = 1.8 if method != "MALIBO" else 2.4

            ax.plot(x_c, yu, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.plot(x_c, yl, color=color, linestyle=ls, linewidth=lw)

        ax.set_title(station_names[k], fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("$x/c$", fontsize=10)
        ax.set_ylabel("$y/c$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.32, 0.35)
        ax.set_aspect("equal", adjustable="box")

    # 在顶部设置全局统一 Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=11,
    )

    plt.suptitle(
        "Cross-Sectional Airfoil Profile Evolution Across 6 Control Stations\n"
        "(IEA-22-280-RWT 22MW Rotor Blade)",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_station_airfoils.png")
    pdf_path = os.path.join(output_dir, "blade_station_airfoils.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已导出 6 站位翼型对比图: {png_path} & {pdf_path}")


# ==============================================================================
# 7. 3D 叶片空间透视渲染图 (3D Blade Surface Perspective)
# ==============================================================================

def plot_3d_blade_rendering(data: Dict[str, Any], output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    x0 = get_baseline_blade_params()

    # 获取 MALIBO 最优叶片
    malibo_runs = results.get("MALIBO", [])
    if malibo_runs:
        best_r = max(malibo_runs, key=lambda r: r["best_score"])
        x_opt = np.array(best_r["best_x"])
    else:
        x_opt = x0.copy()

    X0, Y0, Z0 = loft_blade_3d(x0, n_span=32, n_chord=50)
    Xopt, Yopt, Zopt = loft_blade_3d(x_opt, n_span=32, n_chord=50)

    fig = plt.figure(figsize=(12, 6.5), dpi=300)

    # 子图 1: 基准构型 Baseline
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        Z0, X0, Y0,
        rstride=1, cstride=1,
        cmap="coolwarm",
        edgecolor="#222222",
        linewidth=0.15,
        alpha=0.92,
        antialiased=True,
    )
    ax1.set_title("Baseline Blade Geometry (IEA 22MW)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Span $r$ (m)", fontsize=9.5, labelpad=8)
    ax1.set_ylabel("Chord $x$ (m)", fontsize=9.5, labelpad=8)
    ax1.set_zlabel("Thickness $y$ (m)", fontsize=9.5, labelpad=6)
    ax1.view_init(elev=20, azim=-65)
    ax1.set_box_aspect([3.5, 1.0, 0.45])

    # 子图 2: MALIBO 优化构型
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        Zopt, Xopt, Yopt,
        rstride=1, cstride=1,
        cmap="viridis",
        edgecolor="#222222",
        linewidth=0.15,
        alpha=0.92,
        antialiased=True,
    )
    ax2.set_title("MALIBO Optimized Blade Geometry", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Span $r$ (m)", fontsize=9.5, labelpad=8)
    ax2.set_ylabel("Chord $x$ (m)", fontsize=9.5, labelpad=8)
    ax2.set_zlabel("Thickness $y$ (m)", fontsize=9.5, labelpad=6)
    ax2.view_init(elev=20, azim=-65)
    ax2.set_box_aspect([3.5, 1.0, 0.45])

    fig.colorbar(surf2, ax=[ax1, ax2], shrink=0.55, aspect=15, pad=0.04, label="Lofted Surface Coordinate")
    plt.suptitle("3D Lofted Blade Geometric Surface Comparison", fontsize=13, fontweight="bold", y=0.96)

    png_path = os.path.join(output_dir, "blade_3d_rendering.png")
    pdf_path = os.path.join(output_dir, "blade_3d_rendering.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已导出 3D 渲染图: {png_path} & {pdf_path}")


def generate_all_figures(json_path: str = "results/benchmark_results.json", output_dir: str = "figures"):
    data = load_results(json_path)
    print(f"正在基于 {json_path} 批量生成全套出版级学术图表...")
    # 1. 300 NFE 收敛图
    plot_convergence_curves(data, output_dir=output_dir, max_nfe=300)
    # 2. 100 NFE 聚焦收敛图 (严格满足用户前 100 NFE 聚焦要求)
    plot_convergence_curves(data, output_dir=output_dir, max_nfe=100)
    # 3. 柱状图
    plot_bar_comparison(data, output_dir=output_dir)
    # 4. 箱线图
    plot_boxplot_comparison(data, output_dir=output_dir)
    # 5. 展向弦长与扭角分布
    plot_chord_twist_distributions(data, output_dir=output_dir)
    # 6. 6 截面翼型轮廓对比
    plot_station_airfoils(data, output_dir=output_dir)
    # 7. 3D 叶片空间透视渲染图
    plot_3d_blade_rendering(data, output_dir=output_dir)
    print("全套图表生成完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制 3D 风力机叶片全量学术图表")
    parser.add_argument("--input", type=str, default="results/benchmark_results.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    generate_all_figures(args.input, args.output_dir)
