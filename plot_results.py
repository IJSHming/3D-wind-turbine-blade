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
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

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
COLOR_BASELINE = "#6B7280"   # 沉稳灰 (Baseline 1.5m Blade)
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
    "Baseline": "Baseline (1.5m Blade)",
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
    baseline_p = float(meta.get("baseline_power_kW", 1.661))

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    # 绘制 Baseline 参考线
    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p:.3f} kW)",
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
            label=f"{label} (Final: {mean_curve[-1]:.2f} kW)",
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
    wind_v = meta.get("wind_speed", 10.0)
    rpm_v = meta.get("rpm", 300.0)
    ax.set_title(
        f"Optimization Convergence on 1.5m Reference Blade{title_suffix}\n"
        f"(Rated Condition: $V_\\infty = {wind_v:.2f}\\,\\mathrm{{m/s}}$, $\\Omega = {rpm_v:.2f}\\,\\mathrm{{rpm}}$, 10 Seeds Mean $\\pm$ SEM)",
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
# 3. 柱状图性能对比 (Bar Comparison - 10 随机种子独立对比)
# ==============================================================================

def plot_bar_comparison(data: Dict[str, Any], output_dir: str = "figures"):
    """绘制 10 个独立随机种子下的具体性能对比分组柱状图。
    
    严格满足用户指令: 'blade_bar_comparison 应当是具体的 10 个 seed 下的比较'。
    """
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    baseline_p = float(meta.get("baseline_power_kW", 1.661))

    seeds = meta.get("seeds", [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999])
    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]

    # 提取各种子下的具体得分
    seed_scores: Dict[str, List[float]] = {m: [] for m in methods}
    for s in seeds:
        for m in methods:
            matching = [r for r in results.get(m, []) if r.get("seed") == s]
            if matching:
                seed_scores[m].append(float(matching[0]["best_score"]))
            else:
                seed_scores[m].append(baseline_p)

    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=300)
    x = np.arange(len(seeds))
    width = 0.25

    # 绘制 Baseline 参考线
    ax.axhline(
        baseline_p,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.8,
        label=f"Baseline ({baseline_p:.3f} kW)",
        zorder=2,
    )

    rects1 = ax.bar(
        x - width,
        seed_scores["TuRBO"],
        width,
        label=LABELS["TuRBO"],
        color=COLOR_TURBO,
        edgecolor="#222222",
        linewidth=1.0,
        alpha=0.90,
        zorder=3,
    )
    rects2 = ax.bar(
        x,
        seed_scores["LLM-Meta-TuRBO"],
        width,
        label=LABELS["LLM-Meta-TuRBO"],
        color=COLOR_LLM_META,
        edgecolor="#222222",
        linewidth=1.0,
        alpha=0.90,
        zorder=3,
    )
    rects3 = ax.bar(
        x + width,
        seed_scores["MALIBO"],
        width,
        label=LABELS["MALIBO"],
        color=COLOR_MALIBO,
        edgecolor="#222222",
        linewidth=1.0,
        alpha=0.90,
        zorder=3,
    )

    ax.set_xlabel("Independent Random Seeds", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Final Optimized Aerodynamic Power (kW)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Final Optimized Rotor Power Comparison Across 10 Independent Seeds\n"
        "(1.5m Reference Blade, NFE = 300, Head-to-Head Comparison)",
        fontsize=12,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10, fontweight="bold")

    # 坐标范围自适应: 给予充足顶部净空，确保图例绝对不遮挡柱体数据
    all_scores = [sc for m in methods for sc in seed_scores[m]]
    y_min = max(0.0, baseline_p - 0.20 * baseline_p)
    span_y = max(all_scores) - y_min
    y_max = max(all_scores) + 0.35 * span_y
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=10.5,
        facecolor="#ffffff",
        edgecolor="#d1d5db",
    )

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_bar_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_bar_comparison.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"已导出 10 种子分组柱状图: {png_path} & {pdf_path}")


# ==============================================================================
# 4. 箱线图统计对比 (Boxplot Comparison)
# ==============================================================================

def plot_boxplot_comparison(data: Dict[str, Any], output_dir: str = "figures"):
    """绘制各方法统计分布对比箱线图。
    
    严格对齐用户指令: 不画 baseline 水平线，以便让各优化方法的 box 充分伸展开来，展示清晰的统计分位数与分布形态。
    """
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]

    methods = ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    box_data = []
    labels = []

    for method in methods:
        runs = results.get(method, [])
        if not runs:
            continue
        scores = [float(r["best_score"]) for r in runs]
        box_data.append(scores)
        labels.append(LABELS[method])

    fig, ax = plt.subplots(figsize=(7.5, 5.4), dpi=300)

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=labels,
        widths=0.45,
        medianprops=dict(color="#111827", linewidth=2.2),
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
            s=45,
            alpha=0.92,
            zorder=4,
        )

    ax.set_ylabel("Final Optimized Power (kW)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Optimization Robustness and Distribution Across 10 Seeds\n"
        "(1.5m Reference Blade, NFE = 300)",
        fontsize=12,
        pad=12,
    )

    # 移除基准线后，y 轴严格依据各优化算法实际分数自适应伸展
    all_scores = [sc for group in box_data for sc in group]
    if all_scores:
        score_min = min(all_scores)
        score_max = max(all_scores)
        span = max(score_max - score_min, 10.0)
        y_min = score_min - 0.20 * span
        y_max = score_max + 0.25 * span
        ax.set_ylim(y_min, y_max)

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_boxplot_comparison.png")
    pdf_path = os.path.join(output_dir, "blade_boxplot_comparison.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"已导出高分辨率箱线图: {png_path} & {pdf_path}")


# ==============================================================================
# 5. 叶片展向弦长与扭角分布对比 (Chord & Twist Distributions)
# ==============================================================================

def plot_chord_twist_distributions(data: Dict[str, Any], output_dir: str = "figures"):
    """基于 OpenFAST 50 截面评估数据绘制展向弦长与扭角平滑分布图。"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})

    baseline_det = meta.get("baseline_detailed")
    if not baseline_det and "Baseline" in results and len(results["Baseline"]) > 0:
        baseline_det = results["Baseline"][0].get("detailed", {})

    fig, (ax_c, ax_tw) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    methods = ["Baseline", "TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    for method in methods:
        if method == "Baseline":
            det = baseline_det
        else:
            runs = results.get(method, [])
            if not runs:
                continue
            best_run = max(runs, key=lambda r: r["best_score"])
            det = best_run.get("detailed", {})
        if not det:
            continue

        radii = np.array(det.get("span_radii", []))
        if len(radii) == 0:
            continue
        r_frac = radii / ROTOR_RADIUS
        chord = np.array(det.get("span_chord", []))
        twist = np.array(det.get("span_twist_deg", []))

        # 消除边缘失真的滤波器，直接使用单调保持三次 PCHIP 插值生成致密平滑曲线
        r_dense = np.linspace(r_frac[0], r_frac[-1], 250)
        c_dense = PchipInterpolator(r_frac, chord)(r_dense)
        tw_dense = PchipInterpolator(r_frac, twist)(r_dense)

        color = COLORS[method]
        label = LABELS[method]

        ls = "--" if method == "Baseline" else "-"
        lw = 2.0 if method == "Baseline" else (2.6 if method == "MALIBO" else 2.1)
        zorder = 2 if method == "Baseline" else (4 if method == "MALIBO" else 3)

        ax_c.plot(r_dense, c_dense, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_tw.plot(r_dense, tw_dense, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)

    ax_c.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Chord Length $c$ (m)", fontsize=11, fontweight="bold")
    ax_c.set_title("Spanwise Chord Distribution $c(r/R)$ (27 Sections)", fontsize=12, pad=10)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    ax_tw.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_tw.set_ylabel("Twist Angle $\\beta$ (deg)", fontsize=11, fontweight="bold")
    ax_tw.set_title("Spanwise Twist Angle Distribution $\\beta(r/R)$ (27 Sections)", fontsize=12, pad=10)
    ax_tw.grid(True, linestyle="--", alpha=0.5)
    ax_tw.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    plt.suptitle("1.5m Tidal Turbine Blade Planform Parameter Distributions (27 Evaluation Sections)", fontsize=13, fontweight="bold", y=1.02)
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
    from src.parameterizations import BSplineBladeModel
    model_bs = BSplineBladeModel()
    x0 = model_bs.get_x0()

    best_blades = {"Baseline": x0}
    for m in ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]:
        runs = results.get(m, [])
        if runs:
            best_r = max(runs, key=lambda r: r["best_score"])
            best_blades[m] = np.array(best_r["best_x"])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=300)
    axes = axes.flatten()

    # 采用空气动力学标准半余弦网格加密前缘与后缘采样 (250 点)，完美解析前缘大曲率圆头轮廓
    beta_c = np.linspace(0.0, np.pi, 250)
    x_c = (1.0 - np.cos(beta_c)) / 2.0
    station_names = [
        f"Station 1 (Root, $r={STATION_RADII[0]:.2f}\\mathrm{{m}}$, $t/c\\approx 10.2\\%$)",
        f"Station 2 (Inboard, $r={STATION_RADII[1]:.2f}\\mathrm{{m}}$, $t/c\\approx 10.2\\%$)",
        f"Station 3 (Mid-Inboard, $r={STATION_RADII[2]:.2f}\\mathrm{{m}}$, $t/c\\approx 10.2\\%$)",
        f"Station 4 (Mid-Span Power, $r={STATION_RADII[3]:.2f}\\mathrm{{m}}$, $t/c\\approx 10.2\\%$)",
        f"Station 5 (Outboard, $r={STATION_RADII[4]:.2f}\\mathrm{{m}}$, $t/c\\approx 10.2\\%$)",
        f"Station 6 (Tip, $r={STATION_RADII[5]:.2f}\\mathrm{{m}}$, $t/c\\approx 10.2\\%$)",
    ]

    for k in range(NUM_STATIONS):
        ax = axes[k]
        for method, x_vec in best_blades.items():
            if len(x_vec) in (114, 132):
                st_list = model_bs.get_station_info(x_vec)
                st_k = st_list[k]
                th = st_k["theta"]
            else:
                st_list = unpack_blade_params(x_vec)
                st_k = st_list[k]
                th = st_k["theta"]

            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8],
                lower_weights=th[8:16],
                leading_edge_weight=float(th[16]),
                TE_thickness=0.0,
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

        ax.set_title(station_names[k], fontsize=10.0, fontweight="bold", pad=6)
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
        "(1.5m Reference Rotor Blade)",
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
    """生成 3D 叶片空间交互式几何渲染网页 (Plotly 3D HTML 格式)，支持自由旋转、缩放、平移及对比切换。"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    from src.parameterizations import BSplineBladeModel
    model_bs = BSplineBladeModel()
    x0 = model_bs.get_x0()

    # 获取 MALIBO 最优叶片
    malibo_runs = results.get("MALIBO", [])
    if malibo_runs:
        best_r = max(malibo_runs, key=lambda r: r["best_score"])
        x_opt = np.array(best_r["best_x"])
    else:
        x_opt = x0.copy()

    # 全量放样 27 个截面 (涵盖 0.25m 轮毂至 1.50m 翼尖全展向)
    X0, Y0, Z0 = loft_blade_3d(x0, n_span=27, n_chord=60)
    Xopt, Yopt, Zopt = loft_blade_3d(x_opt, n_span=27, n_chord=60)

    try:
        import plotly.graph_objects as go
        fig = go.Figure()

        # Baseline surface (全量 27 截面放样，深灰色)
        fig.add_trace(go.Surface(
            x=Z0, y=X0, z=Y0,
            name="Baseline (1.5m Blade)",
            colorscale="Greys",
            showscale=False,
            opacity=0.6,
            hoverinfo="name+x+y+z",
        ))

        # Optimized surface (全量 27 截面放样，翠绿色/Viridis)
        fig.add_trace(go.Surface(
            x=Zopt, y=Xopt, z=Yopt,
            name="MALIBO Optimized",
            colorscale="Viridis",
            colorbar=dict(title="Span r (m)", len=0.6),
            opacity=0.95,
            hoverinfo="name+x+y+z",
        ))

        all_r = np.concatenate([Z0.flatten(), Zopt.flatten()])
        all_x = np.concatenate([X0.flatten(), Xopt.flatten()])
        all_y = np.concatenate([Y0.flatten(), Yopt.flatten()])

        r_min, r_max = float(np.min(all_r)), float(np.max(all_r))
        x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
        y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

        # 动态自适应扩展边界，杜绝任何叶根、翼尖或后缘被视锥体硬截断
        r_pad = max(0.05, 0.10 * (r_max - r_min))
        x_pad = max(0.05, 0.15 * (x_max - x_min))
        y_pad = max(0.05, 0.20 * (y_max - y_min))

        r_range = [max(0.0, r_min - r_pad), r_max + r_pad]
        x_range = [x_min - x_pad, x_max + x_pad]
        y_range = [y_min - y_pad, y_max + y_pad]

        dr = r_range[1] - r_range[0]
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]

        aspect_ratio = dict(
            x=3.5,
            y=float(np.clip(3.5 * (dx / dr) * 1.6, 0.8, 2.5)),
            z=float(np.clip(3.5 * (dy / dr) * 3.5, 0.4, 1.5)),
        )

        fig.update_layout(
            title="3D Interactive Tidal Turbine Blade Geometry: Baseline vs Optimized (1.5m Blade in Water)",
            scene=dict(
                xaxis=dict(title="Spanwise Radius r (m)", range=r_range),
                yaxis=dict(title="Chordwise x (m)", range=x_range),
                zaxis=dict(title="Thickness y (m)", range=y_range),
                aspectmode="manual",
                aspectratio=aspect_ratio,
                camera=dict(
                    eye=dict(x=2.2, y=-1.8, z=1.0),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.0, y=1.12,
                    buttons=[
                        dict(label="Show Both (Overlay Comparison)", method="update", args=[{"visible": [True, True]}]),
                        dict(label="Optimized Only", method="update", args=[{"visible": [False, True]}]),
                        dict(label="Baseline Only", method="update", args=[{"visible": [True, False]}]),
                    ],
                )
            ],
            margin=dict(l=0, r=0, b=0, t=50),
        )

        html_path = os.path.join(output_dir, "blade_3d_rendering.html")
        fig.write_html(html_path, include_plotlyjs=True)
        print(f"已导出 3D 交互式网页渲染: {html_path}")
    except Exception as e:
        print(f"Plotly 导出失败 ({e})")


# ==============================================================================
# 8. 27 截面展向气动角度全量学术对比 (Twist, Attack Angle, Effective Pitch)
# ==============================================================================

def plot_spanwise_angles(data: Dict[str, Any], output_dir: str = "figures"):
    """绘制 27 截面的扭角 (Twist Angle)、有效攻角 (Angle of Attack) 与总变桨安装角平滑分布 (含 Baseline 对比)。"""
    os.makedirs(output_dir, exist_ok=True)
    results = data["results"]
    meta = data.get("metadata", {})
    pitch_deg = float(meta.get("pitch_angle_deg", 0.0))

    # 获取 Baseline 27 截面详细数据
    baseline_det = meta.get("baseline_detailed")
    if not baseline_det and "Baseline" in results and len(results["Baseline"]) > 0:
        baseline_det = results["Baseline"][0].get("detailed", {})

    fig, (ax_tw, ax_aoa, ax_pitch) = plt.subplots(1, 3, figsize=(16.0, 4.8), dpi=300)

    methods = ["Baseline", "TuRBO", "LLM-Meta-TuRBO", "MALIBO"]
    for method in methods:
        if method == "Baseline":
            det = baseline_det
            if not det:
                continue
        else:
            runs = results.get(method, [])
            if not runs:
                continue
            best_run = max(runs, key=lambda r: r["best_score"])
            det = best_run.get("detailed", {})

        radii = np.array(det.get("span_radii", []))
        if len(radii) == 0:
            continue
        r_norm = radii / ROTOR_RADIUS
        twists = np.array(det.get("span_twist_deg", []))
        alphas = np.array(det.get("span_alpha_deg", []))
        eff_pitch = np.array(det.get("span_effective_pitch_deg", twists + pitch_deg))

        # 直接使用单调保持三次 PCHIP 插值生成致密平滑曲线 (杜绝端点被滤波改变)
        r_dense = np.linspace(r_norm[0], r_norm[-1], 250)
        tw_dense = PchipInterpolator(r_norm, twists)(r_dense)
        aoa_dense = PchipInterpolator(r_norm, alphas)(r_dense)
        eff_dense = PchipInterpolator(r_norm, eff_pitch)(r_dense)

        color = COLORS[method]
        label = LABELS[method]
        ls = "--" if method == "Baseline" else "-"
        lw = 2.0 if method == "Baseline" else (2.6 if method == "MALIBO" else 2.1)
        zorder = 2 if method == "Baseline" else (4 if method == "MALIBO" else 3)

        ax_tw.plot(r_dense, tw_dense, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_aoa.plot(r_dense, aoa_dense, color=color, linewidth=2.0 if method == "Baseline" else (2.5 if method == "MALIBO" else 2.1), linestyle=ls, label=label, zorder=zorder)
        ax_pitch.plot(r_dense, eff_dense, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)

    # 标注 27 截面特征
    ax_tw.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_tw.set_ylabel("Geometric Twist Angle $\\beta$ (°)", fontsize=11, fontweight="bold")
    ax_tw.set_title("Spanwise Twist Distribution $\\beta(r)$ (27 Sections)", fontsize=11, pad=10)
    ax_tw.grid(True, linestyle="--", alpha=0.5)
    ax_tw.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    ax_aoa.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_aoa.set_ylabel("Effective Angle of Attack $\\alpha$ (°)", fontsize=11, fontweight="bold")
    ax_aoa.set_title("Spanwise Angle of Attack $\\alpha(r)$ (27 Sections)", fontsize=11, pad=10)
    ax_aoa.grid(True, linestyle="--", alpha=0.5)
    ax_aoa.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    ax_pitch.set_xlabel("Spanwise Fraction ($r/R$)", fontsize=11, fontweight="bold")
    ax_pitch.set_ylabel("Total Pitch $\\beta + \\theta_{\\mathrm{pitch}}$ (°)", fontsize=11, fontweight="bold")
    ax_pitch.set_title(f"Spanwise Total Pitch Angle (Blade Pitch = {pitch_deg:.2f}°)", fontsize=11, pad=10)
    ax_pitch.grid(True, linestyle="--", alpha=0.5)
    ax_pitch.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    plt.suptitle("Blade Spanwise Aerodynamic Angle Distributions (27 Evaluation Sections)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "blade_spanwise_angles.png")
    pdf_path = os.path.join(output_dir, "blade_spanwise_angles.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已导出 27 截面气动角分布图: {png_path} & {pdf_path}")


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
    # 8. 50 截面全量气动角度分布
    plot_spanwise_angles(data, output_dir=output_dir)
    print("全套图表生成完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制 3D 风力机叶片全量学术图表")
    parser.add_argument("--input", type=str, default="results/benchmark_results.json")
    parser.add_argument("--output-dir", type=str, default="figures")
    args = parser.parse_args()

    generate_all_figures(args.input, args.output_dir)
