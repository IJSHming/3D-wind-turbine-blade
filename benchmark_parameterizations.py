"""3D 风力机叶片不同参数化方法与约束优化硬件对比评测 (benchmark_parameterizations.py):
系统性对比评估以下三种参数化建模方法在 L40S GPU + 10核 CPU 硬件平台上的综合表现：
1. Standard 6-Station Model (D=120): 经典 6 截面离散控制 + 升级的前缘半径与曲率平滑约束
2. B-Spline Modal Model (D=132): 6 核心翼型站位 CST(108维) + 12 阶 B 样条弦长 + 12 阶 B 样条扭角 (天然保证 C² 连续)
3. Discrete 29-Station Model (D=580): 29 截面直接高维离散控制 (580维) + 差分曲率平滑约束

对比评测维度：
- 算力耗时：单步评估耗时 (s/step)、总优化耗时 (s)
- 收敛能力：100 NFE 收敛曲线、最终输出气动功率增益 (%)、Cp 与推力比
- 工业可制造性与平滑度：弦长二阶差分方差 Var(Δ²c)、扭角二阶差分方差 Var(Δ²β)、二阶导拐点振荡次数 (Zero-crossings)、前缘尖角抑制效果
"""

import json
import os
import time
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

import aerosandbox as asb
from src.evaluator import WindTurbineBladeEvaluator
from src.optimizers.turbo import TuRBOOptimizer
from src.parameterizations import (
    BSplineBladeModel,
    Stations6BladeModel,
    Stations25BladeModel,
)


def run_parameterization_benchmark(
    nfe: int = 100,
    seed: int = 42,
    output_dir: str = "results",
    fig_dir: str = "figures",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print(f"开始执行 3D 风机叶片参数化表示与平滑度基准评测 (NFE={nfe}, Seed={seed})")
    print("=" * 80)

    configs = [
        ("Stations6_D120", "Standard 6-Station Model (D=120)", Stations6BladeModel),
        ("BSpline_D132", "B-Spline Modal Model (D=132)", BSplineBladeModel),
        ("Stations25_D500", "Discrete 25-Station Model (D=500)", Stations25BladeModel),
    ]

    benchmark_summary: Dict[str, Any] = {
        "metadata": {
            "nfe": nfe,
            "seed": seed,
            "wind_speed": 16.26,
            "rpm": 7.55,
            "pitch_angle_deg": 0.0,
            "neuralfoil_model": "xxxlarge",
        },
        "models": {},
    }

    for key, label, model_cls in configs:
        print(f"\n>>> 正在评测模型: {label} ...")
        param_model = model_cls()
        evaluator = WindTurbineBladeEvaluator(
            wind_speed=16.26,
            rpm=7.55,
            pitch_angle_deg=0.0,
            param_model=param_model,
        )

        x0 = param_model.get_x0()
        bounds = param_model.get_bounds()
        dim = param_model.dim

        # 记录初始基线详尽评估
        baseline_det = evaluator.evaluate_detailed(x0)
        p0 = float(baseline_det["power_kW"])

        # 启动 TuRBO-1 优化器
        t_start = time.time()
        optimizer = TuRBOOptimizer(
            evaluator=evaluator,
            x0=x0,
            bounds=bounds,
            max_evals=nfe,
            seed=seed,
        )
        res = optimizer.run()
        t_total = time.time() - t_start
        t_per_step = t_total / float(nfe)

        best_x = np.array(res["best_x"])
        det_opt = evaluator.evaluate_detailed(best_x)

        # 提取 58 截面弦长与扭角分布
        span_r = np.array(det_opt["span_radii"])
        span_c = np.array(det_opt["span_chord"])
        span_tw = np.array(det_opt["span_twist_deg"])

        # 计算几何二阶差分平滑度指标 (反映曲率与波浪感)
        d2_c = np.diff(span_c, 2)
        d2_tw = np.diff(span_tw, 2)

        var_d2_c = float(np.var(d2_c))
        max_d2_c = float(np.max(np.abs(d2_c)))
        var_d2_tw = float(np.var(d2_tw))
        max_d2_tw = float(np.max(np.abs(d2_tw)))

        # 统计二阶导变号次数 (拐点正负震荡波浪数)
        s_d2_c = np.sign(d2_c)
        s_d2_c = s_d2_c[s_d2_c != 0]
        c_inflections = int(np.sum(s_d2_c[:-1] != s_d2_c[1:])) if len(s_d2_c) > 1 else 0

        s_d2_tw = np.sign(d2_tw)
        s_d2_tw = s_d2_tw[s_d2_tw != 0]
        tw_inflections = int(np.sum(s_d2_tw[:-1] != s_d2_tw[1:])) if len(s_d2_tw) > 1 else 0

        p_opt = float(det_opt["power_kW"])
        gain_pct = (p_opt - p0) / p0 * 100.0

        # 前缘尖角抑制检验: 提取控制站位最小前缘半径
        min_le_r = float("inf")
        if key == "Stations6_D120":
            from src.blade_geometry import unpack_blade_params
            st_list = unpack_blade_params(best_x)
            thetas = [s["theta"] for s in st_list]
        elif hasattr(param_model, "thetas_0"):
            thetas = best_x[:108].reshape((6, 18))
        elif key == "Stations25_D500":
            thetas = best_x.reshape((25, 20))[:, :18]
        else:
            thetas = []

        for th in thetas:
            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8], lower_weights=th[8:16],
                leading_edge_weight=float(th[16]), TE_thickness=float(th[17]),
            )
            try:
                le = float(k_af.LE_radius())
                min_le_r = min(min_le_r, le)
            except Exception:
                pass

        model_result = {
            "name": label,
            "dim": dim,
            "t_total_s": round(t_total, 3),
            "t_per_step_s": round(t_per_step, 4),
            "initial_power_kW": round(p0, 2),
            "best_power_kW": round(p_opt, 2),
            "power_gain_pct": round(gain_pct, 2),
            "thrust_kN": round(float(det_opt["thrust_kN"]), 2),
            "thrust_ratio": round(float(det_opt["thrust_kN"]) / float(baseline_det["thrust_kN"]), 4),
            "feasible": bool(det_opt.get("feasible", False)),
            "var_d2_chord": round(var_d2_c, 6),
            "max_abs_d2_chord": round(max_d2_c, 4),
            "chord_inflections": c_inflections,
            "var_d2_twist": round(var_d2_tw, 6),
            "max_abs_d2_twist": round(max_d2_tw, 4),
            "twist_inflections": tw_inflections,
            "min_le_radius": round(min_le_r, 5),
            "history": [round(float(v), 2) for v in res["history"]],
            "span_r": [round(float(v), 2) for v in span_r],
            "span_chord": [round(float(v), 4) for v in span_c],
            "span_twist_deg": [round(float(v), 4) for v in span_tw],
            "best_x": [round(float(v), 5) for v in best_x],
        }

        benchmark_summary["models"][key] = model_result

        print(f"  ✓ 耗时: 总计 {t_total:.2f}s ({t_per_step*1000:.1f}ms/步)")
        print(f"  ✓ 气动功率: {p0:.1f} kW -> {p_opt:.1f} kW (+{gain_pct:.2f}%)")
        print(f"  ✓ 平滑度: Var(Δ²c)={var_d2_c:.6f} (拐点:{c_inflections}), Var(Δ²β)={var_d2_tw:.6f} (拐点:{tw_inflections})")
        print(f"  ✓ 最小前缘半径: {min_le_r:.4f}, 可行解: {det_opt.get('feasible', False)}")

    # 保存基准评测数据
    json_path = os.path.join(output_dir, "param_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)
    print(f"\n基准评测结果已持久化保存至: {json_path}")

    # 生成对比图表
    plot_benchmark_figures(benchmark_summary, fig_dir=fig_dir)

    # 打印学术对比表格
    print_benchmark_table(benchmark_summary)

    return benchmark_summary


def plot_benchmark_figures(summary: Dict[str, Any], fig_dir: str = "figures"):
    models = summary["models"]
    nfe = summary["metadata"]["nfe"]

    palette = {
        "Stations6_D120": "#2ca02c",   # 绿色
        "BSpline_D132": "#1f77b4",     # 蓝色 (推荐)
        "Stations25_D500": "#d62728",  # 红色
    }

    # =========================================================================
    # 图 1: 收敛曲线对比 (Power vs NFE)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    steps = np.arange(1, nfe + 1)

    for k, m in models.items():
        hist = np.array(m["history"][:nfe])
        color = palette.get(k, "#333333")
        label = f"{m['name']} (Final: {m['best_power_kW']:.1f} kW, {m['t_per_step_s']*1000:.1f} ms/step)"
        ax.plot(steps, hist, label=label, color=color, linewidth=2.2)

    ax.set_title("TuRBO-1 Convergence Comparison Across Parameterizations (L40S GPU)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Rotor Aerodynamic Power (kW)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.0)
    plt.tight_layout()

    conv_png = os.path.join(fig_dir, "param_benchmark_convergence.png")
    conv_pdf = os.path.join(fig_dir, "param_benchmark_convergence.pdf")
    fig.savefig(conv_png, dpi=300)
    fig.savefig(conv_pdf)
    plt.close(fig)
    print(f"已生成收敛对比图: {conv_png} & {conv_pdf}")

    # =========================================================================
    # 图 2: 弦长与扭角分布及其二阶导数曲率平滑度对比 (4 子图深入剖析波浪感)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=300)

    ax_c = axes[0, 0]
    ax_d2c = axes[0, 1]
    ax_tw = axes[1, 0]
    ax_d2tw = axes[1, 1]

    for k, m in models.items():
        color = palette.get(k, "#333333")
        r = np.array(m["span_r"])
        c = np.array(m["span_chord"])
        tw = np.array(m["span_twist_deg"])

        # 1. 弦长分布
        ax_c.plot(r, c, color=color, linewidth=2.0, label=m["name"])

        # 2. 弦长二阶差分 (曲率振荡)
        r_mid_c = r[1:-1]
        d2_c = np.diff(c, 2)
        ax_d2c.plot(r_mid_c, d2_c, color=color, linewidth=1.8, marker="o", markersize=3, label=f"{m['name']} (Var={m['var_d2_chord']:.4f})")

        # 3. 扭角分布
        ax_tw.plot(r, tw, color=color, linewidth=2.0, label=m["name"])

        # 4. 扭角二阶差分
        r_mid_tw = r[1:-1]
        d2_tw = np.diff(tw, 2)
        ax_d2tw.plot(r_mid_tw, d2_tw, color=color, linewidth=1.8, marker="s", markersize=3, label=f"{m['name']} (Var={m['var_d2_twist']:.4f})")

    ax_c.set_title("(a) Optimized Chord Distribution $c(r)$", fontsize=10.5, fontweight="bold")
    ax_c.set_xlabel("Spanwise Radius $r$ (m)", fontsize=9.5)
    ax_c.set_ylabel("Chord Length $c$ (m)", fontsize=9.5)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=8.5)

    ax_d2c.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax_d2c.set_title(r"(b) Chord Second Difference $\Delta^2 c$ (Waviness/Curvature)", fontsize=10.5, fontweight="bold")
    ax_d2c.set_xlabel("Spanwise Radius $r$ (m)", fontsize=9.5)
    ax_d2c.set_ylabel(r"$\Delta^2 c$ ($\mathrm{m}$)", fontsize=9.5)
    ax_d2c.grid(True, linestyle="--", alpha=0.5)
    ax_d2c.legend(fontsize=8.5)

    ax_tw.set_title(r"(c) Optimized Twist Distribution $\beta(r)$", fontsize=10.5, fontweight="bold")
    ax_tw.set_xlabel("Spanwise Radius $r$ (m)", fontsize=9.5)
    ax_tw.set_ylabel(r"Twist Angle $\beta$ (deg)", fontsize=9.5)
    ax_tw.grid(True, linestyle="--", alpha=0.5)
    ax_tw.legend(fontsize=8.5)

    ax_d2tw.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax_d2tw.set_title(r"(d) Twist Second Difference $\Delta^2 \beta$ (Inflection Oscillations)", fontsize=10.5, fontweight="bold")
    ax_d2tw.set_xlabel("Spanwise Radius $r$ (m)", fontsize=9.5)
    ax_d2tw.set_ylabel(r"$\Delta^2 \beta$ ($\mathrm{deg}$)", fontsize=9.5)
    ax_d2tw.grid(True, linestyle="--", alpha=0.5)
    ax_d2tw.legend(fontsize=8.5)

    plt.suptitle("Geometric Smoothness & Curvature Waviness Diagnostic (58 Aerodynamic Sections)", fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout()

    smooth_png = os.path.join(fig_dir, "param_benchmark_smoothness.png")
    smooth_pdf = os.path.join(fig_dir, "param_benchmark_smoothness.pdf")
    fig.savefig(smooth_png, dpi=300)
    fig.savefig(smooth_pdf)
    plt.close(fig)
    print(f"已生成平滑度与波浪感诊断图: {smooth_png} & {smooth_pdf}")


def print_benchmark_table(summary: Dict[str, Any]):
    models = summary["models"]
    print("\n" + "=" * 110)
    print("【评测总结】3D 风力机叶片不同参数化表示方法性能、耗时与几何光滑度全面对比")
    print("=" * 110)
    header = (
        f"{'参数化模型':<32} | {'维度 D':<6} | {'单步耗时':<9} | {'总耗时':<7} | "
        f"{'初始功率':<9} | {'最优功率':<9} | {'增益':<6} | "
        f"{'Var(Δ²c)':<10} | {'Var(Δ²β)':<10} | {'拐点振荡数':<10}"
    )
    print(header)
    print("-" * 110)
    for k, m in models.items():
        row = (
            f"{m['name']:<32} | {m['dim']:<6} | "
            f"{m['t_per_step_s']*1000:6.1f} ms | {m['t_total_s']:5.1f} s | "
            f"{m['initial_power_kW']:8.1f} kW | {m['best_power_kW']:8.1f} kW | "
            f"+{m['power_gain_pct']:4.2f}% | "
            f"{m['var_d2_chord']:10.6f} | {m['var_d2_twist']:10.6f} | "
            f"c:{m['chord_inflections']}, tw:{m['twist_inflections']}"
        )
        print(row)
    print("=" * 110)


if __name__ == "__main__":
    run_parameterization_benchmark(nfe=100, seed=42)
