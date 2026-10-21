"""3D 风力机叶片全量基准测试脚本 (benchmark.py):
严格对齐学术对比规范与工作区标准：
在统一基准工况 (V_inf = 9.0 m/s, rpm = 10.0, 300 次 NFE 函数评估预算) 下，
完整评估 3 种前沿优化方法在 10 个独立随机种子下的表现：
1. TuRBO (基于算法发明者官方推荐的高维信赖域贝叶斯优化器);
2. LLM-Meta-TuRBO (结合风机气动领域大模型元控制与激进偏移量注入);
3. MALIBO (多工况元学习无似然贝叶斯元优化器).

产出数据持久化至 results/benchmark_results.json 与 results/benchmark_results.npz。
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.blade_geometry import (
    get_baseline_blade_params,
    interpolate_blade_span,
    loft_blade_3d,
    unpack_blade_params,
)
from src.evaluator import WindTurbineBladeEvaluator
from src.llm_agent import LLMMetaBladeAgent
from src.malibo.config import ResFFNConfig, MALIBOBladeConfig
from src.malibo.models import MALIBOMetaModel
from src.malibo.optimizer import MALIBOBladeOptimizer
from src.optimizers.llm_turbo import LLMMetaTuRBOOptimizer
from src.optimizers.turbo import TuRBOOptimizer


# 10 个标准独立随机种子 (与学术基准完全一致)
BENCHMARK_SEEDS = [42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]


def run_benchmark(
    wind_speed: float = 11.0,
    rpm: float = 7.061,
    max_evals: int = 300,
    seeds: List[int] = BENCHMARK_SEEDS,
    output_json: str = "results/benchmark_results.json",
    output_npz: str = "results/benchmark_results.npz",
    malibo_checkpoint: str = "weights/malibo_checkpoint.pt",
    skip_turbo: bool = False,
    skip_llm_meta: bool = False,
    skip_malibo: bool = False,
) -> Dict[str, Any]:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    evaluator = WindTurbineBladeEvaluator(wind_speed=wind_speed, rpm=rpm)
    x0 = get_baseline_blade_params()
    base_det = evaluator.evaluate_detailed(x0)
    baseline_power = float(base_det["power_kW"])
    baseline_cp = float(base_det["Cp"])
    baseline_torque = float(base_det["torque_kNm"])

    print("=" * 84)
    print(" 3D 风力机叶片气动优化基准测试 (3D-Wind-Turbine-Blade Benchmark)")
    print(f" 测试工况: 风速 V_inf = {wind_speed:.1f} m/s, 转速 = {rpm:.3f} rpm (omega = {evaluator.omega:.4f} rad/s)")
    print(f" 评估预算: NFE = {max_evals} 次 / 种子, 随机种子集合 ({len(seeds)}个): {seeds}")
    print(f" 基准构型 (IEA-22-280-RWT 22MW Scale): 功率 = {baseline_power:.2f} kW ({baseline_power/1e3:.3f} MW), Cp = {baseline_cp:.4f}, 转矩 = {baseline_torque:.2f} kN*m")
    print("=" * 84)

    # 读取已有结果支持断点续跑
    existing_results: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                saved_blob = json.load(f)
                existing_results = saved_blob.get("results", {})
        except Exception:
            existing_results = {}

    all_results: Dict[str, List[Dict[str, Any]]] = {
        "TuRBO": existing_results.get("TuRBO", []),
        "LLM-Meta-TuRBO": existing_results.get("LLM-Meta-TuRBO", []),
        "MALIBO": existing_results.get("MALIBO", []),
    }

    # ==========================================================================
    # 1. 运行 TuRBO
    # ==========================================================================
    if not skip_turbo:
        print(f"\n[1/3] 启动 TuRBO-1 优化实验 (10 种子)...")
        existing_seeds = {r["seed"] for r in all_results["TuRBO"]}
        for idx, s in enumerate(seeds):
            if s in existing_seeds:
                print(f"  --> [TuRBO Seed {s}] 已存在，跳过。")
                continue
            print(f"  >>> [TuRBO Seed {s:4d} ({idx+1}/{len(seeds)})] 正在执行 {max_evals} 步优化...")
            opt = TuRBOOptimizer(evaluator=evaluator, x0=x0, max_evals=max_evals, seed=s)
            res = opt.run()
            all_results["TuRBO"].append(res)
            print(
                f"      完成! 耗时: {res['elapsed_seconds']:.2f}s | "
                f"最终功率: {res['best_score']:.2f} kW ({res['detailed']['power_MW']:.3f} MW) | "
                f"Cp: {res['detailed']['Cp']:.4f} (+{(res['detailed']['Cp'] - baseline_cp)/baseline_cp*100:.2f}%)"
            )
            # 中途持久化
            _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, max_evals, seeds, output_json, output_npz)

    # ==========================================================================
    # 2. 运行 LLM-Meta-TuRBO
    # ==========================================================================
    if not skip_llm_meta:
        print(f"\n[2/3] 启动 LLM-Meta-TuRBO 优化实验 (10 种子)...")
        llm_agent = LLMMetaBladeAgent()
        existing_seeds = {r["seed"] for r in all_results["LLM-Meta-TuRBO"]}
        for idx, s in enumerate(seeds):
            if s in existing_seeds:
                print(f"  --> [LLM-Meta-TuRBO Seed {s}] 已存在，跳过。")
                continue
            print(f"  >>> [LLM-Meta-TuRBO Seed {s:4d} ({idx+1}/{len(seeds)})] 正在执行 {max_evals} 步优化...")
            opt = LLMMetaTuRBOOptimizer(
                evaluator=evaluator,
                x0=x0,
                max_evals=max_evals,
                seed=s,
                llm_agent=llm_agent,
            )
            res = opt.run()
            all_results["LLM-Meta-TuRBO"].append(res)
            n_inter = len(res.get("intervention_history", []))
            print(
                f"      完成! 耗时: {res['elapsed_seconds']:.2f}s | 干预次数: {n_inter} | "
                f"最终功率: {res['best_score']:.2f} kW ({res['detailed']['power_MW']:.3f} MW) | "
                f"Cp: {res['detailed']['Cp']:.4f} (+{(res['detailed']['Cp'] - baseline_cp)/baseline_cp*100:.2f}%)"
            )
            _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, max_evals, seeds, output_json, output_npz)

    # ==========================================================================
    # 3. 运行 MALIBO
    # ==========================================================================
    if not skip_malibo:
        print(f"\n[3/3] 启动 MALIBO 元贝叶斯优化实验 (10 种子)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(malibo_checkpoint):
            raise FileNotFoundError(f"未找到 MALIBO 预训练模型权重: {malibo_checkpoint}，请先执行 train_malibo.py！")

        ckpt = torch.load(malibo_checkpoint, map_location=device, weights_only=False)
        in_dim = ckpt.get("in_dim", 120)
        resffn_cfg = ResFFNConfig(in_dim=in_dim, feature_dim=ckpt["feature_dim"])
        malibo_model = MALIBOMetaModel(num_tasks=ckpt["num_tasks"], config=resffn_cfg)
        malibo_model.load_state_dict(ckpt["model_state_dict"])
        malibo_model.eval().to(device)

        existing_seeds = {r["seed"] for r in all_results["MALIBO"]}
        for idx, s in enumerate(seeds):
            if s in existing_seeds:
                print(f"  --> [MALIBO Seed {s}] 已存在，跳过。")
                continue
            print(f"  >>> [MALIBO Seed {s:4d} ({idx+1}/{len(seeds)})] 正在执行 {max_evals} 步优化...")
            opt = MALIBOBladeOptimizer(
                meta_model=malibo_model,
                evaluator=evaluator,
                x0=x0,
                max_evals=max_evals,
                seed=s,
                device=device,
            )
            res = opt.run()
            all_results["MALIBO"].append(res)
            print(
                f"      完成! 耗时: {res['elapsed_seconds']:.2f}s | "
                f"最终功率: {res['best_score']:.2f} kW ({res['detailed']['power_MW']:.3f} MW) | "
                f"Cp: {res['detailed']['Cp']:.4f} (+{(res['detailed']['Cp'] - baseline_cp)/baseline_cp*100:.2f}%)"
            )
            _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, max_evals, seeds, output_json, output_npz)

    # 打印最终统计汇总表格
    print_benchmark_summary_table(all_results, baseline_power, baseline_cp)
    return all_results


def _save_checkpoint(
    results: Dict[str, List[Dict[str, Any]]],
    baseline_power: float,
    baseline_cp: float,
    wind_speed: float,
    rpm: float,
    max_evals: int,
    seeds: List[int],
    output_json: str,
    output_npz: str,
):
    """持久化保存 JSON 与 NPZ 数据"""
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wind_speed": wind_speed,
            "rpm": rpm,
            "max_evals": max_evals,
            "seeds": seeds,
            "baseline_power_kW": baseline_power,
            "baseline_Cp": baseline_cp,
        },
        "results": results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 导出便于 NumPy 直接向量化加载的 NPZ 文件
    npz_dict = {}
    for method, data_list in results.items():
        if not data_list:
            continue
        scores = [r["best_score"] for r in data_list]
        cps = [r["detailed"]["Cp"] for r in data_list]
        times = [r["elapsed_seconds"] for r in data_list]
        histories = np.array([r["history"] for r in data_list])
        npz_dict[f"{method}_scores"] = np.array(scores)
        npz_dict[f"{method}_cps"] = np.array(cps)
        npz_dict[f"{method}_times"] = np.array(times)
        npz_dict[f"{method}_histories"] = histories

    np.savez_compressed(output_npz, **npz_dict)


def print_benchmark_summary_table(
    results: Dict[str, List[Dict[str, Any]]],
    baseline_power: float,
    baseline_cp: float,
):
    """以专业科研格式输出统计分析对比表格"""
    print("\n" + "=" * 96)
    print(" " * 28 + "3D 风力机叶片气动优化基准测试统计汇总报告")
    print("=" * 96)
    print(
        f"{'优化方法 (Method)':<18} | {'最佳功率 (kW)':<18} | {'Cp (风能利用率)':<18} | {'功率提升 (%)':<14} | {'平均耗时 (s)':<12}"
    )
    print("-" * 96)
    print(
        f"{'Baseline (NREL 5M)':<18} | {baseline_power:8.2f}            | {baseline_cp:6.4f}             | {'0.00%':<14} | {'--':<12}"
    )

    for method in ["TuRBO", "LLM-Meta-TuRBO", "MALIBO"]:
        data_list = results.get(method, [])
        if not data_list:
            continue
        scores = np.array([r["best_score"] for r in data_list])
        cps = np.array([r["detailed"]["Cp"] for r in data_list])
        times = np.array([r["elapsed_seconds"] for r in data_list])

        mean_p, std_p = np.mean(scores), np.std(scores)
        mean_cp, std_cp = np.mean(cps), np.std(cps)
        mean_t = np.mean(times)
        boost = (mean_p - baseline_power) / baseline_power * 100.0

        p_str = f"{mean_p:.1f} ± {std_p:.1f}"
        cp_str = f"{mean_cp:.4f} ± {std_cp:.4f}"
        boost_str = f"+{boost:.2f}%"
        t_str = f"{mean_t:.2f}s"

        print(f"{method:<18} | {p_str:<18} | {cp_str:<18} | {boost_str:<14} | {t_str:<12}")

    print("=" * 96 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D 风机叶片多方法基准测试")
    parser.add_argument("--wind-speed", type=float, default=11.0)
    parser.add_argument("--rpm", type=float, default=7.061)
    parser.add_argument("--max-evals", type=int, default=300)
    args = parser.parse_args()

    run_benchmark(wind_speed=args.wind_speed, rpm=args.rpm, max_evals=args.max_evals)
