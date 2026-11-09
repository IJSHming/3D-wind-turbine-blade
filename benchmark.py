"""3D 风力机叶片全量基准测试脚本 (benchmark.py):
严格对齐学术对比规范与工作区标准：
在统一基准工况 (V_inf = 11.0 m/s, rpm = 8.00 rpm, pitch = 0.0 deg, 300 次 NFE 函数评估预算) 下，
完整评估 3 种前沿优化方法在 10 个独立随机种子下的表现：
1. TuRBO (基于算法发明者官方推荐的高维信赖域贝叶斯优化器);
2. LLM-Meta-TuRBO (结合风机气动领域大模型元控制与激进偏移量注入);
3. MALIBO (多工况元学习无似然贝叶斯元优化器).

产出数据持久化至 results/benchmark_results.json 与 results/benchmark_results.npz。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing as mp
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


def _worker_init():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_turbo_seed_worker(args: Tuple[int, float, float, float, int, int]) -> Dict[str, Any]:
    s, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals = args
    ev = WindTurbineBladeEvaluator(
        wind_speed=wind_speed,
        rpm=rpm,
        pitch_angle_deg=pitch_angle_deg,
        n_bem_stations=n_bem_stations,
        max_evals=max_evals,
    )
    x0 = ev.baseline_x0
    opt = TuRBOOptimizer(evaluator=ev, x0=x0, max_evals=max_evals, seed=s)
    return opt.run()


def _run_llm_meta_seed_worker(args: Tuple[int, float, float, float, int, int]) -> Dict[str, Any]:
    s, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals = args
    ev = WindTurbineBladeEvaluator(
        wind_speed=wind_speed,
        rpm=rpm,
        pitch_angle_deg=pitch_angle_deg,
        n_bem_stations=n_bem_stations,
        max_evals=max_evals,
    )
    x0 = ev.baseline_x0
    llm_agent = LLMMetaBladeAgent()
    opt = LLMMetaTuRBOOptimizer(
        evaluator=ev,
        x0=x0,
        max_evals=max_evals,
        seed=s,
        llm_agent=llm_agent,
    )
    return opt.run()


def _run_malibo_seed_worker(args: Tuple[int, float, float, float, int, int, str]) -> Dict[str, Any]:
    s, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, malibo_checkpoint = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(malibo_checkpoint, map_location=device, weights_only=False)
    in_dim = ckpt.get("in_dim", 132)
    resffn_cfg = ResFFNConfig(in_dim=in_dim, feature_dim=ckpt["feature_dim"])
    malibo_model = MALIBOMetaModel(num_tasks=ckpt["num_tasks"], config=resffn_cfg)
    malibo_model.load_state_dict(ckpt["model_state_dict"])
    malibo_model.eval().to(device)

    ev = WindTurbineBladeEvaluator(
        wind_speed=wind_speed,
        rpm=rpm,
        pitch_angle_deg=pitch_angle_deg,
        n_bem_stations=n_bem_stations,
        max_evals=max_evals,
    )
    x0 = ev.baseline_x0
    opt = MALIBOBladeOptimizer(
        meta_model=malibo_model,
        evaluator=ev,
        x0=x0,
        max_evals=max_evals,
        seed=s,
        device=device,
    )
    return opt.run()


def run_benchmark(
    wind_speed: float = 11.0,
    rpm: float = 8.00,
    pitch_angle_deg: float = 0.0,
    n_bem_stations: int = 50,
    max_evals: int = 300,
    seeds: List[int] = BENCHMARK_SEEDS,
    output_json: str = "results/benchmark_results.json",
    output_npz: str = "results/benchmark_results.npz",
    malibo_checkpoint: str = "weights/malibo_checkpoint.pt",
    force_rerun: bool = False,
    skip_turbo: bool = False,
    skip_llm_meta: bool = False,
    skip_malibo: bool = False,
    n_workers: int = 5,
) -> Dict[str, Any]:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    evaluator = WindTurbineBladeEvaluator(
        wind_speed=wind_speed,
        rpm=rpm,
        pitch_angle_deg=pitch_angle_deg,
        n_bem_stations=n_bem_stations,
        max_evals=max_evals,
    )
    base_det = evaluator.get_baseline_detailed()
    baseline_power = float(base_det["power_kW"])
    baseline_cp = float(base_det["Cp"])
    baseline_torque = float(base_det["torque_kNm"])

    print("=" * 84)
    print(" 3D 风力机叶片气动优化基准测试 (3D-Wind-Turbine-Blade Benchmark)")
    print(f" 测试工况 (严格对齐 OpenFAST AeroDyn15 10MW):")
    print(f"   风速 V_inf = {wind_speed:.2f} m/s, 转速 = {rpm:.5f} rpm (omega = {evaluator.omega:.5f} rad/s)")
    print(f"   叶片安装/变桨角 = {pitch_angle_deg:.5f} deg, 展向气动离散微元数 = {n_bem_stations} 个截面")
    print(f" 优化参数化模型: B-Spline Modal Model (D=132, 108 CST + 12 Chord + 12 Twist)")
    print(f" 评估预算: NFE = {max_evals} 次 / 种子, 随机种子集合 ({len(seeds)}个): {seeds}")
    print(f" 并发执行工作进程数: {n_workers}")
    print(f" 基准构型 (ModelTest 10 MW Reference Turbine): 功率 = {baseline_power:.2f} kW ({baseline_power/1e3:.3f} MW), Cp = {baseline_cp:.4f}, 转矩 = {baseline_torque:.2f} kN*m")
    print("=" * 84)

    # 读取已有结果支持断点续跑
    existing_results: Dict[str, List[Dict[str, Any]]] = {}
    if not force_rerun and os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                saved_blob = json.load(f)
                existing_results = saved_blob.get("results", {})
        except Exception:
            existing_results = {}

    all_results: Dict[str, List[Dict[str, Any]]] = {
        "TuRBO": [] if force_rerun else existing_results.get("TuRBO", []),
        "LLM-Meta-TuRBO": [] if force_rerun else existing_results.get("LLM-Meta-TuRBO", []),
        "MALIBO": [] if force_rerun else existing_results.get("MALIBO", []),
    }

    ctx = mp.get_context("spawn")

    # ==========================================================================
    # 1. 运行 TuRBO
    # ==========================================================================
    if not skip_turbo:
        print(f"\n[1/3] 启动 TuRBO-1 优化实验 (10 种子)...", flush=True)
        existing_seeds = {r["seed"] for r in all_results["TuRBO"]}
        pending_seeds = [s for s in seeds if s not in existing_seeds]
        for s in seeds:
            if s in existing_seeds:
                print(f"  --> [TuRBO Seed {s}] 已存在，跳过。", flush=True)

        if pending_seeds:
            print(f"  >>> 并发执行 {len(pending_seeds)} 个种子的 TuRBO 优化 (并发数: {n_workers})...", flush=True)
            turbo_args = [
                (s, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals)
                for s in pending_seeds
            ]
            with ctx.Pool(processes=min(n_workers, len(pending_seeds)), initializer=_worker_init) as pool:
                for res in pool.imap_unordered(_run_turbo_seed_worker, turbo_args):
                    all_results["TuRBO"].append(res)
                    print(
                        f"      [TuRBO Seed {res['seed']:4d}] 完成! 耗时: {res['elapsed_seconds']:.2f}s | "
                        f"最终功率: {res['best_score']:.2f} kW ({res['detailed']['power_MW']:.3f} MW) | "
                        f"Cp: {res['detailed']['Cp']:.4f} (+{(res['detailed']['Cp'] - baseline_cp)/baseline_cp*100:.2f}%)",
                        flush=True,
                    )
                    _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, seeds, output_json, output_npz, baseline_detailed=base_det)

            all_results["TuRBO"].sort(key=lambda r: r["seed"])
            _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, seeds, output_json, output_npz, baseline_detailed=base_det)

    # ==========================================================================
    # 2. 运行 LLM-Meta-TuRBO
    # ==========================================================================
    if not skip_llm_meta:
        print(f"\n[2/3] 启动 LLM-Meta-TuRBO 优化实验 (10 种子)...", flush=True)
        existing_seeds = {r["seed"] for r in all_results["LLM-Meta-TuRBO"]}
        pending_seeds = [s for s in seeds if s not in existing_seeds]
        for s in seeds:
            if s in existing_seeds:
                print(f"  --> [LLM-Meta-TuRBO Seed {s}] 已存在，跳过。", flush=True)

        if pending_seeds:
            print(f"  >>> 并发执行 {len(pending_seeds)} 个种子的 LLM-Meta-TuRBO 优化 (并发数: {n_workers})...", flush=True)
            llm_args = [
                (s, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals)
                for s in pending_seeds
            ]
            with ctx.Pool(processes=min(n_workers, len(pending_seeds)), initializer=_worker_init) as pool:
                for res in pool.imap_unordered(_run_llm_meta_seed_worker, llm_args):
                    all_results["LLM-Meta-TuRBO"].append(res)
                    n_inter = len(res.get("intervention_history", []))
                    print(
                        f"      [LLM-Meta-TuRBO Seed {res['seed']:4d}] 完成! 耗时: {res['elapsed_seconds']:.2f}s | 干预次数: {n_inter} | "
                        f"最终功率: {res['best_score']:.2f} kW ({res['detailed']['power_MW']:.3f} MW) | "
                        f"Cp: {res['detailed']['Cp']:.4f} (+{(res['detailed']['Cp'] - baseline_cp)/baseline_cp*100:.2f}%)",
                        flush=True,
                    )
                    _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, seeds, output_json, output_npz, baseline_detailed=base_det)

            all_results["LLM-Meta-TuRBO"].sort(key=lambda r: r["seed"])
            _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, seeds, output_json, output_npz, baseline_detailed=base_det)

    # ==========================================================================
    # 3. 运行 MALIBO
    # ==========================================================================
    if not skip_malibo:
        print(f"\n[3/3] 启动 MALIBO 元贝叶斯优化实验 (10 种子)...", flush=True)
        if not os.path.exists(malibo_checkpoint):
            raise FileNotFoundError(f"未找到 MALIBO 预训练模型权重: {malibo_checkpoint}，请先执行 train_malibo.py！")

        existing_seeds = {r["seed"] for r in all_results["MALIBO"]}
        pending_seeds = [s for s in seeds if s not in existing_seeds]
        for s in seeds:
            if s in existing_seeds:
                print(f"  --> [MALIBO Seed {s}] 已存在，跳过。", flush=True)

        if pending_seeds:
            print(f"  >>> 并发执行 {len(pending_seeds)} 个种子的 MALIBO 优化 (并发数: {n_workers})...", flush=True)
            malibo_args = [
                (s, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, malibo_checkpoint)
                for s in pending_seeds
            ]
            with ctx.Pool(processes=min(n_workers, len(pending_seeds)), initializer=_worker_init) as pool:
                for res in pool.imap_unordered(_run_malibo_seed_worker, malibo_args):
                    all_results["MALIBO"].append(res)
                    print(
                        f"      [MALIBO Seed {res['seed']:4d}] 完成! 耗时: {res['elapsed_seconds']:.2f}s | "
                        f"最终功率: {res['best_score']:.2f} kW ({res['detailed']['power_MW']:.3f} MW) | "
                        f"Cp: {res['detailed']['Cp']:.4f} (+{(res['detailed']['Cp'] - baseline_cp)/baseline_cp*100:.2f}%)",
                        flush=True,
                    )
                    _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, seeds, output_json, output_npz, baseline_detailed=base_det)

            all_results["MALIBO"].sort(key=lambda r: r["seed"])
            _save_checkpoint(all_results, baseline_power, baseline_cp, wind_speed, rpm, pitch_angle_deg, n_bem_stations, max_evals, seeds, output_json, output_npz, baseline_detailed=base_det)

    # 打印最终统计汇总表格
    print_benchmark_summary_table(all_results, baseline_power, baseline_cp)
    return all_results


def _save_checkpoint(
    results: Dict[str, List[Dict[str, Any]]],
    baseline_power: float,
    baseline_cp: float,
    wind_speed: float,
    rpm: float,
    pitch_angle_deg: float,
    n_bem_stations: int,
    max_evals: int,
    seeds: List[int],
    output_json: str,
    output_npz: str,
    baseline_detailed: Optional[Dict[str, Any]] = None,
):
    """持久化保存 JSON 与 NPZ 数据"""
    existing_meta = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
                existing_meta = saved.get("metadata", {})
        except Exception:
            pass

    meta_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wind_speed": wind_speed,
        "rpm": rpm,
        "pitch_angle_deg": pitch_angle_deg,
        "n_bem_stations": n_bem_stations,
        "max_evals": max_evals,
        "seeds": seeds,
        "baseline_power_kW": baseline_power,
        "baseline_Cp": baseline_cp,
    }
    if baseline_detailed is not None:
        meta_payload["baseline_detailed"] = baseline_detailed
    elif "baseline_detailed" in existing_meta:
        meta_payload["baseline_detailed"] = existing_meta["baseline_detailed"]

    payload = {
        "metadata": meta_payload,
        "results": results,
    }
    def _json_default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

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
    print(" 3D 风力机叶片全量优化性能指标统计汇总表 (10 随机种子, 300 NFE, OpenFAST 58 截面基准)")
    print("=" * 96)
    print(f"{'优化方法 (Method)':<18} | {'最佳功率 (kW)':<18} | {'风能系数 (Cp)':<18} | {'功率提升 (%)':<14} | {'平均耗时 (s)':<12}")
    print("-" * 96)
    print(f"{'Baseline (ModelTest 10MW)':<18} | {baseline_power:8.1f}          | {baseline_cp:8.4f}          | {'+0.00%':<14} | {'--':<12}")

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
    parser.add_argument("--rpm", type=float, default=8.00)
    parser.add_argument("--pitch-angle-deg", type=float, default=0.0)
    parser.add_argument("--n-bem-stations", type=int, default=50)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--workers", type=int, default=5, help="并发工作进程数 (默认: 5)")
    parser.add_argument("--force-rerun", action="store_true", help="强制重新运行全部 10 种子实验")
    parser.add_argument("--skip-turbo", action="store_true", help="跳过 TuRBO 测试")
    parser.add_argument("--skip-llm-meta", action="store_true", help="跳过 LLM-Meta-TuRBO 测试")
    parser.add_argument("--skip-malibo", action="store_true", help="跳过 MALIBO 测试")
    args = parser.parse_args()

    run_benchmark(
        wind_speed=args.wind_speed,
        rpm=args.rpm,
        pitch_angle_deg=args.pitch_angle_deg,
        n_bem_stations=args.n_bem_stations,
        max_evals=args.max_evals,
        force_rerun=args.force_rerun,
        skip_turbo=args.skip_turbo,
        skip_llm_meta=args.skip_llm_meta,
        skip_malibo=args.skip_malibo,
        n_workers=args.workers,
    )
