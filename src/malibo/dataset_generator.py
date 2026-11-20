"""MALIBO 多工况物理运行轨迹训练集生成器 (Pan et al. 2024 Meta-Dataset):
面向 NREL Phase VI 63 维 (3 翼型控制站位 + 6 阶 B 样条弦长/扭角) 3D 风力机叶片空间，严格遵守零测试集泄露原则:
- 严格避开目标测试工况 (V_test=7.0 m/s, rpm_test=72.0, pitch=4.815 deg)；
- 涵盖 20 档风速 ([4.50, 9.50] 排除 7.0) x 5 档关联转速 ([0.92..1.08] * (72/7)*V)
  x 6 档独立变桨角 ([2.5, 3.3, 4.1, 5.5, 6.3, 7.2] deg 排除 4.815) 共 100 个耦合任务；
- Multi-Start 初始构型多样性: 40% 任务从基准构型出发，60% 任务从经几何校验的不同可行扰动叶片出发；
- 深度轨迹采集: 单任务 200 NFE (31 全余弦加密 BEM 截面), 持久化至 malibo_dataset/ (受 .gitignore 保护)。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import time
from typing import Any, Dict, List, Tuple
import multiprocessing as mp
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch

from src.blade_geometry import BEM_NUM_SECTIONS, DIM, NUM_STATIONS, get_baseline_blade_params, get_blade_bounds
from src.evaluator import WindTurbineBladeEvaluator
from src.optimizers.turbo import matern52_kernel


# 20 个密集环绕且严格避开测试风速 V=7.0 m/s 的风速点
_WIND_SPEEDS_20 = [
    4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75,
    7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50,
]
# 6 档变桨角，严格避开测试基准 pitch=4.815 deg
_PITCH_ANGLES_6 = [2.5, 3.3, 4.1, 5.5, 6.3, 7.2]

CORRELATED_TASKS: List[Tuple[float, float, float]] = []
for _v_idx, _v in enumerate(_WIND_SPEEDS_20):
    _nom_rpm = (72.0 / 7.0) * _v
    for _r_idx, _f in enumerate([0.92, 0.96, 1.00, 1.04, 1.08]):
        _pitch = _PITCH_ANGLES_6[(_v_idx * 5 + _r_idx) % len(_PITCH_ANGLES_6)]
        CORRELATED_TASKS.append((_v, round(_nom_rpm * _f, 2), _pitch))


def _sample_diverse_feasible_start(
    evaluator: WindTurbineBladeEvaluator,
    x_base: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """为部分离线任务生成光滑、合法的多样化可行起始叶片 (Multi-Start Diversity, D=63)."""
    bell = np.sin(np.pi * (np.arange(8) + 0.5) / 8.0)
    n_cst_total = NUM_STATIONS * 17
    for _ in range(40):
        cand = x_base.copy()
        # 光滑低频 CST 弯度与厚度扰动 (3 控制站位: Root / Mid / Tip)
        for k in range(NUM_STATIONS):
            idx = k * 17
            d_cam = rng.uniform(-0.02, 0.035)
            d_thk = rng.uniform(-0.025, 0.015)
            cand[idx : idx + 8] += (d_cam + 0.5 * d_thk) * bell
            cand[idx + 8 : idx + 16] += (d_cam - 0.5 * d_thk) * bell
            cand[idx + 16] += 0.2 * d_thk
        # 保证单调非增弦长与扭角 B 样条控制点偏移
        dc_root = rng.uniform(-0.02, 0.05)
        dc_tip = rng.uniform(-0.04, min(0.015, dc_root))
        cand[n_cst_total : n_cst_total + 6] += np.linspace(dc_root, dc_tip, 6)

        dtw_root = rng.uniform(-3.5, 2.0)
        dtw_tip = rng.uniform(max(-3.0, dtw_root - 1.5), min(1.5, dtw_root))
        cand[n_cst_total + 6 : n_cst_total + 12] += np.linspace(dtw_root, dtw_tip, 6)
        cand = np.clip(cand, lb, ub)

        is_fatal, _, metrics = evaluator.validate_geometry(cand)
        is_geom_ok, pen, _ = evaluator.calculate_penalties(metrics)
        if not is_fatal and is_geom_ok and pen == 0.0:
            if evaluator.evaluate(cand) > 0.0:
                return cand
    return x_base.copy()


class BladeTuRBOCollector:
    """适用于 B-Spline Modal 高阶参数化 (D=114) 3D 叶片高质量收敛轨迹收集的 TuRBO-1 优化器"""

    def __init__(
        self,
        evaluator: WindTurbineBladeEvaluator,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_evals: int = 200,
        n_init: int = 15,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.dim = len(x0)
        self.dim_range = self.ub - self.lb
        self.max_evals = max_evals
        self.n_init = n_init
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.L = 0.8
        self.L_min = 0.5**7
        self.L_max = 1.6
        self.success_tol = 3
        self.failure_tol = 6
        self.prob_perturb = min(20.0 / self.dim, 1.0)
        self.n_candidates = 5000

        self.n_success = 0
        self.n_failure = 0

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lb) / (self.dim_range + 1e-12)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lb + u * self.dim_range

    def _adjust_trust_region(self, improved: bool):
        if improved:
            self.n_success += 1
            self.n_failure = 0
        else:
            self.n_success = 0
            self.n_failure += 1

        if self.n_success >= self.success_tol:
            self.L = min(self.L * 2.0, self.L_max)
            self.n_success = 0
        elif self.n_failure >= self.failure_tol:
            self.L = self.L / 2.0
            self.n_failure = 0

        if self.L < self.L_min:
            self.L = 0.8
            self.n_success = 0
            self.n_failure = 0

    def run_trajectory(self) -> Dict[str, Any]:
        """运行完整 TuRBO 轨迹并记录采样对"""
        d = self.dim
        X_unit_all: List[np.ndarray] = []
        Theta_all: List[np.ndarray] = []
        Y_penalized_all: List[float] = []
        Y_minimize_all: List[float] = []
        feasible_all: List[bool] = []

        best_score = -np.inf
        best_x = self.x0.copy()

        # 1. 评估基准/初始构型
        score0 = self.evaluator.evaluate(self.x0)
        best_score = score0
        X_unit_all.append(self.to_unit(self.x0))
        Theta_all.append(self.x0.copy())
        Y_penalized_all.append(score0)
        Y_minimize_all.append(-score0)
        feasible_all.append(score0 > 0.0)

        # 2. 初始微扰采样 (归一化 [0, 1]^D 空间对称微扰)
        u0 = self.to_unit(self.x0)
        for _ in range(1, self.n_init):
            if len(X_unit_all) >= self.max_evals:
                break
            noise_u = self.rng.uniform(-0.05, 0.05, size=d)
            u_cand = np.clip(u0 + noise_u, 0.0, 1.0)
            x_cand = self.from_unit(u_cand)
            score = self.evaluator.evaluate(x_cand)

            if score > best_score and (score > 0.0 or best_score <= 0.0):
                best_score = score
                best_x = x_cand.copy()

            X_unit_all.append(u_cand)
            Theta_all.append(x_cand.copy())
            Y_penalized_all.append(score)
            Y_minimize_all.append(-score)
            feasible_all.append(score > 0.0)

        # 3. TuRBO 主循环 (结合离线前置致命几何过滤以提升深度收敛流形密度)
        while len(X_unit_all) < self.max_evals:
            X_arr = np.array(X_unit_all, dtype=np.float64)
            Y_arr = np.array(Y_penalized_all, dtype=np.float64)

            Y_gp = np.maximum(Y_arr, -1.0)
            y_mean = np.mean(Y_gp)
            y_std = np.std(Y_gp) + 1e-8
            Y_norm = (Y_gp - y_mean) / y_std

            x_best_u = self.to_unit(best_x)
            lb_tr = np.clip(x_best_u - self.L / 2.0, 0.0, 1.0)
            ub_tr = np.clip(x_best_u + self.L / 2.0, 0.0, 1.0)

            cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(self.n_candidates, d))
            mask = self.rng.rand(self.n_candidates, d) < self.prob_perturb
            mask_all_false = ~np.any(mask, axis=1)
            rand_dims = self.rng.randint(0, d, size=np.sum(mask_all_false))
            mask[mask_all_false, rand_dims] = True
            candidates = np.where(mask, cand_raw, x_best_u)

            K_train = matern52_kernel(X_arr, X_arr, lengthscale=0.35, outputscale=1.0)
            K_train += 1e-4 * np.eye(len(X_arr))

            try:
                c_factor = cho_factor(K_train, lower=True)
                alpha_weights = cho_solve(c_factor, Y_norm)

                K_cand_train = matern52_kernel(candidates, X_arr, lengthscale=0.35, outputscale=1.0)
                mu_cand = K_cand_train @ alpha_weights

                K_diag = np.ones(self.n_candidates)
                v = cho_solve(c_factor, K_cand_train.T)
                var_cand = np.maximum(1e-6, K_diag - np.sum(K_cand_train.T * v, axis=0))
                sigma_cand = np.sqrt(var_cand)

                ts_scores = mu_cand + sigma_cand * self.rng.randn(self.n_candidates)
                top_order = np.argsort(ts_scores)[::-1]
                x_next_u = candidates[int(top_order[0])]
                for idx in top_order[:40]:
                    x_try = self.from_unit(candidates[int(idx)])
                    is_fatal, _, metrics = self.evaluator.validate_geometry(x_try)
                    is_geom_ok, _, _ = self.evaluator.calculate_penalties(metrics)
                    if not is_fatal and is_geom_ok:
                        x_next_u = candidates[int(idx)]
                        break
            except Exception:
                x_next_u = candidates[int(self.rng.randint(0, self.n_candidates))]

            x_next = self.from_unit(x_next_u)
            score_before = best_score
            new_score = self.evaluator.evaluate(x_next)

            if new_score > best_score and (new_score > 0.0 or best_score <= 0.0):
                best_score = new_score
                best_x = x_next.copy()

            X_unit_all.append(x_next_u.copy())
            Theta_all.append(x_next.copy())
            Y_penalized_all.append(new_score)
            Y_minimize_all.append(-new_score)
            feasible_all.append(new_score > 0.0)

            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved)

        return {
            "X_unit": np.array(X_unit_all, dtype=np.float32),
            "Theta": np.array(Theta_all, dtype=np.float32),
            "Y_penalized": np.array(Y_penalized_all, dtype=np.float32),
            "Y_minimize": np.array(Y_minimize_all, dtype=np.float32),
            "is_feasible": np.array(feasible_all, dtype=bool),
            "best_y": best_score,
        }


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


def _run_single_task_worker(task_info: Tuple[int, float, float, float, int, int, str]) -> Dict[str, Any]:
    """单任务多进程工作函数"""
    task_id, V, rpm, pitch_deg, max_evals, seed, output_dir = task_info
    t0 = time.time()
    task_name = f"task_{task_id:02d}_V{V:.2f}_rpm{rpm:.2f}_p{pitch_deg:.1f}"
    tasks_dir = Path(output_dir) / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    npz_file = tasks_dir / f"{task_name}.npz"

    if npz_file.exists():
        try:
            cached = np.load(npz_file)
            print(
                f"  [Task {task_id+1:2d}/{len(CORRELATED_TASKS)}] Loaded cached task from {npz_file.name}",
                flush=True,
            )
            return {
                "task_id": task_id,
                "task_name": task_name,
                "wind_speed": V,
                "rpm": rpm,
                "pitch_angle_deg": pitch_deg,
                "X_unit": torch.from_numpy(cached["X_unit"]),
                "Theta": torch.from_numpy(cached["Theta"]),
                "Y_penalized": torch.from_numpy(cached["Y_penalized"]),
                "Y_minimize": torch.from_numpy(cached["Y_minimize"]),
                "is_feasible": torch.from_numpy(cached["is_feasible"]),
                "best_y": float(cached["best_y"]) if "best_y" in cached else float(np.max(cached["Y_penalized"])),
                "elapsed": float(cached["elapsed"]) if "elapsed" in cached else 0.0,
            }
        except Exception:
            pass

    evaluator = WindTurbineBladeEvaluator(
        wind_speed=V,
        rpm=rpm,
        pitch_angle_deg=pitch_deg,
        n_bem_stations=BEM_NUM_SECTIONS,
    )
    x_base = evaluator.baseline_x0.copy()
    lb, ub = evaluator.param_model.get_bounds()

    # 60% 任务采用多样化可行初始叶片 (Multi-Start)，40% 保留标准基准
    rng_start = np.random.RandomState(seed + 7919)
    if (task_id % 5) in (1, 3, 4):
        x0 = _sample_diverse_feasible_start(evaluator, x_base, lb, ub, rng_start)
        start_tag = "MultiStart"
    else:
        x0 = x_base
        start_tag = "Baseline"

    collector = BladeTuRBOCollector(
        evaluator=evaluator,
        x0=x0,
        lb=lb,
        ub=ub,
        max_evals=max_evals,
        n_init=15,
        seed=seed,
    )
    traj = collector.run_trajectory()
    elapsed = time.time() - t0

    np.savez_compressed(
        npz_file,
        task_id=task_id,
        task_name=task_name,
        wind_speed=V,
        rpm=rpm,
        pitch_angle_deg=pitch_deg,
        X_unit=traj["X_unit"],
        Theta=traj["Theta"],
        Y_penalized=traj["Y_penalized"],
        Y_minimize=traj["Y_minimize"],
        is_feasible=traj["is_feasible"],
        best_y=traj["best_y"],
        elapsed=elapsed,
    )

    feas_rate = float(np.mean(traj["is_feasible"])) * 100.0
    print(
        f"  [Task {task_id+1:2d}/{len(CORRELATED_TASKS)}] V={V:5.2f}, rpm={rpm:6.2f}, p={pitch_deg:4.1f} ({start_tag:10s}) | "
        f"Best={traj['best_y']:.3f} kW | Feas={feas_rate:4.1f}% | Time={elapsed:.1f}s",
        flush=True,
    )

    return {
        "task_id": task_id,
        "task_name": task_name,
        "wind_speed": V,
        "rpm": rpm,
        "pitch_angle_deg": pitch_deg,
        "X_unit": torch.from_numpy(traj["X_unit"]),
        "Theta": torch.from_numpy(traj["Theta"]),
        "Y_penalized": torch.from_numpy(traj["Y_penalized"]),
        "Y_minimize": torch.from_numpy(traj["Y_minimize"]),
        "is_feasible": torch.from_numpy(traj["is_feasible"]),
        "best_y": traj["best_y"],
        "elapsed": elapsed,
    }


def generate_malibo_grid_dataset(
    output_dir: str = "malibo_dataset",
    evals_per_task: int = 200,
    n_workers: int = 10,
) -> str:
    """生成包含 100 个多风速/多转速/多变桨角/Multi-Start 的高密度 MALIBO 元训练数据集"""
    out_path = Path(output_dir)
    tasks_dir = out_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    from src.parameterizations import BSplineBladeModel
    dim = BSplineBladeModel().dim
    print(f"=== 开始生成增强版 MALIBO {len(CORRELATED_TASKS)} 工况元训练数据集 (D={dim}, Multi-Start + Multi-Pitch) ===")
    print(f"严格避开测试工况 (V=7.0 m/s, rpm=72.0, pitch=4.815 deg)")
    print(f"并发进程数: {n_workers}, 每工况评估预算: {evals_per_task} 步 (预计总样本: {len(CORRELATED_TASKS)*evals_per_task:,})")

    task_args = [
        (task_id, V, rpm, pitch_deg, evals_per_task, 42 + task_id * 17, output_dir)
        for task_id, (V, rpm, pitch_deg) in enumerate(CORRELATED_TASKS)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        bundled_tasks = pool.map(_run_single_task_worker, task_args)

    bundled_tasks.sort(key=lambda x: x["task_id"])

    bundled_pt_path = out_path / "malibo_blade_train_dataset.pt"
    dim = int(bundled_tasks[0]["X_unit"].shape[1])
    torch.save(
        {
            "num_tasks": len(bundled_tasks),
            "in_dim": dim,
            "tasks": bundled_tasks,
        },
        bundled_pt_path,
    )

    total_time = time.time() - t_start
    total_samples = sum(len(t["Y_penalized"]) for t in bundled_tasks)
    print(f"=== 增强版 MALIBO 训练集生成完成 ===")
    print(f"总工况数: {len(bundled_tasks)}, 样本总量: {total_samples}, 总耗时: {total_time:.2f}s")
    print(f"已持久化至: {bundled_pt_path}")

    return str(bundled_pt_path)


if __name__ == "__main__":
    generate_malibo_grid_dataset()
