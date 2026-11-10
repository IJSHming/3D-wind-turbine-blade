"""MALIBO 风速-转速强相关物理运行轨迹训练集生成器：
基于变速风电机组最优叶尖速比 (TSR) 控制特性：
在 Region II 运行区内，风电机组转速与风速物理强正相关：
    Omega_nominal(V) = (Omega_rated / V_rated) * V = (300.0 / 10.0) * V = 30.0 * V (rpm)
围绕对齐 1.5m 叶片风电机组基准测试工况 (V_test = 10.0 m/s, rpm_test = 300.0 rpm) 构建高密度物理扰动网格：
- 选取 20 个密集环绕且严格避开 10.0 m/s 的风速点: 7.0 ~ 13.0 m/s;
- 对每个风速点，设置 5 档正相关转速 [-6%, -3%, 0%, +3%, +6%]:
  共生成 20 x 5 = 100 个物理强耦合互斥训练任务；
- 严格对齐当前采用的 B-Spline Modal 高阶平滑模态参数化 (D=114, 102 CST + 6 弦长 B 样条 + 6 扭角 B 样条)；
- 采用 TuRBO-1 贝叶斯信赖域优化器进行高维空间高效收敛探索，
  收集高品质、具有明确气动梯度改进方向的贝叶斯先验轨迹数据；
- 产出数据持久化至 malibo_dataset/ 目录 (受 .gitignore 保护)。
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
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import torch

from src.blade_geometry import DIM, get_baseline_blade_params, get_blade_bounds
from src.evaluator import WindTurbineBladeEvaluator
from src.optimizers.turbo import matern52_kernel


# 20 个密集环绕测试工况 (V=1.5 m/s, rpm=40.0) 的代表性潮流水速与 5 档转速定义 (共 100 个任务)
# 严格避开测试工况 (V=1.5 m/s)
_WIND_SPEEDS_20 = [
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
    1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00
]
CORRELATED_TASKS: List[Tuple[float, float]] = []
for _v in _WIND_SPEEDS_20:
    _nom_rpm = (40.0 / 1.5) * _v
    for f in [0.94, 0.97, 1.00, 1.03, 1.06]:
        CORRELATED_TASKS.append((_v, round(_nom_rpm * f, 2)))


class BladeTuRBOCollector:
    """适用于 B-Spline Modal 高阶参数化 (D=132) 3D 叶片高质量气动收敛轨迹收集的 TuRBO-1 优化器"""

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

        # TuRBO 超参数 (Eriksson et al. 2019 标准推荐)
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

        # 1. 评估基准构型
        score0 = self.evaluator.evaluate(self.x0)
        best_score = score0
        X_unit_all.append(self.to_unit(self.x0))
        Theta_all.append(self.x0.copy())
        Y_penalized_all.append(score0)
        Y_minimize_all.append(-score0)
        feasible_all.append(score0 > 0.0)

        # 2. 初始微扰采样
        for _ in range(1, self.n_init):
            if len(X_unit_all) >= self.max_evals:
                break
            noise = self.rng.uniform(-0.04, 0.04, size=d)
            x_cand = np.clip(self.x0 + noise, self.lb, self.ub)
            score = self.evaluator.evaluate(x_cand)

            if score > best_score:
                best_score = score
                best_x = x_cand.copy()

            u_cand = self.to_unit(x_cand)
            X_unit_all.append(u_cand)
            Theta_all.append(x_cand.copy())
            Y_penalized_all.append(score)
            Y_minimize_all.append(-score)
            feasible_all.append(score > 0.0)

        # 3. TuRBO 主循环
        while len(X_unit_all) < self.max_evals:
            X_arr = np.array(X_unit_all, dtype=np.float64)
            Y_arr = np.array(Y_penalized_all, dtype=np.float64)

            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr) + 1e-8
            Y_norm = (Y_arr - y_mean) / y_std

            x_best_u = self.to_unit(best_x)
            lb_tr = np.clip(x_best_u - self.L / 2.0, 0.0, 1.0)
            ub_tr = np.clip(x_best_u + self.L / 2.0, 0.0, 1.0)

            cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(self.n_candidates, d))
            mask = self.rng.rand(self.n_candidates, d) < self.prob_perturb
            mask_all_false = ~np.any(mask, axis=1)
            rand_dims = self.rng.randint(0, d, size=np.sum(mask_all_false))
            mask[mask_all_false, rand_dims] = True
            candidates = np.where(mask, cand_raw, x_best_u)

            # GP 后验
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
                best_cand_idx = int(np.argmax(ts_scores))
                x_next_u = candidates[best_cand_idx]
            except Exception:
                x_next_u = candidates[0]

            x_next = self.from_unit(x_next_u)
            score_before = best_score
            new_score = self.evaluator.evaluate(x_next)

            if new_score > best_score:
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


def _run_single_task_worker(task_info: Tuple[int, float, float, int, int, str]) -> Dict[str, Any]:
    """单任务多进程工作函数"""
    task_id, V, rpm, max_evals, seed, output_dir = task_info
    t0 = time.time()
    task_name = f"task_{task_id:02d}_V{V:.1f}_rpm{rpm:.2f}"
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

    evaluator = WindTurbineBladeEvaluator(wind_speed=V, rpm=rpm)
    x0 = evaluator.baseline_x0.copy()
    lb, ub = evaluator.param_model.get_bounds()

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

    # 保存单个任务数据
    np.savez_compressed(
        npz_file,
        task_id=task_id,
        task_name=task_name,
        wind_speed=V,
        rpm=rpm,
        X_unit=traj["X_unit"],
        Theta=traj["Theta"],
        Y_penalized=traj["Y_penalized"],
        Y_minimize=traj["Y_minimize"],
        is_feasible=traj["is_feasible"],
        best_y=traj["best_y"],
        elapsed=elapsed,
    )

    print(
        f"  [Task {task_id+1:2d}/{len(CORRELATED_TASKS)}] V={V:4.1f}m/s, rpm={rpm:4.2f} | "
        f"Samples: {len(traj['Y_penalized'])} | Best Power: {traj['best_y']:.1f} kW | "
        f"Time: {elapsed:.2f}s",
        flush=True,
    )

    return {
        "task_id": task_id,
        "task_name": task_name,
        "wind_speed": V,
        "rpm": rpm,
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
    evals_per_task: int = 100,
    n_workers: int = 8,
) -> str:
    """生成包含 100 个密集环绕工况的完整 MALIBO 预热元训练数据集 (多进程 TuRBO 加速)"""
    out_path = Path(output_dir)
    tasks_dir = out_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    from src.parameterizations import BSplineBladeModel
    dim = BSplineBladeModel().dim
    print(f"=== 开始生成 MALIBO {len(CORRELATED_TASKS)} 个密集环绕工况训练数据集 (D={dim}, TuRBO-1 轨迹) ===")
    print(f"工况轨迹涵盖 20 档风速与 5 档正相关转速，严格避开测试工况 (V=10.0, rpm=300.0)")
    print(f"并发进程数: {n_workers}, 每工况评估预算: {evals_per_task} 步")

    task_args = [
        (task_id, V, rpm, evals_per_task, 42 + task_id * 17, output_dir)
        for task_id, (V, rpm) in enumerate(CORRELATED_TASKS)
    ]

    # 多进程并发生成
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        bundled_tasks = pool.map(_run_single_task_worker, task_args)

    # 排序保持 task_id 顺序一致
    bundled_tasks.sort(key=lambda x: x["task_id"])

    # 打包整体 .pt 文件供 PyTorch DataLoader 秒级直接载入
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
    print(f"=== MALIBO 训练集生成完成 ===")
    print(f"总工况数: {len(bundled_tasks)}, 样本总量: {total_samples}, 总耗时: {total_time:.2f}s")
    print(f"已持久化至: {bundled_pt_path}")

    return str(bundled_pt_path)


if __name__ == "__main__":
    generate_malibo_grid_dataset()
