"""MALIBO 风速-转速强相关物理运行轨迹训练集生成器：
基于变速风电机组最优叶尖速比 (TSR) 控制特性：
在 Region II 运行区内，风电机组转速与风速物理强正相关：
    Omega_nominal(V) = (lambda_opt / R) * V * (30 / pi) approx 0.6419 * V (rpm)
围绕额定测试工况 (V_test = 11.0 m/s, rpm_test = 7.061 rpm) 构建环绕物理扰动网格：
- 选取 7 个代表性风速点: [7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5] m/s (严格避开 11.0 m/s);
- 对每个风速点，依据真实风机湍流与工况波动设置 3 档正相关转速 [-5%, 0%, +5%]:
  共生成 7 x 3 = 21 个物理强耦合互斥训练任务；
- 采用粒子群优化器 (PSO) 进行高维 (D=120) 空间多峰探索，收集高多样性、高质量气动优化轨迹数据；
- 产出数据持久化至 malibo_dataset/ 目录 (受 .gitignore 保护)。
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from src.blade_geometry import DIM, get_baseline_blade_params, get_blade_bounds
from src.evaluator import WindTurbineBladeEvaluator


# 21 个环绕测试工况 (V=11.0, rpm=7.061) 的物理强相关工况定义 (V, rpm)
CORRELATED_TASKS: List[Tuple[float, float]] = [
    # V = 7.5 m/s (Omega_nom = 4.81 rpm)
    (7.5, 4.57), (7.5, 4.81), (7.5, 5.05),
    # V = 8.5 m/s (Omega_nom = 5.46 rpm)
    (8.5, 5.18), (8.5, 5.46), (8.5, 5.73),
    # V = 9.5 m/s (Omega_nom = 6.10 rpm)
    (9.5, 5.79), (9.5, 6.10), (9.5, 6.40),
    # V = 10.5 m/s (Omega_nom = 6.74 rpm)
    (10.5, 6.40), (10.5, 6.74), (10.5, 7.08),
    # V = 11.5 m/s (Omega_nom = 7.38 rpm)
    (11.5, 7.01), (11.5, 7.38), (11.5, 7.75),
    # V = 12.5 m/s (Omega_nom = 8.02 rpm)
    (12.5, 7.62), (12.5, 8.02), (12.5, 8.43),
    # V = 13.5 m/s (Omega_nom = 8.67 rpm)
    (13.5, 8.23), (13.5, 8.67), (13.5, 9.10),
]


class BladePSOOptimizer:
    """适用于 120 维 3D 风力机叶片多样性探索的粒子群优化器 (PSO)"""

    def __init__(
        self,
        evaluator: WindTurbineBladeEvaluator,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        n_particles: int = 15,
        n_steps: int = 10,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.x0 = x0
        self.lb = lb
        self.ub = ub
        self.dim = len(x0)
        self.dim_range = ub - lb
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.rng = np.random.RandomState(seed)

        # PSO 超参数 (标准 Clerc-Kennedy 紧缩因子收缩超参)
        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494

    def run_trajectory(self) -> Dict[str, Any]:
        """运行 PSO 探索并记录全程采样的 (x, y) 样本对"""
        d = self.dim
        # 归一化搜索空间 [0, 1]^d
        to_unit = lambda x: (x - self.lb) / (self.dim_range + 1e-12)
        from_unit = lambda u: self.lb + u * self.dim_range

        # 初始化粒子位置与速度
        u_particles = np.zeros((self.n_particles, d), dtype=np.float64)
        # 粒子 0 初始化为基准叶片构型
        u_particles[0] = to_unit(self.x0)
        # 其余粒子在基准周围微扰分布
        for i in range(1, self.n_particles):
            noise = self.rng.uniform(-0.06, 0.06, size=d)
            u_particles[i] = np.clip(u_particles[0] + noise, 0.0, 1.0)

        velocities = self.rng.uniform(-0.02, 0.02, size=(self.n_particles, d))

        pbest_u = u_particles.copy()
        pbest_y = np.full(self.n_particles, -np.inf)
        gbest_u = u_particles[0].copy()
        gbest_y = -np.inf

        X_unit_all: List[np.ndarray] = []
        Theta_all: List[np.ndarray] = []
        Y_penalized_all: List[float] = []
        Y_minimize_all: List[float] = []
        feasible_all: List[bool] = []

        # 初始步评估
        for i in range(self.n_particles):
            x_real = from_unit(u_particles[i])
            score = self.evaluator.evaluate(x_real)

            X_unit_all.append(u_particles[i].copy())
            Theta_all.append(x_real.copy())
            Y_penalized_all.append(score)
            Y_minimize_all.append(-score)
            feasible_all.append(score > 0.0)

            if score > pbest_y[i]:
                pbest_y[i] = score
                pbest_u[i] = u_particles[i].copy()
            if score > gbest_y:
                gbest_y = score
                gbest_u = u_particles[i].copy()

        # PSO 迭代推进
        for step in range(self.n_steps):
            r1 = self.rng.rand(self.n_particles, d)
            r2 = self.rng.rand(self.n_particles, d)

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_u - u_particles)
                + self.c2 * r2 * (gbest_u - u_particles)
            )
            velocities = np.clip(velocities, -0.08, 0.08)
            u_particles = np.clip(u_particles + velocities, 0.0, 1.0)

            for i in range(self.n_particles):
                x_real = from_unit(u_particles[i])
                score = self.evaluator.evaluate(x_real)

                X_unit_all.append(u_particles[i].copy())
                Theta_all.append(x_real.copy())
                Y_penalized_all.append(score)
                Y_minimize_all.append(-score)
                feasible_all.append(score > 0.0)

                if score > pbest_y[i]:
                    pbest_y[i] = score
                    pbest_u[i] = u_particles[i].copy()
                if score > gbest_y:
                    gbest_y = score
                    gbest_u = u_particles[i].copy()

        return {
            "X_unit": np.array(X_unit_all, dtype=np.float32),
            "Theta": np.array(Theta_all, dtype=np.float32),
            "Y_penalized": np.array(Y_penalized_all, dtype=np.float32),
            "Y_minimize": np.array(Y_minimize_all, dtype=np.float32),
            "is_feasible": np.array(feasible_all, dtype=bool),
            "best_y": gbest_y,
        }


def generate_malibo_grid_dataset(
    output_dir: str = "malibo_dataset",
    evals_per_task: int = 150,
) -> str:
    """生成包含 21 个物理强相关工况的完整 MALIBO 预热元训练数据集"""
    out_path = Path(output_dir)
    tasks_dir = out_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    x0 = get_baseline_blade_params()
    lb, ub = get_blade_bounds(x0)

    bundled_tasks = []
    t_start = time.time()

    print(f"=== 开始生成 MALIBO 21 个强相关工况训练数据集 (D={DIM}) ===")
    print(f"工况轨迹涵盖 7 档风速与 3 档正相关转速，严格避开测试工况 (V=11.0, rpm=7.061)")

    for task_id, (V, rpm) in enumerate(CORRELATED_TASKS):
        t0 = time.time()
        task_name = f"task_{task_id:02d}_V{V:.1f}_rpm{rpm:.2f}"

        evaluator = WindTurbineBladeEvaluator(wind_speed=V, rpm=rpm)
        pso = BladePSOOptimizer(
            evaluator=evaluator,
            x0=x0,
            lb=lb,
            ub=ub,
            n_particles=15,
            n_steps=(evals_per_task // 15) - 1,
            seed=42 + task_id * 17,
        )
        traj = pso.run_trajectory()
        elapsed_task = time.time() - t0

        # 保存单个任务数据
        npz_file = tasks_dir / f"{task_name}.npz"
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
        )

        bundled_tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "wind_speed": V,
            "rpm": rpm,
            "X_unit": torch.from_numpy(traj["X_unit"]),
            "Theta": torch.from_numpy(traj["Theta"]),
            "Y_penalized": torch.from_numpy(traj["Y_penalized"]),
            "Y_minimize": torch.from_numpy(traj["Y_minimize"]),
            "is_feasible": torch.from_numpy(traj["is_feasible"]),
        })

        print(
            f"  [Task {task_id+1:2d}/{len(CORRELATED_TASKS)}] V={V:4.1f}m/s, rpm={rpm:4.2f} | "
            f"Samples: {len(traj['Y_penalized'])} | Best Power: {traj['best_y']:.1f} kW | "
            f"Time: {elapsed_task:.2f}s",
            flush=True,
        )

    # 打包整体 .pt 文件供 PyTorch DataLoader 秒级直接载入
    bundled_pt_path = out_path / "malibo_blade_train_dataset.pt"
    torch.save(
        {
            "num_tasks": len(bundled_tasks),
            "in_dim": DIM,
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
