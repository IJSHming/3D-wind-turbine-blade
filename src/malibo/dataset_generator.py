"""MALIBO 风速-转速网格训练集生成器：
设计环绕测试工况 (V_test=9.0 m/s, rpm_test=10.0) 的风速 vs 转速网格 (Grid):
- 风速集合: [7.0, 8.0, 10.0, 11.0] m/s (避开 9.0)
- 转速集合: [8.0, 9.0, 11.0, 12.0] rpm (避开 10.0)
- 共 4 x 4 = 16 个互斥独立工况任务；
- 采用粒子群优化器 (PSO, 惯性权重 w=0.72, 认知/社会因子 c1=c2=1.49, 种群 15) 进行高效多峰探索，
  收集高多样性、高质量气动优化轨迹数据；
- 产出数据持久化至 malibo_dataset/ 目录 (受 .gitignore 保护)。
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from src.blade_geometry import get_baseline_blade_params, get_blade_bounds
from src.evaluator import WindTurbineBladeEvaluator


# 环绕测试工况 (V=9.0, rpm=10.0) 的工况网格
GRID_WIND_SPEEDS: List[float] = [7.0, 8.0, 10.0, 11.0]
GRID_RPMS: List[float] = [8.0, 9.0, 11.0, 12.0]


class BladePSOOptimizer:
    """适用于 3D 风力机叶片多样性探索的粒子群优化器 (PSO)"""

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
            noise = self.rng.uniform(-0.08, 0.08, size=d)
            u_particles[i] = np.clip(u_particles[0] + noise, 0.0, 1.0)

        velocities = self.rng.uniform(-0.03, 0.03, size=(self.n_particles, d))

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
            velocities = np.clip(velocities, -0.10, 0.10)
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
    """生成包含 16 个网格工况的完整 MALIBO 预热元训练数据集"""
    out_path = Path(output_dir)
    tasks_dir = out_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    x0 = get_baseline_blade_params()
    lb, ub = get_blade_bounds(x0)

    task_id = 0
    bundled_tasks = []
    t_start = time.time()

    print(f"=== 开始生成 MALIBO 16 工况训练数据集 ===")
    print(f"网格配置: 风速 {GRID_WIND_SPEEDS} m/s, 转速 {GRID_RPMS} rpm")
    print(f"测试工况 (V=9.0, rpm=10.0) 严格被避开并包围。")

    for V in GRID_WIND_SPEEDS:
        for rpm in GRID_RPMS:
            t0 = time.time()
            task_name = f"task_{task_id:02d}_V{V:.1f}_rpm{rpm:.1f}"

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
                f"  [Task {task_id:2d}/16] V={V:4.1f}m/s, rpm={rpm:4.1f} | "
                f"Samples: {len(traj['Y_penalized'])} | Best Power: {traj['best_y']:.1f} kW | "
                f"Time: {elapsed_task:.2f}s",
                flush=True,
            )
            task_id += 1

    # 打包整体 .pt 文件供 PyTorch DataLoader 秒级直接载入
    bundled_pt_path = out_path / "malibo_blade_train_dataset.pt"
    torch.save(
        {
            "num_tasks": len(bundled_tasks),
            "in_dim": 60,
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
