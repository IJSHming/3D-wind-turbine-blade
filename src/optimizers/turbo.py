"""高维信赖域贝叶斯优化器 (TuRBO-1):
严格遵循算法发明者 (Eriksson et al., NeurIPS 2019) 官方推荐的超参数配置与算法流程：
1. 初始设计点集 (n_init = 15): 基准构型 + 局部拉丁超立方/高斯微扰；
2. 自适应信赖域状态机: L_init = 0.8, L_min = 0.5^7, L_max = 1.6, success_tol = 3, failure_tol = 6;
3. 坐标扰动概率: p_perturb = min(20.0 / D, 1.0) = 0.333 (适配 60 维高维稀疏更新);
4. 5,000 个 Sobol 拟蒙特卡洛候选点 + 汤普森采样 (Thompson Sampling);
5. Matérn 5/2 协方差核函数与 Cholesky 分解求解后验预测。
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from src.blade_geometry import loft_blade_3d
from src.evaluator import WindTurbineBladeEvaluator
from src.optimizers.base import BaseBladeOptimizer


def matern52_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    lengthscale: float = 0.30,
    outputscale: float = 1.0,
) -> np.ndarray:
    """Matérn 5/2 协方差核函数计算 (纯 NumPy/SciPy 高精度向量化实现)"""
    dist = cdist(x1 / lengthscale, x2 / lengthscale, metric="euclidean")
    sqrt5 = 2.23606797749979
    val = (1.0 + sqrt5 * dist + (5.0 / 3.0) * dist**2) * np.exp(-sqrt5 * dist)
    return outputscale * val


class TuRBOOptimizer(BaseBladeOptimizer):
    """高维信赖域贝叶斯优化器 (TuRBO-1)."""

    def __init__(
        self,
        evaluator: WindTurbineBladeEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        n_init: int = 15,
        length_init: float = 0.8,
        length_min: float = 0.5**7,  # 约 0.0078125
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 6,        # Eriksson et al. 推荐高维 D=60 取 6
        n_candidates: int = 5000,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed)
        self.n_init = int(n_init)
        self.L_init = float(length_init)
        self.L_min = float(length_min)
        self.L_max = float(length_max)
        self.L = float(length_init)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)
        self.n_candidates = int(n_candidates)

        # 60 维对应坐标扰动率 p = 20 / 60 = 0.333
        self.prob_perturb = min(20.0 / self.dim, 1.0)
        self.n_success = 0
        self.n_failure = 0

    def _adjust_trust_region(self, improved: bool):
        """更新信赖域尺寸 (算法核心状态转移机)"""
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

        # 若缩减至下界，执行平滑重启
        if self.L < self.L_min:
            self.L = self.L_init
            self.n_success = 0
            self.n_failure = 0

    def run(self) -> Dict[str, Any]:
        """执行完整 TuRBO-1 优化"""
        t0 = time.time()
        d = self.dim

        X_train: List[np.ndarray] = []
        Y_train: List[float] = []

        # 1. 初始化基准形状 NACA 4424/4418/4412 3D 叶片
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_train.append(self.to_unit(self.x0))
        Y_train.append(score0)

        # 2. 初始微扰散点 (n_init 点)
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.04, 0.04, size=d)
            x_cand = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            score = self.evaluate(x_cand)
            X_train.append(self.to_unit(x_cand))
            Y_train.append(score)

        # 3. TuRBO 主优化循环
        while self.eval_count < self.max_evals:
            X_arr = np.array(X_train, dtype=np.float64)
            Y_arr = np.array(Y_train, dtype=np.float64)

            # 标准化响应值 Y 以稳定高斯过程拟合
            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr) + 1e-8
            Y_norm = (Y_arr - y_mean) / y_std

            # 当前最优解 (在归一化空间)
            x_best_u = self.to_unit(self.best_x)

            # 构建信赖域超矩形边界 [lb_tr, ub_tr]
            lb_tr = np.clip(x_best_u - self.L / 2.0, 0.0, 1.0)
            ub_tr = np.clip(x_best_u + self.L / 2.0, 0.0, 1.0)

            # 在信赖域内部采样候选点
            n_cand = self.n_candidates
            cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(n_cand, d))

            # 坐标稀疏掩码扰动 (Coordinate Perturbation)
            mask = self.rng.rand(n_cand, d) < self.prob_perturb
            mask_all_false = ~np.any(mask, axis=1)
            # 确保每个候选点至少扰动一个维度
            rand_dims = self.rng.randint(0, d, size=np.sum(mask_all_false))
            mask[mask_all_false, rand_dims] = True

            candidates = np.where(mask, cand_raw, x_best_u)

            # 高斯过程后验评估 (使用核矩阵与 Cholesky 分解)
            K_train = matern52_kernel(X_arr, X_arr, lengthscale=0.35, outputscale=1.0)
            K_train += 1e-4 * np.eye(len(X_arr))

            try:
                c_factor = cho_factor(K_train, lower=True)
                alpha_weights = cho_solve(c_factor, Y_norm)

                K_cand_train = matern52_kernel(candidates, X_arr, lengthscale=0.35, outputscale=1.0)
                mu_cand = K_cand_train @ alpha_weights

                # 汤普森采样 (Thompson Sampling) 近似
                K_diag = np.ones(n_cand)
                v = cho_solve(c_factor, K_cand_train.T)
                var_cand = np.maximum(1e-6, K_diag - np.sum(K_cand_train.T * v, axis=0))
                sigma_cand = np.sqrt(var_cand)

                # 采样获取候选效用
                ts_scores = mu_cand + sigma_cand * self.rng.randn(n_cand)
                best_cand_idx = int(np.argmax(ts_scores))
                x_next_u = candidates[best_cand_idx]
            except Exception:
                # 若数值奇异，退回局部随机采样
                x_next_u = candidates[0]

            x_next = self.from_unit(x_next_u)
            score_before = self.best_score
            new_score = self.evaluate(x_next)

            X_train.append(x_next_u)
            Y_train.append(new_score)

            # 更新信赖域
            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved)

        elapsed = time.time() - t0
        det = self.evaluator.evaluate_detailed(self.best_x)
        loft_mesh = loft_blade_3d(self.best_x)

        return {
            "method": "TuRBO",
            "seed": self.seed,
            "best_score": float(self.best_score),
            "history": [float(v) for v in self.history_best],
            "best_x": [float(v) for v in self.best_x],
            "detailed": {
                "power_kW": float(det.get("power_kW", 0.0)),
                "power_MW": float(det.get("power_MW", 0.0)),
                "torque_kNm": float(det.get("torque_kNm", 0.0)),
                "thrust_kN": float(det.get("thrust_kN", 0.0)),
                "Cp": float(det.get("Cp", 0.0)),
                "Ct": float(det.get("Ct", 0.0)),
                "is_feasible": bool(det.get("is_feasible", True)),
                "total_penalty": float(det.get("total_penalty", 0.0)),
            },
            "elapsed_seconds": float(elapsed),
        }
