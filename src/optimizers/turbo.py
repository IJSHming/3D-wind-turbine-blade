"""高维信赖域贝叶斯优化器 (TuRBO-1):
严格遵循 Eriksson et al. (NeurIPS 2019) 算法配置:
1. 初始设计点集 (n_init = 15): 基准构型 + 归一化超立方 [0, 1]^D 局部扰动;
2. 自适应信赖域状态机: L_init = 0.8, L_min = 0.5^7, L_max = 1.6, success_tol = 3, failure_tol = 6;
3. 坐标扰动概率: p_perturb = min(20.0 / D, 1.0) (D=114 时约为 0.1754);
4. 5,000 个信赖域候选点 + Matérn 5/2 GP Cholesky 后验汤普森采样 (Thompson Sampling).
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from src.evaluator import WindTurbineBladeEvaluator
from src.optimizers.base import BaseBladeOptimizer


def matern52_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    lengthscale: float = 0.30,
    outputscale: float = 1.0,
) -> np.ndarray:
    """Matérn 5/2 协方差核函数计算 (纯 NumPy/SciPy 向量化实现)."""
    dist = cdist(x1 / lengthscale, x2 / lengthscale, metric="euclidean")
    sqrt5 = 2.23606797749979
    val = (1.0 + sqrt5 * dist + (5.0 / 3.0) * dist ** 2) * np.exp(-sqrt5 * dist)
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
        length_min: float = 0.5 ** 7,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 6,
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

        self.prob_perturb = min(20.0 / self.dim, 1.0)
        self.n_success = 0
        self.n_failure = 0

    def _adjust_trust_region(self, improved: bool) -> None:
        """更新信赖域尺寸 (成功连续扩倍、失败连续收缩与下界重启)."""
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
            self.L = self.L_init
            self.n_success = 0
            self.n_failure = 0

    def run(self) -> Dict[str, Any]:
        """执行完整 TuRBO-1 优化."""
        t0 = time.time()
        d = self.dim

        X_train: List[np.ndarray] = []
        Y_train: List[float] = []

        # 1. 评估基准叶片构型 x0
        score0 = self.evaluate_initial()
        u0 = self.to_unit(self.x0)
        X_train.append(u0)
        Y_train.append(score0)

        # 2. 在 [0, 1]^D 归一化空间进行尺度一致的初始微扰采样 (n_init - 1 点)
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            u_cand = np.clip(u0 + self.rng.uniform(-0.04, 0.04, size=d), 0.0, 1.0)
            x_cand = self.from_unit(u_cand)
            score = self.evaluate(x_cand)
            X_train.append(u_cand)
            Y_train.append(score)

        # 3. TuRBO 主优化循环
        while self.eval_count < self.max_evals:
            X_arr = np.array(X_train, dtype=np.float64)
            Y_arr = np.array(Y_train, dtype=np.float64)

            # 截断极端负惩罚离群值再标准化，防止拉伸 y_std 掩盖可行域内部梯度信号
            Y_gp = np.maximum(Y_arr, -1.0)
            y_mean = float(np.mean(Y_gp))
            y_std = float(np.std(Y_gp)) + 1e-8
            Y_norm = (Y_gp - y_mean) / y_std

            x_best_u = self.to_unit(self.best_x)
            lb_tr = np.clip(x_best_u - self.L / 2.0, 0.0, 1.0)
            ub_tr = np.clip(x_best_u + self.L / 2.0, 0.0, 1.0)

            n_cand = self.n_candidates
            cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(n_cand, d))

            mask = self.rng.rand(n_cand, d) < self.prob_perturb
            mask_all_false = ~np.any(mask, axis=1)
            rand_dims = self.rng.randint(0, d, size=int(np.sum(mask_all_false)))
            mask[mask_all_false, rand_dims] = True

            candidates = np.where(mask, cand_raw, x_best_u)

            K_train = matern52_kernel(X_arr, X_arr, lengthscale=0.35, outputscale=1.0)
            K_train += 1e-4 * np.eye(len(X_arr))

            try:
                c_factor = cho_factor(K_train, lower=True)
                alpha_weights = cho_solve(c_factor, Y_norm)

                K_cand_train = matern52_kernel(candidates, X_arr, lengthscale=0.35, outputscale=1.0)
                mu_cand = K_cand_train @ alpha_weights

                K_diag = np.ones(n_cand)
                v = cho_solve(c_factor, K_cand_train.T)
                var_cand = np.maximum(1e-6, K_diag - np.sum(K_cand_train.T * v, axis=0))
                sigma_cand = np.sqrt(var_cand)

                ts_scores = mu_cand + sigma_cand * self.rng.randn(n_cand)
                best_cand_idx = int(np.argmax(ts_scores))
                x_next_u = candidates[best_cand_idx]
            except Exception:
                x_next_u = candidates[0]

            x_next = self.from_unit(x_next_u)
            score_before = self.best_score
            new_score = self.evaluate(x_next)

            X_train.append(x_next_u)
            Y_train.append(new_score)

            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved)

        elapsed = time.time() - t0
        det = self.evaluator.evaluate_detailed(self.best_x)

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
                "is_feasible": bool(det.get("is_feasible", det.get("feasible", False))),
                "total_penalty": float(det.get("total_penalty", det.get("total_penalty_kW", 0.0))),
                "span_radii": [float(v) for v in det.get("span_radii", [])],
                "span_chord": [float(v) for v in det.get("span_chord", [])],
                "span_twist_deg": [float(v) for v in det.get("span_twist_deg", [])],
                "span_alpha_deg": [float(v) for v in det.get("span_alpha_deg", [])],
                "blade_pitch_deg": float(det.get("blade_pitch_deg", 0.0)),
                "span_effective_pitch_deg": [float(v) for v in det.get("span_effective_pitch_deg", [])],
            },
            "elapsed_seconds": float(elapsed),
        }
