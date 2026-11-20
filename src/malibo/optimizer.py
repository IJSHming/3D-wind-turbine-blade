"""MALIBO 贝叶斯元优化器算法驱动实现 (src/malibo/optimizer.py):
严格对齐原论文 Pan et al. (ICML 2024, Section 4, Algorithm 1 & Appendix G):
- 彻底移除信赖域 (Trust Region)、稀疏坐标掩码 (prob_perturb)、gamma 退火与人工探索方差地板，
  确保作为会议论文纯原版 MALIBO 基线与原论文算法完全一致 (为后续期刊论文算法改进保留纯净参照基准):
  1. 初始化协议 (n_init = 15): 评估标准 NREL Phase VI 基准构型 x0 及 14 个对称微扰样本 (与 TuRBO-1 / LLM-Meta-TuRBO 严格对齐);
  2. 目标任务拉普拉斯自适应 (Algorithm 1 lines 4-6): 固定 gamma = 1/3, L-BFGS 求解 z_MAP (Eq. 6),
     构建精确负 Hessian 精度矩阵 Sigma_N^-1 (Eq. 7), 抽取纯净 Thompson 样本 z_hat ~ N(z_MAP, Sigma_N) (Eq. 9);
  3. 残差梯度提升集成 (Algorithm 1 line 7, Eq. 10): 基于 70/30 验证早停 GBDT 拟合 LFBO 残差修正项 C_GB(x; z_hat);
  4. 全局随机搜索极大化 (Appendix G): 在完整单位超立方 [0, 1]^D 上均匀随机采样 5,120 个全局候选点并选取采集极大值点 x*。
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.evaluator import WindTurbineBladeEvaluator
from src.malibo.adaptation import LaplaceTaskAdapter
from src.malibo.config import MALIBOBladeConfig
from src.malibo.models import MALIBOMetaModel
from src.malibo.residual_booster import MALIBOResidualBooster
from src.optimizers.base import BaseBladeOptimizer


class MALIBOBladeOptimizer(BaseBladeOptimizer):
    """MALIBO 3D 风力机叶片贝叶斯元优化器 (Strictly Aligned with Pan et al., ICML 2024 Algorithm 1)"""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        evaluator: WindTurbineBladeEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        config: Optional[MALIBOBladeConfig] = None,
        max_evals: int = 300,
        seed: int = 42,
        n_init: int = 15,
        device: Optional[torch.device] = None,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed)
        self.meta_model = meta_model
        self.config = config if config is not None else MALIBOBladeConfig()
        self.n_init = int(n_init)

        if device is None:
            self.device = next(meta_model.parameters()).device
        else:
            self.device = device

        self.meta_model.eval().to(self.device)

        opt_cfg = self.config.optimization
        self.n_candidates = int(opt_cfg.n_candidates)  # Appendix G: 5,120
        self.gamma = float(opt_cfg.gamma)              # Appendix G: fixed 1/3

        self.adapter = LaplaceTaskAdapter(
            meta_model=self.meta_model,
            config=self.config.adaptation,
            gamma=self.gamma,
            device=self.device,
        )
        self.residual_booster = MALIBOResidualBooster(
            meta_model=self.meta_model,
            config=self.config.booster,
            device=self.device,
            seed=self.seed,
        )

    def get_warm_start_candidate(self, n_search: int = 5120) -> np.ndarray:
        """Algorithm 1 line 1: 通过全局 5,120 随机搜索极大化任务无关均值预测网络 g_omega(x)."""
        with torch.no_grad():
            cand_u = self.rng.uniform(0.0, 1.0, size=(n_search, self.dim))
            cand_tensor = torch.from_numpy(cand_u).float().to(self.device)
            mean_vals = self.meta_model.predict_mean(cand_tensor).squeeze(-1)
            best_idx = int(torch.argmax(mean_vals).item())
            return self.from_unit(cand_u[best_idx])

    def run(self) -> Dict[str, Any]:
        """运行严格对齐 Pan et al. (ICML 2024) Algorithm 1 的纯全局 MALIBO 元贝叶斯优化."""
        t0 = time.time()
        d = self.dim

        X_unit_list: List[np.ndarray] = []
        Y_minimize_list: List[float] = []

        # 1. 评估基准叶片构型 x0
        score0 = self.evaluate_initial()
        u0 = self.to_unit(self.x0)
        X_unit_list.append(u0)
        Y_minimize_list.append(-score0)

        # 2. 初始设计采样 (与 TuRBO-1 / LLM-Meta-TuRBO 共享一致的 n_init=15 初始化协议)
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise_u = self.rng.uniform(-0.04, 0.04, size=d)
            u_cand = np.clip(u0 + noise_u, 0.0, 1.0)
            x_cand = self.from_unit(u_cand)
            score = self.evaluate(x_cand)
            X_unit_list.append(u_cand)
            Y_minimize_list.append(-score)

        # 3. MALIBO Algorithm 1 主优化循环 (纯全局 [0, 1]^D 5,120 随机搜索, 无信赖域)
        while self.eval_count < self.max_evals:
            step = self.eval_count + 1

            # 截断极端不可行离群惩罚 (y_min <= 1.0) 并标准化，与元训练尺度严格一致
            y_clean = np.minimum(np.array(Y_minimize_list, dtype=np.float64), 1.0)
            y_mean = float(np.mean(y_clean))
            y_std_val = max(float(np.std(y_clean)), 1e-4)
            y_norm = (y_clean - y_mean) / y_std_val
            tau_std = float(np.quantile(y_norm, self.gamma))

            x_obs_np = np.array(X_unit_list, dtype=np.float64)
            x_obs_tensor = torch.from_numpy(x_obs_np.astype(np.float32)).to(self.device)
            y_obs_tensor = torch.from_numpy(y_norm.astype(np.float32)).to(self.device)

            # Algorithm 1 lines 4-6: 在线拉普拉斯 MAP + 精度矩阵 + Thompson 采样 z_hat ~ N(z_MAP, Sigma_N)
            z_hat, _z_map, _L_prec = self.adapter.adapt_and_sample(
                x_obs=x_obs_tensor,
                y_obs_std=y_obs_tensor,
                tau_std=tau_std,
                rng=self.rng,
            )

            # Algorithm 1 line 7 (Eq. 10 & Appendix G): 70/30 早停残差梯度提升树
            self.residual_booster.fit_with_early_stopping(
                x_obs=x_obs_np,
                y_std=y_norm,
                tau_std=tau_std,
                z_sample=z_hat,
                step_seed=int(self.seed * 1009 + step),
            )

            # Appendix G: 全局 [0, 1]^D 上 5,120 个均匀随机采样点极大化采集函数 C_GB(x; z_hat)
            candidates_u = self.rng.uniform(0.0, 1.0, size=(self.n_candidates, d))
            acquisition_scores = self.residual_booster.predict_utility(
                x_cand=candidates_u,
                z_sample=z_hat,
                use_raw_margin=True,
            )

            best_cand_idx = int(np.argmax(acquisition_scores))
            x_next_u = candidates_u[best_cand_idx]
            x_next = self.from_unit(x_next_u)

            new_score = self.evaluate(x_next)
            X_unit_list.append(x_next_u)
            Y_minimize_list.append(-new_score)

        elapsed = time.time() - t0
        det = self.evaluator.evaluate_detailed(self.best_x)

        return {
            "method": "MALIBO",
            "seed": self.seed,
            "best_score": float(self.best_score),
            "history": [float(v) for v in self.history_best],
            "best_x": [float(v) for v in self.best_x],
            "detailed": {
                "power_kW": float(det.get("power_kW", 0.0)),
                "power_MW": float(det.get("power_MW", 0.0)),
                "torque_Nm": float(det.get("torque_Nm", 0.0)),
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
