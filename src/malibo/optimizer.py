"""MALIBO 贝叶斯元优化器算法驱动实现 (Algorithm 1):
严格对齐 Pan et al. (2024, Section 4 & Algorithm 1) 并适配 114 维 3D 叶片连续设计空间:
1. 初始化预算对齐 (n_init = 15): 与 TuRBO-1 / LLM-Meta-TuRBO 完全一致的基准 + 14 个 [0,1]^D 对称微扰；
2. 首个采集步元均值先验混合探索 (Warm-Start Pool @ Step 16): 在更广 Sobol 域引入任务无关均值网络 m(phi(x))；
3. 目标工况在线拉普拉斯自适应: 求解 z_MAP 与 Laplace 精度协方差，抽取 Thompson 采样向量 z_hat ~ q(z)；
4. 分类对数几率残差提升树: 基于早停 GBDT 分类器在连续 decision_function 边距上叠加残差校准 C_GB(x; z_hat)；
5. 准蒙特卡洛 Sobol 极大化: 在适配 D=114 的自适应信赖域 (tr_len_init=0.80, failure_tol=6) 内通过 5,120 个低差异 Sobol 样本极大化采集效用。
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from src.evaluator import WindTurbineBladeEvaluator
from src.malibo.adaptation import LaplaceTaskAdapter
from src.malibo.config import MALIBOBladeConfig
from src.malibo.models import MALIBOMetaModel
from src.malibo.residual_booster import MALIBOResidualBooster
from src.optimizers.base import BaseBladeOptimizer


class MALIBOBladeOptimizer(BaseBladeOptimizer):
    """MALIBO 3D 风力机叶片贝叶斯元优化器"""

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
        self.tr_len = opt_cfg.tr_len_init
        self.tr_len_restart = opt_cfg.tr_len_restart
        self.tr_len_min = opt_cfg.tr_len_min
        self.tr_len_max = opt_cfg.tr_len_max
        self.success_tolerance = opt_cfg.success_tolerance
        self.failure_tolerance = opt_cfg.failure_tolerance
        self.prob_perturb = opt_cfg.prob_perturb
        self.n_candidates = opt_cfg.n_candidates
        self.gamma_init = opt_cfg.gamma_init
        self.gamma_min = opt_cfg.gamma_min
        self.exploration_floor = opt_cfg.exploration_floor

        self.success_counter = 0
        self.failure_counter = 0

        self.adapter = LaplaceTaskAdapter(
            meta_model=self.meta_model,
            config=self.config.adaptation,
            gamma=self.gamma_init,
            device=self.device,
        )
        self.residual_booster = MALIBOResidualBooster(
            meta_model=self.meta_model,
            config=self.config.booster,
            device=self.device,
            seed=self.seed,
        )

    def _adjust_trust_region(self, improved: bool):
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter >= self.success_tolerance:
            self.tr_len = min(self.tr_len * 2.0, self.tr_len_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.tr_len = self.tr_len / 2.0
            self.failure_counter = 0

        if self.tr_len < self.tr_len_min:
            self.tr_len = self.tr_len_restart
            self.success_counter = 0
            self.failure_counter = 0

    def _sample_sobol_tr_candidates(
        self,
        center_u: np.ndarray,
        tr_width: float,
        n_cand: int,
        prob_perturb: float,
        step_idx: int,
    ) -> np.ndarray:
        """在指定置信域内生成低差异 Sobol 稀疏掩码候选点集."""
        d = self.dim
        lb_tr = np.clip(center_u - tr_width / 2.0, 0.0, 1.0)
        ub_tr = np.clip(center_u + tr_width / 2.0, 0.0, 1.0)

        engine = SobolEngine(dimension=d, scramble=True, seed=int(self.seed * 1009 + step_idx * 37))
        sobol_raw = engine.draw(n_cand).numpy().astype(np.float64)
        cand_raw = lb_tr + (ub_tr - lb_tr) * sobol_raw

        mask = self.rng.rand(n_cand, d) < prob_perturb
        mask_all_false = ~np.any(mask, axis=1)
        if np.any(mask_all_false):
            rand_dims = self.rng.randint(0, d, size=int(np.sum(mask_all_false)))
            mask[mask_all_false, rand_dims] = True

        return np.where(mask, cand_raw, center_u)

    def get_warm_start_candidate(self, n_search: int = 5120) -> np.ndarray:
        """从预训练元均值网络 m(phi(x)) 中搜寻热启动高价值区域推荐点 x0_meta."""
        with torch.no_grad():
            u0 = self.to_unit(self.x0)
            cand_u = self._sample_sobol_tr_candidates(
                center_u=u0,
                tr_width=0.45,
                n_cand=n_search,
                prob_perturb=0.35,
                step_idx=0,
            )
            cand_u[0] = u0
            cand_tensor = torch.from_numpy(cand_u).float().to(self.device)
            mean_vals = self.meta_model.predict_mean(cand_tensor).squeeze(-1)
            best_idx = int(torch.argmax(mean_vals).item())
            return self.from_unit(cand_u[best_idx])

    def run(self) -> Dict[str, Any]:
        """运行完整 MALIBO 元贝叶斯优化流程."""
        t0 = time.time()
        d = self.dim

        X_unit_list: List[np.ndarray] = []
        Y_minimize_list: List[float] = []

        # 1. 评估基准叶片构型 x0
        score0 = self.evaluate_initial()
        u0 = self.to_unit(self.x0)
        X_unit_list.append(u0)
        Y_minimize_list.append(-score0)

        # 2. 初始微扰散点 (归一化 [0, 1]^D 空间对称微扰，严格对齐 n_init = 15 初始化预算)
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise_u = self.rng.uniform(-0.04, 0.04, size=d)
            u_cand = np.clip(u0 + noise_u, 0.0, 1.0)
            x_cand = self.from_unit(u_cand)
            score = self.evaluate(x_cand)
            X_unit_list.append(u_cand)
            Y_minimize_list.append(-score)

        # 3. MALIBO 在线元贝叶斯主循环
        while self.eval_count < self.max_evals:
            score_before = self.best_score
            step = self.eval_count + 1

            # 动态退火分位数 gamma_t 与探索方差地板
            progress = min(1.0, float(self.eval_count) / float(self.max_evals))
            gamma_curr = float(self.gamma_init * (1.0 - progress) + self.gamma_min * progress)
            floor_curr = float(self.exploration_floor * max(0.02, (1.0 - progress) ** 2))

            # 截断不可行惩罚离群值 (score <= -0.25 -> -score <= 1.0)，保持可行域排序与权重尺度精准
            y_clean_min = np.minimum(np.array(Y_minimize_list, dtype=np.float64), 1.0)
            tau_curr = float(np.quantile(y_clean_min, gamma_curr))

            x_obs_tensor = torch.from_numpy(np.array(X_unit_list, dtype=np.float32)).to(self.device)
            y_obs_tensor = torch.from_numpy(y_clean_min.astype(np.float32)).to(self.device)

            # 在线拉普拉斯自适应求解 z_MAP 与 Thompson 采样嵌入 z_hat ~ N(z_MAP, Sigma)
            z_hat, z_map, sigma_z = self.adapter.adapt_and_sample(
                x_obs=x_obs_tensor,
                y_obs=y_obs_tensor,
                gamma_current=gamma_curr,
                exploration_floor=floor_curr,
            )

            # 拟合早停残差分类弱学习器 (Pan et al., 2024 Eq. 10)
            x_obs_np = np.array(X_unit_list, dtype=np.float64)
            self.residual_booster.fit_with_early_stopping(
                x_obs=x_obs_np,
                y_minimize=y_clean_min,
                tau=tau_curr,
                z_sample=z_hat,
            )

            # 在自适应信赖域内构建 5,120 个准蒙特卡洛 Sobol 候选集
            x_best_u = self.to_unit(self.best_x)
            if step == self.n_init + 1:
                # 首个主动采集步 (Step 16): 混合更广半径 Sobol 热启动候选池与本地 TR 候选池
                half = self.n_candidates // 2
                c_tr = self._sample_sobol_tr_candidates(
                    center_u=x_best_u, tr_width=self.tr_len, n_cand=half,
                    prob_perturb=self.prob_perturb, step_idx=step,
                )
                c_warm = self._sample_sobol_tr_candidates(
                    center_u=u0, tr_width=0.50, n_cand=self.n_candidates - half,
                    prob_perturb=min(0.30, self.prob_perturb * 1.5), step_idx=step + 999,
                )
                candidates_u = np.vstack([c_tr, c_warm])
            else:
                candidates_u = self._sample_sobol_tr_candidates(
                    center_u=x_best_u, tr_width=self.tr_len, n_cand=self.n_candidates,
                    prob_perturb=self.prob_perturb, step_idx=step,
                )

            # 计算连续综合效用边界 C_GB(x; z_hat) = m(phi(x)) + z_hat^T phi(x) + sum r_m(x)
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

            improved = (new_score > score_before + 1e-3)
            self._adjust_trust_region(improved)

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
