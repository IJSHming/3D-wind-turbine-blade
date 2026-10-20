"""MALIBO 贝叶斯元优化器算法驱动实现 (Algorithm 1):
严格对齐 Pan et al. (2024, Section 4 & Algorithm 1):
1. 初始热启动推荐 (Warm-Start):
   利用跨工况预训练的任务无关均值网络 x_0 = argmax_{x} m(phi(x)) 提取元气动全局高效用区域；
2. 目标工况在线拉普拉斯自适应:
   在测试目标工况 (V=9.0 m/s, rpm=10.0) 上实时求解 z_MAP 与拉普拉斯协方差 Sigma_z，
   执行 Thompson 采样抽取多样化任务嵌入 z_hat ~ q(z)；
3. 残差梯度提升弱学习器集成:
   拟合元 Logit 与当前工况真实功率响应之间的局部差异 C_GB(x)；
4. 准蒙特卡洛极大化:
   在自适应信赖域内生成 5,120 个准蒙特卡洛候选点，极大化综合后验效用函数：
   alpha(x) = m(phi(x)) + z_hat^T phi(x) + C_GB(x)。
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.blade_geometry import loft_blade_3d
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

        # 自适应信赖域超参数
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

        # 组件封装
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

    def get_warm_start_candidate(self, n_search: int = 5000) -> np.ndarray:
        """从预训练元均值网络 m(phi(x)) 中高精度搜寻热启动全局推荐点 x0_meta"""
        with torch.no_grad():
            cand_u = torch.rand(n_search, self.dim, device=self.device)
            mean_vals = self.meta_model.predict_mean(cand_u).squeeze(-1)
            best_idx = int(torch.argmax(mean_vals).item())
            best_u = cand_u[best_idx].cpu().numpy()
        return self.from_unit(best_u)

    def run(self) -> Dict[str, Any]:
        """运行完整 MALIBO 元贝叶斯优化流程"""
        t0 = time.time()
        d = self.dim

        X_unit_list: List[np.ndarray] = []
        Y_minimize_list: List[float] = []

        # 1. 评估基准叶片构型 x0
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_unit_list.append(self.to_unit(self.x0))
        Y_minimize_list.append(-score0)

        # 2. 评估任务无关元均值网络推荐的热启动点 (Warm-Start Recommendation)
        x_meta_warm = self.get_warm_start_candidate()
        score_warm = self.evaluate(x_meta_warm)
        X_unit_list.append(self.to_unit(x_meta_warm))
        Y_minimize_list.append(-score_warm)

        # 3. 初始散点探索 (补足至 n_init 步)
        while self.eval_count < self.n_init and self.eval_count < self.max_evals:
            noise = self.rng.uniform(-0.04, 0.04, size=d)
            x_cand = np.clip(self.best_x + noise, self.lower_bounds, self.upper_bounds)
            score = self.evaluate(x_cand)
            X_unit_list.append(self.to_unit(x_cand))
            Y_minimize_list.append(-score)

        # 4. MALIBO 在线元贝叶斯主循环
        while self.eval_count < self.max_evals:
            score_before = self.best_score
            step = self.eval_count

            # 动态退火分位数 gamma_t (前期注重探索，后期聚焦前沿极值)
            progress = min(1.0, float(step) / float(self.max_evals))
            gamma_curr = float(self.gamma_init * (1.0 - progress) + self.gamma_min * progress)

            x_obs_tensor = torch.from_numpy(np.array(X_unit_list, dtype=np.float32)).to(self.device)
            y_obs_tensor = torch.from_numpy(np.array(Y_minimize_list, dtype=np.float32)).to(self.device)

            # 在线拉普拉斯自适应求解 z_MAP 与 Thompson 采样嵌入 z_hat
            z_hat, z_map, sigma_z = self.adapter.adapt_and_sample(
                x_obs=x_obs_tensor,
                y_obs=y_obs_tensor,
                gamma_current=gamma_curr,
                exploration_floor=self.exploration_floor,
            )

            # 拟合残差弱学习器
            x_obs_np = np.array(X_unit_list, dtype=np.float64)
            y_obs_np = np.array(Y_minimize_list, dtype=np.float64)
            self.residual_booster.fit(x_obs_np, y_obs_np, z_map)

            # 构建信赖域采样候选空间
            x_best_u = self.to_unit(self.best_x)
            lb_tr = np.clip(x_best_u - self.tr_len / 2.0, 0.0, 1.0)
            ub_tr = np.clip(x_best_u + self.tr_len / 2.0, 0.0, 1.0)

            n_cand = self.n_candidates
            cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(n_cand, d))

            # 坐标稀疏掩码扰动
            mask = self.rng.rand(n_cand, d) < self.prob_perturb
            mask_all_false = ~np.any(mask, axis=1)
            rand_dims = self.rng.randint(0, d, size=np.sum(mask_all_false))
            mask[mask_all_false, rand_dims] = True
            candidates_u = np.where(mask, cand_raw, x_best_u)

            # 计算元采集效用: alpha(x) = m(phi(x)) + z_hat^T phi(x) + C_GB(x)
            with torch.no_grad():
                cand_tensor = torch.from_numpy(candidates_u).float().to(self.device)
                meta_logits = self.meta_model(cand_tensor, z=z_hat).cpu().numpy()

            booster_corr = self.residual_booster.predict(candidates_u)
            acquisition_scores = meta_logits + booster_corr

            # 选择极大化采集效用的最优候选点
            best_cand_idx = int(np.argmax(acquisition_scores))
            x_next_u = candidates_u[best_cand_idx]
            x_next = self.from_unit(x_next_u)

            new_score = self.evaluate(x_next)
            X_unit_list.append(x_next_u)
            Y_minimize_list.append(-new_score)

            # 更新信赖域状态
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
                "torque_kNm": float(det.get("torque_kNm", 0.0)),
                "thrust_kN": float(det.get("thrust_kN", 0.0)),
                "Cp": float(det.get("Cp", 0.0)),
                "Ct": float(det.get("Ct", 0.0)),
                "is_feasible": bool(det.get("is_feasible", True)),
                "total_penalty": float(det.get("total_penalty", 0.0)),
            },
            "elapsed_seconds": float(elapsed),
        }
