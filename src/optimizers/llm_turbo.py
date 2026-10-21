"""LLM-Meta-TuRBO 3D 风力机叶片优化器：
基于大语言模型宏观定向元控制的高维信赖域贝叶斯优化器 (TuRBO-1)。
结合风机叶片气动学专家先验与宏观几何变形注入，在保持基准 TuRBO 超参对齐的同时，
显著提升前中期局部探索效率与收敛品质。

关键特性：
1. 前中期元优化干预 (step <= 110 触发)，后期完全交由 TuRBO-1 进行高精度纯贝叶斯收敛；
2. 宏观方向偏移量 (Directional Strategy) 注入与工业几何强约束前置检验；
3. 保留前期介入时倾向于直接给予优化器一个偏移量的激进特性 (Aggressive Offset Injection)；
4. 汤普森候选采样方向性漂移倾斜，兼顾全局气动先验与局部贝叶斯不确定性探索。
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from src.blade_geometry import loft_blade_3d, unpack_blade_params
from src.evaluator import WindTurbineBladeEvaluator
from src.llm_agent import LLMMetaBladeAgent
from src.optimizers.base import BaseBladeOptimizer
from src.optimizers.turbo import matern52_kernel


def convert_directional_strategy_to_blade_delta(
    strat: Dict[str, Any]
) -> np.ndarray:
    """将 LLM 输出的宏观气动方向策略转换为 120 维叶片参数向量偏移量 delta。"""
    delta = np.zeros(120, dtype=np.float64)

    # 6 个站位对应的宏观方向映射权重
    # (camber_scale, thick_scale, chord_scale, twist_scale, c_key, t_key, ch_key, tw_key)
    station_configs = [
        # Station 1: Root transition (AF07)
        (0.7, 0.8, 0.7, 1.0, "camber_shift_root", "thickness_shift_root", "chord_shift_root", "twist_shift_root"),
        # Station 2: Shoulder max chord (AF11)
        (1.0, 1.0, 1.0, 0.9, "camber_shift_root", "thickness_shift_root", "chord_shift_root", "twist_shift_root"),
        # Station 3: Mid-inboard (AF22)
        (0.9, 1.0, 0.9, 0.9, "camber_shift_mid", "thickness_shift_mid", "chord_shift_mid", "twist_shift_mid"),
        # Station 4: Mid-span primary power (AF34)
        (1.0, 1.0, 1.0, 1.0, "camber_shift_mid", "thickness_shift_mid", "chord_shift_mid", "twist_shift_mid"),
        # Station 5: Outboard high Mach (AF45)
        (1.0, 1.0, 1.0, 1.0, "camber_shift_tip", "thickness_shift_tip", "chord_shift_tip", "twist_shift_tip"),
        # Station 6: Tip washout (AF54)
        (0.7, 0.8, 0.5, 1.1, "camber_shift_tip", "thickness_shift_tip", "chord_shift_tip", "twist_shift_tip"),
    ]

    for k in range(6):
        c_sc, t_sc, ch_sc, tw_sc, c_key, t_key, ch_key, tw_key = station_configs[k]
        s_camber = float(strat.get(c_key, 0.0)) * c_sc
        s_thick = float(strat.get(t_key, 0.0)) * t_sc
        s_chord = float(strat.get(ch_key, 0.0)) * ch_sc
        s_twist = float(strat.get(tw_key, 0.0)) * tw_sc

        idx = k * 20
        # 1. 翼型上表面权重 (正弦加权拱起吸力面，增强前中部吸力峰)
        u_weights = np.sin(np.pi * (np.arange(8) + 0.5) / 8.0)
        delta[idx : idx + 8] = s_camber * u_weights + s_thick * 0.5

        # 2. 翼型下表面权重 (后部抬升加权，增强后部加载与下表面反弯)
        l_weights = (np.arange(8) + 1.0) / 8.0
        delta[idx + 8 : idx + 16] = s_camber * l_weights * 0.8 - s_thick * 0.5

        # 3. 前缘与后缘厚度保持微弱扰动
        delta[idx + 16] = 0.0
        delta[idx + 17] = 0.0

        # 4. 弦长偏移量
        delta[idx + 18] = s_chord

        # 5. 扭角偏移量
        delta[idx + 19] = s_twist

    return delta


class LLMMetaTuRBOOptimizer(BaseBladeOptimizer):
    """基于大语言模型宏观定向元控制的高维信赖域贝叶斯优化器 (LLM-Meta-TuRBO)。"""

    def __init__(
        self,
        evaluator: WindTurbineBladeEvaluator,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_evals: int = 300,
        seed: int = 42,
        n_init: int = 15,
        length_init: float = 0.8,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        success_tol: int = 3,
        failure_tol: int = 6,
        n_candidates: int = 5000,
        llm_agent: Optional[LLMMetaBladeAgent] = None,
        milestone_steps: Tuple[int, ...] = (16, 35, 65, 105, 140),
        stag_threshold: int = 12,
        min_intervention_interval: int = 20,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed)
        self.n_init = int(n_init)
        self.L_init = float(length_init)
        self.L_min = float(length_min)
        self.L_max = float(length_max)
        self.base_L = float(length_init)
        self.L = float(length_init)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)
        self.n_candidates = int(n_candidates)

        self.prob_perturb = min(20.0 / self.dim, 1.0)
        self.temperature = 1.0
        self.llm_agent = llm_agent if llm_agent is not None else LLMMetaBladeAgent()

        self.milestone_steps = set(milestone_steps)
        self.stag_threshold = int(stag_threshold)
        self.min_intervention_interval = int(min_intervention_interval)
        self.last_intervention_step = 0
        self.guide_direction_unit: Optional[np.ndarray] = None
        self.guide_direction_steps_left: int = 0
        self.intervention_history: List[Dict[str, Any]] = []

        self.n_success = 0
        self.n_failure = 0

    def _adjust_trust_region(self, improved: bool):
        if improved:
            self.n_success += 1
            self.n_failure = 0
        else:
            self.n_success = 0
            self.n_failure += 1

        if self.n_success >= self.success_tol:
            self.base_L = min(self.base_L * 2.0, self.L_max)
            self.n_success = 0
        elif self.n_failure >= self.failure_tol:
            self.base_L = self.base_L / 2.0
            self.n_failure = 0

        if self.base_L < self.L_min:
            self.base_L = self.L_init
            self.n_success = 0
            self.n_failure = 0

        self.L = self.base_L

    def run(self) -> Dict[str, Any]:
        """运行 LLM-Meta-TuRBO 优化流程"""
        t0 = time.time()
        d = self.dim

        X_train: List[np.ndarray] = []
        Y_train: List[float] = []

        # 1. 初始化基准形状
        score0 = self.evaluate(self.x0)
        self.initial_score = score0
        X_train.append(self.to_unit(self.x0))
        Y_train.append(score0)

        # 2. 初始微扰散点
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            noise = self.rng.uniform(-0.04, 0.04, size=d)
            x_cand = np.clip(self.x0 + noise, self.lower_bounds, self.upper_bounds)
            score = self.evaluate(x_cand)
            X_train.append(self.to_unit(x_cand))
            Y_train.append(score)

        # 3. 主循环
        while self.eval_count < self.max_evals:
            step = self.eval_count + 1
            score_before = self.best_score

            # 判断是否触发 LLM 元优化干预 (限定 step <= 150 前中期)
            need_intervention = (
                (step <= 150)
                and (
                    (step in self.milestone_steps)
                    or (
                        self.stagnation_counter >= self.stag_threshold
                        and (step - self.last_intervention_step) >= self.min_intervention_interval
                    )
                )
            )

            if need_intervention and self.llm_agent is not None:
                self.last_intervention_step = step
                det = self.evaluator.evaluate_detailed(self.best_x)
                stations = unpack_blade_params(self.best_x)

                state_dict = {
                    "step": step,
                    "max_evals": self.max_evals,
                    "budget_left": self.max_evals - self.eval_count,
                    "wind_speed": self.evaluator.wind_speed,
                    "rpm": self.evaluator.rpm,
                    "omega": self.evaluator.omega,
                    "init_power_kW": self.initial_score,
                    "best_power_kW": self.best_score,
                    "best_torque_kNm": det.get("torque_kNm", 0.0),
                    "best_thrust_kN": det.get("thrust_kN", 0.0),
                    "best_Cp": det.get("Cp", 0.0),
                    "best_Ct": det.get("Ct", 0.0),
                    "stagnation_counter": self.stagnation_counter,
                    "tr_length": self.L,
                    "success_counter": self.n_success,
                    "success_tolerance": self.success_tol,
                    "station_metrics": [
                        {
                            "radius": s["radius"],
                            "chord": s["chord"],
                            "twist": s["twist"],
                            "t_max": det.get("geom_metrics", {}).get("station_metrics", [{}])[k].get("t_max", 0.20) if k < len(det.get("geom_metrics", {}).get("station_metrics", [])) else 0.20,
                        }
                        for k, s in enumerate(stations)
                    ],
                }

                action = self.llm_agent.decide_action(state_dict)
                self.intervention_history.append({"step": step, "action": action})

                # 更新元参数
                scale_L = action.get("length_scale_factor", 1.0)
                self.base_L = float(np.clip(self.base_L * scale_L, self.L_min, self.L_max))
                self.L = self.base_L
                self.prob_perturb = float(action.get("prob_perturb", min(20.0 / d, 1.0)))
                self.temperature = float(action.get("exploration_weight", 1.0))

                # 宏观几何形变偏移量计算
                strat = action.get("directional_strategy", {})
                delta_dir = convert_directional_strategy_to_blade_delta(strat)

                # ==============================================================
                # 激进偏移量注入 (Aggressive Offset Injection)
                # ==============================================================
                cand_to_eval = None
                chosen_delta = None
                for s_fac in [1.0, 0.5, 0.25]:
                    trial_x = np.clip(self.best_x + s_fac * delta_dir, self.lower_bounds, self.upper_bounds)
                    is_fatal, _, metrics = self.evaluator.validate_geometry(trial_x)
                    is_feasible, total_penalty, _ = self.evaluator.calculate_penalties(metrics)
                    if not is_fatal and is_feasible and total_penalty == 0.0:
                        cand_to_eval = trial_x
                        chosen_delta = s_fac * delta_dir
                        break

                if cand_to_eval is not None and chosen_delta is not None:
                    # 设定方向引导向量 (在 unit 归一化空间)
                    u_best = self.to_unit(self.best_x)
                    u_cand = self.to_unit(cand_to_eval)
                    diff_u = u_cand - u_best
                    norm_u = np.linalg.norm(diff_u)
                    if norm_u > 1e-4:
                        self.guide_direction_unit = diff_u / norm_u
                        self.guide_direction_steps_left = 15

                    # 作为高优先级样本直接进行一次真实评估
                    cand_score = self.evaluate(cand_to_eval)
                    X_train.append(self.to_unit(cand_to_eval))
                    Y_train.append(cand_score)

                    improved = (cand_score > score_before + 1e-3)
                    self._adjust_trust_region(improved)

                    if self.eval_count >= self.max_evals:
                        break

            # 正常 TuRBO 贝叶斯候选采样阶段
            X_arr = np.array(X_train, dtype=np.float64)
            Y_arr = np.array(Y_train, dtype=np.float64)

            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr) + 1e-8
            Y_norm = (Y_arr - y_mean) / y_std

            x_best_u = self.to_unit(self.best_x)
            lb_tr = np.clip(x_best_u - self.L / 2.0, 0.0, 1.0)
            ub_tr = np.clip(x_best_u + self.L / 2.0, 0.0, 1.0)

            n_cand = self.n_candidates
            cand_raw = self.rng.uniform(lb_tr, ub_tr, size=(n_cand, d))

            # 方向性引导漂移倾斜
            if self.guide_direction_steps_left > 0 and self.guide_direction_unit is not None:
                drift = 0.20 * self.L * self.guide_direction_unit
                cand_raw = np.clip(cand_raw + drift, lb_tr, ub_tr)
                self.guide_direction_steps_left -= 1

            mask = self.rng.rand(n_cand, d) < self.prob_perturb
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

                K_diag = np.ones(n_cand)
                v = cho_solve(c_factor, K_cand_train.T)
                var_cand = np.maximum(1e-6, K_diag - np.sum(K_cand_train.T * v, axis=0))
                sigma_cand = np.sqrt(var_cand)

                # 结合 LLM 自适应探索温度
                effective_temp = self.temperature
                ts_scores = mu_cand + (sigma_cand * effective_temp) * self.rng.randn(n_cand)
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
            "method": "LLM-Meta-TuRBO",
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
            "intervention_history": self.intervention_history,
        }
