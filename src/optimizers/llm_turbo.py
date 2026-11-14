"""LLM-Meta-TuRBO 3D 风力机叶片优化器：
基于本地大语言模型宏观定向元控制的高维信赖域贝叶斯优化器 (TuRBO-1)。
修复关键工程与公平性问题：
1. 修复 station_metrics 中错误读取 't_max' 而非真实键值 'max_t_rel' 的 bug;
2. 禁止永久覆写 self.base_L，使 LLM 的 scale_L 作为临时信赖域放缩因子叠加于 TuRBO 原生状态机之上;
3. 修正 CST 弯度/厚度形变核函数为平滑对称正弦钟形包络，消除下表面后缘局部非单调凸包;
4. 仅在约 30% 候选切片内注入引导方向采样 (其余 70% 保留纯 TuRBO 全维探索);
5. 对 GP 训练目标执行离群惩罚下界截断 (np.maximum(Y_arr, -1.0))，防止不可行离群点掩盖可行域梯度。
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.blade_geometry import unpack_blade_params
from src.evaluator import WindTurbineBladeEvaluator
from src.llm_agent import LLMMetaBladeAgent
from src.optimizers.base import BaseBladeOptimizer
from src.optimizers.turbo import matern52_kernel


def convert_directional_strategy_to_blade_delta(
    strat: Optional[Dict[str, Any]],
    dim: int = 114,
) -> np.ndarray:
    """将 LLM 输出的宏观气动方向策略转换为叶片参数向量偏移量 delta."""
    delta = np.zeros(dim, dtype=np.float64)
    if not strat or not isinstance(strat, dict):
        return delta

    station_configs = [
        (0.7, 0.8, 0.7, 1.0, "camber_shift_root", "thickness_shift_root", "chord_shift_root", "twist_shift_root"),
        (1.0, 1.0, 1.0, 0.9, "camber_shift_root", "thickness_shift_root", "chord_shift_root", "twist_shift_root"),
        (0.9, 1.0, 0.9, 0.9, "camber_shift_mid",  "thickness_shift_mid",  "chord_shift_mid",  "twist_shift_mid"),
        (1.0, 1.0, 1.0, 1.0, "camber_shift_mid",  "thickness_shift_mid",  "chord_shift_mid",  "twist_shift_mid"),
        (1.0, 1.0, 1.0, 1.0, "camber_shift_tip",  "thickness_shift_tip",  "chord_shift_tip",  "twist_shift_tip"),
        (0.7, 0.8, 0.6, 1.0, "camber_shift_tip",  "thickness_shift_tip",  "chord_shift_tip",  "twist_shift_tip"),
    ]

    # 平滑对称 Bernstein 包络，前缘与后缘端光滑趋零，避免尾部鼓包/波纹
    bell = np.sin(np.pi * (np.arange(8) + 0.5) / 8.0)

    if dim == 114:
        for k in range(6):
            c_sc, t_sc, _, _, c_key, t_key, _, _ = station_configs[k]
            s_camber = float(strat.get(c_key, 0.0)) * c_sc
            s_thick = float(strat.get(t_key, 0.0)) * t_sc

            idx = k * 17
            delta[idx : idx + 8] = (s_camber + 0.5 * s_thick) * bell
            delta[idx + 8 : idx + 16] = (s_camber - 0.5 * s_thick) * bell
            delta[idx + 16] = 0.25 * s_thick

        c_root = float(strat.get("chord_shift_root", 0.0))
        c_mid = float(strat.get("chord_shift_mid", 0.0))
        c_tip = float(strat.get("chord_shift_tip", 0.0))
        delta[102 : 104] = c_root * 0.85
        delta[104 : 106] = c_mid * 1.0
        delta[106 : 108] = c_tip * 0.70

        tw_root = float(strat.get("twist_shift_root", 0.0))
        tw_mid = float(strat.get("twist_shift_mid", 0.0))
        tw_tip = float(strat.get("twist_shift_tip", 0.0))
        delta[108 : 110] = tw_root * 0.90
        delta[110 : 112] = tw_mid * 1.0
        delta[112 : 114] = tw_tip * 1.0
    else:
        for k in range(6):
            c_sc, t_sc, ch_sc, tw_sc, c_key, t_key, ch_key, tw_key = station_configs[k]
            s_camber = float(strat.get(c_key, 0.0)) * c_sc
            s_thick = float(strat.get(t_key, 0.0)) * t_sc
            s_chord = float(strat.get(ch_key, 0.0)) * ch_sc
            s_twist = float(strat.get(tw_key, 0.0)) * tw_sc

            idx = k * 20
            delta[idx : idx + 8] = (s_camber + 0.5 * s_thick) * bell
            delta[idx + 8 : idx + 16] = (s_camber - 0.5 * s_thick) * bell
            delta[idx + 16] = 0.25 * s_thick
            delta[idx + 17] = 0.0
            delta[idx + 18] = s_chord
            delta[idx + 19] = s_twist

    return delta


class LLMMetaTuRBOOptimizer(BaseBladeOptimizer):
    """基于大语言模型宏观定向元控制的高维信赖域贝叶斯优化器 (LLM-Meta-TuRBO)."""

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
        llm_agent: Optional[LLMMetaBladeAgent] = None,
        milestone_steps: Tuple[int, ...] = (16, 35, 65, 105, 140),
        stag_threshold: int = 14,
        min_intervention_interval: int = 20,
    ):
        super().__init__(evaluator, x0, bounds, max_evals, seed)
        self.n_init = int(n_init)
        self.L_init = float(length_init)
        self.L_min = float(length_min)
        self.L_max = float(length_max)
        self.base_L = float(length_init)
        self.scale_L_mult = 1.0
        self.scale_L_steps_left = 0
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

    def _adjust_trust_region(self, improved: bool) -> None:
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
            self.scale_L_mult = 1.0
            self.scale_L_steps_left = 0

        if self.scale_L_steps_left > 0:
            self.scale_L_steps_left -= 1
            if self.scale_L_steps_left == 0:
                self.scale_L_mult = 1.0

        self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))

    def run(self) -> Dict[str, Any]:
        """运行 LLM-Meta-TuRBO 优化流程."""
        t0 = time.time()
        d = self.dim

        X_train: List[np.ndarray] = []
        Y_train: List[float] = []

        # 1. 评估基准叶片构型 x0
        score0 = self.evaluate_initial()
        u0 = self.to_unit(self.x0)
        X_train.append(u0)
        Y_train.append(score0)

        # 2. 在 [0, 1]^D 归一化空间进行初始微扰散点 (n_init = 15, 与 TuRBO/MALIBO 严格对齐)
        for _ in range(1, self.n_init):
            if self.eval_count >= self.max_evals:
                break
            u_cand = np.clip(u0 + self.rng.uniform(-0.04, 0.04, size=d), 0.0, 1.0)
            x_cand = self.from_unit(u_cand)
            score = self.evaluate(x_cand)
            X_train.append(u_cand)
            Y_train.append(score)

        # 3. 主循环
        while self.eval_count < self.max_evals:
            step = self.eval_count + 1
            score_before = self.best_score

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
                if hasattr(self.evaluator, "param_model") and hasattr(self.evaluator.param_model, "get_station_info"):
                    stations = self.evaluator.param_model.get_station_info(self.best_x)
                else:
                    stations = unpack_blade_params(self.best_x)

                st_metrics_list = det.get("geom_metrics", {}).get("station_metrics", [])
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
                    "max_allowable_thrust_kN": self.evaluator.max_allowable_thrust_kN,
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
                            "max_t_rel": st_metrics_list[k].get("max_t_rel", 0.21) if k < len(st_metrics_list) else 0.21,
                        }
                        for k, s in enumerate(stations)
                    ],
                }

                action = self.llm_agent.decide_action(state_dict)
                inter_item = {"step": step, "action": action}
                if hasattr(self.llm_agent, "interaction_log") and len(self.llm_agent.interaction_log) > 0:
                    last_log = self.llm_agent.interaction_log[-1]
                    inter_item["user_prompt"] = last_log.get("user_prompt")
                    inter_item["raw_response"] = last_log.get("raw_response")
                self.intervention_history.append(inter_item)

                # 临时叠加信赖域尺度修正而不破坏 TuRBO 原生 base_L
                scale_L = float(action.get("length_scale_factor", 1.0))
                self.scale_L_mult = scale_L
                self.scale_L_steps_left = 12
                self.L = float(np.clip(self.base_L * self.scale_L_mult, self.L_min, self.L_max))
                self.prob_perturb = float(action.get("prob_perturb", min(20.0 / d, 1.0)))
                self.temperature = float(action.get("exploration_weight", 1.0))

                strat = action.get("directional_strategy")
                delta_dir = convert_directional_strategy_to_blade_delta(strat, dim=d)

                cand_to_eval = None
                if np.linalg.norm(delta_dir) > 1e-8:
                    for s_fac in (1.0, 0.5, 0.25):
                        trial_x = np.clip(self.best_x + s_fac * delta_dir, self.lower_bounds, self.upper_bounds)
                        is_fatal, _, metrics = self.evaluator.validate_geometry(trial_x)
                        is_geom_ok, total_penalty, _ = self.evaluator.calculate_penalties(metrics)
                        if not is_fatal and is_geom_ok and total_penalty == 0.0:
                            cand_to_eval = trial_x
                            break

                if cand_to_eval is not None:
                    u_best = self.to_unit(self.best_x)
                    u_cand = self.to_unit(cand_to_eval)
                    diff_u = u_cand - u_best
                    norm_u = float(np.linalg.norm(diff_u))
                    if norm_u > 1e-4:
                        self.guide_direction_unit = diff_u / norm_u
                        self.guide_direction_steps_left = 15

                    cand_score = self.evaluate(cand_to_eval)
                    X_train.append(u_cand)
                    Y_train.append(cand_score)

                    improved = (cand_score > score_before + 1e-3)
                    self._adjust_trust_region(improved)

                    if self.eval_count >= self.max_evals:
                        break
                    continue

            # 正常 TuRBO 贝叶斯候选采样阶段
            X_arr = np.array(X_train, dtype=np.float64)
            Y_arr = np.array(Y_train, dtype=np.float64)

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

            # 在 30% 候选切片内注入有界 LLM 方向性倾斜采样，70% 保留纯信赖域随机探索
            if (
                self.guide_direction_unit is not None
                and self.guide_direction_steps_left > 0
                and step <= 165
            ):
                n_dir = min(1500, n_cand // 3)
                step_scale = self.rng.uniform(0.05 * self.L, 0.65 * self.L, size=(n_dir, 1))
                dir_cands = (
                    x_best_u
                    + step_scale * self.guide_direction_unit
                    + self.rng.normal(0.0, 0.03 * self.L, size=(n_dir, d))
                )
                candidates[:n_dir] = np.clip(dir_cands, lb_tr, ub_tr)
                self.guide_direction_steps_left -= 1

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

                ts_scores = mu_cand + (sigma_cand * self.temperature) * self.rng.randn(n_cand)
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
            "intervention_history": self.intervention_history,
        }
