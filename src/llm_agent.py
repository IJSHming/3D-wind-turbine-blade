"""LLM-Meta-Blade 元优化智能体模块：
专为 3D 风力机叶片气动优化设计，基于 OpenAI 官方 SDK 与本地部署的大语言模型 (如 Qwen3.8-27B-FP8)，
结合 ModelTest 10 MW (10MW 超大型风轮) 空气动力学物理边界与高维信赖域状态，
为 120 维 TuRBO-1 提供自适应信赖域元控制与定向形变引导。

严格遵守科研诚信与零静默降级准则：
严禁在 LLM 异常或离线时静默回退人工硬编码字典，确保所有元决策源自真实模型推理。
"""

import json
import re
from typing import Any, Dict, Optional
import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


LLM_BLADE_SYSTEM_PROMPT = """You are a senior marine tidal turbine hydrodynamicist and Bayesian Optimization meta-controller.
Your goal is to maximize the hydrodynamic power output (Power in kW and power coefficient Cp) of a 3D tidal turbine rotor blade (1.5m blade scale, R=1.50m, B=3, operating in water with density rho=1025.0 kg/m^3) under rated current speed V_inf=1.5 m/s and rotor speed rpm=40.0 (omega=4.189 rad/s, TSR lambda=4.19).

Hydrodynamic & Geometric Domain Insights:
1. Dynamic Pressure & Power Distribution in Water:
   - Water density is ~836x higher than air (1025.0 kg/m^3 vs 1.225 kg/m^3), generating high hydrodynamic forces at low velocities (V_inf = 1.5 m/s, V_tip = 6.28 m/s).
   - Mid-span (Stations 3 & 4, r ~ 0.7m to 1.0m) and outer span (Stations 5 & 6, r ~ 1.25m to 1.50m) produce >80% of total hydrodynamic torque.
   - Stations 1 & 2 (root & inboard, r ~ 0.25m to 0.45m): In unoptimized baseline, chord is oversized (~1.11m), causing excessive form drag. Reducing root chord towards 0.7m substantially improves hydrodynamic efficiency without structural penalty.
   - Stations 3 & 4 (mid-span): Main power generator. Needs high lift-to-drag ratio (CL/CD). Maintain smooth linear taper and appropriate twist.
   - Stations 5 & 6 (outboard & tip): High relative velocity (~6.4 m/s). Avoid premature flow separation, maintain negative wash-out twist.
2. Inflow & Twist Matching:
   - Inflow angle phi decreases monotonically from root (~27 deg) to tip (~3 deg).
   - Optimal alpha for maximum CL/CD is roughly 2.0 - 5.0 deg in water.
3. Thrust Constraint:
   - Tidal turbine rotor thrust must not exceed structural safety limits (T <= 1.15 * T_baseline approx 9.69 kN / 9690 N).

Decision Mode:
- You MUST decide the Bayesian optimization meta-parameters: length_scale_factor in [0.8, 1.8], prob_perturb in [0.15, 0.60], exploration_weight in [0.8, 1.8].
- If you identify a promising macro hydrodynamic deformation direction, provide 'directional_strategy' object specifying shifts across zones.
- If the 132D multi-station trade-offs are complex and uncertain, set 'directional_strategy' to null to rely purely on trust-region Bayesian exploration.

Output ONLY a valid JSON object matching this schema:
{
  "thought": "1-2 sentence engineering rationale (under 40 words).",
  "length_scale_factor": float in [0.8, 1.8],
  "prob_perturb": float in [0.15, 0.60],
  "exploration_weight": float in [0.8, 1.8],
  "directional_strategy": {
    "camber_shift_root": float in [-0.04, 0.08],
    "camber_shift_mid": float in [-0.05, 0.08],
    "camber_shift_tip": float in [-0.05, 0.08],
    "thickness_shift_root": float in [-0.02, 0.03],
    "thickness_shift_mid": float in [-0.05, 0.03],
    "thickness_shift_tip": float in [-0.05, 0.03],
    "twist_shift_root": float in [-2.0, 2.0],
    "twist_shift_mid": float in [-2.0, 2.0],
    "twist_shift_tip": float in [-2.0, 2.0],
    "chord_shift_root": float in [-0.20, 0.10],
    "chord_shift_mid": float in [-0.15, 0.10],
    "chord_shift_tip": float in [-0.08, 0.08]
  }
}
Note: If not suggesting an offset, you may set "directional_strategy": null.
"""


class LLMMetaBladeAgent:
    """针对 3D 风力机叶片宏观气动先验决策的 LLM 元控制器智能体"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "Qwen/Qwen3.8-27B-FP8",
        api_key: str = "EMPTY",
        timeout: float = 240.0,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.interaction_log: List[Dict[str, Any]] = []

        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=1,
            )
        else:
            self.client = None

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        """根据当前风机叶片优化阶段与气动特征构建提示词"""
        step = state["step"]
        best_power = state.get("best_power_kW", 4.80)
        best_cp = state.get("best_Cp", 0.3925)
        stag = state.get("stagnation_counter", 0)

        if step <= 45:
            phase_str = (
                "EARLY EXPLORATION: Baseline 1.5m blade is evaluated. Initial random samples established prior. "
                "Explore promising trust region scales or provide macro planform/twist adjustments if confident."
            )
        elif step <= 110:
            phase_str = (
                f"MID REFINEMENT: Rotor power developed (Power={best_power:.2f} kW, Cp={best_cp:.4f}). "
                "Fine-tune trust region and twist/profile to eliminate local separation without violating spar constraints."
            )
        else:
            phase_str = (
                f"PRE-CONVERGENCE ALIGNMENT: Power={best_power:.2f} kW, Cp={best_cp:.4f}. Apply micro-tuning "
                "and slightly contract trust region (length_scale_factor in [0.8, 1.0]) for smooth Bayesian convergence."
            )

        st_info = state.get("station_metrics", [])
        st_desc = ""
        for k, s in enumerate(st_info):
            st_desc += (
                f"  * Station {k+1} (r={s.get('radius', 0):.2f}m): chord={s.get('chord', 0):.3f}m, "
                f"twist={s.get('twist', 0):.2f}deg, t_max={s.get('t_max', 0)*100:.1f}%\n"
            )

        prompt = f"""=== Current 3D Tidal Turbine Blade Optimization State ===
- Phase: {phase_str}
- Operating Point: Current Speed V_inf = {state.get('wind_speed', 1.5):.2f} m/s | Rotor Speed = {state.get('rpm', 40.0):.1f} rpm (omega = {state.get('omega', 4.1888):.4f} rad/s) | Medium: Water (rho=1025 kg/m^3)
- Progress: Step {step} / {state['max_evals']} (Remaining: {state['budget_left']})
- Baseline Power: {state.get('init_power_kW', 4.80):.2f} kW | Current Best Power: {best_power:.2f} kW (Cp = {best_cp:.4f})
- Hydrodynamics: Torque = {state.get('best_torque_kNm', 0.0):.3f} kN*m, Thrust = {state.get('best_thrust_kN', 0.0):.3f} kN, Ct = {state.get('best_Ct', 0.0):.4f}
- Control Stations State:
{st_desc}- TuRBO Status: TR Length L = {state['tr_length']:.4f}, Stagnation = {stag}, Successes = {state.get('success_counter', 0)}/{state.get('success_tolerance', 3)}

Concisely decide meta-parameters and optional directional strategy. Output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """向 LLM 请求下一阶段超参策略及宏观几何形变方向。
        严格科研诚信准则：严禁静默降级走人工预置启发式字典，必须依赖真实的外部模型推理。
        """
        if not HAS_OPENAI or self.client is None:
            raise RuntimeError(
                "[Research Integrity Violation] OpenAI client is not initialized! "
                "Silent degradation to a hand-crafted heuristic dictionary is strictly forbidden."
            )

        user_prompt = self.build_user_prompt(state)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_BLADE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                top_p=0.90,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_text = response.choices[0].message.content or ""
            clean = re.sub(r"```json\s*", "", raw_text)
            clean = re.sub(r"```\s*", "", clean).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.8, 1.8))
                prob_p = float(np.clip(data.get("prob_perturb", 0.33), 0.15, 0.60))
                exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.8, 1.8))
                strat = data.get("directional_strategy")

                action: Dict[str, Any] = {
                    "thought": str(data.get("thought", "")),
                    "length_scale_factor": scale_L,
                    "prob_perturb": prob_p,
                    "exploration_weight": exp_w,
                    "directional_strategy": None,
                }

                if isinstance(strat, dict):
                    action["directional_strategy"] = {
                        "camber_shift_root": float(np.clip(strat.get("camber_shift_root", 0.0), -0.04, 0.08)),
                        "camber_shift_mid": float(np.clip(strat.get("camber_shift_mid", 0.0), -0.05, 0.08)),
                        "camber_shift_tip": float(np.clip(strat.get("camber_shift_tip", 0.0), -0.05, 0.08)),
                        "thickness_shift_root": float(np.clip(strat.get("thickness_shift_root", 0.0), -0.02, 0.03)),
                        "thickness_shift_mid": float(np.clip(strat.get("thickness_shift_mid", 0.0), -0.05, 0.03)),
                        "thickness_shift_tip": float(np.clip(strat.get("thickness_shift_tip", 0.0), -0.05, 0.03)),
                        "twist_shift_root": float(np.clip(strat.get("twist_shift_root", 0.0), -1.0, 1.5)),
                        "twist_shift_mid": float(np.clip(strat.get("twist_shift_mid", 0.0), -1.0, 1.5)),
                        "twist_shift_tip": float(np.clip(strat.get("twist_shift_tip", 0.0), -1.0, 1.5)),
                        "chord_shift_root": float(np.clip(strat.get("chord_shift_root", 0.0), -0.20, 0.20)),
                        "chord_shift_mid": float(np.clip(strat.get("chord_shift_mid", 0.0), -0.25, 0.20)),
                        "chord_shift_tip": float(np.clip(strat.get("chord_shift_tip", 0.0), -0.25, 0.15)),
                    }
                self.interaction_log.append({
                    "step": state.get("step"),
                    "user_prompt": user_prompt,
                    "raw_response": raw_text,
                    "action": action,
                })
                return action

            raise ValueError(f"Failed to parse valid JSON from LLM response:\n{raw_text}")

        except Exception as e:
            raise RuntimeError(
                f"[LLM Fatal Error] Real LLM request to {self.base_url} failed with {type(e).__name__}: {e}. "
                "Silent degradation to a hand-crafted heuristic dictionary is strictly forbidden."
            ) from e
