"""LLM-Meta-Blade 元优化智能体模块：
专为 3D 风力机叶片气动优化设计，基于 OpenAI 官方 SDK 与本地部署的大语言模型 (如 Qwen3.8-27B-FP8)，
结合 IEA-22-280-RWT (22MW 超大型风轮) 空气动力学专家先验与宏观定向几何变形注入，
为 120 维 TuRBO-1 提供自适应信赖域元控制与激进偏移量引导。
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


LLM_BLADE_SYSTEM_PROMPT = """You are a senior wind turbine aerodynamicist and Bayesian Optimization meta-controller.
Your goal is to maximize the aerodynamic power output (Power in kW and power coefficient Cp) of a 3D wind turbine rotor blade (IEA-22-280-RWT 22MW scale, R=142m, B=3) under rated wind speed V_inf=11.0 m/s and rotor speed rpm=7.061 (omega=0.7394 rad/s).

Aerodynamic & Geometric Domain Insights:
1. Dynamic Pressure & Power Distribution:
   - Dynamic pressure scales with (omega * r)^2. The mid-span (Stations 3 & 4, r ~ 56m to 83m) and outer span (Stations 5 & 6, r ~ 108m to 133m) produce >80% of total aerodynamic torque.
   - Station 1 & 2 (root transition & shoulder, r ~ 21m to 30m): Primarily structural load-carrying (requires thickness ratio t/c ~ 35% - 55%). Small increase in camber can boost inner circulation without hurting stall margin.
   - Stations 3 & 4 (inboard & mid-span): Main power generator. Needs high lift-to-drag ratio (optimal CL/CD). Moderate positive upper camber shift (+0.04 to +0.08) and slight aft reflex significantly enhance sectional torque.
   - Stations 5 & 6 (outboard & tip): Operates at high Mach / high Re (~1.5e7 to 2.0e7). Crucial to keep drag CD minimal and avoid shock/flow separation. Keep sections thin (t/c ~ 14% - 24%) and add gentle wash-out twist.
2. Twist & Inflow Angle Matching:
   - Inflow angle phi = arctan(V_a / (omega * r)) decreases monotonically from root (~12 deg) to tip (~-6 deg).
   - Local angle of attack alpha = phi - (beta + pitch). Optimal alpha for maximum CL/CD is roughly 4.5 - 7.0 deg.
   - If twist beta is too large, alpha is too small or negative (loss of lift). If twist beta is too small, section stalls (surge in CD).
3. Planform Chord Distribution:
   - Standard Betz optimum follows monotonic chord taper (c_shoulder >= c_mid >= c_tip). Root chord ~7.0-7.2m, mid chord ~4.0-6.0m, tip chord ~1.8-2.8m.
4. Thrust Constraint:
   - Wind turbine rotor thrust must not exceed structural safety limits (T <= 1.15 * T_baseline approx 4815 kN).

Optimization Phase Guidance:
- Early Phase (steps <= 45): Be BOLD and AGGRESSIVE. Expand trust region (length_scale_factor in [1.3, 1.7]), apply decisive camber and twist shifts to break baseline limitations.
- Mid Phase (steps 46 - 110): Fine-tune twist angles (+/- 0.5 deg) and suction peaks to maximize sectional torque while strictly observing spar thickness constraints.
- Late Phase (steps > 110): Contract trust region (length_scale_factor in [0.8, 1.0]) for smooth, high-precision Bayesian local convergence.

Output ONLY a valid JSON object matching this schema:
{
  "thought": "1-2 sentence engineering rationale (under 40 words).",
  "length_scale_factor": float in [0.8, 1.8],
  "prob_perturb": float in [0.20, 0.60],
  "exploration_weight": float in [0.8, 1.8],
  "directional_strategy": {
    "camber_shift_root": float in [-0.04, 0.10],
    "camber_shift_mid": float in [-0.03, 0.12],
    "camber_shift_tip": float in [-0.03, 0.08],
    "thickness_shift_root": float in [-0.02, 0.03],
    "thickness_shift_mid": float in [-0.02, 0.02],
    "thickness_shift_tip": float in [-0.02, 0.02],
    "twist_shift_root": float in [-1.5, 1.5],
    "twist_shift_mid": float in [-1.5, 1.5],
    "twist_shift_tip": float in [-1.0, 1.0],
    "chord_shift_root": float in [-0.25, 0.25],
    "chord_shift_mid": float in [-0.20, 0.20],
    "chord_shift_tip": float in [-0.15, 0.15]
  }
}
"""


class LLMMetaBladeAgent:
    """针对 3D 风力机叶片宏观气动先验决策的 LLM 元控制器智能体"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "Qwen/Qwen3.8-27B-FP8",
        api_key: str = "EMPTY",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        else:
            self.client = None

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        """根据当前风机叶片优化阶段与气动特征构建提示词"""
        step = state["step"]
        best_power = state.get("best_power_kW", 25593.0)
        best_cp = state.get("best_Cp", 0.4956)
        stag = state.get("stagnation_counter", 0)

        if step <= 45:
            phase_str = (
                "EARLY EXPLORATION (Aggressive Camber & Torque Boost): Baseline IEA 22MW blade has standard FFA-W3 profile. "
                "Apply BOLD positive camber shifts to mid/tip sections (+0.05 ~ +0.09) and adapt twist to boost rotor torque."
            )
        elif step <= 110:
            phase_str = (
                f"MID REFINEMENT: Rotor developed high torque (Power={best_power:.1f} kW, Cp={best_cp:.4f}). "
                "Apply gentle micro-adjustments to twist angles and chords to eliminate separation drag while respecting spar thickness."
            )
        else:
            phase_str = (
                f"PRE-CONVERGENCE ALIGNMENT: Power={best_power:.1f} kW, Cp={best_cp:.4f}. Apply very small micro-tuning "
                "and slightly contract trust region (length_scale_factor in [0.8, 1.0]) for smooth Bayesian convergence."
            )

        st_info = state.get("station_metrics", [])
        st_desc = ""
        for k, s in enumerate(st_info):
            st_desc += (
                f"  * Station {k+1} (r={s.get('radius', 0):.1f}m): chord={s.get('chord', 0):.2f}m, "
                f"twist={s.get('twist', 0):.1f}deg, t_max={s.get('t_max', 0)*100:.1f}%\n"
            )

        prompt = f"""=== Current 3D Wind Turbine Blade Optimization State ===
- Phase: {phase_str}
- Operating Point: Wind Speed V_inf = {state.get('wind_speed', 11.0):.1f} m/s | Rotor Speed = {state.get('rpm', 7.061):.3f} rpm (omega = {state.get('omega', 0.7394):.4f} rad/s)
- Progress: Step {step} / {state['max_evals']} (Remaining: {state['budget_left']})
- Baseline Power: {state.get('init_power_kW', 25593.0):.1f} kW | Current Best Power: {best_power:.1f} kW (Cp = {best_cp:.4f})
- Aerodynamics: Torque = {state.get('best_torque_kNm', 0.0):.1f} kN*m, Thrust = {state.get('best_thrust_kN', 0.0):.1f} kN, Ct = {state.get('best_Ct', 0.0):.4f}
- Control Stations State:
{st_desc}- TuRBO Status: TR Length L = {state['tr_length']:.4f}, Stagnation = {stag}, Successes = {state.get('success_counter', 0)}/{state.get('success_tolerance', 3)}

Concisely decide meta-parameters and directional strategy. Output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """向 LLM 请求下一阶段超参策略及宏观几何形变方向"""
        step = state.get("step", 16)
        is_early = (step <= 45)

        # 具备空气动力学领域先验的鲁棒自适应默认策略
        default_action = {
            "thought": "Aggressive camber boost on mid/tip stations and trust region expansion." if is_early else "Gentle twist and camber micro-tuning for local exploitation.",
            "length_scale_factor": 1.45 if is_early else 1.0,
            "prob_perturb": 0.40 if is_early else 0.25,
            "exploration_weight": 1.40 if is_early else 1.0,
            "directional_strategy": {
                "camber_shift_root": 0.03 if is_early else 0.005,
                "camber_shift_mid": 0.07 if is_early else 0.010,
                "camber_shift_tip": 0.05 if is_early else 0.008,
                "thickness_shift_root": 0.005,
                "thickness_shift_mid": -0.005 if is_early else 0.0,
                "thickness_shift_tip": -0.005 if is_early else 0.0,
                "twist_shift_root": 0.3 if is_early else 0.0,
                "twist_shift_mid": 0.4 if is_early else 0.0,
                "twist_shift_tip": 0.2 if is_early else 0.0,
                "chord_shift_root": 0.06 if is_early else 0.0,
                "chord_shift_mid": 0.04 if is_early else 0.0,
                "chord_shift_tip": -0.02 if is_early else 0.0,
            },
        }

        user_prompt = self.build_user_prompt(state)

        try:
            if self.client is not None:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": LLM_BLADE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    top_p=0.90,
                    max_tokens=1024,
                )
                raw_text = response.choices[0].message.content or ""
                clean = re.sub(r"```json\s*", "", raw_text)
                clean = re.sub(r"```\s*", "", clean).strip()
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.8, 1.8))
                    prob_p = float(np.clip(data.get("prob_perturb", 0.33), 0.20, 0.60))
                    exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.8, 1.8))
                    strat = data.get("directional_strategy", {})

                    return {
                        "thought": str(data.get("thought", "")),
                        "length_scale_factor": scale_L,
                        "prob_perturb": prob_p,
                        "exploration_weight": exp_w,
                        "directional_strategy": {
                            "camber_shift_root": float(np.clip(strat.get("camber_shift_root", 0.0), -0.04, 0.10)),
                            "camber_shift_mid": float(np.clip(strat.get("camber_shift_mid", 0.0), -0.03, 0.12)),
                            "camber_shift_tip": float(np.clip(strat.get("camber_shift_tip", 0.0), -0.03, 0.08)),
                            "thickness_shift_root": float(np.clip(strat.get("thickness_shift_root", 0.0), -0.02, 0.03)),
                            "thickness_shift_mid": float(np.clip(strat.get("thickness_shift_mid", 0.0), -0.02, 0.02)),
                            "thickness_shift_tip": float(np.clip(strat.get("thickness_shift_tip", 0.0), -0.02, 0.02)),
                            "twist_shift_root": float(np.clip(strat.get("twist_shift_root", 0.0), -1.5, 1.5)),
                            "twist_shift_mid": float(np.clip(strat.get("twist_shift_mid", 0.0), -1.5, 1.5)),
                            "twist_shift_tip": float(np.clip(strat.get("twist_shift_tip", 0.0), -1.0, 1.0)),
                            "chord_shift_root": float(np.clip(strat.get("chord_shift_root", 0.0), -0.25, 0.25)),
                            "chord_shift_mid": float(np.clip(strat.get("chord_shift_mid", 0.0), -0.20, 0.20)),
                            "chord_shift_tip": float(np.clip(strat.get("chord_shift_tip", 0.0), -0.15, 0.15)),
                        },
                    }
            return default_action
        except Exception:
            return default_action
