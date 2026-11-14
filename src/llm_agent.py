"""LLM-Meta-Blade 元优化智能体模块：
专为 NREL Phase VI 3D 风力机叶片气动优化设计，调用本地 SGLang 部署的大语言模型
(如 Qwen/Qwen3.8-27B-FP8)，结合空气动力学物理边界与高维信赖域状态，
为 114 维 TuRBO-1 提供自适应信赖域元控制与定向形变引导。

严格遵守科研诚信与零静默降级准则：
严禁在 LLM 异常或离线时静默回退人工硬编码启发式策略，且严禁在提示词中泄漏任何参考真值/最优目标提示。
"""

import json
import re
from typing import Any, Dict, List
import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


LLM_BLADE_SYSTEM_PROMPT = """You are a senior wind turbine aerodynamicist and Bayesian Optimization meta-controller.
Your goal is to maximize the aerodynamic rotor power output (Power in kW and power coefficient Cp) of a 3D horizontal-axis wind turbine blade modeled on the NREL Phase VI (Sequence H) configuration:
- Rotor: R = 5.029 m (D = 10.058 m), B = 2 upwind blades, aerodynamic S809 lifting span r in [1.2575 m, 5.0290 m] (27 spanwise stations, 0.30c pitch axis).
- Flow & Operating Point: Uniform air inflow V_inf = 7.0 m/s (rho = 1.225 kg/m^3), rotor speed = 72.0 rpm (omega = 7.5398 rad/s, TSR = 5.42), reference blade pitch = 4.815 deg (3.0 deg tip pitch).

Aerodynamic & Structural Domain Insights:
1. Spanwise Energy Capture & Load Distribution:
   - Inboard/root region (Stations 1-2, r ~ 1.26 m to 1.98 m): Lower local rotational speed and steep inflow angle phi; high baseline twist (~20 deg down to ~7.7 deg) and thick baseline S809 section (~21.1% t/c). Keep structural thickness >= 10-12% and avoid excessive root drag or stall.
   - Mid-span region (Stations 3-4, r ~ 2.71 m to 3.58 m): Primary torque-producing annulus. Performance depends strongly on sectional lift-to-drag ratio (CL/CD), matched angle of attack via twist tuning, smooth camber, and controlled chord taper.
   - Outboard/tip region (Stations 5-6, r ~ 4.30 m to 5.03 m): Highest dynamic pressure (tip speed ~37.9 m/s) and dominant induced-drag/tip-loss sensitivity. Moderate thinning (above structural limits ~7-8%), moderate camber, smooth tip wash-out, and controlled chord prevent tip stall and thrust overload.
2. Geometric & Structural Constraints (Strictly Enforced):
   - Airfoil relative thickness limits taper smoothly from root [12%, 28%] to tip [7%, 24%]; closed trailing edge TE=0; smooth non-wavy upper surface; monotonic non-increasing chord and twist from root to tip.
   - Rotor axial thrust must strictly respect the structural bound: T <= 1.15 * T_baseline. Any violation marks the design infeasible.

Decision Mode:
- Decide trust-region Bayesian optimization hyperparameters adapted to D=114:
  * length_scale_factor in [0.70, 1.50]
  * prob_perturb in [0.12, 0.32] (note baseline D=114 coordinate rate is ~0.175)
  * exploration_weight in [0.75, 1.40]
- Optionally provide 'directional_strategy' specifying physically smooth macro shifts across root/mid/tip zones, or set 'directional_strategy' to null when purely local GP exploration is preferable.
- When near or exceeding the thrust margin, avoid aggressive positive chord/camber increases at outboard stations that trigger thrust overload; prefer efficiency (CL/CD) via mild twist matching, camber refinement, or drag-reducing chord/thickness balance.

Output ONLY a valid JSON object matching this exact schema:
{
  "thought": "1-2 sentence aerodynamic/BO rationale (under 45 words).",
  "length_scale_factor": float in [0.70, 1.50],
  "prob_perturb": float in [0.12, 0.32],
  "exploration_weight": float in [0.75, 1.40],
  "directional_strategy": {
    "camber_shift_root": float in [-0.04, 0.06],
    "camber_shift_mid": float in [-0.04, 0.06],
    "camber_shift_tip": float in [-0.04, 0.06],
    "thickness_shift_root": float in [-0.04, 0.03],
    "thickness_shift_mid": float in [-0.04, 0.03],
    "thickness_shift_tip": float in [-0.04, 0.03],
    "twist_shift_root": float in [-4.0, 2.0],
    "twist_shift_mid": float in [-3.0, 2.0],
    "twist_shift_tip": float in [-2.5, 2.5],
    "chord_shift_root": float in [-0.12, 0.12],
    "chord_shift_mid": float in [-0.10, 0.10],
    "chord_shift_tip": float in [-0.08, 0.08]
  }
}
"""


class LLMMetaBladeAgent:
    """针对 3D 风力机叶片宏观气动先验决策的 LLM 元控制器智能体."""

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
        """根据当前 NREL Phase VI 优化阶段与气动特征构建提示词."""
        step = state["step"]
        best_power = state.get("best_power_kW", 0.0)
        best_cp = state.get("best_Cp", 0.0)
        stag = state.get("stagnation_counter", 0)
        max_thrust = state.get("max_allowable_thrust_kN", 0.0)

        if step <= 45:
            phase_str = (
                "EARLY EXPLORATION: NREL Phase VI baseline and initial samples evaluated. "
                "Adjust trust-region scale/perturbation and suggest physically consistent macro shifts if helpful."
            )
        elif step <= 105:
            phase_str = (
                f"MID REFINEMENT: Current feasible optimum reaches Power={best_power:.3f} kW (Cp={best_cp:.4f}). "
                "Refine trust region and guide smooth spanwise twist/profile adjustments within thrust safety limits."
            )
        else:
            phase_str = (
                f"LATE EXPLOITATION: Power={best_power:.3f} kW (Cp={best_cp:.4f}). "
                "Focus on local trust-region refinement (length_scale_factor <= 1.0) for tight convergence."
            )

        st_info = state.get("station_metrics", [])
        st_desc = ""
        for k, s in enumerate(st_info):
            st_desc += (
                f"  * Station {k+1} (r={s.get('radius', 0):.2f}m): "
                f"chord={s.get('chord', 0):.3f}m, twist={s.get('twist', 0):.2f}deg, "
                f"t/c={s.get('max_t_rel', 0)*100:.1f}%\n"
            )

        prompt = f"""=== Current NREL Phase VI 3D Wind Turbine Blade Optimization State ===
- Phase: {phase_str}
- Operating Point: Wind Speed V_inf = {state.get('wind_speed', 7.0):.2f} m/s | Rotor Speed = {state.get('rpm', 72.0):.1f} rpm (omega = {state.get('omega', 7.5398):.4f} rad/s) | Medium: Air (rho=1.225 kg/m^3)
- Progress: Step {step} / {state['max_evals']} (Remaining: {state['budget_left']})
- Baseline Power: {state.get('init_power_kW', 0.0):.3f} kW | Current Best Feasible Power: {best_power:.3f} kW (Cp = {best_cp:.4f})
- Aerodynamic Loads: Torque = {state.get('best_torque_kNm', 0.0):.4f} kN*m, Thrust = {state.get('best_thrust_kN', 0.0):.4f} kN (Limit <= {max_thrust:.4f} kN), Ct = {state.get('best_Ct', 0.0):.4f}
- Control Stations State:
{st_desc}- TuRBO Status: Effective TR Length L = {state['tr_length']:.4f}, Stagnation = {stag}, Successes = {state.get('success_counter', 0)}/{state.get('success_tolerance', 3)}

Concisely decide meta-parameters and optional directional strategy. Output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """向本地部署的真实 LLM 请求下一阶段超参策略与宏观气动方向 (严禁静默降级)."""
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
                temperature=0.6,
                top_p=0.90,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_text = response.choices[0].message.content or ""
            clean = re.sub(r"```json\s*", "", raw_text)
            clean = re.sub(r"```\s*", "", clean).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.70, 1.50))
                prob_p = float(np.clip(data.get("prob_perturb", 20.0 / 114.0), 0.12, 0.32))
                exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.75, 1.40))
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
                        "camber_shift_root": float(np.clip(strat.get("camber_shift_root", 0.0), -0.04, 0.06)),
                        "camber_shift_mid": float(np.clip(strat.get("camber_shift_mid", 0.0), -0.04, 0.06)),
                        "camber_shift_tip": float(np.clip(strat.get("camber_shift_tip", 0.0), -0.04, 0.06)),
                        "thickness_shift_root": float(np.clip(strat.get("thickness_shift_root", 0.0), -0.04, 0.03)),
                        "thickness_shift_mid": float(np.clip(strat.get("thickness_shift_mid", 0.0), -0.04, 0.03)),
                        "thickness_shift_tip": float(np.clip(strat.get("thickness_shift_tip", 0.0), -0.04, 0.03)),
                        "twist_shift_root": float(np.clip(strat.get("twist_shift_root", 0.0), -4.0, 2.0)),
                        "twist_shift_mid": float(np.clip(strat.get("twist_shift_mid", 0.0), -3.0, 2.0)),
                        "twist_shift_tip": float(np.clip(strat.get("twist_shift_tip", 0.0), -2.5, 2.5)),
                        "chord_shift_root": float(np.clip(strat.get("chord_shift_root", 0.0), -0.12, 0.12)),
                        "chord_shift_mid": float(np.clip(strat.get("chord_shift_mid", 0.0), -0.10, 0.10)),
                        "chord_shift_tip": float(np.clip(strat.get("chord_shift_tip", 0.0), -0.08, 0.08)),
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
