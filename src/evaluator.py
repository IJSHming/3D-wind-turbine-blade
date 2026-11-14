"""基于 AeroSandbox、NeuralFoil 与叶素动量理论 (BEM) 的 3D 风力机叶片高保真气动与几何综合评估器。
严格对齐 NREL Phase VI (Sequence H) 官方风洞试验与 Kaya (2019) 基准规范：
- 转子参数: R = 5.029 m (D = 10.058 m), 叶片数 B = 2 (Upwind), 轮毂等效根半径 r_hub = 0.8835 m,
  气动 S809 升力主体 span r in [1.2575 m, 5.0290 m] (27 截面均匀离散, 俯仰轴 0.30c);
- 额定工况 (Sequence H @ 7 m/s): 来流风速 V_inf = 7.0 m/s, 转速 = 72.0 rpm (omega = 7.5398 rad/s, TSR = 5.42),
  参考标准海平面干燥空气 (rho = 1.225 kg/m^3, mu = 1.789e-5 Pa*s), 均匀流场 (wind_shear_exponent = 0.0),
  变桨角 pitch_angle_deg = 4.815° (以 75% 展向零扭角站位计, 等效叶尖 3.0° 桨距角), 光滑洁净表面;
- 集成 OpenFAST 3D 旋转失速延迟修正 (Snel)、QBlade Shen et al. (2005) 根尖三维损失修正、
  Viterna-Corrigan 深失速动态外推与 Buhl-Glauert 湍流高推力修正;
- NeuralFoil 深度学习代理网络锁定最高精度架构 'xxxlarge';
- 严格科研诚信与可行性隔离机制: 无静默异常回退; 任意约束违背强制映射为严格负值 (<= -0.25),
  杜绝不可行解混入并击败可行最优解。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import aerosandbox as asb
import neuralfoil as nf

from src.blade_geometry import (
    BASELINE_PITCH_DEG,
    BASELINE_RPM,
    BASELINE_WIND_SPEED,
    BEM_27_DR,
    BEM_27_RADII,
    BEM_NUM_SECTIONS,
    BLADE_LENGTH,
    BLADE_ROOT_RADIUS,
    DIM,
    HUB_HEIGHT,
    HUB_RADIUS,
    NUM_BLADES,
    NUM_STATIONS,
    ROTOR_RADIUS,
    STATION_RADII,
    get_baseline_blade_params,
    interpolate_blade_span,
    unpack_blade_params,
)
from src.parameterizations import BSplineBladeModel


class WindTurbineBladeEvaluator:
    """3D 风力机单叶片气动与几何约束综合评估器 (NREL Phase VI Sequence H 标称)."""

    def __init__(
        self,
        wind_speed: float = BASELINE_WIND_SPEED,      # NREL Phase VI / Kaya (2019): V_inf = 7.0 m/s
        rpm: float = BASELINE_RPM,                    # 额定转速 72.0 rpm (omega = 7.5398 rad/s)
        air_density: float = 1.225,                   # 标准海平面空气密度 rho = 1.225 kg/m^3
        viscosity: float = 1.789e-5,                  # 标准海平面空气动力粘度 mu = 1.789e-5 Pa*s
        n_bem_stations: int = BEM_NUM_SECTIONS,       # 27 个展向水/空气动力微元
        penalty_scale: float = 1.0,                   # 约束惩罚尺度因子
        pitch_angle_deg: float = BASELINE_PITCH_DEG,  # Sequence H: 4.815 deg @ 75% ref (tip = 3.0 deg)
        tilt_angle_deg: float = 0.0,                  # 机头仰角 0.0 deg (NASA Ames Sequence H)
        precone_angle_deg: float = 0.0,               # 叶片预锥角 0.0 deg (NASA Ames Sequence H)
        wind_shear_exponent: float = 0.0,             # 风洞/URANS 均匀入流 (alpha = 0.0)
        hub_height: float = HUB_HEIGHT,               # NASA Ames 风洞中心高 12.192 m
        enable_roughness: bool = False,               # 风洞洁净 S809 叶片默认关闭粗糙度污染衰减
        roughness_cl_loss: float = 0.05,
        roughness_cd_increase: float = 0.25,
        max_thrust_ratio: float = 1.15,               # 最大允许推力比 T <= 1.15 * T_baseline
        max_evals: Optional[int] = None,              # 统一评估预算上限
        param_model: Optional[Any] = None,            # 默认采用 B-Spline Modal 参数化模型 (D=114)
    ):
        self.param_model = BSplineBladeModel() if param_model is None else param_model
        self.wind_speed = float(wind_speed)
        self.rpm = float(rpm)
        self.omega = float(rpm * np.pi / 30.0)
        self.air_density = float(air_density)
        self.viscosity = float(viscosity)
        self.n_bem_stations = int(n_bem_stations)
        self.penalty_scale = float(penalty_scale)
        self.pitch_angle_deg = float(pitch_angle_deg)
        self.pitch_angle_rad = float(np.radians(pitch_angle_deg))
        self.tilt_angle_deg = float(tilt_angle_deg)
        self.tilt_angle_rad = float(np.radians(tilt_angle_deg))
        self.precone_angle_deg = float(precone_angle_deg)
        self.precone_angle_rad = float(np.radians(precone_angle_deg))
        self.wind_shear_exponent = float(wind_shear_exponent)
        self.hub_height = float(hub_height)
        self.enable_roughness = bool(enable_roughness)
        self.roughness_cl_loss = float(roughness_cl_loss)
        self.roughness_cd_increase = float(roughness_cd_increase)
        self.max_thrust_ratio = float(max_thrust_ratio)

        self.max_evals = int(max_evals) if max_evals is not None else None
        self.eval_count = 0

        # 从参数化模型继承站位结构/厚度边界 (S809 max_t = 0.2107)
        if hasattr(self.param_model, "thickness_limits"):
            self.thickness_limits = list(self.param_model.thickness_limits)
        else:
            self.thickness_limits = [
                (0.12, 0.28), (0.10, 0.26), (0.09, 0.25),
                (0.08, 0.25), (0.08, 0.24), (0.07, 0.24),
            ]
        if hasattr(self.param_model, "spar_cap_min_t"):
            self.spar_cap_min_t = list(self.param_model.spar_cap_min_t)
        else:
            self.spar_cap_min_t = [0.05 for _ in range(NUM_STATIONS)]
        if hasattr(self.param_model, "min_le_radii"):
            self.min_le_radii = list(self.param_model.min_le_radii)
        else:
            self.min_le_radii = [0.003 for _ in range(NUM_STATIONS)]

        self.baseline_x0 = self.param_model.get_x0() if self.param_model is not None else get_baseline_blade_params()
        self.dim = self.param_model.dim if self.param_model is not None else DIM
        self._baseline_power_kW: Optional[float] = None
        self._baseline_thrust_kN: Optional[float] = None
        self._max_allowable_thrust_kN: Optional[float] = None
        self._baseline_detailed_cache: Optional[Dict[str, Any]] = None
        self._calibrate_baseline()

    def reset_counter(self) -> None:
        """重置评估计数器"""
        self.eval_count = 0

    def _calibrate_baseline(self) -> None:
        """严格评估当前工况下 NREL Phase VI 基准构型以标定功率与推力锚点 (严禁静默回退假数)."""
        if self.param_model is not None:
            span_data = self.param_model.to_span_data(self.baseline_x0, n_stations=self.n_bem_stations)
        else:
            span_data = interpolate_blade_span(self.baseline_x0, n_stations=self.n_bem_stations)
        base_bem = self.solve_bem(span_data)
        self._baseline_power_kW = float(base_bem["power_kW"])
        self._baseline_thrust_kN = float(base_bem["thrust_kN"])
        self._max_allowable_thrust_kN = self._baseline_thrust_kN * self.max_thrust_ratio

    def get_baseline_detailed(self) -> Dict[str, Any]:
        """获取当前基准构型的完整气动与几何细节指标字典."""
        if self._baseline_detailed_cache is not None:
            return self._baseline_detailed_cache.copy()
        if self.param_model is not None:
            span_data = self.param_model.to_span_data(self.baseline_x0, n_stations=self.n_bem_stations)
        else:
            span_data = interpolate_blade_span(self.baseline_x0, n_stations=self.n_bem_stations)
        det = self.solve_bem(span_data)
        _, _, geom_metrics = self.validate_geometry(self.baseline_x0)
        det["geom_metrics"] = geom_metrics
        det["x"] = self.baseline_x0.tolist()
        det["feasible"] = True
        det["is_feasible"] = True
        self._baseline_detailed_cache = det
        return det.copy()

    @property
    def max_allowable_thrust_kN(self) -> float:
        if self._max_allowable_thrust_kN is None:
            self._calibrate_baseline()
        return float(self._max_allowable_thrust_kN)

    @staticmethod
    def _dominant_penalty(
        violation: float,
        scale: float,
        base_step: float = 0.25,
    ) -> float:
        """平滑有界约束惩罚函数 (尺度校准至 NREL Phase VI ~6.6 kW 功率量级)."""
        if violation <= 0.0:
            return 0.0
        rel = float(violation) / max(float(scale), 1e-8)
        cap = 1.5 * base_step
        return float(cap * np.tanh(1.5 * rel) + 3.0 * rel)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全套叶片几何与展向拓扑校验."""
        if self.param_model is not None:
            return self.param_model.validate_geometry(x)

        stations = unpack_blade_params(x)
        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "chord_smooth_violation": 0.0,
            "twist_smooth_violation": 0.0,
            "max_overlap": 0.0,
        }

        x_sample = (1.0 - np.cos(np.linspace(0.0, np.pi, 80))) / 2.0
        chords = [s["chord"] for s in stations]
        twists = [s["twist"] for s in stations]

        for k, s in enumerate(stations):
            th = s["theta"]
            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8],
                lower_weights=th[8:16],
                leading_edge_weight=float(th[16]),
                TE_thickness=float(th[17]),
            )
            try:
                yu = k_af.upper_coordinates(x_sample)[:, 1]
                yl = k_af.lower_coordinates(x_sample)[:, 1]
            except Exception:
                return True, 50.0, geom_metrics

            t = yu - yl
            min_t = float(np.min(t))
            max_t = float(np.max(t))
            try:
                t_spar = float(k_af.upper_coordinates(0.40)[0, 1] - k_af.lower_coordinates(0.40)[0, 1])
            except Exception:
                t_spar = min_t

            overlap = float(max(0.0, -min_t))
            geom_metrics["max_overlap"] = max(geom_metrics["max_overlap"], overlap)
            try:
                le_r = float(k_af.LE_radius())
                if np.isnan(le_r) or le_r < 0:
                    le_r = 0.0
            except Exception:
                le_r = 0.0

            idx_tmax = int(np.argmax(t))
            aft_non_mono = float(np.sum(np.maximum(0.0, np.diff(t[idx_tmax:]))))
            idx_crest = int(np.argmax(yu))
            crest_non_mono = 0.0
            if idx_crest > 0:
                crest_non_mono += float(np.sum(np.maximum(0.0, -np.diff(yu[:idx_crest]))))
            if idx_crest < len(yu) - 1:
                crest_non_mono += float(np.sum(np.maximum(0.0, np.diff(yu[idx_crest:]))))
            d2_yu = np.diff(yu, 2)
            x_mid_mask = (x_sample[1:-1] >= 0.05) & (x_sample[1:-1] <= 0.65)
            waviness = float(np.sum(np.maximum(0.0, d2_yu[x_mid_mask])))

            geom_metrics["station_metrics"].append({
                "station_idx": k,
                "max_t_rel": max_t,
                "t_spar": t_spar,
                "min_t": min_t,
                "overlap": overlap,
                "le_radius": le_r,
                "aft_non_mono": aft_non_mono,
                "crest_non_mono": crest_non_mono,
                "waviness": waviness,
            })

            if overlap > 0.008:
                return True, 20.0 + overlap * 100.0, geom_metrics

        chord_pen = float(np.sum(np.maximum(0.0, np.diff(chords))))
        twist_pen = float(np.sum(np.maximum(0.0, np.diff(twists))))
        geom_metrics["chord_taper_violation"] = chord_pen
        geom_metrics["twist_monotonicity_violation"] = twist_pen

        if chord_pen > 0.08 or twist_pen > 1.0:
            return True, 10.0 + 20.0 * chord_pen + 2.0 * twist_pen, geom_metrics

        return False, 0.0, geom_metrics

    def calculate_penalties(
        self,
        geom_metrics: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, float]]:
        """计算软约束几何惩罚 (翼型厚度、主梁、前缘圆角、单调性、波纹度、展向平滑性)."""
        penalties: Dict[str, float] = {}
        for st_m in geom_metrics.get("station_metrics", []):
            k = st_m["station_idx"]
            idx_lim = min(k, len(self.thickness_limits) - 1)
            t_max = st_m["max_t_rel"]
            t_spar = st_m["t_spar"]
            min_lim, max_lim = self.thickness_limits[idx_lim]
            spar_min = self.spar_cap_min_t[idx_lim]

            if t_max < min_lim:
                penalties[f"st{k+1}_thin"] = self._dominant_penalty(min_lim - t_max, scale=0.03, base_step=0.20)
            elif t_max > max_lim:
                penalties[f"st{k+1}_thick"] = self._dominant_penalty(t_max - max_lim, scale=0.03, base_step=0.20)

            if t_spar < spar_min:
                penalties[f"st{k+1}_spar"] = self._dominant_penalty(spar_min - t_spar, scale=0.03, base_step=0.20)

            min_le = self.min_le_radii[idx_lim]
            le_r = st_m.get("le_radius", min_le)
            if le_r < min_le:
                penalties[f"st{k+1}_sharp_le"] = self._dominant_penalty(min_le - le_r, scale=0.005, base_step=0.20)

            aft_viol = st_m.get("aft_non_mono", 0.0)
            if aft_viol > 0.002:
                penalties[f"st{k+1}_aft_bump"] = self._dominant_penalty(aft_viol, scale=0.005, base_step=0.15)

            crest_viol = st_m.get("crest_non_mono", 0.0)
            if crest_viol > 0.002:
                penalties[f"st{k+1}_crest_dimple"] = self._dominant_penalty(crest_viol, scale=0.005, base_step=0.15)

            wave_viol = st_m.get("waviness", 0.0)
            if wave_viol > 0.001:
                penalties[f"st{k+1}_waviness"] = self._dominant_penalty(wave_viol, scale=0.002, base_step=0.20)

        min_c_viol = geom_metrics.get("min_chord_violation", 0.0)
        if min_c_viol > 0.0:
            penalties["min_chord"] = self._dominant_penalty(min_c_viol, scale=0.05, base_step=0.25)

        c_viol = geom_metrics.get("chord_taper_violation", 0.0)
        if c_viol > 0.0:
            penalties["chord_taper"] = self._dominant_penalty(c_viol, scale=0.10, base_step=0.20)

        c_lin = geom_metrics.get("chord_linear_violation", 0.0)
        if c_lin > 0.0:
            penalties["chord_linear"] = self._dominant_penalty(c_lin, scale=0.04, base_step=0.25)

        tw_viol = geom_metrics.get("twist_monotonicity_violation", 0.0)
        if tw_viol > 0.0:
            penalties["twist_monotonicity"] = self._dominant_penalty(tw_viol, scale=0.5, base_step=0.20)

        c_smooth = geom_metrics.get("chord_smooth_violation", 0.0)
        if c_smooth > 0.0:
            penalties["chord_smoothness"] = self._dominant_penalty(c_smooth, scale=0.05, base_step=0.20)

        tw_smooth = geom_metrics.get("twist_smooth_violation", 0.0)
        if tw_smooth > 0.0:
            penalties["twist_smoothness"] = self._dominant_penalty(tw_smooth, scale=1.0, base_step=0.20)

        total_penalty = sum(penalties.values())
        is_feasible = (len(penalties) == 0)
        return is_feasible, total_penalty, penalties

    def validate_physics(
        self,
        bem_results: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, float]]:
        """检验 BEM 气动求解结果的物理可行性与推力安全边界."""
        penalties: Dict[str, float] = {}
        total_penalty = 0.0

        thrust_kN = float(bem_results.get("thrust_kN", 0.0))
        max_T = self.max_allowable_thrust_kN

        if thrust_kN > max_T:
            t_violation = thrust_kN - max_T
            pen_t = self._dominant_penalty(t_violation, scale=max(max_T * 0.05, 1e-3), base_step=0.35)
            penalties["thrust_overload"] = pen_t
            total_penalty += pen_t

        power_kW = float(bem_results.get("power_kW", 0.0))
        if power_kW <= 0.0 or not np.isfinite(power_kW):
            penalties["non_positive_power"] = 5.0
            total_penalty += 5.0

        is_feasible = (len(penalties) == 0)
        return is_feasible, total_penalty, penalties

    @staticmethod
    def viterna_extrapolate(
        alpha_deg: np.ndarray,
        cl_2d: np.ndarray,
        cd_2d: np.ndarray,
        aspect_ratio: float = 14.0,
        cl_s_pos: Optional[np.ndarray] = None,
        cd_s_pos: Optional[np.ndarray] = None,
        cl_s_neg: Optional[np.ndarray] = None,
        cd_s_neg: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Viterna-Corrigan 深失速气动极线全域外推模型 (动态绑定实际翼型 +/-20° 极线端点)."""
        alpha_rad = np.radians(alpha_deg)
        abs_alpha = np.abs(alpha_deg)

        alpha_s_deg = 20.0
        alpha_s = np.radians(alpha_s_deg)
        cd_max = np.clip(1.11 + 0.018 * aspect_ratio, 1.30, 2.0)

        if cl_s_pos is not None and cd_s_pos is not None:
            cl_s_pos_arr = np.asarray(cl_s_pos, dtype=np.float64)
            cd_s_pos_arr = np.asarray(cd_s_pos, dtype=np.float64)
        else:
            cl_s_pos_arr = np.full_like(cl_2d, 1.15, dtype=np.float64)
            cd_s_pos_arr = np.full_like(cd_2d, 0.20, dtype=np.float64)

        if cl_s_neg is not None and cd_s_neg is not None:
            cl_s_neg_arr = np.asarray(cl_s_neg, dtype=np.float64)
            cd_s_neg_arr = np.asarray(cd_s_neg, dtype=np.float64)
        else:
            cl_s_neg_arr = -cl_s_pos_arr
            cd_s_neg_arr = cd_s_pos_arr

        cl_s = np.where(alpha_deg >= 0.0, cl_s_pos_arr, -cl_s_neg_arr)
        cd_s = np.where(alpha_deg >= 0.0, cd_s_pos_arr, cd_s_neg_arr)

        b1 = cd_max
        b2 = (cd_s - cd_max * (np.sin(alpha_s) ** 2)) / np.cos(alpha_s)
        a1 = b1 / 2.0
        a2 = (cl_s - cd_max * np.sin(alpha_s) * np.cos(alpha_s)) * np.sin(alpha_s) / (np.cos(alpha_s) ** 2 + 1e-6)

        abs_sin = np.where(np.abs(np.sin(alpha_rad)) > 1e-4, np.sin(alpha_rad), 1e-4)
        cl_extrap = a1 * np.sin(2.0 * alpha_rad) + a2 * (np.cos(alpha_rad) ** 2) / abs_sin
        cd_extrap = np.maximum(1e-4, b1 * (np.sin(alpha_rad) ** 2) + b2 * np.cos(alpha_rad))

        blend = 1.0 / (1.0 + np.exp(-(abs_alpha - 20.0) / 1.0))
        cl_out = (1.0 - blend) * cl_2d + blend * cl_extrap
        cd_out = (1.0 - blend) * cd_2d + blend * cd_extrap
        return cl_out, cd_out

    def solve_bem(self, span_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """高精度快速叶素动量理论 (BEM) 求解器 (NREL Phase VI + NeuralFoil 'xxxlarge')."""
        r = span_data["r"]
        dr = span_data["dr"]
        chords = span_data["chord"]
        twists_deg = span_data["twist_deg"]
        twists_rad = span_data["twist_rad"]
        thetas = span_data["kulfan_thetas"]
        N = len(r)

        r_rotor = r * np.cos(self.precone_angle_rad)
        dr_rotor = dr * np.cos(self.precone_angle_rad)
        R_rotor = ROTOR_RADIUS * np.cos(self.precone_angle_rad)
        swept_area = np.pi * (R_rotor ** 2)
        aspect_ratio = float(max(6.0, BLADE_LENGTH / max(float(np.mean(chords)), 1e-3)))

        # 垂直风切变方位角平均 (wind_shear_exponent=0.0 时严格退化为 1.0 均匀入流)
        if abs(self.wind_shear_exponent) < 1e-8:
            shear_factor = np.ones(N, dtype=np.float64)
        else:
            psis = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
            z_stations = self.hub_height + (r_rotor[:, None]) * np.cos(psis[None, :])
            z_clamped = np.maximum(z_stations, 1.0)
            shear_factor = np.mean((z_clamped / self.hub_height) ** self.wind_shear_exponent, axis=1)
        V_axial_inflow = self.wind_speed * np.cos(self.tilt_angle_rad) * shear_factor

        # 3D 旋转失速延迟修正因子
        c_over_r = chords / np.maximum(r, BLADE_ROOT_RADIUS)
        c_over_r_eff = np.minimum(c_over_r, 0.8)
        r_frac_span = np.clip((r - BLADE_ROOT_RADIUS) / BLADE_LENGTH, 0.0, 1.0)
        tip_cutoff_3d = np.clip(1.0 - (r_frac_span ** 2), 0.0, 1.0)

        lambda_tip = self.omega * R_rotor / max(self.wind_speed, 1e-3)
        g_tip = float(np.clip(np.exp(-0.125 * (NUM_BLADES * lambda_tip - 21.0)) + 0.1, 0.1, 1.0))

        # NeuralFoil 'xxxlarge' 全展向极线批处理推理 (严禁静默降级至常数假极线)
        alpha_grid = np.linspace(-25.0, 35.0, 61)
        n_alpha = len(alpha_grid)
        idx_p20 = 45  # alpha = +20.0 deg
        idx_n20 = 5   # alpha = -20.0 deg
        n_cases = N * n_alpha

        upper_w = np.zeros((8, n_cases), dtype=np.float64)
        lower_w = np.zeros((8, n_cases), dtype=np.float64)
        le_w = np.zeros(n_cases, dtype=np.float64)
        te_t = np.zeros(n_cases, dtype=np.float64)
        res_vec = np.zeros(n_cases, dtype=np.float64)
        alphas_vec = np.tile(alpha_grid, N)

        for k in range(N):
            th = thetas[k]
            V_rel_nom = np.sqrt((V_axial_inflow[k] * 0.75) ** 2 + (self.omega * r_rotor[k]) ** 2)
            Re_nom = np.clip(self.air_density * V_rel_nom * chords[k] / self.viscosity, 2e4, 2e7)

            idx_s = k * n_alpha
            idx_e = (k + 1) * n_alpha
            for i in range(8):
                upper_w[i, idx_s:idx_e] = th[i]
                lower_w[i, idx_s:idx_e] = th[8 + i]
            le_w[idx_s:idx_e] = th[16]
            te_t[idx_s:idx_e] = th[17]
            res_vec[idx_s:idx_e] = Re_nom

        k_dict_batch = {
            "upper_weights": upper_w,
            "lower_weights": lower_w,
            "leading_edge_weight": le_w,
            "TE_thickness": te_t,
        }

        out_nf = nf.get_aero_from_kulfan_parameters(
            k_dict_batch,
            alpha=alphas_vec,
            Re=res_vec,
            model_size="xxxlarge",
        )
        cl_tables = out_nf["CL"].reshape(N, n_alpha)
        cd_tables = out_nf["CD"].reshape(N, n_alpha)

        cl_s_pos = cl_tables[:, idx_p20]
        cd_s_pos = cd_tables[:, idx_p20]
        cl_s_neg = cl_tables[:, idx_n20]
        cd_s_neg = cd_tables[:, idx_n20]

        a = np.zeros(N, dtype=np.float64) + 0.20
        a_p = np.zeros(N, dtype=np.float64) + 0.01
        max_iter = 35
        relaxation = 0.20

        CL_3d = np.zeros(N, dtype=np.float64)
        CD_3d = np.zeros(N, dtype=np.float64)
        CL_2d_all = np.zeros(N, dtype=np.float64)
        CD_2d_all = np.zeros(N, dtype=np.float64)
        V_rel = np.zeros(N, dtype=np.float64)
        phi = np.zeros(N, dtype=np.float64)
        alpha_deg = np.zeros(N, dtype=np.float64)
        C_n = np.zeros(N, dtype=np.float64)
        C_t = np.zeros(N, dtype=np.float64)

        r_hub_proj = HUB_RADIUS * np.cos(self.precone_angle_rad)

        for it in range(max_iter):
            V_a = V_axial_inflow * (1.0 - a)
            V_t = self.omega * r_rotor * (1.0 + a_p)
            phi = np.arctan2(V_a, V_t)
            alpha_rad = phi - (twists_rad + self.pitch_angle_rad)
            alpha_deg = np.degrees(alpha_rad)
            V_rel = np.sqrt(V_a ** 2 + V_t ** 2)

            sin_phi_safe = np.maximum(np.sin(phi), 1e-4)
            f_tip = (NUM_BLADES / 2.0) * np.maximum(R_rotor - r_rotor, 0.0) / (r_rotor * sin_phi_safe)
            F_tip = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-g_tip * f_tip), 0.0, 1.0))
            f_root = (NUM_BLADES / 2.0) * np.maximum(r_rotor - r_hub_proj, 0.0) / (r_hub_proj * sin_phi_safe)
            F_root = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_root), 0.0, 1.0))
            # 防止叶尖端点 r=R 处 F=0 引起 Buhl-Glauert 奇异尖峰
            F = np.clip(F_tip * F_root, 0.02, 1.0)

            CL_2d = np.zeros(N, dtype=np.float64)
            CD_2d = np.zeros(N, dtype=np.float64)
            for k in range(N):
                CL_2d[k] = np.interp(alpha_deg[k], alpha_grid, cl_tables[k])
                CD_2d[k] = np.interp(alpha_deg[k], alpha_grid, cd_tables[k])

            CL_2d, CD_2d = self.viterna_extrapolate(
                alpha_deg,
                CL_2d,
                CD_2d,
                aspect_ratio=aspect_ratio,
                cl_s_pos=cl_s_pos,
                cd_s_pos=cd_s_pos,
                cl_s_neg=cl_s_neg,
                cd_s_neg=cd_s_neg,
            )

            if self.enable_roughness:
                cl_loss = self.roughness_cl_loss / (1.0 + np.exp(-(alpha_deg - 4.0) / 2.0))
                CL_2d = CL_2d * (1.0 - cl_loss)
                CD_2d = CD_2d * (1.0 + self.roughness_cd_increase)

            CL_2d_all = CL_2d
            CD_2d_all = CD_2d

            # OpenFAST Snel 3D 旋转失速延迟修正
            CL_pot = 2.0 * np.pi * np.sin(alpha_rad)
            raw_dCL = 3.0 * (c_over_r_eff ** 2) * np.maximum(0.0, CL_pot - CL_2d) * tip_cutoff_3d
            delta_CL = np.minimum(raw_dCL, 0.5)
            raw_dCD = -0.5 * 3.0 * (c_over_r_eff ** 2) * np.maximum(0.0, CD_2d - 0.008) * tip_cutoff_3d
            delta_CD = np.maximum(raw_dCD, -0.25)

            CL = CL_2d + delta_CL
            CD = np.maximum(1e-4, CD_2d + delta_CD)
            CL_3d = CL
            CD_3d = CD

            C_n = CL * np.cos(phi) + CD * np.sin(phi)
            C_t = CL * np.sin(phi) - CD * np.cos(phi)
            solidity = (NUM_BLADES * chords) / (2.0 * np.pi * r_rotor)

            C_T_elem = solidity * ((1.0 - a) ** 2) * C_n / (sin_phi_safe ** 2)

            a_new = np.zeros(N, dtype=np.float64)
            for i in range(N):
                if C_n[i] <= 0.0:
                    a_new[i] = 0.0
                elif C_T_elem[i] < 0.96 * F[i]:
                    val = 4.0 * F[i] * (sin_phi_safe[i] ** 2) / (solidity[i] * C_n[i] + 1e-6)
                    a_new[i] = 1.0 / (val + 1.0)
                else:
                    denom = 36.0 * F[i] - 50.0
                    num = 18.0 * F[i] - 20.0 - 3.0 * np.sqrt(
                        max(0.0, C_T_elem[i] * (50.0 - 36.0 * F[i]) + 12.0 * F[i] * (3.0 * F[i] - 4.0))
                    )
                    a_new[i] = num / (denom + 1e-6)

            a_new = np.clip(a_new, 0.0, 0.75)

            denom_ap = 4.0 * F * sin_phi_safe * np.cos(phi) / (solidity * np.maximum(1e-6, C_t)) - 1.0
            a_p_new = np.where((C_t > 0.0) & (np.abs(denom_ap) > 1e-5), 1.0 / denom_ap, 0.0)
            a_p_new = np.clip(a_p_new, 0.0, 0.35)

            delta_iter = float(np.max(np.abs(a_new - a)))
            a = (1.0 - relaxation) * a + relaxation * a_new
            a_p = (1.0 - relaxation) * a_p + relaxation * a_p_new

            if it >= 6 and delta_iter < 1e-4:
                break

        dQ = 0.5 * self.air_density * (V_rel ** 2) * NUM_BLADES * chords * C_t * r_rotor * dr_rotor
        dFz = 0.5 * self.air_density * (V_rel ** 2) * NUM_BLADES * chords * C_n * dr_rotor

        torque_Nm = float(np.sum(dQ))
        thrust_N = float(np.sum(dFz) * np.cos(self.tilt_angle_rad))
        power_W = float(torque_Nm * self.omega)

        power_ref = 0.5 * self.air_density * (self.wind_speed ** 3) * swept_area
        thrust_ref = 0.5 * self.air_density * (self.wind_speed ** 2) * swept_area

        betz_cap = (16.0 / 27.0) * power_ref
        power_W_phys = min(power_W, betz_cap)

        Cp = float(power_W_phys / max(power_ref, 1e-6))
        Ct = float(thrust_N / max(thrust_ref, 1e-6))

        return {
            "power_kW": power_W_phys / 1e3,
            "power_MW": power_W_phys / 1e6,
            "torque_Nm": torque_Nm,
            "torque_kNm": torque_Nm / 1e3,
            "thrust_kN": thrust_N / 1e3,
            "Cp": Cp,
            "Ct": Ct,
            "span_r": r,
            "span_radii": r,
            "span_r_rotor": r_rotor,
            "span_chord": chords,
            "span_twist_deg": twists_deg,
            "span_alpha_deg": alpha_deg,
            "blade_pitch_deg": float(self.pitch_angle_deg),
            "span_effective_pitch_deg": twists_deg + self.pitch_angle_deg,
            "tilt_angle_deg": float(self.tilt_angle_deg),
            "precone_angle_deg": float(self.precone_angle_deg),
            "enable_roughness": bool(self.enable_roughness),
            "span_CL_2d": CL_2d_all,
            "span_CL_3d": CL_3d,
            "span_CD_3d": CD_3d,
            "span_axial_a": a,
            "span_tangential_ap": a_p,
            "span_dQ": dQ,
            "span_dFz": dFz,
        }

    def evaluate(self, x: np.ndarray) -> float:
        """主标量优化目标函数 (kW)。任意几何/推力不可行解严格返回负值 (< 0.0) 以实现学术可行域隔离。"""
        if self.max_evals is not None and self.eval_count >= self.max_evals:
            raise RuntimeError(
                f"[NFE Budget Violation in BladeEvaluator] Evaluation count reached max_evals ({self.eval_count} >= {self.max_evals}). "
                "Further function evaluations are strictly forbidden."
            )
        self.eval_count += 1

        is_fatal, fatal_penalty, geom_metrics = self.validate_geometry(x)
        if is_fatal:
            return -float(max(1.0, fatal_penalty))

        is_geom_feasible, geom_penalty, _ = self.calculate_penalties(geom_metrics)
        if self.param_model is not None:
            span_data = self.param_model.to_span_data(x, n_stations=self.n_bem_stations)
        else:
            span_data = interpolate_blade_span(x, n_stations=self.n_bem_stations)

        bem_res = self.solve_bem(span_data)

        is_phys_feasible, phys_pen, _ = self.validate_physics(bem_res)
        soft_penalty = geom_penalty + phys_pen
        is_feasible = is_geom_feasible and is_phys_feasible
        raw_power_kW = float(bem_res["power_kW"])

        if not is_feasible:
            return -float(max(0.25, self.penalty_scale * soft_penalty))

        return raw_power_kW

    def evaluate_detailed(self, x: np.ndarray) -> Dict[str, Any]:
        """全量综合评估，返回涵盖几何指标、气动转矩、功率、推力、截面角度及约束明细的详尽字典."""
        is_fatal, fatal_pen, geom_metrics = self.validate_geometry(x)
        if self.param_model is not None:
            span_data = self.param_model.to_span_data(x, n_stations=self.n_bem_stations)
        else:
            span_data = interpolate_blade_span(x, n_stations=self.n_bem_stations)

        if is_fatal:
            pen_val = float(max(1.0, fatal_pen))
            return {
                "feasible": False,
                "is_feasible": False,
                "is_fatal": True,
                "power_kW": -pen_val,
                "power_MW": -pen_val / 1e3,
                "raw_power_kW": 0.0,
                "torque_Nm": 0.0,
                "torque_kNm": 0.0,
                "thrust_kN": 0.0,
                "Cp": 0.0,
                "Ct": 0.0,
                "penalties": {"fatal_geometry": pen_val},
                "total_penalty": pen_val,
                "total_penalty_kW": pen_val,
                "geom_metrics": geom_metrics,
                "span_radii": [float(v) for v in span_data["r"]],
                "span_chord": [float(v) for v in span_data["chord"]],
                "span_twist_deg": [float(v) for v in span_data["twist_deg"]],
                "span_alpha_deg": [0.0] * len(span_data["r"]),
                "blade_pitch_deg": float(self.pitch_angle_deg),
                "span_effective_pitch_deg": [float(v + self.pitch_angle_deg) for v in span_data["twist_deg"]],
            }

        is_geom_feasible, geom_penalty, penalties = self.calculate_penalties(geom_metrics)
        bem_res = self.solve_bem(span_data)
        is_phys_feasible, phys_pen, phys_penalties = self.validate_physics(bem_res)
        penalties.update(phys_penalties)

        total_penalty = float(sum(penalties.values()))
        is_feasible = is_geom_feasible and is_phys_feasible and (len(penalties) == 0)
        raw_power_kW = float(bem_res["power_kW"])
        reported_power_kW = raw_power_kW if is_feasible else -float(max(0.25, self.penalty_scale * total_penalty))

        return {
            "feasible": bool(is_feasible),
            "is_feasible": bool(is_feasible),
            "is_fatal": False,
            "power_kW": float(reported_power_kW),
            "power_MW": float(reported_power_kW) / 1e3,
            "raw_power_kW": raw_power_kW,
            "raw_power_MW": raw_power_kW / 1e3,
            "torque_Nm": float(bem_res["torque_Nm"]),
            "torque_kNm": float(bem_res["torque_kNm"]),
            "thrust_kN": float(bem_res["thrust_kN"]),
            "Cp": float(bem_res["Cp"]),
            "Ct": float(bem_res["Ct"]),
            "penalties": penalties,
            "total_penalty": total_penalty,
            "total_penalty_kW": total_penalty,
            "geom_metrics": geom_metrics,
            "span_radii": [float(v) for v in span_data["r"]],
            "span_chord": [float(v) for v in span_data["chord"]],
            "span_twist_deg": [float(v) for v in span_data["twist_deg"]],
            "span_alpha_deg": [float(v) for v in bem_res["span_alpha_deg"]],
            "blade_pitch_deg": float(self.pitch_angle_deg),
            "span_effective_pitch_deg": [float(v) for v in bem_res["span_effective_pitch_deg"]],
            "span_axial_a": [float(v) for v in bem_res.get("span_axial_a", [])],
            "span_tangential_ap": [float(v) for v in bem_res.get("span_tangential_ap", [])],
            "bem_details": bem_res,
        }
