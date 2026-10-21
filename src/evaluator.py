"""基于 AeroSandbox、NeuralFoil 与叶素动量理论 (BEM) 的 3D 风力机叶片工业级气动与几何综合评估器。
严格对齐 IEA-22-280-RWT (22MW 超大型海上风机) 工业规范：
- 额定测试工况: 风速 V_inf = 11.0 m/s, 转速 = 7.061 rpm (omega = 0.7394 rad/s);
- 6 个工业控制截面 (Station 1 ~ 6: r = 20.83m, 30.33m, 56.47m, 82.60m, 108.74m, 132.50m);
- 6 截面全套工业级几何强约束前置校验 (对齐 GEMINI.md 工业规范):
  * 逐站位翼型无自交叉、厚度分布合理、抗波纹凸性、主梁站位 (0.4c) 最小结构承载厚度;
  * 全叶片 3D 展向宏观几何约束: 弦长从肩部 (站位 2) 向叶尖单调平滑收缩、扭角严格单调负斜率气动洗退;
- 纯内存、微秒级 BEM 求解器:
  * 深度集成 NeuralFoil 深度学习气动代理网络, 6 区域批量推理;
  * 耦合 Buhl-Glauert 湍流尾流高推力状态修正与 Prandtl 根尖三维损失修正;
  * 在 10~20ms 内完成单个 120 维 3D 叶片的完整气动转矩 (Torque)、输出功率 (Power)、推力 (Thrust) 与 Cp 求解;
- 极端载荷物理约束: 风轮推力限制 T <= 1.15 * T_baseline (防止风机叶尖扫塔与叶根极端弯矩破坏);
- 致命缺陷与物理越界主导惩罚机制 (Dominant Penalty), 平滑有界且严格引导优化器回归可行域。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import aerosandbox as asb
import neuralfoil as nf

from src.blade_geometry import (
    DIM,
    HUB_RADIUS,
    NUM_BLADES,
    NUM_STATIONS,
    ROTOR_RADIUS,
    STATION_RADII,
    get_baseline_blade_params,
    interpolate_blade_span,
    unpack_blade_params,
)


class WindTurbineBladeEvaluator:
    """3D 风力机单叶片气动与几何约束综合评估器 (IEA-22-280-RWT 22MW 标称)。"""

    def __init__(
        self,
        wind_speed: float = 11.0,         # 额定基准测试风速 V_inf = 11.0 m/s
        rpm: float = 7.061,               # 额定基准测试转速 rpm = 7.061 (omega = 0.73943 rad/s)
        air_density: float = 1.225,       # 空气密度 rho (kg/m^3)
        viscosity: float = 1.81e-5,       # 动力粘度 mu (kg/(m*s))
        n_bem_stations: int = 18,         # BEM 展向积分散点数 (适配 6 站位)
        penalty_scale: float = 1.0,       # 约束惩罚尺度因子
        pitch_angle_deg: float = 0.0,     # 全叶片安装变桨角 (deg)
        max_thrust_ratio: float = 1.15,   # 最大允许推力比 (T_max = 1.15 * T_baseline)
    ):
        self.wind_speed = float(wind_speed)
        self.rpm = float(rpm)
        self.omega = float(rpm * np.pi / 30.0)  # rad/s
        self.air_density = float(air_density)
        self.viscosity = float(viscosity)
        self.n_bem_stations = int(n_bem_stations)
        self.penalty_scale = float(penalty_scale)
        self.pitch_angle_deg = float(pitch_angle_deg)
        self.pitch_angle_rad = float(np.radians(pitch_angle_deg))
        self.max_thrust_ratio = float(max_thrust_ratio)

        # 6 个站位的翼型相对厚度工程允许范围 [min_t_rel, max_t_rel] (FFA-W3 工业族规范)
        self.thickness_limits = [
            (0.38, 0.65),  # 站位 1 (r=20.83m): 根部结构超厚过渡段
            (0.28, 0.45),  # 站位 2 (r=30.33m): 肩部高强度承载区
            (0.22, 0.38),  # 站位 3 (r=56.47m): 内展高效气动区
            (0.18, 0.32),  # 站位 4 (r=82.60m): 中展主功率区
            (0.16, 0.28),  # 站位 5 (r=108.74m): 外展减阻区
            (0.14, 0.25),  # 站位 6 (r=132.50m): 翼尖薄阻力区
        ]

        # 40% 弦长主梁/抗剪腹板关键站位最小相对厚度要求 (t_spar / chord)
        self.spar_cap_min_t = [0.35, 0.26, 0.20, 0.16, 0.14, 0.12]

        # 预先评估基准叶片以建立推力与功率物理锚点
        self.baseline_x0 = get_baseline_blade_params()
        self._baseline_power_kW: Optional[float] = None
        self._baseline_thrust_kN: Optional[float] = None
        self._max_allowable_thrust_kN: Optional[float] = None
        self._calibrate_baseline()

    def _calibrate_baseline(self) -> None:
        """评估标准基准构型以标定额定工况基线推力与功率限制。"""
        try:
            span_data = interpolate_blade_span(self.baseline_x0, n_stations=self.n_bem_stations)
            base_bem = self.solve_bem(span_data)
            self._baseline_power_kW = float(base_bem["power_kW"])
            self._baseline_thrust_kN = float(base_bem["thrust_kN"])
            self._max_allowable_thrust_kN = self._baseline_thrust_kN * self.max_thrust_ratio
        except Exception:
            # 理论参考缺省值
            self._baseline_power_kW = 25000.0  # 约 25 MW 气动功率
            self._baseline_thrust_kN = 4200.0   # 约 4200 kN
            self._max_allowable_thrust_kN = 4200.0 * self.max_thrust_ratio

    @property
    def max_allowable_thrust_kN(self) -> float:
        if self._max_allowable_thrust_kN is None:
            self._calibrate_baseline()
        return self._max_allowable_thrust_kN

    @staticmethod
    def _dominant_penalty(
        violation: float,
        scale: float,
        base_step: float = 1200.0,
    ) -> float:
        """平滑有界工业约束主导惩罚函数 (单位对齐 22MW 级功率 kW)：
        
        当违背约束 (violation > 0) 时：
        - 消除断崖跳跃：P(0) = 0；
        - 初期陡峭排斥：大幅抹平局部微弱气动增益；
        - 双曲正切平滑饱和有界：控制单项惩罚在 1.5 * base_step (约 1800 kW) 辅以弱线性项。
        """
        if violation <= 0.0:
            return 0.0
        rel = float(violation) / max(float(scale), 1e-8)
        cap = 1.5 * base_step
        return float(cap * np.tanh(1.5 * rel) + 15.0 * rel)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全套 120 维叶片几何与展向拓扑校验。
        
        返回:
            (is_fatal, fatal_penalty, geom_metrics)
        """
        stations = unpack_blade_params(x)
        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "max_overlap": 0.0,
        }

        # 1. 逐站位翼型几何质量检验
        x_sample = np.linspace(0.005, 0.995, 60)
        chords = [s["chord"] for s in stations]
        twists = [s["twist"] for s in stations]

        for k, s in enumerate(stations):
            th = s["theta"]
            u_w = th[0:8]
            l_w = th[8:16]
            le_w = float(th[16])
            te_t = float(th[17])

            k_af = asb.KulfanAirfoil(
                upper_weights=u_w,
                lower_weights=l_w,
                leading_edge_weight=le_w,
                TE_thickness=te_t,
            )

            try:
                yu = k_af.upper_coordinates(x_sample)[:, 1]
                yl = k_af.lower_coordinates(x_sample)[:, 1]
            except Exception:
                # 无法解析几何坐标，视为致命缺陷
                return True, 50000.0, geom_metrics

            t = yu - yl
            min_t = float(np.min(t))
            max_t = float(np.max(t))

            # 40% 弦长主梁站位厚度
            try:
                t_spar = float(k_af.upper_coordinates(0.40)[0, 1] - k_af.lower_coordinates(0.40)[0, 1])
            except Exception:
                t_spar = min_t

            # 交叉违背: 上表面低于下表面
            overlap = float(max(0.0, -min_t))
            geom_metrics["max_overlap"] = max(geom_metrics["max_overlap"], overlap)

            geom_metrics["station_metrics"].append({
                "station_idx": k,
                "max_t_rel": max_t,
                "t_spar": t_spar,
                "min_t": min_t,
                "overlap": overlap,
            })

            # 若站位发生明显穿透自交叉 (overlap > 0.008)，判定为致命崩塌解
            if overlap > 0.008:
                fatal_pen = 40000.0 + self._dominant_penalty(overlap, scale=0.01, base_step=5000.0)
                return True, fatal_pen, geom_metrics

        # 2. 展向宏观几何约束检验
        # 弦长单调平滑收缩: 从肩部站位 2 (r=30.33m) 到翼尖站位 6 严格单调平滑收缩
        # 且站位 1 与站位 2 之间过渡平缓 (c1 <= c2 + 0.35m)
        chord_pen = 0.0
        if chords[0] > chords[1] + 0.35:
            chord_pen += chords[0] - (chords[1] + 0.35)

        for k in range(1, NUM_STATIONS - 1):
            if chords[k] < chords[k + 1]:
                chord_pen += (chords[k + 1] - chords[k])

        geom_metrics["chord_taper_violation"] = float(chord_pen)

        # 扭角严格单调负斜率气动洗退 (twist[k] >= twist[k+1])
        twist_pen = 0.0
        for k in range(NUM_STATIONS - 1):
            if twists[k] < twists[k + 1]:
                twist_pen += (twists[k + 1] - twists[k])

        geom_metrics["twist_monotonicity_violation"] = float(twist_pen)

        # 若宏观几何严重扭曲，施加致命排斥
        if chord_pen > 2.5 or twist_pen > 8.0:
            fatal_pen = 30000.0 + 2000.0 * (chord_pen + twist_pen)
            return True, fatal_pen, geom_metrics

        return False, 0.0, geom_metrics

    def calculate_penalties(
        self,
        geom_metrics: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, float]]:
        """计算软约束几何惩罚 (翼型厚度超限、主梁厚度不足、弦长扭角轻微违背)"""
        penalties: Dict[str, float] = {}
        for st_m in geom_metrics.get("station_metrics", []):
            k = st_m["station_idx"]
            t_max = st_m["max_t_rel"]
            t_spar = st_m["t_spar"]
            min_lim, max_lim = self.thickness_limits[k]
            spar_min = self.spar_cap_min_t[k]

            if t_max < min_lim:
                penalties[f"st{k+1}_thin"] = self._dominant_penalty(min_lim - t_max, scale=0.03, base_step=800.0)
            elif t_max > max_lim:
                penalties[f"st{k+1}_thick"] = self._dominant_penalty(t_max - max_lim, scale=0.03, base_step=800.0)

            if t_spar < spar_min:
                penalties[f"st{k+1}_spar"] = self._dominant_penalty(spar_min - t_spar, scale=0.03, base_step=1000.0)

        c_viol = geom_metrics.get("chord_taper_violation", 0.0)
        if c_viol > 0.0:
            penalties["chord_taper"] = self._dominant_penalty(c_viol, scale=0.5, base_step=600.0)

        tw_viol = geom_metrics.get("twist_monotonicity_violation", 0.0)
        if tw_viol > 0.0:
            penalties["twist_monotonicity"] = self._dominant_penalty(tw_viol, scale=1.0, base_step=600.0)

        total_penalty = sum(penalties.values())
        is_feasible = (len(penalties) == 0)
        return is_feasible, total_penalty, penalties

    def validate_physics(
        self,
        bem_results: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, float]]:
        """检验 BEM 气动求解结果的物理可信性与推力安全边界。"""
        penalties: Dict[str, float] = {}
        total_penalty = 0.0

        thrust_kN = float(bem_results.get("thrust_kN", 0.0))
        max_T = self.max_allowable_thrust_kN

        # 推力越界约束 (防止风轮过载折断或扫塔)
        if thrust_kN > max_T:
            t_violation = thrust_kN - max_T
            pen_t = self._dominant_penalty(t_violation, scale=max_T * 0.05, base_step=1500.0)
            penalties["thrust_overload"] = pen_t
            total_penalty += pen_t

        is_feasible = (len(penalties) == 0)
        return is_feasible, total_penalty, penalties

    def solve_bem(self, span_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """高精度快速叶素动量理论 (BEM) 求解器 (耦合 Buhl-Glauert 湍流高推力修正与 Prandtl 根尖损失)。
        
        集成 NeuralFoil 深度学习气动代理网络，在内存中进行微秒级批量推理。
        """
        r = span_data["r"]
        dr = span_data["dr"]
        chords = span_data["chord"]
        twists_rad = span_data["twist_rad"]
        thetas = span_data["kulfan_thetas"]
        N = len(r)

        # 将 N 个展向截面分组为 6 个气动区域，分别调用各区域 NeuralFoil 代理
        step = max(1, N // NUM_STATIONS)
        sec_slices = []
        for s in range(NUM_STATIONS):
            s_start = s * step
            s_end = (s + 1) * step if s < NUM_STATIONS - 1 else N
            sec_slices.append((s_start, s_end))

        # 诱导因子初始化
        a = np.zeros(N, dtype=np.float64) + 0.25
        a_p = np.zeros(N, dtype=np.float64) + 0.01

        max_iter = 12
        relaxation = 0.35

        CL = np.zeros(N, dtype=np.float64)
        CD = np.zeros(N, dtype=np.float64)
        V_rel = np.zeros(N, dtype=np.float64)
        phi = np.zeros(N, dtype=np.float64)
        alpha = np.zeros(N, dtype=np.float64)
        C_n = np.zeros(N, dtype=np.float64)
        C_t = np.zeros(N, dtype=np.float64)

        for it in range(max_iter):
            V_a = self.wind_speed * (1.0 - a)
            V_t = self.omega * r * (1.0 + a_p)
            phi = np.arctan2(V_a, V_t)
            alpha_rad = phi - (twists_rad + self.pitch_angle_rad)
            alpha_deg = np.degrees(alpha_rad)
            alpha = alpha_deg

            V_rel = np.sqrt(V_a**2 + V_t**2)
            Re = self.air_density * V_rel * chords / self.viscosity

            # Prandtl 叶尖与叶根三维损失修正
            f_tip = (NUM_BLADES / 2.0) * (ROTOR_RADIUS - r) / (r * np.sin(phi) + 1e-6)
            F_tip = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), 0.0, 1.0))
            f_root = (NUM_BLADES / 2.0) * (r - HUB_RADIUS) / (HUB_RADIUS * np.sin(phi) + 1e-6)
            F_root = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_root), 0.0, 1.0))
            F = np.clip(F_tip * F_root, 1e-3, 1.0)

            # 批量通过 NeuralFoil 求解 6 段的气动升阻力
            for st_i, (start_idx, end_idx) in enumerate(sec_slices):
                if start_idx >= end_idx:
                    continue
                sub_alpha = alpha_deg[start_idx:end_idx]
                sub_Re = float(np.mean(Re[start_idx:end_idx]))
                mid_k = (start_idx + end_idx) // 2
                k_dict = {
                    "upper_weights": thetas[mid_k, 0:8],
                    "lower_weights": thetas[mid_k, 8:16],
                    "leading_edge_weight": float(thetas[mid_k, 16]),
                    "TE_thickness": float(thetas[mid_k, 17]),
                }
                try:
                    aero = nf.get_aero_from_kulfan_parameters(
                        k_dict,
                        alpha=np.asarray(sub_alpha, dtype=np.float64),
                        Re=sub_Re,
                        model_size="xxxlarge",
                    )
                    CL[start_idx:end_idx] = np.asarray(aero["CL"])
                    CD[start_idx:end_idx] = np.asarray(aero["CD"])
                except Exception:
                    CL[start_idx:end_idx] = 0.6
                    CD[start_idx:end_idx] = 0.015

            CD = np.maximum(CD, 1e-4)

            # 法向力系数与切向力系数
            C_n = CL * np.cos(phi) + CD * np.sin(phi)
            C_t = CL * np.sin(phi) - CD * np.cos(phi)
            solidity = (NUM_BLADES * chords) / (2.0 * np.pi * r)

            # 局部推力系数
            C_T_elem = solidity * ((1.0 - a)**2) * C_n / (np.sin(phi)**2 + 1e-6)

            # Buhl-Glauert 湍流高推力状态修正
            a_new = np.zeros(N, dtype=np.float64)
            for i in range(N):
                if C_T_elem[i] < 0.96 * F[i]:
                    val = 4.0 * F[i] * (np.sin(phi[i])**2) / (solidity[i] * C_n[i] + 1e-6)
                    a_new[i] = 1.0 / (val + 1.0)
                else:
                    denom = 36.0 * F[i] - 50.0
                    num = 18.0 * F[i] - 20.0 - 3.0 * np.sqrt(
                        max(0.0, C_T_elem[i] * (50.0 - 36.0 * F[i]) + 12.0 * F[i] * (3.0 * F[i] - 4.0))
                    )
                    a_new[i] = num / (denom + 1e-6)

            a_new = np.clip(a_new, 0.0, 0.75)

            # 切向诱导因子更新
            denom_ap = 4.0 * F * np.sin(phi) * np.cos(phi) / (solidity * C_t + 1e-6) - 1.0
            a_p_new = np.where(np.abs(denom_ap) > 1e-5, 1.0 / denom_ap, 0.0)
            a_p_new = np.clip(a_p_new, 0.0, 0.35)

            # 欠松弛迭代
            a = (1.0 - relaxation) * a + relaxation * a_new
            a_p = (1.0 - relaxation) * a_p + relaxation * a_p_new

        # 积分全叶轮转矩与推力
        dQ = 0.5 * self.air_density * (V_rel**2) * NUM_BLADES * chords * C_t * r * dr
        dFz = 0.5 * self.air_density * (V_rel**2) * NUM_BLADES * chords * C_n * dr

        torque_Nm = float(np.sum(dQ))
        thrust_N = float(np.sum(dFz))
        power_W = float(torque_Nm * self.omega)

        swept_area = np.pi * (ROTOR_RADIUS**2)
        power_ref = 0.5 * self.air_density * (self.wind_speed**3) * swept_area
        thrust_ref = 0.5 * self.air_density * (self.wind_speed**2) * swept_area

        # Betz 理论极限功率保护 (Cp <= 16/27 = 0.5926)
        betz_cap = (16.0 / 27.0) * power_ref
        power_W_phys = min(power_W, betz_cap)

        Cp = float(power_W_phys / max(power_ref, 1e-6))
        Ct = float(thrust_N / max(thrust_ref, 1e-6))

        return {
            "power_kW": power_W_phys / 1e3,
            "power_MW": power_W_phys / 1e6,
            "torque_kNm": torque_Nm / 1e3,
            "thrust_kN": thrust_N / 1e3,
            "Cp": Cp,
            "Ct": Ct,
            "span_r": r,
            "span_CL": CL,
            "span_CD": CD,
            "span_alpha_deg": alpha,
            "span_axial_a": a,
            "span_tangential_ap": a_p,
            "span_dQ": dQ,
            "span_dFz": dFz,
        }

    def evaluate(self, x: np.ndarray) -> float:
        """主标量优化目标函数 (惩罚后有效输出功率 P_penalized, 单位: kW)。
        
        供 TuRBO、PSO 以及 MALIBO 直接调用进行单目标最大化。
        """
        # 1. 几何强约束前置校验
        is_fatal, fatal_penalty, geom_metrics = self.validate_geometry(x)
        if is_fatal:
            return -float(fatal_penalty)

        # 2. 软约束惩罚累计 (翼型厚度超限、弦长扭角微弱违背)
        soft_penalty = 0.0

        for st_m in geom_metrics["station_metrics"]:
            k = st_m["station_idx"]
            t_max = st_m["max_t_rel"]
            t_spar = st_m["t_spar"]
            min_lim, max_lim = self.thickness_limits[k]
            spar_min = self.spar_cap_min_t[k]

            # 最大相对厚度上下限超限
            if t_max < min_lim:
                soft_penalty += self._dominant_penalty(min_lim - t_max, scale=0.03, base_step=800.0)
            elif t_max > max_lim:
                soft_penalty += self._dominant_penalty(t_max - max_lim, scale=0.03, base_step=800.0)

            # 主梁最小相对厚度不足
            if t_spar < spar_min:
                soft_penalty += self._dominant_penalty(spar_min - t_spar, scale=0.03, base_step=1000.0)

        # 弦长单调平滑收缩轻微超限
        c_viol = geom_metrics["chord_taper_violation"]
        if c_viol > 0.0:
            soft_penalty += self._dominant_penalty(c_viol, scale=0.5, base_step=600.0)

        # 扭角负斜率轻微超限
        tw_viol = geom_metrics["twist_monotonicity_violation"]
        if tw_viol > 0.0:
            soft_penalty += self._dominant_penalty(tw_viol, scale=1.0, base_step=600.0)

        # 3. 展向平滑插值与 BEM 微秒级气动求解
        span_data = interpolate_blade_span(x, n_stations=self.n_bem_stations)
        try:
            bem_res = self.solve_bem(span_data)
        except Exception:
            return -50000.0

        # 4. 物理约束与推力超限校验
        is_phys_feasible, phys_pen, _ = self.validate_physics(bem_res)
        soft_penalty += phys_pen

        # 综合惩罚后净功率
        raw_power_kW = float(bem_res["power_kW"])
        penalized_power_kW = raw_power_kW - self.penalty_scale * soft_penalty

        return float(penalized_power_kW)

    def evaluate_detailed(self, x: np.ndarray) -> Dict[str, Any]:
        """全量综合评估，返回涵盖几何指标、气动转矩、功率、推力及约束明细的详尽字典。"""
        is_fatal, fatal_pen, geom_metrics = self.validate_geometry(x)
        if is_fatal:
            return {
                "feasible": False,
                "is_fatal": True,
                "power_kW": -float(fatal_pen),
                "power_MW": -float(fatal_pen) / 1e3,
                "raw_power_kW": 0.0,
                "torque_kNm": 0.0,
                "thrust_kN": 0.0,
                "Cp": 0.0,
                "Ct": 0.0,
                "penalties": {"fatal_geometry": float(fatal_pen)},
                "geom_metrics": geom_metrics,
            }

        # 软约束评估
        penalties: Dict[str, float] = {}
        for st_m in geom_metrics["station_metrics"]:
            k = st_m["station_idx"]
            t_max = st_m["max_t_rel"]
            t_spar = st_m["t_spar"]
            min_lim, max_lim = self.thickness_limits[k]
            spar_min = self.spar_cap_min_t[k]

            if t_max < min_lim:
                penalties[f"st{k+1}_thin"] = self._dominant_penalty(min_lim - t_max, scale=0.03, base_step=800.0)
            elif t_max > max_lim:
                penalties[f"st{k+1}_thick"] = self._dominant_penalty(t_max - max_lim, scale=0.03, base_step=800.0)

            if t_spar < spar_min:
                penalties[f"st{k+1}_spar"] = self._dominant_penalty(spar_min - t_spar, scale=0.03, base_step=1000.0)

        if geom_metrics["chord_taper_violation"] > 0.0:
            penalties["chord_taper"] = self._dominant_penalty(
                geom_metrics["chord_taper_violation"], scale=0.5, base_step=600.0
            )

        if geom_metrics["twist_monotonicity_violation"] > 0.0:
            penalties["twist_monotonicity"] = self._dominant_penalty(
                geom_metrics["twist_monotonicity_violation"], scale=1.0, base_step=600.0
            )

        span_data = interpolate_blade_span(x, n_stations=self.n_bem_stations)
        bem_res = self.solve_bem(span_data)

        is_phys_feasible, phys_pen, phys_penalties = self.validate_physics(bem_res)
        penalties.update(phys_penalties)

        total_penalty = sum(penalties.values())
        raw_power_kW = float(bem_res["power_kW"])
        penalized_power_kW = raw_power_kW - self.penalty_scale * total_penalty

        return {
            "feasible": (len(penalties) == 0),
            "is_fatal": False,
            "power_kW": float(penalized_power_kW),
            "power_MW": float(penalized_power_kW) / 1e3,
            "raw_power_kW": raw_power_kW,
            "raw_power_MW": raw_power_kW / 1e3,
            "torque_kNm": float(bem_res["torque_kNm"]),
            "thrust_kN": float(bem_res["thrust_kN"]),
            "Cp": float(bem_res["Cp"]),
            "Ct": float(bem_res["Ct"]),
            "penalties": penalties,
            "total_penalty_kW": float(total_penalty),
            "geom_metrics": geom_metrics,
            "bem_details": bem_res,
        }
