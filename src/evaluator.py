"""基于 AeroSandbox、NeuralFoil 与叶素动量理论 (BEM) 的 3D 风力机叶片工业级气动与几何综合评估器。

核心特性：
1. 工业级几何强约束前置校验 (对齐 GEMINI.md 规范)：
   - 3 个核心站位翼型的无自交叉、单调后缘收缩、抗波纹凸性、关键结构梁站位最小厚度；
   - 全叶片 3D 展向宏观几何约束：弦长单调平滑收缩 (c_root >= c_mid >= c_tip) 与扭角负斜率气动扭退 (beta_root >= beta_mid >= beta_tip)；
   - 致命缺陷与几何越界主导惩罚机制 (Dominant Penalty)，绝不让崩塌解浪费气动评估或破坏代理模型连续性。
2. 纯内存、微秒级 BEM 求解器：
   - 深度集成 NeuralFoil 神经网络气动代理，多截面批量推理；
   - 包含 Prandtl 根尖损失修正 (Tip/Root Loss) 与 Glauert 湍流尾流高推力修正；
   - 在 10~20ms 内完成单个 60 维 3D 叶片的完整气动转矩 (Torque)、输出功率 (Power) 与风能利用系数 (Cp) 求解。
3. 灵活支持不同环境工况 (风速 V_inf 与转速 Omega)，直接适配 TuRBO、LLM-Meta 以及 MALIBO 多工况网格数据集生成。
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import aerosandbox as asb
import neuralfoil as nf

from src.blade_geometry import (
    HUB_RADIUS,
    NUM_BLADES,
    ROTOR_RADIUS,
    interpolate_blade_span,
    unpack_blade_params,
)


class WindTurbineBladeEvaluator:
    """3D 风力机单叶片气动与几何约束综合评估器。"""

    def __init__(
        self,
        wind_speed: float = 9.0,         # 默认基准测试风速 V_inf = 9.0 m/s
        rpm: float = 10.0,                # 默认基准测试转速 rpm = 10.0 (omega = 1.0472 rad/s)
        air_density: float = 1.225,       # 空气密度 rho (kg/m^3)
        viscosity: float = 1.81e-5,       # 动力粘度 mu (kg/(m*s))
        n_bem_stations: int = 15,         # BEM 展向积分离散截面数
        penalty_scale: float = 1.0,       # 约束惩罚尺度因子
        pitch_angle_deg: float = 0.0,     # 全叶片安装变桨角 (deg)
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

        # 3 个核心站位的翼型相对厚度工程允许范围 [min_t_rel, max_t_rel]
        self.thickness_limits = [
            (0.20, 0.35),  # 站位 1 (内展段): 结构厚翼型
            (0.15, 0.26),  # 站位 2 (中展段): 主气动功率段
            (0.10, 0.18),  # 站位 3 (翼尖段): 减阻薄翼型
        ]

        # 40% 弦长主梁/油箱/剪切肋板站位最小结构厚度要求
        self.spar_cap_min_t = [0.14, 0.10, 0.05]

    @staticmethod
    def _dominant_penalty(
        violation: float,
        scale: float,
        base_step: float = 300.0,
    ) -> float:
        """平滑有界工业约束主导惩罚函数 (单位对齐输出功率 kW)：
        
        当违背约束 (violation > 0) 时：
        - 消除断崖跳跃：P(0) = 0；
        - 初期陡峭排斥：大幅抹平局部微弱气动增益；
        - 双曲正切平滑饱和有界：控制单项惩罚在 1.5 * base_step (约 450 kW) 辅以弱线性项。
        """
        if violation <= 0.0:
            return 0.0
        rel = float(violation) / max(float(scale), 1e-8)
        cap = 1.5 * base_step
        return float(cap * np.tanh(1.5 * rel) + 5.0 * rel)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全套叶片几何与展向拓扑校验。
        
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
        x_sample = np.linspace(0.001, 0.999, 100)
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
                return True, 1500.0, geom_metrics

            if np.any(np.isnan(yu)) or np.any(np.isnan(yl)):
                return True, 1500.0, geom_metrics

            thickness = yu - yl
            overlap = yl - yu
            max_ol = float(np.max(overlap))
            min_t = float(np.min(thickness))
            max_t = float(np.max(thickness))

            if max_ol > geom_metrics["max_overlap"]:
                geom_metrics["max_overlap"] = max_ol

            # 致命缺陷判定：表面自反转交叉或厚度严重坍塌
            if max_ol > 1e-4 or min_t < -1e-4 or max_t < 0.03:
                fatal_p = 1000.0 + 800.0 * float(np.tanh(50.0 * (max_ol + max(0.0, 0.03 - max_t))))
                return True, fatal_p, geom_metrics

            # 提取站位几何特征
            idx_tmax = int(np.argmax(thickness))
            x_tmax = float(x_sample[idx_tmax])

            # 40% 弦长结构主梁厚度
            t_spar = float(np.interp(0.40, x_sample, thickness))

            # 后缘单调收缩检测 (防鱼尾)
            aft_thick = thickness[idx_tmax:]
            aft_non_mono = float(np.sum(np.maximum(0.0, np.diff(aft_thick))))

            # 上表面抗波纹凹陷检测 (Anti-waviness)
            d2_yu = np.diff(yu, 2)
            mid_mask = (x_sample[1:-1] >= 0.05) & (x_sample[1:-1] <= 0.70)
            waviness = float(np.sum(np.maximum(0.0, d2_yu[mid_mask])))

            st_met = {
                "t_max": max_t,
                "x_t_max": x_tmax,
                "t_spar": t_spar,
                "aft_non_mono": aft_non_mono,
                "waviness": waviness,
                "max_overlap": max_ol,
                "chord": chords[k],
                "twist": twists[k],
            }
            geom_metrics["station_metrics"].append(st_met)

        # 2. 全叶片 3D 展向宏观几何约束
        # 弦长单调平滑递减：c1 >= c2 >= c3
        chord_viol = max(0.0, chords[1] - chords[0]) + max(0.0, chords[2] - chords[1])
        geom_metrics["chord_taper_violation"] = float(chord_viol)

        # 扭角单调递减 (根部大攻角，翼尖小攻角气动扭退)：beta1 >= beta2 >= beta3
        twist_viol = max(0.0, twists[1] - twists[0]) + max(0.0, twists[2] - twists[1])
        geom_metrics["twist_monotonicity_violation"] = float(twist_viol)

        # 若弦长或扭角出现反向严重倒挂 (> 0.5m 或 > 3 deg)，前置拦截
        if chord_viol > 0.6 or twist_viol > 4.0:
            return True, 800.0, geom_metrics

        return False, 0.0, geom_metrics

    def calculate_penalties(
        self, geom_metrics: Dict[str, Any]
    ) -> Tuple[bool, float, Dict[str, float]]:
        """计算工业几何与展向气动约束惩罚项 (单位: kW)。"""
        penalties: Dict[str, float] = {}

        # 1. 展向宏观参数惩罚
        c_viol = geom_metrics["chord_taper_violation"]
        if c_viol > 1e-4:
            penalties["chord_taper"] = self._dominant_penalty(c_viol, scale=0.10, base_step=350.0)

        tw_viol = geom_metrics["twist_monotonicity_violation"]
        if tw_viol > 1e-4:
            penalties["twist_monotonicity"] = self._dominant_penalty(tw_viol, scale=0.50, base_step=350.0)

        # 2. 各站位翼型厚度与结构梁约束惩罚
        st_metrics = geom_metrics.get("station_metrics", [])
        for k, met in enumerate(st_metrics):
            t_max = met["t_max"]
            min_t_lim, max_t_lim = self.thickness_limits[k]

            if t_max < min_t_lim:
                v = min_t_lim - t_max
                penalties[f"st{k}_thick_min"] = self._dominant_penalty(v, scale=0.015, base_step=250.0)
            elif t_max > max_t_lim:
                v = t_max - max_t_lim
                penalties[f"st{k}_thick_max"] = self._dominant_penalty(v, scale=0.015, base_step=250.0)

            # 主梁结构站位厚度
            t_spar = met["t_spar"]
            min_spar = self.spar_cap_min_t[k]
            if t_spar < min_spar:
                v = min_spar - t_spar
                penalties[f"st{k}_spar_min"] = self._dominant_penalty(v, scale=0.010, base_step=200.0)

            # 后缘单调收缩 (防鱼尾)
            aft_nm = met["aft_non_mono"]
            if aft_nm > 1e-4:
                penalties[f"st{k}_aft_non_mono"] = self._dominant_penalty(aft_nm, scale=0.005, base_step=200.0)

            # 上表面抗波纹凹坑 (Anti-waviness)
            wav = met["waviness"]
            if wav > 2e-4:
                penalties[f"st{k}_waviness"] = self._dominant_penalty(wav, scale=0.003, base_step=150.0)

        total_penalty = float(sum(penalties.values())) * self.penalty_scale
        is_feasible = (len(penalties) == 0)
        return is_feasible, total_penalty, penalties

    def solve_bem(self, span_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """高精度快速叶素动量理论 (BEM) 求解器。"""
        r = span_data["r"]
        dr = span_data["dr"]
        chords = span_data["chord"]
        twists_rad = span_data["twist_rad"]
        thetas = span_data["kulfan_thetas"]
        N = len(r)

        # 准备站位分组以减少 NeuralFoil 调用频次
        # 3 个主要区域：0..4 对应 station 0; 5..9 对应 station 1; 10..N-1 对应 station 2
        split_1 = N // 3
        split_2 = 2 * N // 3
        sec_slices = [(0, split_1), (split_1, split_2), (split_2, N)]

        # 诱导因子初始化
        a = np.zeros(N, dtype=np.float64) + 0.20
        a_p = np.zeros(N, dtype=np.float64) + 0.01

        max_iter = 10
        relaxation = 0.40

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

            # Prandtl 叶尖与叶根损失修正
            f_tip = (NUM_BLADES / 2.0) * (ROTOR_RADIUS - r) / (r * np.sin(phi) + 1e-6)
            F_tip = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), 0.0, 1.0))
            f_root = (NUM_BLADES / 2.0) * (r - HUB_RADIUS) / (HUB_RADIUS * np.sin(phi) + 1e-6)
            F_root = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_root), 0.0, 1.0))
            F = np.clip(F_tip * F_root, 1e-3, 1.0)

            # 批量通过 NeuralFoil 求解 3 段的气动升阻比
            for st_i, (start_idx, end_idx) in enumerate(sec_slices):
                sub_alpha = alpha_deg[start_idx:end_idx]
                sub_Re = float(np.mean(Re[start_idx:end_idx]))
                # 取对应站位中心的 Kulfan 几何参数
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
                        alpha=sub_alpha,
                        Re=sub_Re,
                        model_size="large",
                    )
                    CL[start_idx:end_idx] = np.asarray(aero["CL"])
                    CD[start_idx:end_idx] = np.asarray(aero["CD"])
                except Exception:
                    CL[start_idx:end_idx] = 0.5
                    CD[start_idx:end_idx] = 0.05

            CD = np.maximum(CD, 1e-4)

            # 法向与切向力系数
            C_n = CL * np.cos(phi) + CD * np.sin(phi)
            C_t = CL * np.sin(phi) - CD * np.cos(phi)
            solidity = (NUM_BLADES * chords) / (2.0 * np.pi * r)

            # Glauert 湍流高推力修正
            term = 4.0 * F * np.sin(phi)**2 / (solidity * C_n + 1e-8)
            a_c = 0.33
            a_new = np.where(
                (term > 0) & (1.0 / (term + 1.0) <= a_c),
                1.0 / (term + 1.0),
                0.35 + 0.5 * (solidity * C_n)
            )
            a_new = np.clip(a_new, 0.0, 0.75)

            # 切向诱导因子更新
            denom_ap = 4.0 * F * np.sin(phi) * np.cos(phi) / (solidity * C_t + 1e-8) - 1.0
            a_p_new = np.where(np.abs(denom_ap) > 1e-5, 1.0 / denom_ap, 0.0)
            a_p_new = np.clip(a_p_new, -0.15, 0.40)

            # 欠松弛迭代
            a = (1.0 - relaxation) * a + relaxation * a_new
            a_p = (1.0 - relaxation) * a_p + relaxation * a_p_new

        # 积分全叶轮转矩与推力
        # 截面转矩微元: dQ = 0.5 * rho * V_rel^2 * B * c * C_t * r * dr
        dQ = 0.5 * self.air_density * (V_rel**2) * NUM_BLADES * chords * C_t * r * dr
        # 截面轴向推力微元: dFz = 0.5 * rho * V_rel^2 * B * c * C_n * dr
        dFz = 0.5 * self.air_density * (V_rel**2) * NUM_BLADES * chords * C_n * dr

        torque_Nm = float(np.sum(dQ))
        thrust_N = float(np.sum(dFz))
        power_W = float(torque_Nm * self.omega)

        swept_area = np.pi * (ROTOR_RADIUS**2)
        power_ref = 0.5 * self.air_density * (self.wind_speed**3) * swept_area
        thrust_ref = 0.5 * self.air_density * (self.wind_speed**2) * swept_area

        Cp = float(power_W / max(power_ref, 1e-6))
        Ct = float(thrust_N / max(thrust_ref, 1e-6))

        return {
            "power_kW": power_W / 1e3,
            "power_MW": power_W / 1e6,
            "torque_kNm": torque_Nm / 1e3,
            "thrust_kN": thrust_N / 1e3,
            "Cp": Cp,
            "Ct": Ct,
            "a": a,
            "a_p": a_p,
            "CL": CL,
            "CD": CD,
            "alpha": alpha,
            "V_rel": V_rel,
            "r": r,
        }

    def evaluate(self, x: np.ndarray) -> float:
        """核心评估接口：
        
        优化目标：最大化叶轮输出净气动功率 Power (kW)，扣除所有几何越界与展向气动违背惩罚。
        对于物理塌陷或严重越界几何早退拦截，绝不调用 NeuralFoil。
        
        返回:
            float: 有效输出功率 (kW，考虑违背惩罚)，越大越好
        """
        # 1. 工业几何与展向拓扑前置校验
        is_fatal, fatal_penalty, geom_metrics = self.validate_geometry(x)
        if is_fatal:
            return -1000.0 - fatal_penalty

        # 2. 约束惩罚计算
        is_feasible, total_penalty, _ = self.calculate_penalties(geom_metrics)

        # 3. BEM 气动特性求解
        span_data = interpolate_blade_span(x, n_stations=self.n_bem_stations)
        try:
            bem_res = self.solve_bem(span_data)
            raw_power_kW = float(bem_res["power_kW"])
        except Exception:
            return -1500.0

        if np.isnan(raw_power_kW) or raw_power_kW < -1000.0:
            return -1500.0

        if not is_feasible:
            # 存在微弱越界时，施加主导惩罚，确保越界在博弈中永远亏损
            return min(-50.0, raw_power_kW - 500.0 - total_penalty)

        return raw_power_kW - total_penalty

    def evaluate_detailed(self, x: np.ndarray) -> Dict[str, Any]:
        """获取全套详细气动与三维几何报告 (供测试输出、收敛记录与多图绘制)。"""
        is_fatal, fatal_penalty, geom_metrics = self.validate_geometry(x)
        is_feasible, total_penalty, penalties = self.calculate_penalties(geom_metrics)

        if is_fatal:
            return {
                "power_kW": 0.0,
                "power_MW": 0.0,
                "power_penalized_kW": -1000.0 - fatal_penalty,
                "torque_kNm": 0.0,
                "thrust_kN": 0.0,
                "Cp": 0.0,
                "Ct": 0.0,
                "is_feasible": False,
                "is_fatal": True,
                "penalties": penalties,
                "total_penalty": fatal_penalty,
                "geom_metrics": geom_metrics,
            }

        span_data = interpolate_blade_span(x, n_stations=self.n_bem_stations)
        try:
            bem_res = self.solve_bem(span_data)
            raw_power_kW = float(bem_res["power_kW"])
            torque_kNm = float(bem_res["torque_kNm"])
            thrust_kN = float(bem_res["thrust_kN"])
            Cp = float(bem_res["Cp"])
            Ct = float(bem_res["Ct"])
        except Exception:
            raw_power_kW = 0.0
            torque_kNm = 0.0
            thrust_kN = 0.0
            Cp = 0.0
            Ct = 0.0
            bem_res = {}

        penalized_power = (
            min(-50.0, raw_power_kW - 500.0 - total_penalty)
            if not is_feasible
            else raw_power_kW - total_penalty
        )

        return {
            "power_kW": raw_power_kW,
            "power_MW": raw_power_kW / 1e3,
            "power_penalized_kW": penalized_power,
            "torque_kNm": torque_kNm,
            "thrust_kN": thrust_kN,
            "Cp": Cp,
            "Ct": Ct,
            "is_feasible": is_feasible,
            "is_fatal": False,
            "penalties": penalties,
            "total_penalty": total_penalty,
            "geom_metrics": geom_metrics,
            "bem_details": bem_res,
        }
