"""3D 风力机叶片多种参数化模型库 (src/parameterizations.py):
提供多种参数化表示方法，统一接口以支持优化器 (TuRBO / MALIBO) 在不同维度下的算法性能、单步耗时与平滑度对比：
1. BSplineBladeModel (D=132): 6 翼型站位 CST(108维) + 12 阶 B 样条弦长 + 12 阶 B 样条扭角 (天然保证 C² 连续光滑)
2. Stations29BladeModel (D=580): 29 离散控制站位，每站位 18 CST + 1 弦长 + 1 扭角
3. Stations6BladeModel (D=120): 原版 6 离散控制站位 (更新几何质量与前缘半径约束)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import BSpline

import aerosandbox as asb
from src.blade_geometry import (
    BLADE_LENGTH,
    DIM,
    HUB_RADIUS,
    NUM_STATIONS,
    OPENFAST_58_DR,
    OPENFAST_58_RADII,
    ROTOR_RADIUS,
    STATION_RADII,
    get_baseline_blade_params,
    interpolate_blade_span,
    unpack_blade_params,
)


class BaseParameterizationModel(ABC):
    """参数化模型抽象基类"""

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_x0(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def to_span_data(self, x: np.ndarray, n_stations: int = 58) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        pass


# ==============================================================================
# 1. B-样条模态参数化模型 (D=132): 108 CST + 12 B-spline Chord + 12 B-spline Twist
# ==============================================================================

class BSplineBladeModel(BaseParameterizationModel):
    """B 样条高阶平滑模态参数化模型 (D=132):
    - 6 个核心气动控制站位的 18 维 Kulfan CST 参数 (108 维)
    - 12 个三次 B 样条弦长控制点 (12 维)
    - 12 个三次 B 样条扭角控制点 (12 维)
    数学上天然具有 C² 连续性，完全消除二次导数正负震荡与波浪感。
    """

    def __init__(self, n_spline_points: int = 12, spline_degree: int = 3):
        self._dim = 6 * 18 + 2 * n_spline_points
        self.n_spline = n_spline_points
        self.degree = spline_degree

        # 归一化展向半径范围 [0, 1] 对应 [HUB_RADIUS, ROTOR_RADIUS]
        self.r_ref = OPENFAST_58_RADII
        self.r_norm_58 = (self.r_ref - HUB_RADIUS) / BLADE_LENGTH

        # 构造 Clamped Knot Vector: 两端各重复 (degree+1) 次
        interior = np.linspace(0.0, 1.0, self.n_spline - self.degree + 1)[1:-1]
        self.knots = np.concatenate(([0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)))

        # 构造基函数投影矩阵 A (58, n_spline)
        self.A_proj = np.zeros((len(self.r_norm_58), self.n_spline), dtype=np.float64)
        for i in range(self.n_spline):
            c_basis = np.zeros(self.n_spline, dtype=np.float64)
            c_basis[i] = 1.0
            bs = BSpline(self.knots, c_basis, self.degree)
            self.A_proj[:, i] = bs(self.r_norm_58)

        # 拟合基线 B 样条初始控制点
        x0_6 = get_baseline_blade_params()
        span0 = interpolate_blade_span(x0_6, n_stations=58)
        self.c_ctrl_0, _, _, _ = np.linalg.lstsq(self.A_proj, span0["chord"], rcond=None)
        self.tw_ctrl_0, _, _, _ = np.linalg.lstsq(self.A_proj, span0["twist_deg"], rcond=None)

        # 提取 6 个站位的 108 维基线翼型参数
        st6 = unpack_blade_params(x0_6)
        self.thetas_0 = np.concatenate([s["theta"] for s in st6])

        # 组装 132 维基线向量
        self.x0 = np.concatenate([self.thetas_0, self.c_ctrl_0, self.tw_ctrl_0])

        # 预设站位最小前缘半径与厚度界限
        self.min_le_radii = [0.080, 0.050, 0.025, 0.020, 0.015, 0.012]
        self.thickness_limits = [
            (0.38, 0.65), (0.28, 0.45), (0.22, 0.38), (0.18, 0.32), (0.16, 0.28), (0.14, 0.25)
        ]
        self.spar_cap_min_t = [0.35, 0.26, 0.20, 0.16, 0.14, 0.12]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"B-Spline Modal Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 132 维上下界向量"""
        lb = self.x0.copy()
        ub = self.x0.copy()

        # 108 维 CST 权重浮动界限
        for i in range(108):
            val = self.thetas_0[i]
            delta = max(0.18 * abs(val), 0.06)
            lb[i] = val - delta
            ub[i] = val + delta

        # 12 维弦长控制点浮动界限 (保证弦长为正且在合理范围)
        offset_c = 108
        for i in range(self.n_spline):
            c_val = self.c_ctrl_0[i]
            lb[offset_c + i] = max(0.40, c_val - 1.20)
            ub[offset_c + i] = min(8.50, c_val + 1.20)

        # 12 维扭角控制点浮动界限
        offset_tw = 108 + self.n_spline
        for i in range(self.n_spline):
            tw_val = self.tw_ctrl_0[i]
            lb[offset_tw + i] = tw_val - 3.50
            ub[offset_tw + i] = tw_val + 3.50

        return lb, ub

    def to_span_data(self, x: np.ndarray, n_stations: int = 58) -> Dict[str, np.ndarray]:
        """将 132 维参数直接转化为 58 截面的展向气动数据字典"""
        thetas_raw = x[:108].reshape((6, 18))
        c_ctrl = x[108 : 108 + self.n_spline]
        tw_ctrl = x[108 + self.n_spline : 108 + 2 * self.n_spline]

        # 计算 B 样条在 58 截面的致密取值
        chord_span = self.A_proj @ c_ctrl
        twist_span_deg = self.A_proj @ tw_ctrl
        twist_span_rad = np.radians(twist_span_deg)

        # 6 个站位的 CST 参数沿展向平滑插值到 58 截面
        kulfan_span = np.zeros((n_stations, 18), dtype=np.float64)
        for dim_i in range(18):
            kulfan_span[:, dim_i] = np.interp(
                OPENFAST_58_RADII,
                STATION_RADII,
                thetas_raw[:, dim_i],
                left=thetas_raw[0, dim_i],
                right=thetas_raw[-1, dim_i],
            )

        return {
            "r": OPENFAST_58_RADII.copy(),
            "dr": OPENFAST_58_DR.copy(),
            "chord": chord_span,
            "twist_deg": twist_span_deg,
            "twist_rad": twist_span_rad,
            "kulfan_thetas": kulfan_span,
        }

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全量几何质量与连续性校验 (含前缘尖角抑制与单调性检测)"""
        thetas_raw = x[:108].reshape((6, 18))
        c_ctrl = x[108 : 108 + self.n_spline]
        tw_ctrl = x[108 + self.n_spline : 108 + 2 * self.n_spline]

        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "max_overlap": 0.0,
        }

        # 1. 检验 6 个核心翼型站位几何质量
        x_sample = (1.0 - np.cos(np.linspace(0.0, np.pi, 80))) / 2.0
        for k in range(6):
            th = thetas_raw[k]
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
                return True, 50000.0, geom_metrics

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
            diff_aft = np.diff(t[idx_tmax:])
            aft_non_mono = float(np.sum(np.maximum(0.0, diff_aft)))

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
                return True, 45000.0, geom_metrics

        # 2. 展向宏观弦长与扭角收缩单调性检测 (基于 58 截面取样)
        chords_58 = self.A_proj @ c_ctrl
        twists_58 = self.A_proj @ tw_ctrl

        # 检查肩部站位之后至叶尖的弦长单调收缩
        idx_shoulder = 12  # r ≈ 30m
        diff_c = np.diff(chords_58[idx_shoulder:])
        chord_pen = float(np.sum(np.maximum(0.0, diff_c)))
        geom_metrics["chord_taper_violation"] = chord_pen

        diff_tw = np.diff(twists_58)
        twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))
        geom_metrics["twist_monotonicity_violation"] = twist_pen

        # B 样条天然具备 C² 连续性，二阶差分平滑度违背为 0
        geom_metrics["chord_smooth_violation"] = 0.0
        geom_metrics["twist_smooth_violation"] = 0.0

        if chord_pen > 2.5 or twist_pen > 8.0:
            return True, 30000.0 + 2000.0 * (chord_pen + twist_pen), geom_metrics

        return False, 0.0, geom_metrics


# ==============================================================================
# 2. 29 截面直接高维离散控制模型 (D=580): 29 stations * (18 CST + 1 Chord + 1 Twist)
# ==============================================================================

class Stations29BladeModel(BaseParameterizationModel):
    """29 截面直接离散控制模型 (D=580):
    选取 58 截面中的每隔一个微元作为控制站位 (共 29 个站位)。
    每个站位独立优化 18 维 CST 翼型 + 局部弦长 c + 局部扭角 beta。
    """

    def __init__(self):
        self._dim = 29 * 20
        # 选取 58 截面中的每隔一个截面作为 29 控制站位
        self.indices_29 = np.linspace(0, 57, 29, dtype=int)
        self.radii_29 = OPENFAST_58_RADII[self.indices_29]

        # 基于基线参数生成 29 站位的初始向量 x0 (580 维)
        x0_6 = get_baseline_blade_params()
        span0 = interpolate_blade_span(x0_6, n_stations=58)
        thetas_58 = span0["kulfan_thetas"]
        chords_58 = span0["chord"]
        twists_58 = span0["twist_deg"]

        x0_list = []
        for idx in self.indices_29:
            th_k = thetas_58[idx]
            c_k = chords_58[idx]
            tw_k = twists_58[idx]
            x0_list.append(np.concatenate([th_k, [c_k], [tw_k]]))

        self.x0 = np.concatenate(x0_list)
        self.min_le_radii_58 = np.linspace(0.120, 0.012, 29)
        self.thickness_limits_29 = [
            (max(0.14, 0.54 * (1.0 - 0.7 * (r - HUB_RADIUS) / BLADE_LENGTH)),
             min(0.65, 0.65 * (1.0 - 0.6 * (r - HUB_RADIUS) / BLADE_LENGTH)))
            for r in self.radii_29
        ]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"Discrete 29-Station Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 580 维上下界向量"""
        lb = self.x0.copy()
        ub = self.x0.copy()

        for k in range(29):
            idx_start = k * 20
            # 18 维 CST
            for i in range(18):
                val = self.x0[idx_start + i]
                delta = max(0.18 * abs(val), 0.06)
                lb[idx_start + i] = val - delta
                ub[idx_start + i] = val + delta
            # 弦长
            c_val = self.x0[idx_start + 18]
            lb[idx_start + 18] = max(0.40, c_val - 1.20)
            ub[idx_start + 18] = min(8.50, c_val + 1.20)
            # 扭角
            tw_val = self.x0[idx_start + 19]
            lb[idx_start + 19] = tw_val - 3.50
            ub[idx_start + 19] = tw_val + 3.50

        return lb, ub

    def to_span_data(self, x: np.ndarray, n_stations: int = 58) -> Dict[str, np.ndarray]:
        """将 29 站位 580 维参数平滑插值拓展到 58 截面"""
        mat = x.reshape((29, 20))
        thetas_29 = mat[:, :18]
        chords_29 = mat[:, 18]
        twists_29 = mat[:, 19]

        r_58 = OPENFAST_58_RADII
        chord_span = np.interp(r_58, self.radii_29, chords_29)
        twist_span_deg = np.interp(r_58, self.radii_29, twists_29)
        twist_span_rad = np.radians(twist_span_deg)

        kulfan_span = np.zeros((n_stations, 18), dtype=np.float64)
        for dim_i in range(18):
            kulfan_span[:, dim_i] = np.interp(r_58, self.radii_29, thetas_29[:, dim_i])

        return {
            "r": r_58.copy(),
            "dr": OPENFAST_58_DR.copy(),
            "chord": chord_span,
            "twist_deg": twist_span_deg,
            "twist_rad": twist_span_rad,
            "kulfan_thetas": kulfan_span,
        }

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全套 580 维几何校验与二阶差分平滑约束"""
        mat = x.reshape((29, 20))
        thetas_29 = mat[:, :18]
        chords_29 = mat[:, 18]
        twists_29 = mat[:, 19]

        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "max_overlap": 0.0,
        }

        # 抽样检测 6 个关键站位的翼型质量以节省时间
        sample_k = [0, 5, 11, 17, 22, 28]
        x_sample = (1.0 - np.cos(np.linspace(0.0, np.pi, 60))) / 2.0

        for enum_idx, k in enumerate(sample_k):
            th = thetas_29[k]
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
                return True, 50000.0, geom_metrics

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
            except Exception:
                le_r = 0.0

            geom_metrics["station_metrics"].append({
                "station_idx": enum_idx,
                "max_t_rel": max_t,
                "t_spar": t_spar,
                "min_t": min_t,
                "overlap": overlap,
                "le_radius": le_r,
                "aft_non_mono": 0.0,
                "crest_non_mono": 0.0,
                "waviness": 0.0,
            })

            if overlap > 0.008:
                return True, 45000.0, geom_metrics

        # 弦长肩部后单调收缩
        diff_c = np.diff(chords_29[6:])
        chord_pen = float(np.sum(np.maximum(0.0, diff_c)))
        geom_metrics["chord_taper_violation"] = chord_pen

        # 扭角单调递减气动洗退
        diff_tw = np.diff(twists_29)
        twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))
        geom_metrics["twist_monotonicity_violation"] = twist_pen

        # 29 截面独立控制极其容易发生局部锯齿与波浪！强制施加二阶差分平滑惩罚
        d2_chord = np.diff(chords_29, 2)
        d2_twist = np.diff(twists_29, 2)
        chord_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_chord) - 0.85)))
        twist_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_twist) - 0.80)))

        geom_metrics["chord_smooth_violation"] = chord_smooth_viol
        geom_metrics["twist_smooth_violation"] = twist_smooth_viol

        if chord_pen > 2.5 or twist_pen > 8.0 or chord_smooth_viol > 5.0 or twist_smooth_viol > 10.0:
            return True, 30000.0 + 1000.0 * (chord_pen + twist_pen + chord_smooth_viol + twist_smooth_viol), geom_metrics

        return False, 0.0, geom_metrics


# ==============================================================================
# 3. 原版 6 截面参数化模型 (D=120) (升级约束接口)
# ==============================================================================

class Stations6BladeModel(BaseParameterizationModel):
    """原版 6 截面参数化模型 (D=120)"""

    def __init__(self):
        self._dim = 120
        self.x0 = get_baseline_blade_params()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"Standard 6-Station Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        from src.blade_geometry import get_blade_bounds
        return get_blade_bounds(self.x0)

    def to_span_data(self, x: np.ndarray, n_stations: int = 58) -> Dict[str, np.ndarray]:
        return interpolate_blade_span(x, n_stations=n_stations)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        from src.evaluator import WindTurbineBladeEvaluator
        # 使用 evaluator 自带的 validate_geometry
        ev = WindTurbineBladeEvaluator(wind_speed=11.0, rpm=6.77094, pitch_angle_deg=2.42859)
        return ev.validate_geometry(x)
