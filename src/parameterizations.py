"""3D 风力机叶片多种参数化模型库 (src/parameterizations.py):
提供多种参数化表示方法，统一接口以支持优化器 (TuRBO / MALIBO) 在不同维度下的算法性能、单步耗时与平滑度对比：
1. BSplineBladeModel (D=132): 6 翼型站位 CST(108维) + 12 阶 B 样条弦长 + 12 阶 B 样条扭角 (天然保证 C² 连续光滑)
2. Stations25BladeModel (D=500): 25 离散控制站位 (恰好为 50 截面的一半)，每站位 18 CST + 1 弦长 + 1 扭角
3. Stations6BladeModel (D=120): 原版 6 离散控制站位 (更新 10MW 几何质量与前缘半径约束)
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
    OPENFAST_NUM_SECTIONS,
    OPENFAST_50_DR,
    OPENFAST_50_RADII,
    OPENFAST_50_CHORD,
    OPENFAST_50_TWIST,
    OPENFAST_50_KULFAN_THETAS,
    ROTOR_RADIUS,
    STATION_INDICES,
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
    def to_span_data(self, x: np.ndarray, n_stations: int = 50) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        pass


# ==============================================================================
# 1. B-样条高阶平滑模态参数化模型 (D=132): 108 CST + 12 B-spline Chord + 12 B-spline Twist
# ==============================================================================

class BSplineBladeModel(BaseParameterizationModel):
    """B 样条高阶平滑模态参数化模型 (D=132):
    - 6 个核心气动控制站位的 18 维 Kulfan CST 参数 (108 维)
    - 12 个三次 B 样条弦长控制点 (12 维, 根部法兰固定)
    - 12 个三次 B 样条扭角控制点 (12 维, 根部安装角固定)
    数学上天然具有 C² 连续性，完全消除二次导数正负震荡与波浪感。
    """

    def __init__(self, n_spline_points: int = 12, spline_degree: int = 3):
        self._dim = 6 * 18 + 2 * n_spline_points
        self.n_spline = n_spline_points
        self.degree = spline_degree

        # 归一化展向半径范围 [0, 1] 对应 [HUB_RADIUS, ROTOR_RADIUS]
        self.r_ref = OPENFAST_50_RADII
        self.r_norm = (self.r_ref - HUB_RADIUS) / BLADE_LENGTH

        # 构造 Clamped Knot Vector: 两端各重复 (degree+1) 次
        interior = np.linspace(0.0, 1.0, self.n_spline - self.degree + 1)[1:-1]
        self.knots = np.concatenate(([0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)))

        # 构造基函数投影矩阵 A (50, n_spline)
        self.A_proj = np.zeros((len(self.r_norm), self.n_spline), dtype=np.float64)
        for i in range(self.n_spline):
            c_basis = np.zeros(self.n_spline, dtype=np.float64)
            c_basis[i] = 1.0
            bs = BSpline(self.knots, c_basis, self.degree)
            self.A_proj[:, i] = bs(self.r_norm)

        # 采用两端固定约束的最小二乘拟合基准 B 样条初始控制点 (保证根部与翼尖端点 100% 比特级吻合)
        def _fit_clamped(y_target: np.ndarray) -> np.ndarray:
            y_sub = y_target - self.A_proj[:, 0] * y_target[0] - self.A_proj[:, -1] * y_target[-1]
            c_mid, _, _, _ = np.linalg.lstsq(self.A_proj[:, 1:-1], y_sub, rcond=None)
            return np.concatenate(([y_target[0]], c_mid, [y_target[-1]]))

        self.c_ctrl_0 = _fit_clamped(OPENFAST_50_CHORD)
        self.tw_ctrl_0 = _fit_clamped(OPENFAST_50_TWIST)

        # 提取 6 个核心气动站位的 108 维基线翼型参数
        x0_6 = get_baseline_blade_params()
        st6 = unpack_blade_params(x0_6)
        self.thetas_0 = np.concatenate([s["theta"] for s in st6])

        # 组装 132 维基线向量
        self.x0 = np.concatenate([self.thetas_0, self.c_ctrl_0, self.tw_ctrl_0])

        # 10MW 翼型几何界限 (相对厚度上限与主梁厚度)
        self.min_le_radii = [0.005, 0.008, 0.008, 0.006, 0.006, 0.004]
        self.thickness_limits = [
            (0.08, 0.22), (0.08, 0.22), (0.08, 0.22), (0.08, 0.22), (0.08, 0.22), (0.08, 0.25)
        ]
        self.spar_cap_min_t = [0.06, 0.06, 0.06, 0.06, 0.06, 0.06]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"B-Spline Modal Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 132 维上下界向量。
        
        工业边界约束设计：
        - 根部法兰 (端点 0) 严格锁定为基线尺寸 (c = 5.082m, twist = 17.3606 deg)，确保与风电机组轮毂变桨轴承物理装配对齐；
        - 中内展与主功率段享有充分的平滑变动空间；
        - 翼尖端点维持气动收缩与洗退约束。
        """
        lb = self.x0.copy()
        ub = self.x0.copy()

        # 1. 108 维 CST 权重浮动界限
        for i in range(108):
            val = self.thetas_0[i]
            delta = max(0.25 * abs(val), 0.12)
            lb[i] = val - delta
            ub[i] = val + delta

        # 2. 12 维弦长控制点浮动界限
        offset_c = 108
        # 端点 0: 轮毂根部装配法兰，严格锁死
        lb[offset_c + 0] = self.c_ctrl_0[0]
        ub[offset_c + 0] = self.c_ctrl_0[0]

        for i in range(1, self.n_spline - 1):
            c_val = self.c_ctrl_0[i]
            lb[offset_c + i] = max(0.80, c_val - 2.00)
            ub[offset_c + i] = min(13.20, c_val + 2.00)

        # 端点 -1: 翼尖收缩区
        c_tip = self.c_ctrl_0[-1]
        lb[offset_c + self.n_spline - 1] = max(0.50, c_tip - 0.35)
        ub[offset_c + self.n_spline - 1] = min(1.60, c_tip + 0.35)

        # 3. 12 维扭角控制点浮动界限
        offset_tw = 108 + self.n_spline
        # 端点 0: 轮毂根部安装角，严格锁死
        lb[offset_tw + 0] = self.tw_ctrl_0[0]
        ub[offset_tw + 0] = self.tw_ctrl_0[0]

        for i in range(1, self.n_spline - 1):
            tw_val = self.tw_ctrl_0[i]
            lb[offset_tw + i] = tw_val - 4.50
            ub[offset_tw + i] = tw_val + 4.50

        # 端点 -1: 翼尖洗退角
        tw_tip = self.tw_ctrl_0[-1]
        lb[offset_tw + self.n_spline - 1] = tw_tip - 3.00
        ub[offset_tw + self.n_spline - 1] = tw_tip + 3.00

        return lb, ub

    def to_span_data(self, x: np.ndarray, n_stations: int = 50) -> Dict[str, np.ndarray]:
        """将 132 维参数直接转化为 50 截面的展向气动数据字典 (兼顾 120 维向后兼容)"""
        if len(x) == 120:
            return interpolate_blade_span(x, n_stations=n_stations)

        thetas_raw = x[:108].reshape((6, 18))
        c_ctrl = x[108 : 108 + self.n_spline]
        tw_ctrl = x[108 + self.n_spline : 108 + 2 * self.n_spline]

        # 计算 B 样条在 50 截面的致密取值 (天然保证全局 C² 连续性)
        chord_span = self.A_proj @ c_ctrl
        twist_span_deg = self.A_proj @ tw_ctrl
        twist_span_rad = np.radians(twist_span_deg)

        # 6 个核心气动站位的 CST 参数沿展向平滑插值到 50 截面
        # 注意: 根部圆柱区 (前两截面 AF01, AF02, r <= 6.0m) 保持官方基准圆柱，不被机翼翼型覆盖
        kulfan_span = OPENFAST_50_KULFAN_THETAS.copy()
        for dim_i in range(18):
            interp_vals = np.interp(
                OPENFAST_50_RADII[2:],
                STATION_RADII,
                thetas_raw[:, dim_i],
                left=thetas_raw[0, dim_i],
                right=thetas_raw[-1, dim_i],
            )
            kulfan_span[2:, dim_i] = interp_vals

        return {
            "r": OPENFAST_50_RADII.copy(),
            "dr": OPENFAST_50_DR.copy(),
            "chord": chord_span,
            "twist_deg": twist_span_deg,
            "twist_rad": twist_span_rad,
            "kulfan_thetas": kulfan_span,
        }

    def get_station_info(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """提取 6 个核心气动站位的几何参数字典列表，供绘图与日志分析。"""
        thetas_raw = x[:108].reshape((6, 18))
        c_ctrl = x[108 : 108 + self.n_spline]
        tw_ctrl = x[108 + self.n_spline : 108 + 2 * self.n_spline]

        chord_span = self.A_proj @ c_ctrl
        twist_span_deg = self.A_proj @ tw_ctrl

        stations_info = []
        for k in range(6):
            sec_idx = int(STATION_INDICES[k])
            th = thetas_raw[k]
            stations_info.append({
                "station_idx": k,
                "global_idx": sec_idx,
                "radius": float(STATION_RADII[k]),
                "chord": float(chord_span[sec_idx]),
                "twist": float(twist_span_deg[sec_idx]),
                "theta": th,
                "kulfan_dict": {
                    "upper_weights": th[0:8],
                    "lower_weights": th[8:16],
                    "leading_edge_weight": float(th[16]),
                    "TE_thickness": float(th[17]),
                },
            })
        return stations_info

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全量几何质量与连续性校验 (含前缘尖角抑制与单调性检测，支持 132 维与 120 维向后兼容)"""
        if len(x) == 120:
            st_list = unpack_blade_params(x)
            thetas_raw = np.array([s["theta"] for s in st_list])
            chords_eval = np.array([s["chord"] for s in st_list])
            twists_eval = np.array([s["twist"] for s in st_list])
            is_120 = True
        else:
            thetas_raw = x[:108].reshape((6, 18))
            c_ctrl = x[108 : 108 + self.n_spline]
            tw_ctrl = x[108 + self.n_spline : 108 + 2 * self.n_spline]
            chords_eval = self.A_proj @ c_ctrl
            twists_eval = self.A_proj @ tw_ctrl
            is_120 = False

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

            if overlap > 0.015:
                return True, 45000.0, geom_metrics

        # 2. 展向宏观弦长与扭角收缩单调性检测
        if is_120:
            diff_c = np.diff(chords_eval[1:])  # 从肩部站位 (idx 1) 开始
            chord_pen = float(np.sum(np.maximum(0.0, diff_c)))
            diff_tw = np.diff(twists_eval)
            twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))
        else:
            # 基于 50 截面取样
            idx_shoulder = 8
            diff_c = np.diff(chords_eval[idx_shoulder:])
            chord_pen = float(np.sum(np.maximum(0.0, diff_c)))
            diff_tw = np.diff(twists_eval)
            twist_pen = float(np.sum(np.maximum(0.0, diff_tw - 0.25)))

        geom_metrics["chord_taper_violation"] = chord_pen
        geom_metrics["twist_monotonicity_violation"] = twist_pen
        geom_metrics["chord_smooth_violation"] = 0.0
        geom_metrics["twist_smooth_violation"] = 0.0

        if chord_pen > 2.5 or twist_pen > 8.0:
            return True, 30000.0 + 2000.0 * (chord_pen + twist_pen), geom_metrics

        return False, 0.0, geom_metrics


# ==============================================================================
# 2. 25 截面直接高维离散控制模型 (D=500): 25 stations * (18 CST + 1 Chord + 1 Twist)
# (恰好为 50 截面的一半，对齐用户探索硬件极限与高维控制的意图)
# ==============================================================================

class Stations25BladeModel(BaseParameterizationModel):
    """25 截面直接离散控制模型 (D=500):
    选取 50 截面中的每隔一个微元作为控制站位 (共 25 个站位)。
    每个站位独立优化 18 维 CST 翼型 + 局部弦长 c + 局部扭角 beta。
    """

    def __init__(self):
        self._dim = 25 * 20
        # 选取 50 截面中的每隔一个截面作为 25 控制站位
        self.indices_25 = np.linspace(0, 49, 25, dtype=int)
        self.radii_25 = OPENFAST_50_RADII[self.indices_25]

        # 基于基线参数生成 25 站位的初始向量 x0 (500 维)
        x0_6 = get_baseline_blade_params()
        span0 = interpolate_blade_span(x0_6, n_stations=50)
        thetas_50 = span0["kulfan_thetas"]
        chords_50 = span0["chord"]
        twists_50 = span0["twist_deg"]

        x0_list = []
        for idx in self.indices_25:
            th_k = thetas_50[idx]
            c_k = chords_50[idx]
            tw_k = twists_50[idx]
            x0_list.append(np.concatenate([th_k, [c_k], [tw_k]]))

        self.x0 = np.concatenate(x0_list)
        self.min_le_radii_50 = np.linspace(0.008, 0.003, 25)
        self.thickness_limits_25 = [
            (0.08, 1.05 if r <= 6.5 else (0.22 if r < 80.0 else 0.25))
            for r in self.radii_25
        ]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"Discrete 25-Station Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 500 维上下界向量"""
        lb = self.x0.copy()
        ub = self.x0.copy()

        for k in range(25):
            idx_start = k * 20
            # 18 维 CST
            for i in range(18):
                val = self.x0[idx_start + i]
                delta = max(0.20 * abs(val), 0.08)
                lb[idx_start + i] = val - delta
                ub[idx_start + i] = val + delta
            # 弦长
            c_val = self.x0[idx_start + 18]
            lb[idx_start + 18] = max(0.50, c_val - 1.50)
            ub[idx_start + 18] = min(13.50, c_val + 1.50)
            # 扭角
            tw_val = self.x0[idx_start + 19]
            lb[idx_start + 19] = tw_val - 3.50
            ub[idx_start + 19] = tw_val + 3.50

        return lb, ub

    def to_span_data(self, x: np.ndarray, n_stations: int = 50) -> Dict[str, np.ndarray]:
        """将 25 站位 500 维参数平滑插值拓展到 50 截面"""
        mat = x.reshape((25, 20))
        thetas_25 = mat[:, :18]
        chords_25 = mat[:, 18]
        twists_25 = mat[:, 19]

        r_50 = OPENFAST_50_RADII
        chord_span = np.interp(r_50, self.radii_25, chords_25)
        twist_span_deg = np.interp(r_50, self.radii_25, twists_25)
        twist_span_rad = np.radians(twist_span_deg)

        kulfan_span = np.zeros((n_stations, 18), dtype=np.float64)
        for dim_i in range(18):
            kulfan_span[:, dim_i] = np.interp(r_50, self.radii_25, thetas_25[:, dim_i])

        return {
            "r": r_50.copy(),
            "dr": OPENFAST_50_DR.copy(),
            "chord": chord_span,
            "twist_deg": twist_span_deg,
            "twist_rad": twist_span_rad,
            "kulfan_thetas": kulfan_span,
        }

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全套 500 维几何校验与二阶差分平滑约束"""
        mat = x.reshape((25, 20))
        thetas_25 = mat[:, :18]
        chords_25 = mat[:, 18]
        twists_25 = mat[:, 19]

        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "max_overlap": 0.0,
        }

        # 抽样检测 6 个关键气动站位的翼型质量 (避开根部圆柱段)
        sample_k = [3, 7, 11, 15, 19, 23]
        x_sample = (1.0 - np.cos(np.linspace(0.0, np.pi, 60))) / 2.0

        for enum_idx, k in enumerate(sample_k):
            th = thetas_25[k]
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

        # 弦长肩部后单调收缩 (肩部约在 index 4: r=19.5m)
        diff_c = np.diff(chords_25[4:])
        chord_pen = float(np.sum(np.maximum(0.0, diff_c)))
        geom_metrics["chord_taper_violation"] = chord_pen

        # 扭角单调递减气动洗退
        diff_tw = np.diff(twists_25)
        twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))
        geom_metrics["twist_monotonicity_violation"] = twist_pen

        # 25 截面独立控制容易发生局部锯齿与波浪，施加二阶差分平滑惩罚
        d2_chord = np.diff(chords_25, 2)
        d2_twist = np.diff(twists_25, 2)
        chord_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_chord) - 2.0)))
        twist_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_twist) - 1.6)))

        geom_metrics["chord_smooth_violation"] = chord_smooth_viol
        geom_metrics["twist_smooth_violation"] = twist_smooth_viol

        if chord_pen > 2.5 or twist_pen > 8.0 or chord_smooth_viol > 5.0 or twist_smooth_viol > 10.0:
            return True, 30000.0 + 1000.0 * (chord_pen + twist_pen + chord_smooth_viol + twist_smooth_viol), geom_metrics

        return False, 0.0, geom_metrics


# 向后兼容别名
Stations29BladeModel = Stations25BladeModel


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

    def to_span_data(self, x: np.ndarray, n_stations: int = 50) -> Dict[str, np.ndarray]:
        return interpolate_blade_span(x, n_stations=n_stations)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        from src.evaluator import WindTurbineBladeEvaluator
        ev = WindTurbineBladeEvaluator(wind_speed=16.26, rpm=7.55, pitch_angle_deg=0.0)
        return ev.validate_geometry(x)
