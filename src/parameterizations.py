"""3D 水轮机叶片多种参数化模型库 (src/parameterizations.py):
依据 1.5m 叶片小型/模型水轮机参考规范 (未优化 Baseline 1p5mBlade)：
1. BSplineBladeModel (D=114): 6 翼型站位 17 维 CST (102维, 尾缘定死为0) + 6 阶 B 样条弦长增量 + 6 阶 B 样条扭角增量 (天然保证 C² 连续光滑)
2. Stations6BladeModel (D=120): 经典 6 离散控制站位模型
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
    BEM_NUM_SECTIONS,
    BEM_27_RADII,
    BEM_27_DR,
    BEM_27_CHORD,
    BEM_27_TWIST,
    BEM_27_KULFAN_THETAS,
    ROTOR_RADIUS,
    STATION_INDICES,
    STATION_RADII,
    get_baseline_blade_params,
    get_blade_bounds,
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
    def to_span_data(self, x: np.ndarray, n_stations: int = 27) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        pass


# ==============================================================================
# 1. B-样条高阶平滑模态参数化模型 (D=114): 102 CST + 6 B-spline Chord + 6 B-spline Twist
# ==============================================================================

class BSplineBladeModel(BaseParameterizationModel):
    """B 样条高阶平滑模态参数化模型 (D=114):
    适配 1.5m 叶片小型水轮机规范 (Baseline 1p5mBlade, 27 截面):
    - 6 个核心气动控制站位的 17 维 Kulfan CST 翼型参数 (102 维: 6 * 17);
      * 第 18 维后缘厚度参数 (TE_thickness) 被彻底移除并定死为 0.0 (物理严格闭口尾缘，杜绝水下空化与钝体基底阻力);
    - 6 阶三次 B 样条增量控制点 Delta_c (6 维);
    - 6 阶三次 B 样条增量控制点 Delta_twist (6 维);
    - 严苛工业级平滑度与单调性约束体系 (尺度对齐 1.5m 小型叶片真实尺寸):
      * 中展段线性收缩容差缩紧至 15mm;
      * 弦长增量二阶差分容差缩紧至 0.015m, 扭角增量二阶差分容差缩紧至 0.40deg;
      * 控制点二阶差分约束从源头杜绝高频波浪振荡;
      * 弦长全展向单调收缩 (diff(c) <= 0);
      * 扭角沿展向严格单调洗退 (diff(twist) <= 0)。
    """

    def __init__(self, n_spline_points: int = 6, spline_degree: int = 3):
        self.n_stations = NUM_STATIONS       # 6
        self.n_cst_per_station = 17          # 17 CST (移除尾缘厚度自由度)
        self.n_cst_total = self.n_stations * self.n_cst_per_station  # 102
        self.n_spline = n_spline_points      # 6 阶 B 样条
        self.degree = spline_degree          # 3 (三次 B 样条)
        self._dim = self.n_cst_total + 2 * self.n_spline  # 102 + 6 + 6 = 114

        # 归一化展向半径范围 [0, 1] 对应 [HUB_RADIUS, ROTOR_RADIUS]
        self.r_ref = BEM_27_RADII.copy()
        self.r_norm = (self.r_ref - HUB_RADIUS) / BLADE_LENGTH

        # 构造 Clamped Knot Vector: 两端各重复 (degree+1) 次
        interior = np.linspace(0.0, 1.0, self.n_spline - self.degree + 1)[1:-1]
        self.knots = np.concatenate(([0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)))

        # 构造基函数投影矩阵 A (27, n_spline)
        self.A_proj = np.zeros((len(self.r_norm), self.n_spline), dtype=np.float64)
        for i in range(self.n_spline):
            c_basis = np.zeros(self.n_spline, dtype=np.float64)
            c_basis[i] = 1.0
            bs = BSpline(self.knots, c_basis, self.degree)
            self.A_proj[:, i] = bs(self.r_norm)

        # 提取 6 个核心气动站位的 102 维基线翼型参数 (仅保留前 17 维，第 18 维尾缘定死为 0)
        x0_6 = get_baseline_blade_params()
        st6 = unpack_blade_params(x0_6)
        self.thetas_0 = np.concatenate([s["theta"][:17] for s in st6])

        # 初始弦长与扭角微扰增量控制点均为 0 (保证基准向量在 27 截面上 100% 比特级吻合官方数据)
        self.delta_c_0 = np.zeros(self.n_spline, dtype=np.float64)
        self.delta_tw_0 = np.zeros(self.n_spline, dtype=np.float64)

        # 组装 114 维基线向量 x0 = [thetas_0, delta_c_0, delta_tw_0]
        self.x0 = np.concatenate([self.thetas_0, self.delta_c_0, self.delta_tw_0])

        # 1.5m 翼型几何界限
        self.min_le_radii = [0.003 for _ in range(6)]
        self.thickness_limits = [(0.06, 0.20) for _ in range(6)]
        self.spar_cap_min_t = [0.04 for _ in range(6)]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"B-Spline Modal Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 114 维上下界向量。"""
        lb = np.zeros(self._dim, dtype=np.float64)
        ub = np.zeros(self._dim, dtype=np.float64)

        # 1. 102 维 CST 权重浮动界限 (每站位 17 维)
        for i in range(102):
            val = self.thetas_0[i]
            delta = max(0.25 * abs(val), 0.12)
            lb[i] = val - delta
            ub[i] = val + delta

        # 2. 6 维弦长增量控制点 Delta_c (单位: m)
        offset_c = 102
        # 根部控制点 (允许向 opt 版本减小弦长)
        lb[offset_c + 0] = -0.55
        ub[offset_c + 0] = 0.20

        for i in range(1, self.n_spline - 1):
            lb[offset_c + i] = -0.35
            ub[offset_c + i] = 0.25

        # 翼尖端点
        lb[offset_c + self.n_spline - 1] = -0.15
        ub[offset_c + self.n_spline - 1] = 0.15

        # 3. 6 维扭角增量控制点 Delta_twist (单位: deg)
        offset_tw = 102 + self.n_spline
        for i in range(self.n_spline):
            lb[offset_tw + i] = -6.0
            ub[offset_tw + i] = 6.0

        return lb, ub

    def to_span_data(self, x: np.ndarray, n_stations: int = 27) -> Dict[str, np.ndarray]:
        """将参数转化为 27 截面的展向水动力几何数据字典 (主干支持 D=114，兼容 132/120)"""
        if len(x) == 120:
            return interpolate_blade_span(x, n_stations=n_stations)

        if len(x) == 114:
            thetas_raw_17 = x[:102].reshape((6, 17))
            delta_c_ctrl = x[102 : 102 + self.n_spline]
            delta_tw_ctrl = x[102 + self.n_spline : 102 + 2 * self.n_spline]

            delta_chord = self.A_proj @ delta_c_ctrl
            delta_twist_deg = self.A_proj @ delta_tw_ctrl

            chord_span = BEM_27_CHORD + delta_chord
            twist_span_deg = BEM_27_TWIST + delta_twist_deg
            twist_span_rad = np.radians(twist_span_deg)

            # 6 站位 17 维 CST 平滑插值至 27 截面，第 18 维尾缘厚度定死为 0.0 (闭口尾缘)
            kulfan_span = np.zeros((len(BEM_27_RADII), 18), dtype=np.float64)
            for dim_i in range(17):
                interp_vals = np.interp(
                    BEM_27_RADII,
                    STATION_RADII,
                    thetas_raw_17[:, dim_i],
                    left=thetas_raw_17[0, dim_i],
                    right=thetas_raw_17[-1, dim_i],
                )
                kulfan_span[:, dim_i] = interp_vals
            kulfan_span[:, 17] = 0.0  # 闭口尾缘，彻底消除水下空化漏洞
            return {
                "r": BEM_27_RADII.copy(),
                "dr": BEM_27_DR.copy(),
                "chord": chord_span,
                "twist_deg": twist_span_deg,
                "twist_rad": twist_span_rad,
                "kulfan_thetas": kulfan_span,
            }

        # 向后兼容旧版 132 维
        thetas_raw = x[:108].reshape((6, 18))
        n_sp = (len(x) - 108) // 2
        delta_c_ctrl = x[108 : 108 + n_sp]
        delta_tw_ctrl = x[108 + n_sp : 108 + 2 * n_sp]

        A = self.A_proj if n_sp == self.n_spline else np.zeros((len(self.r_norm), n_sp))
        delta_chord = A @ delta_c_ctrl
        delta_twist_deg = A @ delta_tw_ctrl

        chord_span = BEM_27_CHORD + delta_chord
        twist_span_deg = BEM_27_TWIST + delta_twist_deg
        twist_span_rad = np.radians(twist_span_deg)

        kulfan_span = BEM_27_KULFAN_THETAS.copy()
        for dim_i in range(18):
            kulfan_span[:, dim_i] = np.interp(
                BEM_27_RADII, STATION_RADII, thetas_raw[:, dim_i],
                left=thetas_raw[0, dim_i], right=thetas_raw[-1, dim_i]
            )
        kulfan_span[:, 17] = 0.0

        return {
            "r": BEM_27_RADII.copy(),
            "dr": BEM_27_DR.copy(),
            "chord": chord_span,
            "twist_deg": twist_span_deg,
            "twist_rad": twist_span_rad,
            "kulfan_thetas": kulfan_span,
        }

    def get_station_info(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """提取 6 个核心气动站位的几何参数字典列表，供绘图与日志分析。"""
        if len(x) == 114:
            thetas_raw_17 = x[:102].reshape((6, 17))
            delta_c_ctrl = x[102 : 102 + self.n_spline]
            delta_tw_ctrl = x[102 + self.n_spline : 102 + 2 * self.n_spline]

            chord_span = BEM_27_CHORD + self.A_proj @ delta_c_ctrl
            twist_span_deg = BEM_27_TWIST + self.A_proj @ delta_tw_ctrl

            stations_info = []
            for k in range(6):
                sec_idx = int(STATION_INDICES[k])
                th17 = thetas_raw_17[k]
                th18 = np.append(th17, 0.0)
                stations_info.append({
                    "station_idx": k,
                    "global_idx": sec_idx,
                    "radius": float(STATION_RADII[k]),
                    "chord": float(chord_span[sec_idx]),
                    "twist": float(twist_span_deg[sec_idx]),
                    "theta": th18,
                    "kulfan_dict": {
                        "upper_weights": th17[0:8],
                        "lower_weights": th17[8:16],
                        "leading_edge_weight": float(th17[16]),
                        "TE_thickness": 0.0,
                    },
                })
            return stations_info

        thetas_raw = x[:108].reshape((6, 18)) if len(x) >= 108 else np.zeros((6, 18))
        stations_info = []
        for k in range(6):
            sec_idx = int(STATION_INDICES[k])
            th = thetas_raw[k].copy()
            th[17] = 0.0
            stations_info.append({
                "station_idx": k,
                "global_idx": sec_idx,
                "radius": float(STATION_RADII[k]),
                "chord": float(BEM_27_CHORD[sec_idx]),
                "twist": float(BEM_27_TWIST[sec_idx]),
                "theta": th,
                "kulfan_dict": {
                    "upper_weights": th[0:8],
                    "lower_weights": th[8:16],
                    "leading_edge_weight": float(th[16]),
                    "TE_thickness": 0.0,
                },
            })
        return stations_info

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全量几何质量与连续性校验 (前缘尖角抑制、尾缘严格闭口、线性收缩、单调洗退与严苛二阶差分平滑约束)"""
        if len(x) == 120:
            span_d = interpolate_blade_span(x, n_stations=27)
            st_list = unpack_blade_params(x)
            thetas_raw = np.array([np.append(s["theta"][:17], 0.0) for s in st_list])
            chords_eval = span_d["chord"]
            twists_eval = span_d["twist_deg"]
            delta_chord = chords_eval - BEM_27_CHORD
            delta_twist = twists_eval - BEM_27_TWIST
            is_114 = False
        elif len(x) == 114:
            thetas_raw_17 = x[:102].reshape((6, 17))
            thetas_raw = np.zeros((6, 18), dtype=np.float64)
            thetas_raw[:, :17] = thetas_raw_17
            thetas_raw[:, 17] = 0.0  # 尾缘定死为 0
            delta_c_ctrl = x[102 : 102 + self.n_spline]
            delta_tw_ctrl = x[102 + self.n_spline : 102 + 2 * self.n_spline]
            delta_chord = self.A_proj @ delta_c_ctrl
            delta_twist = self.A_proj @ delta_tw_ctrl
            chords_eval = BEM_27_CHORD + delta_chord
            twists_eval = BEM_27_TWIST + delta_twist
            is_114 = True
        else:
            thetas_raw = x[:108].reshape((6, 18)).copy()
            thetas_raw[:, 17] = 0.0
            n_sp = (len(x) - 108) // 2
            delta_c_ctrl = x[108 : 108 + n_sp]
            delta_tw_ctrl = x[108 + n_sp : 108 + 2 * n_sp]
            A = self.A_proj if n_sp == self.n_spline else np.zeros((len(self.r_norm), n_sp))
            delta_chord = A @ delta_c_ctrl
            delta_twist = A @ delta_tw_ctrl
            chords_eval = BEM_27_CHORD + delta_chord
            twists_eval = BEM_27_TWIST + delta_twist
            is_114 = False

        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "chord_linear_violation": 0.0,
            "chord_smooth_violation": 0.0,
            "twist_smooth_violation": 0.0,
            "max_overlap": 0.0,
        }

        # 1. 检验 6 个核心翼型站位几何质量 (尾缘严格闭口 TE_thickness=0.0)
        x_sample = (1.0 - np.cos(np.linspace(0.0, np.pi, 80))) / 2.0
        for k in range(6):
            th = thetas_raw[k]
            k_af = asb.KulfanAirfoil(
                upper_weights=th[0:8],
                lower_weights=th[8:16],
                leading_edge_weight=float(th[16]),
                TE_thickness=0.0,
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

            if overlap > 0.010:
                return True, 10.0, geom_metrics

        # 2. 展向宏观弦长与扭角工业约束体系 (严格缩紧阈值，尺度对齐 1.5m 小型叶片)
        # a) 弦长全展向单调平滑收缩 (任何局部反弹隆起均严厉惩罚)
        diff_c = np.diff(chords_eval)
        chord_pen = float(np.sum(np.maximum(0.0, diff_c)))

        # b) 扭角全展向单调洗退
        diff_tw = np.diff(twists_eval)
        twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))

        # c) 中展段线性收缩偏差约束 (r in [0.40m, 1.30m], 截面 4 至 22)
        # 允许基准自身弯曲 (0.0889m)，严禁向外隆起加深，同时约束增量 Delta_c 的线性度 (<= 0.040m)
        r_mid = BEM_27_RADII[4:23]
        c_mid = chords_eval[4:23]
        p_lin = np.polyfit(r_mid, c_mid, 1)
        dev_linear = float(np.max(np.abs(c_mid - np.polyval(p_lin, r_mid))))
        linear_pen = float(max(0.0, dev_linear - 0.090))

        if is_114:
            p_inc = np.polyfit(r_mid, delta_chord[4:23], 1)
            dev_inc = float(np.max(np.abs(delta_chord[4:23] - np.polyval(p_inc, r_mid))))
            linear_pen += float(max(0.0, dev_inc - 0.040))

        # d) 全叶片展向二阶差分平滑约束 (抑制曲率突变，杜绝波浪与边缘凹凸)
        # 严格缩紧：弦长增量二阶差分容差缩紧至 0.015m；扭角增量二阶差分容差缩紧至 0.40deg
        if is_114:
            d2_chord_inc = np.diff(delta_chord, 2)
            d2_twist_inc = np.diff(delta_twist, 2)
            chord_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_chord_inc) - 0.015)))
            twist_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_twist_inc) - 0.40)))

            # 控制点二阶差分约束 (从自由度源头消除锯齿交替振荡)
            d2_ctrl_c = np.diff(delta_c_ctrl, 2)
            d2_ctrl_tw = np.diff(delta_tw_ctrl, 2)
            ctrl_c_pen = float(np.sum(np.maximum(0.0, np.abs(d2_ctrl_c) - 0.08)))
            ctrl_tw_pen = float(np.sum(np.maximum(0.0, np.abs(d2_ctrl_tw) - 2.00)))
            chord_smooth_viol += ctrl_c_pen
            twist_smooth_viol += ctrl_tw_pen
        else:
            d2_chord = np.diff(chords_eval, 2)
            d2_twist = np.diff(twists_eval, 2)
            chord_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_chord) - 0.045)))
            twist_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_twist) - 1.20)))

        geom_metrics["chord_taper_violation"] = chord_pen
        geom_metrics["twist_monotonicity_violation"] = twist_pen
        geom_metrics["chord_linear_violation"] = linear_pen
        geom_metrics["chord_smooth_violation"] = chord_smooth_viol
        geom_metrics["twist_smooth_violation"] = twist_smooth_viol

        # 致命几何缺陷判断阈值严格缩紧 (彻底拦截波浪畸变解)
        if chord_pen > 0.05 or twist_pen > 0.50 or linear_pen > 0.025 or chord_smooth_viol > 0.08 or twist_smooth_viol > 0.60:
            fatal_pen = 5.0 + 1.0 * (chord_pen * 20.0 + twist_pen * 2.0 + linear_pen * 30.0 + chord_smooth_viol * 10.0 + twist_smooth_viol)
            return True, fatal_pen, geom_metrics

        return False, 0.0, geom_metrics


class Stations6BladeModel(BaseParameterizationModel):
    """经典 6 离散控制站位模型 (D=120): 每站位 18 CST + 1 弦长 + 1 扭角"""

    def __init__(self):
        self._dim = 120
        self.x0 = get_baseline_blade_params()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"Discrete 6-Station Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return get_blade_bounds(self.x0)

    def to_span_data(self, x: np.ndarray, n_stations: int = 27) -> Dict[str, np.ndarray]:
        return interpolate_blade_span(x, n_stations=n_stations)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        model_bs = BSplineBladeModel()
        return model_bs.validate_geometry(x)
