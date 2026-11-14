"""3D 风力机叶片高维参数化模型库 (src/parameterizations.py):
严格对齐 NREL Phase VI (Sequence H) 气动主体规范 (27 截面 S809 升力段, r in [1.2575, 5.0290] m):
1. BSplineBladeModel (D=114): 6 翼型站位 17 维闭口尾缘 CST (102 维, TE_thickness=0)
   + 6 阶 B 样条弦长增量 (6 维) + 6 阶 B 样条扭角增量 (6 维)
2. Stations6BladeModel (D=120): 6 离散控制站位对比基线模型
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.interpolate import BSpline

import aerosandbox as asb
from src.blade_geometry import (
    BLADE_LENGTH,
    BLADE_ROOT_RADIUS,
    DIM,
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
    def to_span_data(self, x: np.ndarray, n_stations: int = BEM_NUM_SECTIONS) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        pass


class BSplineBladeModel(BaseParameterizationModel):
    """B 样条高阶平滑模态参数化模型 (D=114):
    适配 NREL Phase VI (S809 气动段 r in [1.2575, 5.0290] m, 27 截面):
    - 6 个核心气动控制站位的 17 维 Kulfan CST 翼型参数 (102 维: 6 * 17, TE_thickness=0);
    - 6 阶三次 B 样条弦长增量控制点 Delta_c (6 维, m);
    - 6 阶三次 B 样条扭角增量控制点 Delta_twist (6 维, deg);
    - 物理约束校准:
      * 全展向弦长单调非增收缩 (diff(c) <= 0), 最小弦长 >= 0.15 m;
      * 全展向扭角单调洗退 (diff(twist) <= 0);
      * 中展段 [4:23] 线性偏离与二阶平滑性检查 (允许平滑弧线平面但阻止局部锯齿/隆起);
      * S809 相对厚度 t/c = 21.07%, 约束区间兼顾结构强度与气动减阻自由度。
    """

    def __init__(self, n_spline_points: int = 6, spline_degree: int = 3):
        self.n_stations = NUM_STATIONS
        self.n_cst_per_station = 17
        self.n_cst_total = self.n_stations * self.n_cst_per_station  # 102
        self.n_spline = n_spline_points
        self.degree = spline_degree
        self._dim = self.n_cst_total + 2 * self.n_spline  # 114

        # 归一化展向半径范围 [0, 1] 精确对应气动段 [BLADE_ROOT_RADIUS, ROTOR_RADIUS]
        self.r_ref = BEM_27_RADII.copy()
        self.r_norm = (self.r_ref - BLADE_ROOT_RADIUS) / BLADE_LENGTH

        # Clamped Uniform Cubic B-Spline Knot Vector
        interior = np.linspace(0.0, 1.0, self.n_spline - self.degree + 1)[1:-1]
        self.knots = np.concatenate(([0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)))

        self.A_proj = np.zeros((len(self.r_norm), self.n_spline), dtype=np.float64)
        for i in range(self.n_spline):
            c_basis = np.zeros(self.n_spline, dtype=np.float64)
            c_basis[i] = 1.0
            bs = BSpline(self.knots, c_basis, self.degree)
            self.A_proj[:, i] = bs(self.r_norm)

        # 提取 6 个核心站位的 102 维基准 S809 翼型参数 (移除第 18 维尾缘厚度)
        x0_6 = get_baseline_blade_params()
        st6 = unpack_blade_params(x0_6)
        self.thetas_0 = np.concatenate([s["theta"][:17] for s in st6])

        self.delta_c_0 = np.zeros(self.n_spline, dtype=np.float64)
        self.delta_tw_0 = np.zeros(self.n_spline, dtype=np.float64)
        self.x0 = np.concatenate([self.thetas_0, self.delta_c_0, self.delta_tw_0])

        # NREL Phase VI (S809 baseline max_t=0.2107) 结构与气动厚度边界
        self.min_le_radii = [0.003 for _ in range(NUM_STATIONS)]
        self.thickness_limits = [
            (0.12, 0.28),
            (0.10, 0.26),
            (0.09, 0.25),
            (0.08, 0.25),
            (0.08, 0.24),
            (0.07, 0.24),
        ]
        self.spar_cap_min_t = [0.05 for _ in range(NUM_STATIONS)]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"B-Spline Modal Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 114 维 NREL Phase VI 设计参数上下界。"""
        lb = np.zeros(self._dim, dtype=np.float64)
        ub = np.zeros(self._dim, dtype=np.float64)

        # 1. 102 维 CST 权重探索空间 (17 params/station * 6)
        for i in range(self.n_cst_total):
            val = float(self.thetas_0[i])
            if (i % 17) == 16:  # LE weight
                delta = max(0.30 * abs(val), 0.10)
            else:
                delta = max(0.30 * abs(val), 0.14)
            lb[i] = val - delta
            ub[i] = val + delta

        # 2. 6 维弦长增量控制点 Delta_c (单位: m; 基准线性锥度 0.737m -> 0.356m)
        offset_c = self.n_cst_total
        lb[offset_c + 0], ub[offset_c + 0] = -0.22, 0.25
        for i in range(1, self.n_spline - 1):
            lb[offset_c + i], ub[offset_c + i] = -0.20, 0.22
        lb[offset_c + self.n_spline - 1], ub[offset_c + self.n_spline - 1] = -0.14, 0.16

        # 3. 6 维扭角增量控制点 Delta_twist (单位: deg; 涵盖 Kaya 2019 根部大幅降扭区)
        offset_tw = self.n_cst_total + self.n_spline
        twist_delta_bounds = [
            (-18.0, 8.0),
            (-15.0, 8.0),
            (-12.0, 10.0),
            (-12.0, 10.0),
            (-10.0, 10.0),
            (-10.0, 10.0),
        ]
        for i in range(self.n_spline):
            lb[offset_tw + i], ub[offset_tw + i] = twist_delta_bounds[i]

        return lb, ub

    def _compute_span_geom(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        thetas_raw_17 = x[: self.n_cst_total].reshape((NUM_STATIONS, 17))
        thetas_raw = np.zeros((NUM_STATIONS, 18), dtype=np.float64)
        thetas_raw[:, :17] = thetas_raw_17
        thetas_raw[:, 17] = 0.0

        delta_c_ctrl = x[self.n_cst_total : self.n_cst_total + self.n_spline]
        delta_tw_ctrl = x[self.n_cst_total + self.n_spline : self._dim]
        delta_chord = self.A_proj @ delta_c_ctrl
        delta_twist = self.A_proj @ delta_tw_ctrl
        chords = BEM_27_CHORD + delta_chord
        twists = BEM_27_TWIST + delta_twist
        return thetas_raw, delta_c_ctrl, delta_tw_ctrl, delta_chord, delta_twist, np.vstack([chords, twists])

    def to_span_data(self, x: np.ndarray, n_stations: int = BEM_NUM_SECTIONS) -> Dict[str, np.ndarray]:
        """将 114 维 (或兼容 120 维) 参数转化为展向 27 截面气动数据字典。"""
        if len(x) == 120:
            return interpolate_blade_span(x, n_stations=n_stations)
        if len(x) != self._dim:
            raise ValueError(f"Expected vector of dimension {self._dim} or 120, got {len(x)}")

        thetas_raw, _, _, _, _, ct = self._compute_span_geom(x)
        chord_span = ct[0]
        twist_span_deg = ct[1]

        kulfan_27 = np.zeros((len(BEM_27_RADII), 18), dtype=np.float64)
        for dim_i in range(17):
            kulfan_27[:, dim_i] = np.interp(
                BEM_27_RADII,
                STATION_RADII,
                thetas_raw[:, dim_i],
                left=thetas_raw[0, dim_i],
                right=thetas_raw[-1, dim_i],
            )
        kulfan_27[:, 17] = 0.0

        if n_stations == BEM_NUM_SECTIONS:
            return {
                "r": BEM_27_RADII.copy(),
                "dr": BEM_27_DR.copy(),
                "chord": chord_span,
                "twist_deg": twist_span_deg,
                "twist_rad": np.radians(twist_span_deg),
                "kulfan_thetas": kulfan_27,
            }

        r = np.linspace(BLADE_ROOT_RADIUS, ROTOR_RADIUS, n_stations)
        bnds = np.zeros(n_stations + 1, dtype=np.float64)
        bnds[0] = r[0]
        bnds[1:-1] = 0.5 * (r[:-1] + r[1:])
        bnds[-1] = r[-1]
        dr = np.diff(bnds)
        chord_n = np.interp(r, BEM_27_RADII, chord_span)
        twist_n = np.interp(r, BEM_27_RADII, twist_span_deg)
        kulfan_n = np.zeros((n_stations, 18), dtype=np.float64)
        for dim_i in range(17):
            kulfan_n[:, dim_i] = np.interp(r, BEM_27_RADII, kulfan_27[:, dim_i])
        return {
            "r": r,
            "dr": dr,
            "chord": chord_n,
            "twist_deg": twist_n,
            "twist_rad": np.radians(twist_n),
            "kulfan_thetas": kulfan_n,
        }

    def get_station_info(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """提取 6 个核心气动控制站位的几何参数字典列表。"""
        if len(x) == 120:
            return unpack_blade_params(x)
        thetas_raw, _, _, _, _, ct = self._compute_span_geom(x)
        chord_span, twist_span_deg = ct[0], ct[1]

        stations_info = []
        for k in range(NUM_STATIONS):
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
                    "TE_thickness": 0.0,
                },
            })
        return stations_info

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """全量几何质量与展向平滑连续性校验。"""
        if len(x) == 120:
            span_d = interpolate_blade_span(x, n_stations=BEM_NUM_SECTIONS)
            st_list = unpack_blade_params(x)
            thetas_raw = np.array([np.append(s["theta"][:17], 0.0) for s in st_list])
            chords_eval = span_d["chord"]
            twists_eval = span_d["twist_deg"]
            delta_chord = chords_eval - BEM_27_CHORD
            delta_twist = twists_eval - BEM_27_TWIST
            delta_c_ctrl = np.zeros(self.n_spline)
            delta_tw_ctrl = np.zeros(self.n_spline)
            is_114 = False
        elif len(x) == self._dim:
            thetas_raw, delta_c_ctrl, delta_tw_ctrl, delta_chord, delta_twist, ct = self._compute_span_geom(x)
            chords_eval, twists_eval = ct[0], ct[1]
            is_114 = True
        else:
            raise ValueError(f"Expected vector of dimension {self._dim} or 120, got {len(x)}")

        geom_metrics: Dict[str, Any] = {
            "station_metrics": [],
            "chord_taper_violation": 0.0,
            "twist_monotonicity_violation": 0.0,
            "chord_linear_violation": 0.0,
            "chord_smooth_violation": 0.0,
            "twist_smooth_violation": 0.0,
            "min_chord_violation": 0.0,
            "max_overlap": 0.0,
        }

        # 1. 检验 6 个站位 17 维闭口尾缘 CST 翼型几何质量
        x_sample = (1.0 - np.cos(np.linspace(0.0, np.pi, 80))) / 2.0
        for k in range(NUM_STATIONS):
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
                return True, 20.0 + overlap * 100.0, geom_metrics

        # 2. 展向弦长与扭角物理平滑约束 (尺度严格对齐 NREL Phase VI 0.356~0.737m 弦长)
        min_c = float(np.min(chords_eval))
        min_chord_viol = float(max(0.0, 0.15 - min_c))
        if min_c <= 0.08:
            return True, 30.0 + (0.08 - min_c) * 100.0, geom_metrics

        diff_c = np.diff(chords_eval)
        chord_pen = float(np.sum(np.maximum(0.0, diff_c)))

        diff_tw = np.diff(twists_eval)
        twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))

        # NREL Phase VI 中展段本身接近线性锥度，限制非物理外凸波动 <= 0.045 m
        r_mid = BEM_27_RADII[4:23]
        c_mid = chords_eval[4:23]
        p_lin = np.polyfit(r_mid, c_mid, 1)
        dev_linear = float(np.max(np.abs(c_mid - np.polyval(p_lin, r_mid))))
        linear_pen = float(max(0.0, dev_linear - 0.045))

        # 二阶差分直接检查实际弦长/扭角曲线平滑性，同时温和限制控制点锯齿
        d2_chord = np.diff(chords_eval, 2)
        d2_twist = np.diff(twists_eval, 2)
        # 基准弦长二阶差分为 0.0, 基准扭角在近根过渡区最大 d2 为 0.763 deg
        chord_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_chord) - 0.020)))
        twist_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_twist) - 1.15)))

        if is_114:
            d2_ctrl_c = np.diff(delta_c_ctrl, 2)
            d2_ctrl_tw = np.diff(delta_tw_ctrl, 2)
            chord_smooth_viol += float(np.sum(np.maximum(0.0, np.abs(d2_ctrl_c) - 0.12)))
            twist_smooth_viol += 0.25 * float(np.sum(np.maximum(0.0, np.abs(d2_ctrl_tw) - 8.5)))

        geom_metrics["chord_taper_violation"] = chord_pen
        geom_metrics["twist_monotonicity_violation"] = twist_pen
        geom_metrics["chord_linear_violation"] = linear_pen
        geom_metrics["chord_smooth_violation"] = chord_smooth_viol
        geom_metrics["twist_smooth_violation"] = twist_smooth_viol
        geom_metrics["min_chord_violation"] = min_chord_viol

        # 致命几何崩塌防线 (阻止严重锯齿反转但不过度截杀平滑优化解)
        if (
            chord_pen > 0.08
            or twist_pen > 1.00
            or linear_pen > 0.06
            or chord_smooth_viol > 0.15
            or twist_smooth_viol > 2.00
        ):
            fatal_pen = 5.0 + 1.0 * (
                chord_pen * 20.0
                + twist_pen * 2.0
                + linear_pen * 30.0
                + chord_smooth_viol * 10.0
                + twist_smooth_viol * 2.0
            )
            return True, fatal_pen, geom_metrics

        return False, 0.0, geom_metrics


class Stations6BladeModel(BaseParameterizationModel):
    """离散 6 控制站位对比模型 (D=120): 每站位 18 CST + 1 弦长 + 1 扭角"""

    def __init__(self):
        self._dim = DIM
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

    def to_span_data(self, x: np.ndarray, n_stations: int = BEM_NUM_SECTIONS) -> Dict[str, np.ndarray]:
        return interpolate_blade_span(x, n_stations=n_stations)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        model_bs = BSplineBladeModel()
        return model_bs.validate_geometry(x)
