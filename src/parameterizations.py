"""3D 风力机叶片高维参数化模型库 (src/parameterizations.py):
严格对齐 NREL Phase VI (Sequence H) 气动主体规范 (31 全余弦加密截面 S809 升力段, r in [1.2575, 5.0290] m):
1. BSplineBladeModel (D=63): 3 个核心翼型控制站位 (Root/Mid/Tip, r = [1.2575, 3.14325, 5.0290] m) 的
   17 维闭口尾缘 CST (51 维, TE_thickness=0, 经保单调三次 PCHIP 样条沿展向 C1 光滑放样)
   + 6 阶三次 B 样条弦长增量 (6 维) + 6 阶三次 B 样条扭角增量 (6 维)
2. Stations3BladeModel (D=60): 3 离散控制站位对比基线模型 (每站位 18 CST + 1 弦长 + 1 扭角)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator

import aerosandbox as asb
from src.blade_geometry import (
    BLADE_LENGTH,
    BLADE_ROOT_RADIUS,
    DIM,
    NUM_STATIONS,
    BEM_NUM_SECTIONS,
    BEM_RADII,
    BEM_DR,
    BEM_CHORD,
    BEM_TWIST,
    BEM_KULFAN_THETAS,
    ROTOR_RADIUS,
    STATION_INDICES,
    STATION_RADII,
    get_baseline_blade_params,
    get_blade_bounds,
    interpolate_blade_span,
    make_cosine_radii,
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
    """B 样条高阶平滑模态参数化模型 (D=63):
    适配 NREL Phase VI (S809 气动段 r in [1.2575, 5.0290] m, 31 全余弦加密截面):
    - 3 个核心翼型控制站位 (头/中/尾: Root r=1.2575m, Mid r=3.1433m, Tip r=5.0290m) 的
      17 维 Kulfan CST 翼型参数 (51 维: 3 * 17, TE_thickness=0);
      展向翼型过渡采用保单调三次 PCHIP 样条 (C1 光滑且零过冲), 彻底消除 3D 侧视表面起伏;
    - 6 阶三次 B 样条弦长增量控制点 Delta_c (6 维, m);
    - 6 阶三次 B 样条扭角增量控制点 Delta_twist (6 维, deg);
    - 物理约束校准:
      * 全 31 余弦展向弦长单调非增收缩 (diff(c) <= 0), 最小弦长 >= 0.15 m;
      * 全 31 余弦展向扭角单调洗退 (diff(twist) <= 0);
      * 在等距 27 点参考网格上评估线性偏离与二阶差分曲率平滑性 (杜绝非均匀余弦步长导致的伪二阶差分偏差);
      * S809 相对厚度 t/c = 21.07%, Root/Mid/Tip 厚度边界平滑递减以兼顾根部结构强度与叶尖减阻自由度。
    """

    def __init__(self, n_spline_points: int = 6, spline_degree: int = 3):
        self.n_stations = NUM_STATIONS
        self.n_cst_per_station = 17
        self.n_cst_total = self.n_stations * self.n_cst_per_station  # 51
        self.n_spline = n_spline_points
        self.degree = spline_degree
        self._dim = self.n_cst_total + 2 * self.n_spline  # 63

        # 1. 31 截面全余弦加密网格 (用于高精度 BEM 求解与全展向单调性校验)
        self.r_ref = BEM_RADII.copy()
        self.r_norm = (self.r_ref - BLADE_ROOT_RADIUS) / BLADE_LENGTH

        # Clamped Uniform Cubic B-Spline Knot Vector
        interior = np.linspace(0.0, 1.0, self.n_spline - self.degree + 1)[1:-1]
        self.knots = np.concatenate(([0.0] * (self.degree + 1), interior, [1.0] * (self.degree + 1)))

        self.A_proj = self._build_bspline_matrix(self.r_norm)

        # 2. 等距 27 点参考网格 (用于物理尺度一致的二阶差分曲率与线性偏离校验)
        self.r_unif = np.linspace(BLADE_ROOT_RADIUS, ROTOR_RADIUS, 27, dtype=np.float64)
        self.r_unif_norm = (self.r_unif - BLADE_ROOT_RADIUS) / BLADE_LENGTH
        self.A_proj_unif = self._build_bspline_matrix(self.r_unif_norm)
        self.base_chord_unif = PchipInterpolator(BEM_RADII, BEM_CHORD)(self.r_unif)
        self.base_twist_unif = PchipInterpolator(BEM_RADII, BEM_TWIST)(self.r_unif)

        # 提取 3 个核心站位 (Root/Mid/Tip) 的 51 维基准 S809 翼型参数 (移除第 18 维尾缘厚度)
        x0_3 = get_baseline_blade_params()
        st3 = unpack_blade_params(x0_3)
        self.thetas_0 = np.concatenate([s["theta"][:17] for s in st3])

        self.delta_c_0 = np.zeros(self.n_spline, dtype=np.float64)
        self.delta_tw_0 = np.zeros(self.n_spline, dtype=np.float64)
        self.x0 = np.concatenate([self.thetas_0, self.delta_c_0, self.delta_tw_0])

        # NREL Phase VI (S809 baseline max_t=0.2107) Root / Mid / Tip 结构与气动厚度边界
        self.min_le_radii = [0.003, 0.003, 0.003]
        self.thickness_limits = [
            (0.12, 0.28),  # Station 1 (Root, r=1.2575 m)
            (0.09, 0.25),  # Station 2 (Mid,  r=3.1433 m)
            (0.07, 0.24),  # Station 3 (Tip,  r=5.0290 m)
        ]
        self.spar_cap_min_t = [0.05, 0.05, 0.05]

    def _build_bspline_matrix(self, r_norm_vec: np.ndarray) -> np.ndarray:
        A = np.zeros((len(r_norm_vec), self.n_spline), dtype=np.float64)
        for i in range(self.n_spline):
            c_basis = np.zeros(self.n_spline, dtype=np.float64)
            c_basis[i] = 1.0
            bs = BSpline(self.knots, c_basis, self.degree)
            A[:, i] = bs(r_norm_vec)
        return A

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"B-Spline Modal Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 63 维 NREL Phase VI 设计参数上下界."""
        lb = np.zeros(self._dim, dtype=np.float64)
        ub = np.zeros(self._dim, dtype=np.float64)

        # 1. 51 维 CST 权重探索空间 (17 params/station * 3)
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

    def _compute_span_geom(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        thetas_raw_17 = x[: self.n_cst_total].reshape((NUM_STATIONS, 17))
        thetas_raw = np.zeros((NUM_STATIONS, 18), dtype=np.float64)
        thetas_raw[:, :17] = thetas_raw_17
        thetas_raw[:, 17] = 0.0

        delta_c_ctrl = x[self.n_cst_total : self.n_cst_total + self.n_spline]
        delta_tw_ctrl = x[self.n_cst_total + self.n_spline : self._dim]
        delta_chord = self.A_proj @ delta_c_ctrl
        delta_twist = self.A_proj @ delta_tw_ctrl
        chords = BEM_CHORD + delta_chord
        twists = BEM_TWIST + delta_twist
        return thetas_raw, delta_c_ctrl, delta_tw_ctrl, delta_chord, delta_twist, np.vstack([chords, twists])

    def to_span_data(self, x: np.ndarray, n_stations: int = BEM_NUM_SECTIONS) -> Dict[str, np.ndarray]:
        """将 63 维 (或兼容 60 维) 参数转化为展向 31 全余弦截面气动数据字典."""
        if len(x) == DIM:
            return interpolate_blade_span(x, n_stations=n_stations)
        if len(x) != self._dim:
            raise ValueError(f"Expected vector of dimension {self._dim} or {DIM}, got {len(x)}")

        thetas_raw, delta_c_ctrl, delta_tw_ctrl, _, _, ct = self._compute_span_geom(x)
        chord_span = ct[0]
        twist_span_deg = ct[1]

        # 使用 C1 保单调三次 PCHIP 样条沿 3 个控制站位 (Root/Mid/Tip) 光滑插值 CST 翼型参数
        pchip_cst = PchipInterpolator(STATION_RADII, thetas_raw[:, :17], axis=0)

        if n_stations == BEM_NUM_SECTIONS:
            kulfan_31 = np.zeros((len(BEM_RADII), 18), dtype=np.float64)
            kulfan_31[:, :17] = pchip_cst(BEM_RADII)
            kulfan_31[:, 17] = 0.0
            return {
                "r": BEM_RADII.copy(),
                "dr": BEM_DR.copy(),
                "chord": chord_span,
                "twist_deg": twist_span_deg,
                "twist_rad": np.radians(twist_span_deg),
                "kulfan_thetas": kulfan_31,
            }

        r, dr = make_cosine_radii(n_stations)
        r_norm_n = (r - BLADE_ROOT_RADIUS) / BLADE_LENGTH
        A_n = self._build_bspline_matrix(r_norm_n)
        ch_base_n = PchipInterpolator(BEM_RADII, BEM_CHORD)(r)
        tw_base_n = PchipInterpolator(BEM_RADII, BEM_TWIST)(r)
        chord_n = ch_base_n + A_n @ delta_c_ctrl
        twist_n = tw_base_n + A_n @ delta_tw_ctrl

        kulfan_n = np.zeros((n_stations, 18), dtype=np.float64)
        kulfan_n[:, :17] = pchip_cst(r)
        kulfan_n[:, 17] = 0.0
        return {
            "r": r,
            "dr": dr,
            "chord": chord_n,
            "twist_deg": twist_n,
            "twist_rad": np.radians(twist_n),
            "kulfan_thetas": kulfan_n,
        }

    def get_station_info(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """提取 3 个核心气动控制站位 (Root / Mid / Tip) 的几何参数字典列表."""
        if len(x) == DIM:
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
        """全量几何质量与展向平滑连续性校验."""
        if len(x) == DIM:
            span_d = interpolate_blade_span(x, n_stations=BEM_NUM_SECTIONS)
            st_list = unpack_blade_params(x)
            thetas_raw = np.array([np.append(s["theta"][:17], 0.0) for s in st_list])
            chords_eval = span_d["chord"]
            twists_eval = span_d["twist_deg"]
            chords_unif = PchipInterpolator(BEM_RADII, chords_eval)(self.r_unif)
            twists_unif = PchipInterpolator(BEM_RADII, twists_eval)(self.r_unif)
            delta_c_ctrl = np.zeros(self.n_spline)
            delta_tw_ctrl = np.zeros(self.n_spline)
            is_bspline = False
        elif len(x) == self._dim:
            thetas_raw, delta_c_ctrl, delta_tw_ctrl, _, _, ct = self._compute_span_geom(x)
            chords_eval, twists_eval = ct[0], ct[1]
            chords_unif = self.base_chord_unif + self.A_proj_unif @ delta_c_ctrl
            twists_unif = self.base_twist_unif + self.A_proj_unif @ delta_tw_ctrl
            is_bspline = True
        else:
            raise ValueError(f"Expected vector of dimension {self._dim} or {DIM}, got {len(x)}")

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

        # 1. 检验 3 个站位 (Root / Mid / Tip) 17 维闭口尾缘 CST 翼型几何质量
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

        # 2. 展向弦长与扭角物理平滑约束 (单调性在全 31 余弦网格检验, 曲率在等距 27 点参考网格检验)
        min_c = float(min(np.min(chords_eval), np.min(chords_unif)))
        min_chord_viol = float(max(0.0, 0.15 - min_c))
        if min_c <= 0.08:
            return True, 30.0 + (0.08 - min_c) * 100.0, geom_metrics

        diff_c = np.diff(chords_eval)
        chord_pen = float(np.sum(np.maximum(0.0, diff_c)))

        diff_tw = np.diff(twists_eval)
        twist_pen = float(np.sum(np.maximum(0.0, diff_tw)))

        # NREL Phase VI 中展段线性偏离检查 (等距参考网格 [4:23], 限制非物理外凸波动 <= 0.045 m)
        r_mid = self.r_unif[4:23]
        c_mid = chords_unif[4:23]
        p_lin = np.polyfit(r_mid, c_mid, 1)
        dev_linear = float(np.max(np.abs(c_mid - np.polyval(p_lin, r_mid))))
        linear_pen = float(max(0.0, dev_linear - 0.045))

        # 等距参考网格上的真实物理二阶差分曲率检查 (杜绝非均匀余弦步长造成的虚假二阶差分)
        d2_chord = np.diff(chords_unif, 2)
        d2_twist = np.diff(twists_unif, 2)
        chord_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_chord) - 0.020)))
        twist_smooth_viol = float(np.sum(np.maximum(0.0, np.abs(d2_twist) - 1.15)))

        if is_bspline:
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

        # 致命几何崩塌防线
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


class Stations3BladeModel(BaseParameterizationModel):
    """离散 3 控制站位对比模型 (D=60): 每站位 18 CST + 1 弦长 + 1 扭角"""

    def __init__(self):
        self._dim = DIM
        self.x0 = get_baseline_blade_params()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"Discrete 3-Station Model (D={self._dim})"

    def get_x0(self) -> np.ndarray:
        return self.x0.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return get_blade_bounds(self.x0)

    def to_span_data(self, x: np.ndarray, n_stations: int = BEM_NUM_SECTIONS) -> Dict[str, np.ndarray]:
        return interpolate_blade_span(x, n_stations=n_stations)

    def validate_geometry(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        model_bs = BSplineBladeModel()
        return model_bs.validate_geometry(x)
