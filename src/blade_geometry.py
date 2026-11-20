"""3D 风力机叶片几何参数化与三维放样工具库：
严格对齐 NREL Phase VI (Sequence H) 官方风洞试验参考规范 (Hand et al., 2001, NREL/TP-500-29955; Kaya, 2019)：
- 风轮半径 R = 5.029 m (直径 10.058 m), 叶片数 B = 2 (Upwind 水平轴风力机);
- 物理轮毂法兰 r_flange = 0.5083 m, 圆杆段 r = 0.5083~0.8835 m, 过渡段 0.8835~1.2575 m;
- 气动升力段 (S809 翼型展向主体): r in [1.2575 m (0.25R), 5.0290 m (1.00R)], 气动展长 L = 3.7715 m;
- 基准测试工况 (Sequence H / Kaya 2019): 来流风速 V_inf = 7.0 m/s, 转速 = 72.0 rpm (omega = 7.5398 rad/s, TSR = 5.42);
- 叶片变桨安装角 pitch = 4.815 deg (以 r/R = 0.75 零扭角参考站位计，等效叶尖 3.0 deg 桨距角), 扭转/俯仰轴位于 30% 弦长处 (0.30c);
- 展向采用 31 截面全余弦加密离散节点 (Full-Cosine Clustering, 根/尖双端加密), BEM 积分精度提升 3 倍;
- 3 个核心翼型控制站位严格布设于纯 S809 升力段的头/中/尾 (Root/Mid/Tip, 索引 [0, 15, 30], r = [1.2575, 3.14325, 5.0290] m),
  并通过保单调三次 PCHIP 样条沿展向 C1 光滑过渡, 彻底消除 3D 叶片侧视波纹起伏;
- 主设计空间: BSplineBladeModel (D=63: 3 * 17 维闭口尾缘 CST + 6 维弦长 B 样条增量 + 6 维扭角 B 样条增量)。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import PchipInterpolator
import aerosandbox as asb


# NREL Phase VI (Sequence H) 官方几何与基准工况常数
ROTOR_RADIUS: float = 5.029           # 风轮外半径 R (m), 直径 D = 10.058 m
HUB_FLANGE_RADIUS: float = 0.5083     # 物理轮毂装配法兰半径 (m)
HUB_RADIUS: float = 0.8835            # 非升力圆柱根杆外端/轮毂损耗等效根半径 r_hub (m, 17.6% span)
BLADE_ROOT_RADIUS: float = 1.2575     # S809 气动升力段起始半径 r_root (m, 25.0% R)
BLADE_LENGTH: float = 3.7715          # 气动升力段净展长 L = R - BLADE_ROOT_RADIUS (m)
HUB_HEIGHT: float = 12.192            # NASA Ames 80x120 ft 风洞轮毂中心高 (m)
NUM_BLADES: int = 2                   # 叶片数 B = 2
PITCH_AXIS_FRAC: float = 0.30         # 俯仰/扭转/放样中心轴位置 (0.30c, Hand et al. 2001 & Kaya 2019)

# 基准额定测试工况 (Kaya 2019 & NREL Phase VI Sequence H @ 7 m/s)
BASELINE_WIND_SPEED: float = 7.0      # m/s
BASELINE_RPM: float = 72.0            # rpm
BASELINE_PITCH_DEG: float = 4.815     # deg (75% span zero-twist ref; tip pitch = 3.0 deg)

# 3 个核心翼型控制站位 (头/中/尾) 与 60 维离散参数维度 (主干 B-Spline 模型为 D=63)
NUM_STATIONS: int = 3
DIM: int = 60
BEM_NUM_SECTIONS: int = 31
OPENFAST_NUM_SECTIONS: int = BEM_NUM_SECTIONS

# 载入基准 NREL Phase VI 31 截面全余弦加密预计算几何数组
_data_dir = Path(__file__).parent.parent / "data" / "baseline_nrel_phase6"
_npz_path = _data_dir / "baseline_31_sections.npz"

if not _npz_path.exists():
    raise FileNotFoundError(
        f"Missing NREL Phase VI baseline dataset: {_npz_path}. "
        "Run data/baseline_nrel_phase6/build_baseline.py first."
    )

_precomputed = np.load(_npz_path)
BEM_BLSPN: np.ndarray = _precomputed["blspn"]
BEM_SWEEP: np.ndarray = _precomputed["sweep"]
BEM_TWIST: np.ndarray = _precomputed["twist"]
BEM_CHORD: np.ndarray = _precomputed["chord"]
BEM_AF_IDS: np.ndarray = _precomputed["af_ids"]
BEM_KULFAN_THETAS: np.ndarray = _precomputed["kulfan_thetas"]
BEM_RADII: np.ndarray = _precomputed["radii"]

# 计算双网格 Voronoi 气动微元宽度 dr (sum(dr) == BLADE_LENGTH == 3.7715 m)
_bnds = np.zeros(len(BEM_RADII) + 1, dtype=np.float64)
_bnds[0] = BEM_RADII[0]
_bnds[1:-1] = 0.5 * (BEM_RADII[:-1] + BEM_RADII[1:])
_bnds[-1] = BEM_RADII[-1]
BEM_DR: np.ndarray = np.diff(_bnds)

# 规范别名 (向后兼容外部引用)
BEM_31_BLSPN = BEM_BLSPN
BEM_31_SWEEP = BEM_SWEEP
BEM_31_TWIST = BEM_TWIST
BEM_31_CHORD = BEM_CHORD
BEM_31_AF_IDS = BEM_AF_IDS
BEM_31_KULFAN_THETAS = BEM_KULFAN_THETAS
BEM_31_RADII = BEM_RADII
BEM_31_DR = BEM_DR

# 3 个核心气动控制站位对应的 31 截面全余弦对称索引 (头/中/尾):
# r = [1.2575 (0.25R), 3.14325 (0.625R), 5.0290 (1.00R)] m
STATION_INDICES: np.ndarray = np.array([0, 15, 30], dtype=np.int64)
STATION_RADII: np.ndarray = BEM_RADII[STATION_INDICES]
STATION_RADIANS_FRAC: np.ndarray = STATION_RADII / ROTOR_RADIUS

# 提取 3 个站位的基准弦长、扭角与 S809 CST 参数
BASELINE_STATION_CHORDS: np.ndarray = BEM_CHORD[STATION_INDICES]
BASELINE_STATION_TWISTS: np.ndarray = BEM_TWIST[STATION_INDICES]
BASELINE_STATION_THETAS: np.ndarray = BEM_KULFAN_THETAS[STATION_INDICES]

# 组装 60 维基准参数向量 (兼容 Stations3BladeModel)
_x0_parts: List[float] = []
for _k in range(NUM_STATIONS):
    _x0_parts.extend(BASELINE_STATION_THETAS[_k])
    _x0_parts.append(float(BASELINE_STATION_CHORDS[_k]))
    _x0_parts.append(float(BASELINE_STATION_TWISTS[_k]))

BASELINE_X0: np.ndarray = np.array(_x0_parts, dtype=np.float64)


def make_cosine_radii(n_stations: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成 [BLADE_ROOT_RADIUS, ROTOR_RADIUS] 上的全余弦加密节点及其 Voronoi 微元宽度 dr."""
    theta = np.linspace(0.0, np.pi, n_stations, dtype=np.float64)
    r = BLADE_ROOT_RADIUS + BLADE_LENGTH * 0.5 * (1.0 - np.cos(theta))
    bnds = np.zeros(n_stations + 1, dtype=np.float64)
    bnds[0] = r[0]
    bnds[1:-1] = 0.5 * (r[:-1] + r[1:])
    bnds[-1] = r[-1]
    dr = np.diff(bnds)
    return r, dr


def get_baseline_blade_params() -> np.ndarray:
    """获取 NREL Phase VI 标准基准 60 维离散设计参数向量 x0 (3 站位 x 20)."""
    return BASELINE_X0.copy()


def unpack_blade_params(x: np.ndarray) -> List[Dict[str, Any]]:
    """将 60 维向量解构为 3 个控制站位 (Root / Mid / Tip) 的结构化字典列表."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) != DIM:
        raise ValueError(f"Expected blade parameters vector of dimension {DIM}, got {len(x)}")

    stations = []
    for k in range(NUM_STATIONS):
        idx = k * 20
        sec_theta = x[idx : idx + 18]
        chord = float(x[idx + 18])
        twist = float(x[idx + 19])

        kulfan_dict = {
            "upper_weights": sec_theta[0:8],
            "lower_weights": sec_theta[8:16],
            "leading_edge_weight": float(sec_theta[16]),
            "TE_thickness": float(sec_theta[17]),
        }

        stations.append({
            "station_idx": k,
            "global_idx": int(STATION_INDICES[k]),
            "r_frac": float(STATION_RADIANS_FRAC[k]),
            "radius": float(STATION_RADII[k]),
            "theta": sec_theta,
            "kulfan_dict": kulfan_dict,
            "chord": chord,
            "twist": twist,
        })
    return stations


def get_blade_bounds(x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """获取 60 维 NREL Phase VI 3 站位叶片参数的工程可行探索上下界."""
    if x0 is None:
        x0 = get_baseline_blade_params()

    lb = np.zeros(DIM, dtype=np.float64)
    ub = np.zeros(DIM, dtype=np.float64)

    # 3 个控制站位弦长工程探索范围 (基准线性收缩: 0.7370 m -> 0.5465 m -> 0.3560 m)
    chord_bounds = [
        (0.50, 0.98),   # Station 1 (Root, r=1.2575m): baseline 0.7370 m
        (0.35, 0.78),   # Station 2 (Mid,  r=3.1433m): baseline 0.5465 m
        (0.20, 0.52),   # Station 3 (Tip,  r=5.0290m): baseline 0.3560 m
    ]
    # 3 个控制站位扭角探索范围 (含 Kaya 2019 最优洗退空间)
    twist_bounds = [
        ( 2.0, 28.0),   # Station 1 (Root, r=1.2575m): baseline 20.040 deg
        (-8.0, 11.0),   # Station 2 (Mid,  r=3.1433m): baseline  1.375 deg
        (-11.0, 7.0),   # Station 3 (Tip,  r=5.0290m): baseline -1.815 deg
    ]

    for k in range(NUM_STATIONS):
        idx = k * 20
        base_sec = x0[idx : idx + 18]

        for j in range(16):
            delta = max(0.30 * abs(float(base_sec[j])), 0.14)
            lb[idx + j] = float(base_sec[j]) - delta
            ub[idx + j] = float(base_sec[j]) + delta

        le_delta = max(0.30 * abs(float(base_sec[16])), 0.10)
        lb[idx + 16] = float(base_sec[16]) - le_delta
        ub[idx + 16] = float(base_sec[16]) + le_delta

        lb[idx + 17] = max(0.0, float(base_sec[17]) - 0.005)
        ub[idx + 17] = float(base_sec[17]) + 0.010

        lb[idx + 18], ub[idx + 18] = chord_bounds[k]
        lb[idx + 19], ub[idx + 19] = twist_bounds[k]

    return lb, ub


def interpolate_blade_span(
    x: np.ndarray,
    n_stations: int = BEM_NUM_SECTIONS,
) -> Dict[str, np.ndarray]:
    """将 3 个控制站位通过 C1 保单调三次 PCHIP 样条光滑拓展到展向 31 全余弦加密截面."""
    stations = unpack_blade_params(x)

    if n_stations == BEM_NUM_SECTIONS:
        r = BEM_RADII.copy()
        dr = BEM_DR.copy()
        ch_base = BEM_CHORD.copy()
        tw_base = BEM_TWIST.copy()
        th_base = BEM_KULFAN_THETAS.copy()
    else:
        r, dr = make_cosine_radii(n_stations)
        ch_base = PchipInterpolator(BEM_RADII, BEM_CHORD)(r)
        tw_base = PchipInterpolator(BEM_RADII, BEM_TWIST)(r)
        th_base = PchipInterpolator(BEM_RADII, BEM_KULFAN_THETAS, axis=0)(r)

    if np.allclose(x, BASELINE_X0, atol=1e-7):
        return {
            "r": r,
            "dr": dr,
            "chord": ch_base,
            "twist_deg": tw_base,
            "twist_rad": np.radians(tw_base),
            "kulfan_thetas": th_base,
        }

    r_ctrl = STATION_RADII.copy()
    delta_chord_ctrl = np.array([s["chord"] - BASELINE_STATION_CHORDS[k] for k, s in enumerate(stations)])
    delta_twist_ctrl = np.array([s["twist"] - BASELINE_STATION_TWISTS[k] for k, s in enumerate(stations)])

    chord_span = ch_base + PchipInterpolator(r_ctrl, delta_chord_ctrl)(r)
    twist_span_deg = tw_base + PchipInterpolator(r_ctrl, delta_twist_ctrl)(r)
    twist_span_rad = np.radians(twist_span_deg)

    thetas_3 = np.array([s["theta"] for s in stations])
    delta_thetas_3 = thetas_3 - BASELINE_STATION_THETAS
    kulfan_span = th_base + PchipInterpolator(r_ctrl, delta_thetas_3, axis=0)(r)

    return {
        "r": r,
        "dr": dr,
        "chord": chord_span,
        "twist_deg": twist_span_deg,
        "twist_rad": twist_span_rad,
        "kulfan_thetas": kulfan_span,
    }


def loft_blade_3d(
    x: np.ndarray,
    n_span: int = BEM_NUM_SECTIONS,
    n_chord: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 3D 叶片表面的结构化空间三维点云坐标网格 (X, Y, Z)。

    严格围绕 NREL Phase VI 官方 30% 弦长俯仰/扭转中心轴 (PITCH_AXIS_FRAC = 0.30) 放样。
    """
    if len(x) == 63:
        from src.parameterizations import BSplineBladeModel
        model_bs = BSplineBladeModel()
        span_data = model_bs.to_span_data(x, n_stations=n_span)
    else:
        span_data = interpolate_blade_span(x, n_stations=n_span)

    r_span = span_data["r"]
    chords = span_data["chord"]
    twists_rad = span_data["twist_rad"]
    thetas = span_data["kulfan_thetas"]
    n_actual = len(r_span)

    X_mesh = np.zeros((n_actual, 2 * n_chord), dtype=np.float64)
    Y_mesh = np.zeros((n_actual, 2 * n_chord), dtype=np.float64)
    Z_mesh = np.zeros((n_actual, 2 * n_chord), dtype=np.float64)

    beta_chord = np.linspace(0.0, np.pi, n_chord)
    x_c = 0.5 * (1.0 - np.cos(beta_chord))

    for i in range(n_actual):
        r_i = r_span[i]
        c_i = chords[i]
        beta_i = twists_rad[i]
        th_i = thetas[i]

        k_af = asb.KulfanAirfoil(
            upper_weights=th_i[0:8],
            lower_weights=th_i[8:16],
            leading_edge_weight=float(th_i[16]),
            TE_thickness=float(th_i[17]),
        )

        yu = k_af.upper_coordinates(x_c)[:, 1]
        yl = k_af.lower_coordinates(x_c)[:, 1]

        # 闭合轮廓：上表面 (后缘 -> 前缘) + 下表面 (前缘 -> 后缘)
        x_sec = np.concatenate([x_c[::-1], x_c])
        y_sec = np.concatenate([yu[::-1], yl])

        # 围绕 NREL Phase VI 30% 弦长 (0.30c) 俯仰扭转轴心缩放
        x_local = (x_sec - PITCH_AXIS_FRAC) * c_i
        y_local = y_sec * c_i

        cos_b = np.cos(beta_i)
        sin_b = np.sin(beta_i)
        x_rot = x_local * cos_b - y_local * sin_b
        y_rot = x_local * sin_b + y_local * cos_b

        X_mesh[i, :] = x_rot
        Y_mesh[i, :] = y_rot
        Z_mesh[i, :] = r_i

    return X_mesh, Y_mesh, Z_mesh
