"""3D 风力机叶片几何参数化与三维放样工具库：
依据 1.5m 叶片小型/模型风力机参考规范 (未优化 Baseline 1p5mBlade)：
- 风轮半径 R = 1.50 m, 轮毂半径 R_hub = 0.25 m, 净叶片长度 L = 1.25 m, 轮毂高度 H = 10.0 m;
- 基准运行测试工况: 风速 V_inf = 10.0 m/s, 转速 = 300.0 rpm (omega = 31.416 rad/s, TSR = 4.71);
- 叶片安装角/变桨角 pitch = 0.0 deg, 预锥角 precone = 0.0 deg, 机头仰角 tilt = 0.0 deg;
- 展向采用 27 截面几何节点定义 (严格对齐导师附件 1p5mBlade modified_data.txt 与 Sheet1~Sheet27);
- 6 个核心气动控制截面 (Station 1 ~ 6: r = 0.25m, 0.45m, 0.70m, 1.00m, 1.25m, 1.50m);
- 每个站位包含: 18 维 Kulfan 翼型参数 (8 上表面 + 8 下表面 + 1 前缘 + 1 后缘厚度) + 1 维弦长 c (m) + 1 维扭角 beta (deg);
- 总设计空间维度: D = 6 * 18 + 2 * 12 = 132 维 (B-Spline Modal 模型) 或 120 维 (离散 6 站位模型);
- 提供标准基准叶片构型 (BASELINE_X0)、工业探索边界映射、沿展向平滑插值与 3D 表面网格放样。
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import aerosandbox as asb


# 1.5m 参考风电机组几何常数
ROTOR_RADIUS: float = 1.50        # 风轮半径 R = R_hub + L (m)
HUB_RADIUS: float = 0.25          # 轮毂半径 R_hub (m)
BLADE_LENGTH: float = 1.25        # 净叶片长度 L (m)
HUB_HEIGHT: float = 10.0          # 轮毂高度 (m)
NUM_BLADES: int = 3               # 叶片数 B

# 6 个核心气动控制站位
NUM_STATIONS: int = 6
DIM: int = 120

# 导师附件官方 27 截面几何节点定义
BEM_NUM_SECTIONS: int = 27
OPENFAST_NUM_SECTIONS: int = 27

# 载入基准 27 截面预计算几何数组
_data_dir = Path(__file__).parent.parent / "data" / "baseline_1p5m"
_npz_path = _data_dir / "baseline_27_sections.npz"

if _npz_path.exists():
    _precomputed = np.load(_npz_path)
    BEM_27_BLSPN: np.ndarray = _precomputed["blspn"]
    BEM_27_SWEEP: np.ndarray = _precomputed["sweep"]
    BEM_27_TWIST: np.ndarray = _precomputed["twist"]
    BEM_27_CHORD: np.ndarray = _precomputed["chord"]
    BEM_27_AF_IDS: np.ndarray = _precomputed["af_ids"]
    BEM_27_KULFAN_THETAS: np.ndarray = _precomputed["kulfan_thetas"]
    BEM_27_RADII: np.ndarray = _precomputed["radii"]
else:
    # 理论备选回退 (27 截面)
    BEM_27_RADII = np.linspace(HUB_RADIUS, ROTOR_RADIUS, BEM_NUM_SECTIONS, dtype=np.float64)
    BEM_27_BLSPN = BEM_27_RADII - HUB_RADIUS
    BEM_27_SWEEP = np.zeros(BEM_NUM_SECTIONS, dtype=np.float64)
    BEM_27_TWIST = np.linspace(27.26, 2.93, BEM_NUM_SECTIONS, dtype=np.float64)
    BEM_27_CHORD = np.linspace(1.1118, 0.2851, BEM_NUM_SECTIONS, dtype=np.float64)
    BEM_27_AF_IDS = np.arange(1, BEM_NUM_SECTIONS + 1, dtype=np.int64)
    BEM_27_KULFAN_THETAS = np.zeros((BEM_NUM_SECTIONS, 18), dtype=np.float64)

# 严格计算双网格 Voronoi 气动微元宽度 dr (sum(dr) == BLADE_LENGTH)
_bnds = np.zeros(len(BEM_27_RADII) + 1, dtype=np.float64)
_bnds[0] = BEM_27_RADII[0]
_bnds[1:-1] = 0.5 * (BEM_27_RADII[:-1] + BEM_27_RADII[1:])
_bnds[-1] = BEM_27_RADII[-1]
BEM_27_DR: np.ndarray = np.diff(_bnds)

# 向后兼容别名
OPENFAST_50_RADII = BEM_27_RADII
OPENFAST_50_DR = BEM_27_DR
OPENFAST_50_BLSPN = BEM_27_BLSPN
OPENFAST_50_CHORD = BEM_27_CHORD
OPENFAST_50_TWIST = BEM_27_TWIST
OPENFAST_50_KULFAN_THETAS = BEM_27_KULFAN_THETAS
OPENFAST_58_RADII = BEM_27_RADII
OPENFAST_58_DR = BEM_27_DR
OPENFAST_BLSPN = BEM_27_BLSPN

# 6 个核心气动控制站位对应的 27 截面索引:
# r = [0.25, 0.45, 0.70, 1.00, 1.25, 1.50]
STATION_INDICES: np.ndarray = np.array([0, 5, 10, 16, 21, 26], dtype=np.int64)
STATION_RADII: np.ndarray = BEM_27_RADII[STATION_INDICES]
STATION_RADIANS_FRAC: np.ndarray = STATION_RADII / ROTOR_RADIUS

# 提取 6 个站位的基准弦长、扭角与翼型参数
BASELINE_STATION_CHORDS: np.ndarray = BEM_27_CHORD[STATION_INDICES]
BASELINE_STATION_TWISTS: np.ndarray = BEM_27_TWIST[STATION_INDICES]
BASELINE_STATION_THETAS: np.ndarray = BEM_27_KULFAN_THETAS[STATION_INDICES]

# 组装 120 维基准参数向量
_x0_parts = []
for _k in range(NUM_STATIONS):
    _x0_parts.extend(BASELINE_STATION_THETAS[_k])
    _x0_parts.append(float(BASELINE_STATION_CHORDS[_k]))
    _x0_parts.append(float(BASELINE_STATION_TWISTS[_k]))

BASELINE_X0: np.ndarray = np.array(_x0_parts, dtype=np.float64)


def get_baseline_blade_params() -> np.ndarray:
    """获取 1.5m 标准基准 120 维设计参数向量 x0。"""
    return BASELINE_X0.copy()


def unpack_blade_params(x: np.ndarray) -> List[Dict[str, Any]]:
    """将 120 维向量解构为 6 个控制站位的结构化字典列表。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) != DIM:
        raise ValueError(f"Expected blade parameters vector of dimension {DIM}, got {len(x)}")

    stations = []
    for k in range(NUM_STATIONS):
        idx = k * 20
        sec_theta = x[idx : idx + 18]
        chord = float(x[idx + 18])
        twist = float(x[idx + 19])

        u_w = sec_theta[0:8]
        l_w = sec_theta[8:16]
        le_w = float(sec_theta[16])
        te_t = float(sec_theta[17])

        kulfan_dict = {
            "upper_weights": u_w,
            "lower_weights": l_w,
            "leading_edge_weight": le_w,
            "TE_thickness": te_t,
        }

        stations.append({
            "station_idx": k,
            "global_idx": int(STATION_INDICES[k]),
            "r_frac": STATION_RADIANS_FRAC[k],
            "radius": STATION_RADII[k],
            "theta": sec_theta,
            "kulfan_dict": kulfan_dict,
            "chord": chord,
            "twist": twist,
        })
    return stations


def get_blade_bounds(x0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """获取 120 维 1.5m 风电机组叶片参数的工程可行探索上下界。"""
    if x0 is None:
        x0 = get_baseline_blade_params()

    lb = np.zeros(DIM, dtype=np.float64)
    ub = np.zeros(DIM, dtype=np.float64)

    # 6 个站位的弦长与扭角工程允许范围 (单位: m, deg)
    chord_bounds = [
        (0.60, 1.25),   # 站位 1 (r=0.25m): 根部，基准 1.1118m
        (0.45, 1.00),   # 站位 2 (r=0.45m): 内展，基准 0.8236m
        (0.35, 0.80),   # 站位 3 (r=0.70m): 中展内侧，基准 0.5838m
        (0.25, 0.60),   # 站位 4 (r=1.00m): 中展主功率区，基准 0.4220m
        (0.18, 0.45),   # 站位 5 (r=1.25m): 外展减阻区，基准 0.3469m
        (0.12, 0.35),   # 站位 6 (r=1.50m): 翼尖薄弦区，基准 0.2851m
    ]
    twist_bounds = [
        (20.0, 35.0),    # 站位 1 (r=0.25m): 根部大安装角，基准 27.2602 deg
        (14.0, 26.0),    # 站位 2 (r=0.45m): 肩部过渡，基准 18.9417 deg
        ( 8.0, 18.0),    # 站位 3 (r=0.70m): 内展段，基准 11.8643 deg
        ( 4.0, 13.0),    # 站位 4 (r=1.00m): 中展段，基准 7.0680 deg
        ( 1.5,  9.0),    # 站位 5 (r=1.25m): 外展负扭洗退，基准 4.6233 deg
        ( 0.0,  7.0),    # 站位 6 (r=1.50m): 翼尖气动洗退，基准 2.9318 deg
    ]

    for k in range(NUM_STATIONS):
        idx = k * 20
        base_sec = x0[idx : idx + 18]

        # 1. 翼型上表面权重 (吸力面控制，扰动范围 +-0.25)
        lb[idx : idx + 8] = base_sec[0:8] - 0.25
        ub[idx : idx + 8] = base_sec[0:8] + 0.25

        # 2. 翼型下表面权重 (压力面控制，扰动范围 +-0.25)
        lb[idx + 8 : idx + 16] = base_sec[8:16] - 0.25
        ub[idx + 8 : idx + 16] = base_sec[8:16] + 0.25

        # 3. 前缘微调权重
        lb[idx + 16] = base_sec[16] - 0.18
        ub[idx + 16] = base_sec[16] + 0.18

        # 4. 后缘厚度
        lb[idx + 17] = max(0.0, base_sec[17] - 0.005)
        ub[idx + 17] = base_sec[17] + 0.010

        # 5. 弦长上下界
        lb[idx + 18], ub[idx + 18] = chord_bounds[k]

        # 6. 扭角上下界
        lb[idx + 19], ub[idx + 19] = twist_bounds[k]

    return lb, ub


def interpolate_blade_span(
    x: np.ndarray,
    n_stations: int = 27,
) -> Dict[str, np.ndarray]:
    """将 6 个控制站位平滑插值拓展到沿展向的 27 截面集合。"""
    stations = unpack_blade_params(x)

    if n_stations == 27 or n_stations == 50:
        r = BEM_27_RADII.copy()
        dr = BEM_27_DR.copy()
        ch_base = BEM_27_CHORD.copy()
        tw_base = BEM_27_TWIST.copy()
        th_base = BEM_27_KULFAN_THETAS.copy()
    else:
        r = np.linspace(HUB_RADIUS, ROTOR_RADIUS, n_stations)
        dr = np.gradient(r)
        ch_base = np.interp(r, BEM_27_RADII, BEM_27_CHORD)
        tw_base = np.interp(r, BEM_27_RADII, BEM_27_TWIST)
        th_base = np.zeros((n_stations, 18), dtype=np.float64)
        for dim_i in range(18):
            th_base[:, dim_i] = np.interp(r, BEM_27_RADII, BEM_27_KULFAN_THETAS[:, dim_i])

    # 若输入为基准向量，严格返回官方原始几何
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

    chord_span = ch_base + np.interp(r, r_ctrl, delta_chord_ctrl)
    twist_span_deg = tw_base + np.interp(r, r_ctrl, delta_twist_ctrl)
    twist_span_rad = np.radians(twist_span_deg)

    thetas_6 = np.array([s["theta"] for s in stations])
    delta_thetas_6 = thetas_6 - BASELINE_STATION_THETAS

    kulfan_span = th_base.copy()
    for dim_i in range(18):
        kulfan_span[:, dim_i] += np.interp(
            r,
            STATION_RADII,
            delta_thetas_6[:, dim_i],
            left=delta_thetas_6[0, dim_i],
            right=delta_thetas_6[-1, dim_i],
        )

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
    n_span: int = 27,
    n_chord: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 3D 叶片表面的结构化空间三维点云坐标网格 (X, Y, Z)。
    
    返回:
        (X, Y, Z): shape (N_span, 2 * n_chord) 网格坐标矩阵 (严禁截断任何展向截面)
    """
    if len(x) in (114, 132):
        from src.parameterizations import BSplineBladeModel
        model_bs = BSplineBladeModel()
        span_data = model_bs.to_span_data(x, n_stations=n_span)
    else:
        span_data = interpolate_blade_span(x, n_stations=n_span)

    r_span = span_data["r"]
    chords = span_data["chord"]
    twists_rad = span_data["twist_rad"]
    thetas = span_data["kulfan_thetas"]
    N_actual = len(r_span)

    X_mesh = np.zeros((N_actual, 2 * n_chord), dtype=np.float64)
    Y_mesh = np.zeros((N_actual, 2 * n_chord), dtype=np.float64)
    Z_mesh = np.zeros((N_actual, 2 * n_chord), dtype=np.float64)

    # 弦向余弦加密采样
    beta_chord = np.linspace(0.0, np.pi, n_chord)
    x_c = 0.5 * (1.0 - np.cos(beta_chord))

    for i in range(N_actual):
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

        try:
            yu = k_af.upper_coordinates(x_c)[:, 1]
            yl = k_af.lower_coordinates(x_c)[:, 1]
        except Exception:
            yu = 0.10 * np.sin(x_c * np.pi)
            yl = -0.10 * np.sin(x_c * np.pi)

        # 闭合轮廓：上表面 (从后缘至前缘) + 下表面 (从前缘至后缘)
        x_sec = np.concatenate([x_c[::-1], x_c])
        y_sec = np.concatenate([yu[::-1], yl])

        # 考虑以 1/4 弦长 (0.25c) 为俯仰扭转轴心并缩放至真实尺寸
        x_local = (x_sec - 0.25) * c_i
        y_local = y_sec * c_i

        # 绕 Z 轴进行局部扭角旋转 (beta_i)
        cos_b = np.cos(beta_i)
        sin_b = np.sin(beta_i)
        x_rot = x_local * cos_b - y_local * sin_b
        y_rot = x_local * sin_b + y_local * cos_b

        X_mesh[i, :] = x_rot
        Y_mesh[i, :] = y_rot
        Z_mesh[i, :] = r_i

    return X_mesh, Y_mesh, Z_mesh
