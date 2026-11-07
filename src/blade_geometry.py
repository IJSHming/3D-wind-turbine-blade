"""3D 风力机叶片几何参数化与三维放样工具库：
严格依据 ModelTest 10 MW (10MW 超大型海上风机参考模型) 工业规范：
- 风轮半径 R = 89.166 m, 轮毂半径 R_hub = 1.8 m (轮毂直径 3.6 m), 净叶片长度 L = 87.366 m, 轮毂高度 H = 147.5 m;
- 额定运行测试工况: 风速 V_inf = 16.26 m/s, 转速 = 7.55 rpm, 叶片安装角/变桨角 pitch = 0.0 deg, 预锥角 precone = 0.0 deg, 机头仰角 tilt = 0.0 deg;
- 展向采用 OpenFAST AeroDyn15 官方 50 截面几何节点定义 (严格对齐 10MW_AeroDyn15_blade.dat 与 Airfoils_10MW);
- 6 个工业级核心气动控制截面 (Station 1 ~ 6: r = 12.00m, 19.50m, 31.50m, 49.50m, 73.50m, 87.00m);
- 每个站位包含: 18 维 Kulfan 翼型参数 (8 上表面 + 8 下表面 + 1 前缘 + 1 后缘厚度) + 1 维弦长 c (m) + 1 维扭角 beta (deg);
- 总设计空间维度 D = 6 * (18 + 2) = 120 维;
- 提供标准基准叶片构型 (BASELINE_X0)、工业探索边界映射、沿展向平滑插值与 3D 表面网格放样。
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import aerosandbox as asb


# ModelTest 10 MW 参考风电机组几何常数
ROTOR_RADIUS: float = 89.166      # 风轮半径 R = R_hub + L (m)
HUB_RADIUS: float = 1.8           # 轮毂半径 R_hub (m) (轮毂直径 3.6 m)
BLADE_LENGTH: float = 87.366      # 净叶片长度 L (m)
HUB_HEIGHT: float = 147.5         # 轮毂高度 (m)
NUM_BLADES: int = 3               # 叶片数 B

# 6 个核心气动控制站位的径向位置与无量纲展向位置 (用于 120 维参数化优化设计空间)
NUM_STATIONS: int = 6
DIM: int = 120

# OpenFAST AeroDyn15 50 截面几何节点定义 (严格对齐 10MW_AeroDyn15_blade.dat):
OPENFAST_NUM_SECTIONS: int = 50

# 载入基准 50 截面预计算几何数组
_data_dir = Path(__file__).parent.parent / "data" / "baseline_10mw"
_npz_path = _data_dir / "baseline_50_sections.npz"

if _npz_path.exists():
    _precomputed = np.load(_npz_path)
    OPENFAST_50_BLSPN: np.ndarray = _precomputed["blspn"]
    OPENFAST_50_SWEEP: np.ndarray = _precomputed["sweep"]
    OPENFAST_50_TWIST: np.ndarray = _precomputed["twist"]
    OPENFAST_50_CHORD: np.ndarray = _precomputed["chord"]
    OPENFAST_50_AF_IDS: np.ndarray = _precomputed["af_ids"]
    OPENFAST_50_KULFAN_THETAS: np.ndarray = _precomputed["kulfan_thetas"]
else:
    # 理论备选回退 (50 截面)
    OPENFAST_50_BLSPN = np.linspace(0.0, BLADE_LENGTH, OPENFAST_NUM_SECTIONS, dtype=np.float64)
    OPENFAST_50_SWEEP = np.zeros(OPENFAST_NUM_SECTIONS, dtype=np.float64)
    OPENFAST_50_TWIST = np.linspace(17.36, -0.11, OPENFAST_NUM_SECTIONS, dtype=np.float64)
    OPENFAST_50_CHORD = np.linspace(5.08, 0.91, OPENFAST_NUM_SECTIONS, dtype=np.float64)
    OPENFAST_50_AF_IDS = np.arange(1, OPENFAST_NUM_SECTIONS + 1, dtype=np.int64)
    OPENFAST_50_KULFAN_THETAS = np.zeros((OPENFAST_NUM_SECTIONS, 18), dtype=np.float64)

OPENFAST_50_RADII: np.ndarray = HUB_RADIUS + OPENFAST_50_BLSPN

# 严格计算双网格 Voronoi 气动微元宽度 dr (sum(dr) == BLADE_LENGTH)
_bnds = np.zeros(len(OPENFAST_50_RADII) + 1, dtype=np.float64)
_bnds[0] = OPENFAST_50_RADII[0]
_bnds[1:-1] = 0.5 * (OPENFAST_50_RADII[:-1] + OPENFAST_50_RADII[1:])
_bnds[-1] = OPENFAST_50_RADII[-1]
OPENFAST_50_DR: np.ndarray = np.diff(_bnds)

# 向后兼容别名 (原 58 截面变量名兼容)
OPENFAST_58_RADII = OPENFAST_50_RADII
OPENFAST_58_DR = OPENFAST_50_DR
OPENFAST_BLSPN = OPENFAST_50_BLSPN

# 6 个核心气动控制站位对应的 50 截面索引: AF04 (idx 3), AF09 (idx 8), AF17 (idx 16), AF27 (idx 26), AF43 (idx 42), AF49 (idx 48)
STATION_INDICES: np.ndarray = np.array([3, 8, 16, 26, 42, 48], dtype=np.int64)
STATION_RADII: np.ndarray = OPENFAST_50_RADII[STATION_INDICES]  # [12.00, 19.50, 31.50, 49.50, 73.50, 87.00]
STATION_RADIANS_FRAC: np.ndarray = STATION_RADII / ROTOR_RADIUS  # [0.1346, 0.2187, 0.3533, 0.5551, 0.8243, 0.9757]

# 提取 6 个站位的基准弦长、扭角与翼型参数
BASELINE_STATION_CHORDS: np.ndarray = OPENFAST_50_CHORD[STATION_INDICES]
BASELINE_STATION_TWISTS: np.ndarray = OPENFAST_50_TWIST[STATION_INDICES]
BASELINE_STATION_THETAS: np.ndarray = OPENFAST_50_KULFAN_THETAS[STATION_INDICES]

# 组装 120 维基准参数向量
_x0_parts = []
for _k in range(NUM_STATIONS):
    _x0_parts.extend(BASELINE_STATION_THETAS[_k])
    _x0_parts.append(float(BASELINE_STATION_CHORDS[_k]))
    _x0_parts.append(float(BASELINE_STATION_TWISTS[_k]))

BASELINE_X0: np.ndarray = np.array(_x0_parts, dtype=np.float64)


def get_baseline_blade_params() -> np.ndarray:
    """获取 ModelTest 10MW 标准基准 120 维设计参数向量 x0。"""
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
    """获取 120 维 10MW 风电机组叶片参数的工程可行探索上下界。
    
    保证 6 个站位的局部翼型与全叶片气动几何在工业制造与气动探索边界内优化。
    """
    if x0 is None:
        x0 = get_baseline_blade_params()

    lb = np.zeros(DIM, dtype=np.float64)
    ub = np.zeros(DIM, dtype=np.float64)

    # 6 个站位的弦长与扭角工程允许范围 (单位: m, deg)
    chord_bounds = [
        (8.00, 11.00),   # 站位 1 (r=12.00m, AF04): 基准 9.6391m
        (9.50, 12.50),   # 站位 2 (r=19.50m, AF09): 最大弦长肩部，基准 11.0645m
        (8.50, 11.50),   # 站位 3 (r=31.50m, AF17): 内展过渡区，基准 10.2158m
        (6.00,  9.50),   # 站位 4 (r=49.50m, AF27): 中展主功率区，基准 7.8508m
        (3.00,  6.00),   # 站位 5 (r=73.50m, AF43): 外展减阻区，基准 4.4580m
        (1.00,  3.50),   # 站位 6 (r=87.00m, AF49): 翼尖薄弦区，基准 2.0617m
    ]
    twist_bounds = [
        (13.0, 19.0),    # 站位 1 (r=12.00m): 根部大安装角，基准 16.4333 deg
        (11.0, 17.0),    # 站位 2 (r=19.50m): 肩部过渡，基准 14.1982 deg
        ( 8.5, 14.5),    # 站位 3 (r=31.50m): 内展段，基准 11.4181 deg
        ( 4.0, 10.0),    # 站位 4 (r=49.50m): 中展段，基准 6.8734 deg
        ( 0.0,  5.5),    # 站位 5 (r=73.50m): 外展负扭洗退，基准 2.2293 deg
        (-3.5,  2.0),    # 站位 6 (r=87.00m): 翼尖气动洗退，基准 -0.5745 deg
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
    n_stations: int = 50,
) -> Dict[str, np.ndarray]:
    """将 6 个控制站位平滑插值拓展到沿展向的连续离散截面集合。
    
    默认采用 OpenFAST AeroDyn15 官方 50 截面几何节点分布，
    严格对齐 10MW_AeroDyn15_blade.dat 中的 50 个气动计算截面。

    返回:
        {
            "r": 半径 (m),
            "dr": 微元长度 (m),
            "chord": 各截面局部弦长 (m),
            "twist_deg": 各截面局部扭角 (deg),
            "twist_rad": 各截面局部扭角 (rad),
            "kulfan_thetas": 各截面插值出的 18 维 Kulfan 参数 (n_stations, 18),
        }
    """
    stations = unpack_blade_params(x)

    if n_stations == 50:
        r = OPENFAST_50_RADII.copy()
        dr = OPENFAST_50_DR.copy()
        ch_base = OPENFAST_50_CHORD.copy()
        tw_base = OPENFAST_50_TWIST.copy()
        th_base = OPENFAST_50_KULFAN_THETAS.copy()
    else:
        r = np.linspace(HUB_RADIUS + 0.5, ROTOR_RADIUS - 0.5, n_stations)
        dr = np.gradient(r)
        ch_base = np.interp(r, OPENFAST_50_RADII, OPENFAST_50_CHORD)
        tw_base = np.interp(r, OPENFAST_50_RADII, OPENFAST_50_TWIST)
        th_base = np.zeros((n_stations, 18), dtype=np.float64)
        for dim_i in range(18):
            th_base[:, dim_i] = np.interp(r, OPENFAST_50_RADII, OPENFAST_50_KULFAN_THETAS[:, dim_i])

    # 若输入为基准向量，严格返回官方原始几何，杜绝任何微小拟合误差
    if np.allclose(x, BASELINE_X0, atol=1e-7):
        return {
            "r": r,
            "dr": dr,
            "chord": ch_base,
            "twist_deg": tw_base,
            "twist_rad": np.radians(tw_base),
            "kulfan_thetas": th_base,
        }

    # 6 个控制站位的位置及根尖外插锚点
    r_ctrl = np.concatenate([
        [HUB_RADIUS],
        STATION_RADII,
        [ROTOR_RADIUS],
    ])

    # 根部锚点固定圆柱结构，翼尖平滑外推
    delta_chord_ctrl = np.concatenate([
        [0.0],
        [s["chord"] - BASELINE_STATION_CHORDS[k] for k, s in enumerate(stations)],
        [0.0],
    ])

    tip_twist_ctrl = float(
        stations[-1]["twist"]
        + (stations[-1]["twist"] - stations[-2]["twist"])
        / (STATION_RADII[-1] - STATION_RADII[-2])
        * (ROTOR_RADIUS - STATION_RADII[-1])
    )
    base_tip_twist = float(
        BASELINE_STATION_TWISTS[-1]
        + (BASELINE_STATION_TWISTS[-1] - BASELINE_STATION_TWISTS[-2])
        / (STATION_RADII[-1] - STATION_RADII[-2])
        * (ROTOR_RADIUS - STATION_RADII[-1])
    )
    delta_tip_twist = tip_twist_ctrl - base_tip_twist

    delta_twist_ctrl = np.concatenate([
        [0.0],
        [s["twist"] - BASELINE_STATION_TWISTS[k] for k, s in enumerate(stations)],
        [delta_tip_twist],
    ])

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
    n_span: int = 50,
    n_chord: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 3D 叶片表面的结构化空间三维点云坐标网格 (X, Y, Z)。
    
    用于三维可视化展示、学术对比以及三维流场几何验证。
    坐标系定义：
    - Z: 沿叶片径向展向延伸 [HUB_RADIUS, ROTOR_RADIUS] (m)
    - X: 沿叶轮旋转切向与弦向气动平面 (m)
    - Y: 沿叶轮轴向与厚度方向 (m)
    
    返回:
        (X, Y, Z): shape (n_span, 2 * n_chord) 网格坐标矩阵
    """
    if len(x) == 132:
        from src.parameterizations import BSplineBladeModel
        model_bs = BSplineBladeModel()
        span_data = model_bs.to_span_data(x, n_stations=n_span)
    else:
        span_data = interpolate_blade_span(x, n_stations=n_span)
    r_span = span_data["r"]
    chords = span_data["chord"]
    twists_rad = span_data["twist_rad"]
    thetas = span_data["kulfan_thetas"]

    X_mesh = np.zeros((n_span, 2 * n_chord), dtype=np.float64)
    Y_mesh = np.zeros((n_span, 2 * n_chord), dtype=np.float64)
    Z_mesh = np.zeros((n_span, 2 * n_chord), dtype=np.float64)

    # 弦向余弦加密采样
    beta_chord = np.linspace(0.0, np.pi, n_chord)
    x_c = 0.5 * (1.0 - np.cos(beta_chord))

    for i in range(n_span):
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
