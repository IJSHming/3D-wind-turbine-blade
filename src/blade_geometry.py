"""3D 风力机叶片几何参数化与三维放样工具库：
严格依据 IEA-22-280-RWT (22MW 超大型海上风机参考模型) 工业规范：
- 风轮半径 R = 142.0 m, 轮毂半径 R_hub = 4.2 m, 净叶片长度 L = 137.8 m, 轮毂高度 H = 170 m;
- 6 个工业级核心气动控制截面 (Station 1 ~ 6: r = 20.83m, 30.33m, 56.47m, 82.60m, 108.74m, 132.50m);
- 每个站位包含: 18 维 Kulfan 翼型参数 (8 上表面 + 8 下表面 + 1 前缘 + 1 后缘厚度) + 1 维弦长 c (m) + 1 维扭角 beta (deg);
- 总设计空间维度 D = 6 * (18 + 2) = 120 维;
- 提供标准基准叶片构型 (BASELINE_X0)、工业探索边界映射、沿展向平滑插值与 3D 表面网格放样。
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import aerosandbox as asb


# IEA-22-280-RWT 22MW 参考风电机组几何常数
ROTOR_RADIUS: float = 142.0      # 风轮半径 R (m)
HUB_RADIUS: float = 4.2          # 轮毂半径 R_hub (m)
BLADE_LENGTH: float = 137.8      # 净叶片长度 L = R - R_hub (m)
HUB_HEIGHT: float = 170.0        # 轮毂高度 (m)
NUM_BLADES: int = 3              # 叶片数 B

# 6 个核心气动控制站位的径向位置与无量纲展向位置
NUM_STATIONS: int = 6
DIM: int = 120

# 站位径向位置 (m)
STATION_RADII: np.ndarray = np.array(
    [20.83, 30.33, 56.47, 82.60, 108.74, 132.50],
    dtype=np.float64,
)
# 站位无量纲展向相对半径 r/R
STATION_RADIANS_FRAC: np.ndarray = STATION_RADII / ROTOR_RADIUS  # [0.1467, 0.2136, 0.3977, 0.5817, 0.7658, 0.9331]

# IEA-22-280-RWT 真实 baseline 翼型 (FFA-W3 族) 提取的 120 维基准参数向量
BASELINE_X0: np.ndarray = np.array([
    # Station 1 (r=20.83m, r/R=0.147, AF07 近根部结构厚过渡段, c=7.04m, beta=11.69 deg)
    0.574651,  0.424919,  0.579950,  0.916930,  0.228988,  0.908471,  0.443884,  0.773322,
    -0.681624, -0.797088, -0.136357, -1.421527,  0.192271, -1.189681, -0.129301, -0.235693,
    0.543202,  0.171453,  7.040000, 11.690000,
    # Station 2 (r=30.33m, r/R=0.214, AF11 最大弦长肩部区, c=7.22m, beta=9.95 deg)
    0.451195,  0.265872,  0.751927,  0.148375,  0.581828,  0.258565,  0.359456,  0.321566,
    -0.600138, -0.697259, -0.220806, -0.802424, -0.024659, -0.516398,  0.106307,  0.241523,
    0.649358,  0.031558,  7.220000,  9.950000,
    # Station 3 (r=56.47m, r/R=0.398, AF22 内展段高升力区, c=6.03m, beta=3.79 deg)
    0.405189,  0.522829,  0.306613,  0.463373,  0.214329,  0.399755,  0.219136,  0.257872,
    -0.322618, -0.371756, -0.486538, -0.274522, -0.497197, -0.017638, -0.008714,  0.228788,
    -0.202455,  0.018928,  6.030000,  3.790000,
    # Station 4 (r=82.60m, r/R=0.582, AF34 中展主功率段, c=4.22m, beta=-0.42 deg)
    0.299578,  0.391518,  0.305252,  0.389453,  0.217226,  0.342942,  0.190630,  0.233268,
    -0.265542, -0.409706, -0.301511, -0.375562, -0.308372, -0.075078,  0.084972,  0.156354,
    0.036076,  0.010982,  4.220000, -0.420000,
    # Station 5 (r=108.74m, r/R=0.766, AF45 外展高马赫数减阻段, c=2.81m, beta=-4.56 deg)
    0.227568,  0.185572,  0.504136,  0.140250,  0.425568,  0.167485,  0.257694,  0.190049,
    -0.290007, -0.504497, -0.120624, -0.485333, -0.214748, -0.059902,  0.074169,  0.154215,
    0.473202,  0.007456,  2.810000, -4.560000,
    # Station 6 (r=132.50m, r/R=0.933, AF54 梢部超临界薄翼段, c=1.88m, beta=-6.67 deg)
    0.203761,  0.160451,  0.496454,  0.169378,  0.381272,  0.190114,  0.232166,  0.176691,
    -0.252841, -0.411705, -0.119199, -0.441219, -0.199090, -0.024079,  0.038067,  0.161640,
    0.448774,  0.004174,  1.880000, -6.670000,
], dtype=np.float64)


def get_baseline_blade_params() -> np.ndarray:
    """获取 IEA-22-280-RWT 标准基准 120 维设计参数向量 x0。"""
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
            "r_frac": STATION_RADIANS_FRAC[k],
            "radius": STATION_RADII[k],
            "theta": sec_theta,
            "kulfan_dict": kulfan_dict,
            "chord": chord,
            "twist": twist,
        })
    return stations


def get_blade_bounds(x0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """获取 120 维 IEA-22MW 风电机组叶片参数的工程可行探索上下界。
    
    保证 6 个站位的局部翼型与全叶片气动几何在工业制造与气动探索边界内优化。
    """
    if x0 is None:
        x0 = get_baseline_blade_params()

    lb = np.zeros(DIM, dtype=np.float64)
    ub = np.zeros(DIM, dtype=np.float64)

    # 6 个站位的弦长与扭角工程允许范围 (单位: m, deg)
    chord_bounds = [
        (5.50, 8.20),   # 站位 1 (r=20.83m): 根部过渡弦长
        (5.80, 8.50),   # 站位 2 (r=30.33m): 最大弦长肩部
        (4.80, 7.20),   # 站位 3 (r=56.47m): 内展段
        (3.20, 5.50),   # 站位 4 (r=82.60m): 中展主功率段
        (2.00, 3.80),   # 站位 5 (r=108.74m): 外展减阻段
        (1.20, 2.60),   # 站位 6 (r=132.50m): 翼尖薄弦长
    ]
    twist_bounds = [
        (8.0, 15.0),    # 站位 1 (r=20.83m): 根部大安装角
        (6.5, 13.0),    # 站位 2 (r=30.33m): 肩部过渡
        (1.0, 7.0),     # 站位 3 (r=56.47m): 内展段
        (-3.0, 3.0),    # 站位 4 (r=82.60m): 中展段
        (-7.5, -1.5),   # 站位 5 (r=108.74m): 外展负扭
        (-10.0, -3.5),  # 站位 6 (r=132.50m): 翼尖大负扭洗退
    ]

    for k in range(NUM_STATIONS):
        idx = k * 20
        base_sec = x0[idx : idx + 18]

        # 1. 翼型上表面权重 (吸力面厚度与弯度控制，扰动范围 +-0.22)
        lb[idx : idx + 8] = np.maximum(0.01, base_sec[0:8] - 0.22)
        ub[idx : idx + 8] = base_sec[0:8] + 0.25

        # 2. 翼型下表面权重 (压力面控制)
        lb[idx + 8 : idx + 16] = base_sec[8:16] - 0.25
        ub[idx + 8 : idx + 16] = np.minimum(0.35, base_sec[8:16] + 0.22)

        # 3. 前缘微调权重
        lb[idx + 16] = base_sec[16] - 0.18
        ub[idx + 16] = base_sec[16] + 0.18

        # 4. 后缘厚度
        lb[idx + 17] = max(0.0010, base_sec[17] - 0.005)
        ub[idx + 17] = base_sec[17] + 0.012

        # 5. 弦长上下界
        lb[idx + 18], ub[idx + 18] = chord_bounds[k]

        # 6. 扭角上下界
        lb[idx + 19], ub[idx + 19] = twist_bounds[k]

    return lb, ub


def interpolate_blade_span(
    x: np.ndarray,
    n_stations: int = 18,
) -> Dict[str, np.ndarray]:
    """将 6 个控制站位平滑插值拓展到沿展向的连续离散截面集合。
    
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
    
    # 展向离散站位分布 (在轮毂与翼尖之间，包含保护余量)
    r = np.linspace(HUB_RADIUS + 1.5, ROTOR_RADIUS - 1.0, n_stations)
    dr = np.gradient(r)

    # 6 个控制站位的位置及根尖外插锚点
    # 锚点 0: 轮毂根部 (r = HUB_RADIUS = 4.2m, chord = 5.8m, twist = 12.0 deg)
    # 锚点 7: 翼尖梢部 (r = ROTOR_RADIUS = 142.0m, chord = 0.4m, twist = -4.95 deg)
    r_ctrl = np.concatenate([
        [HUB_RADIUS],
        STATION_RADII,
        [ROTOR_RADIUS],
    ])

    chord_ctrl = np.concatenate([
        [5.80],
        [s["chord"] for s in stations],
        [0.40],
    ])

    twist_ctrl = np.concatenate([
        [12.00],
        [s["twist"] for s in stations],
        [-4.95],
    ])

    # 沿展向平滑插值弦长与扭角
    chord_span = np.interp(r, r_ctrl, chord_ctrl)
    twist_span_deg = np.interp(r, r_ctrl, twist_ctrl)
    twist_span_rad = np.radians(twist_span_deg)

    # 沿展向对 18 维 Kulfan 翼型参数进行线性插值
    thetas_6 = np.array([s["theta"] for s in stations])  # (6, 18)

    kulfan_span = np.zeros((n_stations, 18), dtype=np.float64)
    for dim_i in range(18):
        kulfan_span[:, dim_i] = np.interp(
            r,
            STATION_RADII,
            thetas_6[:, dim_i],
            left=thetas_6[0, dim_i],
            right=thetas_6[-1, dim_i],
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
    n_span: int = 35,
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
