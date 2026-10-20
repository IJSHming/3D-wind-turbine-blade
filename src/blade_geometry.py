"""3D 风力机叶片几何参数化与三维放样工具库：
涵盖 60 维全叶片设计参数：
- 3 个核心控制站位（Station 1: 内翼段 r/R=0.25; Station 2: 主功率段 r/R=0.55; Station 3: 翼尖段 r/R=0.85）
- 每个站位包含: 18 维 Kulfan 翼型参数 + 1 维弦长 c (m) + 1 维扭角 beta (deg)
- 总维度 D = 3 * (18 + 2) = 60 维。
提供标准基准叶片构型（基于 NREL 5MW 工业几何）、探索边界映射、沿展向插值与 3D 表面点云放样功能。
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import aerosandbox as asb


# NREL 5MW 参考风轮几何常数
ROTOR_RADIUS: float = 63.0       # 风轮半径 R (m)
HUB_RADIUS: float = 1.5          # 轮毂半径 R_hub (m)
NUM_BLADES: int = 3              # 叶片数 B
BLADE_LENGTH: float = ROTOR_RADIUS - HUB_RADIUS  # 净叶片长度 (m)

# 3 个核心气动控制站位的无量纲展向位置
STATION_RADIANS_FRAC: np.ndarray = np.array([0.25, 0.55, 0.85], dtype=np.float64)
STATION_RADII: np.ndarray = HUB_RADIUS + STATION_RADIANS_FRAC * BLADE_LENGTH  # [16.875, 35.325, 53.775] m


def get_baseline_blade_params() -> np.ndarray:
    """构建标准基准 3D 风力机叶片的 60 维设计参数向量 x0。
    
    基准翼型选型对齐大型兆瓦级风机典型构型：
    - 站位 1 (内展段, r/R=0.25): NACA 4424 结构厚翼型 (t/c ~ 24%), 弦长 4.25m, 扭角 11.5 deg
    - 站位 2 (中展段, r/R=0.55): NACA 4418 高效气动翼型 (t/c ~ 18%), 弦长 3.45m, 扭角 5.2 deg
    - 站位 3 (外展段, r/R=0.85): NACA 4412 薄阻力翼型 (t/c ~ 12%), 弦长 2.15m, 扭角 1.2 deg
    
    返回:
        np.ndarray: shape (60,), float64
    """
    af_names = ["naca4424", "naca4418", "naca4412"]
    base_chords = [4.25, 3.45, 2.15]
    base_twists = [11.5, 5.2, 1.2]

    x0_parts = []
    for name, c, tw in zip(af_names, base_chords, base_twists):
        af = asb.Airfoil(name).to_kulfan_airfoil(n_weights_per_side=8)
        kulfan_18 = np.concatenate([
            np.asarray(af.upper_weights, dtype=np.float64),
            np.asarray(af.lower_weights, dtype=np.float64),
            [float(af.leading_edge_weight)],
            [float(af.TE_thickness)],
        ])
        # 拼接 18 维翼型参数 + 弦长 + 扭角 -> 20 维/站位
        sec_20 = np.concatenate([kulfan_18, [c], [tw]])
        x0_parts.append(sec_20)

    x0 = np.concatenate(x0_parts).astype(np.float64)
    return x0


def unpack_blade_params(x: np.ndarray) -> List[Dict[str, Any]]:
    """将 60 维向量解构为 3 个站位的结构化字典列表。"""
    x = np.asarray(x, dtype=np.float64)
    stations = []
    for k in range(3):
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
    """获取 60 维风机叶片参数的工程可行探索上下界。
    
    确保每个站位的几何形状与全叶片展向气动参数在真实工业物理边界内优化。
    """
    if x0 is None:
        x0 = get_baseline_blade_params()

    lb = np.zeros(60, dtype=np.float64)
    ub = np.zeros(60, dtype=np.float64)

    # 3 个站位的弦长与扭角工程允许范围
    chord_bounds = [
        (3.30, 5.20),   # 站位 1 根部过渡区弦长 (m)
        (2.50, 4.30),   # 站位 2 主功率段弦长 (m)
        (1.20, 2.70),   # 站位 3 翼尖段弦长 (m)
    ]
    twist_bounds = [
        (8.0, 16.0),    # 站位 1 扭角 (deg)
        (2.5, 9.0),     # 站位 2 扭角 (deg)
        (-1.0, 4.0),    # 站位 3 扭角 (deg)
    ]

    for k in range(3):
        idx = k * 20
        base_sec = x0[idx : idx + 18]

        # 1. 翼型上表面权重 (正值吸力面控制，适度扰动)
        lb[idx : idx + 8] = np.maximum(0.02, base_sec[0:8] - 0.22)
        ub[idx : idx + 8] = base_sec[0:8] + 0.25

        # 2. 翼型下表面权重 (负值下表面与弯度控制)
        lb[idx + 8 : idx + 16] = base_sec[8:16] - 0.25
        ub[idx + 8 : idx + 16] = np.minimum(0.05, base_sec[8:16] + 0.22)

        # 3. 前缘微调权重 (围绕基准值自适应扰动)
        lb[idx + 16] = base_sec[16] - 0.15
        ub[idx + 16] = base_sec[16] + 0.15

        # 4. 后缘厚度
        lb[idx + 17] = max(0.0010, base_sec[17] - 0.003)
        ub[idx + 17] = base_sec[17] + 0.005

        # 5. 弦长上下界
        lb[idx + 18], ub[idx + 18] = chord_bounds[k]

        # 6. 扭角上下界
        lb[idx + 19], ub[idx + 19] = twist_bounds[k]

    return lb, ub


def interpolate_blade_span(
    x: np.ndarray,
    n_stations: int = 15,
) -> Dict[str, np.ndarray]:
    """将 3 个控制站位平滑插值拓展到沿展向的连续离散截面集合。
    
    返回:
        {
            "r": 半径 (m),
            "dr": 微元长度 (m),
            "chord": 各截面局部弦长 (m),
            "twist_deg": 各截面局部扭角 (deg),
            "twist_rad": 各截面局部扭角 (rad),
            "kulfan_thetas": 各截面插值出的 18 维 Kulfan 参数 (n_stations, 18),
            "station_weights": 各截面对控制站位的权重矩阵 (n_stations, 3)
        }
    """
    stations = unpack_blade_params(x)
    
    # 展向离散站位分布 (在轮毂与翼尖之间，余弦加密以提高根部和尖部精度)
    r = np.linspace(HUB_RADIUS + 0.8, ROTOR_RADIUS - 0.3, n_stations)
    dr = np.gradient(r)

    # 3 个控制站位的位置及外插边界锚点
    # 锚点 0: 轮毂根部 (r = HUB_RADIUS, chord = 4.8m, twist = 13.5 deg)
    # 锚点 4: 翼尖梢部 (r = ROTOR_RADIUS, chord = 1.05m, twist = 0.0 deg)
    r_ctrl = np.array([
        HUB_RADIUS,
        stations[0]["radius"],
        stations[1]["radius"],
        stations[2]["radius"],
        ROTOR_RADIUS,
    ])

    chord_ctrl = np.array([
        stations[0]["chord"] * 1.15,
        stations[0]["chord"],
        stations[1]["chord"],
        stations[2]["chord"],
        max(0.9, stations[2]["chord"] * 0.50),
    ])

    twist_ctrl = np.array([
        stations[0]["twist"] + 2.0,
        stations[0]["twist"],
        stations[1]["twist"],
        stations[2]["twist"],
        min(0.0, stations[2]["twist"] - 1.0),
    ])

    # 沿展向平滑插值弦长与扭角
    chord_span = np.interp(r, r_ctrl, chord_ctrl)
    twist_span_deg = np.interp(r, r_ctrl, twist_ctrl)
    twist_span_rad = np.radians(twist_span_deg)

    # 沿展向对 18 维 Kulfan 翼型参数进行线性权重混合插值
    ctrl_radii_3 = np.array([s["radius"] for s in stations])  # [16.875, 35.325, 53.775]
    thetas_3 = np.array([s["theta"] for s in stations])        # (3, 18)

    kulfan_span = np.zeros((n_stations, 18), dtype=np.float64)
    for dim_i in range(18):
        kulfan_span[:, dim_i] = np.interp(
            r,
            ctrl_radii_3,
            thetas_3[:, dim_i],
            left=thetas_3[0, dim_i],
            right=thetas_3[2, dim_i],
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
    n_span: int = 25,
    n_chord: int = 60,
) -> Dict[str, np.ndarray]:
    """生成 3D 风机叶片的三维放样表面坐标网格 (X, Y, Z)，供三维立体渲染与气动特征可视化。
    
    坐标系定义 (风机行业标准):
    - Z 轴: 沿叶片展向从轮毂指向翼尖 (0 -> 63m)
    - X 轴: 弦向由前缘指向后缘
    - Y 轴: 厚度吸力面/压力面法向 (包含局部扭角倾角旋转)
    
    返回:
        {
            "X_surf": (n_span, n_chord) 网格,
            "Y_surf": (n_span, n_chord) 网格,
            "Z_surf": (n_span, n_chord) 网格,
            "chords": (n_span,),
            "twists": (n_span,),
            "radii": (n_span,),
        }
    """
    span_data = interpolate_blade_span(x, n_stations=n_span)
    radii = span_data["r"]
    chords = span_data["chord"]
    twists_rad = span_data["twist_rad"]
    kulfans = span_data["kulfan_thetas"]

    # 余弦分布翼型弦向采样
    n_pts_half = n_chord // 2
    beta = np.linspace(0, np.pi, n_pts_half)
    xc = 0.5 * (1.0 - np.cos(beta))
    n_surf_pts = 2 * n_pts_half - 1

    X_surf = np.zeros((n_span, n_surf_pts), dtype=np.float64)
    Y_surf = np.zeros((n_span, n_surf_pts), dtype=np.float64)
    Z_surf = np.zeros((n_span, n_surf_pts), dtype=np.float64)

    for i in range(n_span):
        r_val = radii[i]
        c_val = chords[i]
        tw = twists_rad[i]
        th = kulfans[i]

        k_af = asb.KulfanAirfoil(
            upper_weights=th[0:8],
            lower_weights=th[8:16],
            leading_edge_weight=float(th[16]),
            TE_thickness=float(th[17]),
        )

        try:
            yu = k_af.upper_coordinates(xc)[:, 1]
            yl = k_af.lower_coordinates(xc)[:, 1]
        except Exception:
            yu = np.zeros_like(xc)
            yl = np.zeros_like(xc)

        # 构成闭合翼型剖面 (上表面后缘 -> 前缘 -> 下表面后缘)
        x_norm = np.concatenate([xc[::-1], xc[1:]])
        y_norm = np.concatenate([yu[::-1], yl[1:]])

        # 调整前缘对齐至 0，后缘为 1，进行弦长缩放
        x_sec = (x_norm - 0.25) * c_val  # 围绕 1/4 弦线进行扭角旋转
        y_sec = y_norm * c_val

        # 扭角旋转 (绕 1/4 弦线)
        cos_tw = np.cos(tw)
        sin_tw = np.sin(tw)
        x_rot = x_sec * cos_tw - y_sec * sin_tw
        y_rot = x_sec * sin_tw + y_sec * cos_tw

        X_surf[i, :] = x_rot
        Y_surf[i, :] = y_rot
        Z_surf[i, :] = r_val

    return {
        "X_surf": X_surf,
        "Y_surf": Y_surf,
        "Z_surf": Z_surf,
        "chords": chords,
        "twists": np.degrees(twists_rad),
        "radii": radii,
    }
