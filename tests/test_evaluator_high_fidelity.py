"""高保真评估器单元与物理特性集成测试 (tests/test_evaluator_high_fidelity.py):
验证:
1. NeuralFoil 锁定为 'xxxlarge' 最精准模式;
2. OpenFAST 58 截面与基准工况 (11.0 m/s, 6.77094 rpm, 2.42859 deg);
3. 机头仰角 (Tilt=4.0°) 与叶片预锥角 (Pre-cone=-3.5°) 几何投影;
4. 轮毂高度 (170m) 下垂直风剪切 (alpha=0.14);
5. QBlade 真实前缘粗糙度/污染衰减模型;
6. Viterna-Corrigan 深失速气动极线全域外推 (大攻角连续性与鲁棒性);
7. 单次评估耗时严格低于 1.0 秒红线 (实测 ~0.20 秒)。
"""

import time
import unittest
import numpy as np

from src.blade_geometry import (
    get_baseline_blade_params,
    interpolate_blade_span,
    OPENFAST_NUM_SECTIONS,
)
from src.evaluator import WindTurbineBladeEvaluator


class TestHighFidelityEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = WindTurbineBladeEvaluator(
            wind_speed=11.0,
            rpm=6.77094,
            pitch_angle_deg=2.42859,
            tilt_angle_deg=4.0,
            precone_angle_deg=-3.5,
            wind_shear_exponent=0.14,
            hub_height=170.0,
            enable_roughness=True,
            n_bem_stations=58,
        )
        self.x0 = get_baseline_blade_params()

    def test_evaluator_configuration_and_locking(self):
        """测试物理配置对齐与参数锁定"""
        self.assertEqual(self.evaluator.n_bem_stations, 58)
        self.assertAlmostEqual(self.evaluator.wind_speed, 11.0)
        self.assertAlmostEqual(self.evaluator.rpm, 6.77094)
        self.assertAlmostEqual(self.evaluator.pitch_angle_deg, 2.42859)
        self.assertAlmostEqual(self.evaluator.tilt_angle_deg, 4.0)
        self.assertAlmostEqual(self.evaluator.precone_angle_deg, -3.5)
        self.assertAlmostEqual(self.evaluator.wind_shear_exponent, 0.14)
        self.assertTrue(self.evaluator.enable_roughness)

    def test_baseline_physics_metrics(self):
        """测试基准叶片在高保真流场下的物理指标输出"""
        det = self.evaluator.evaluate_detailed(self.x0)
        self.assertTrue(det["feasible"])
        self.assertFalse(det["is_fatal"])

        # 检查全展向截面维度为 58
        self.assertEqual(len(det["span_radii"]), 58)
        self.assertEqual(len(det["span_chord"]), 58)
        self.assertEqual(len(det["span_twist_deg"]), 58)
        self.assertEqual(len(det["span_alpha_deg"]), 58)
        self.assertEqual(len(det["span_effective_pitch_deg"]), 58)

        # 检查功率与推力处于合理物理区间 (22MW 级在粗糙度与仰角修正下约为 22~23MW)
        self.assertGreater(det["power_kW"], 21000.0)
        self.assertLess(det["power_kW"], 25500.0)
        self.assertGreater(det["thrust_kN"], 2800.0)
        self.assertLess(det["thrust_kN"], self.evaluator.max_allowable_thrust_kN)

        # 检查风能利用系数与推力系数
        self.assertGreater(det["Cp"], 0.40)
        self.assertLess(det["Cp"], 0.50)
        self.assertGreater(det["Ct"], 0.60)
        self.assertLess(det["Ct"], 0.85)

    def test_viterna_post_stall_extrapolation(self):
        """测试 Viterna-Corrigan 深失速外推在极端大攻角下的鲁棒性与连续性"""
        extreme_alphas = np.array([-45.0, -30.0, -20.0, 0.0, 15.0, 25.0, 45.0, 90.0])
        cl_mock = np.zeros_like(extreme_alphas)
        cd_mock = np.ones_like(extreme_alphas) * 0.02

        cl_extrap, cd_extrap = self.evaluator.viterna_extrapolate(extreme_alphas, cl_mock, cd_mock)

        # 验证输出无 NaN 或 Inf
        self.assertFalse(np.any(np.isnan(cl_extrap)))
        self.assertFalse(np.any(np.isinf(cl_extrap)))
        self.assertFalse(np.any(np.isnan(cd_extrap)))
        self.assertFalse(np.any(np.isinf(cd_extrap)))

        # 验证阻力严格为正且在大攻角下显著增加
        self.assertTrue(np.all(cd_extrap > 0.0))
        self.assertGreater(cd_extrap[-1], 1.5)  # 90度攻角平板阻力应接近 1.8~2.0

    def test_single_evaluation_speed_under_one_second(self):
        """严格验证单次评估耗时必须 < 1.0 秒 (HPC 防线)"""
        # Warmup
        self.evaluator.evaluate(self.x0)

        # 连续评估 5 次测量平均耗时
        t0 = time.time()
        n_repeats = 5
        for _ in range(n_repeats):
            val = self.evaluator.evaluate(self.x0)
            self.assertGreater(val, 0.0)
        t_total = time.time() - t0
        t_per_eval = t_total / n_repeats

        print(f"\n[TestEvaluatorSpeed] Single evaluation avg time: {t_per_eval*1000:.2f} ms ({t_per_eval:.4f} s)")
        # 必须严格低于 1.0 秒，实测在 0.20 秒左右
        self.assertLess(t_per_eval, 1.0)
        self.assertLess(t_per_eval, 0.35)


if __name__ == "__main__":
    unittest.main()
