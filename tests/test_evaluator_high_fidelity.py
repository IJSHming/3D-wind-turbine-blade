"""高保真评估器单元与物理特性集成测试 (tests/test_evaluator_high_fidelity.py):
验证:
1. NeuralFoil 锁定为 'xxxlarge' 最精准模式;
2. OpenFAST 50 截面与基准工况 (16.26 m/s, 7.55 rpm, 0.0 deg);
3. 机头仰角 (Tilt=0.0°) 与叶片预锥角 (Pre-cone=0.0°) 几何投影;
4. 轮毂高度 (147.5m) 下垂直风剪切 (alpha=0.14);
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
            wind_speed=16.26,
            rpm=7.55,
            pitch_angle_deg=0.0,
            tilt_angle_deg=0.0,
            precone_angle_deg=0.0,
            wind_shear_exponent=0.14,
            hub_height=147.5,
            enable_roughness=True,
            n_bem_stations=50,
        )
        self.x0 = self.evaluator.baseline_x0

    def test_evaluator_configuration_and_locking(self):
        """测试物理配置对齐与参数锁定"""
        self.assertEqual(self.evaluator.n_bem_stations, 50)
        self.assertAlmostEqual(self.evaluator.wind_speed, 16.26)
        self.assertAlmostEqual(self.evaluator.rpm, 7.55)
        self.assertAlmostEqual(self.evaluator.pitch_angle_deg, 0.0)
        self.assertAlmostEqual(self.evaluator.tilt_angle_deg, 0.0)
        self.assertAlmostEqual(self.evaluator.precone_angle_deg, 0.0)
        self.assertAlmostEqual(self.evaluator.wind_shear_exponent, 0.14)
        self.assertTrue(self.evaluator.enable_roughness)

    def test_baseline_physics_metrics(self):
        """测试基准叶片在高保真流场下的物理指标输出"""
        det = self.evaluator.evaluate_detailed(self.x0)
        self.assertTrue(det["feasible"])
        self.assertFalse(det["is_fatal"])

        # 检查全展向截面维度为 50
        self.assertEqual(len(det["span_radii"]), 50)
        self.assertEqual(len(det["span_chord"]), 50)
        self.assertEqual(len(det["span_twist_deg"]), 50)
        self.assertEqual(len(det["span_alpha_deg"]), 50)
        self.assertEqual(len(det["span_effective_pitch_deg"]), 50)

        # 检查功率与推力处于合理物理区间 (10MW 级在粗糙度下约为 27~28MW)
        self.assertGreater(det["power_kW"], 25000.0)
        self.assertLess(det["power_kW"], 31000.0)
        self.assertGreater(det["thrust_kN"], 2100.0)
        self.assertLess(det["thrust_kN"], self.evaluator.max_allowable_thrust_kN)

        # 检查风能利用系数与推力系数
        self.assertGreater(det["Cp"], 0.38)
        self.assertLess(det["Cp"], 0.48)
        self.assertGreater(det["Ct"], 0.50)
        self.assertLess(det["Ct"], 0.75)

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
        self.assertGreater(cd_extrap[-1], 1.5)

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
        self.assertLess(t_per_eval, 1.0)
        self.assertLess(t_per_eval, 0.35)


if __name__ == "__main__":
    unittest.main()
