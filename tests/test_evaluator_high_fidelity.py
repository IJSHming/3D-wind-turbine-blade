"""高保真评估器单元与物理特性集成测试 (tests/test_evaluator_high_fidelity.py):
验证 NREL Phase VI (Sequence H) 物理对齐:
1. NeuralFoil 锁定为 'xxxlarge' 最精准模式;
2. 27 截面 S809 叶片与基准工况 (7.0 m/s, 72.0 rpm, 4.815 deg, air rho=1.225);
3. 机头仰角与叶片预锥角几何投影、垂直风剪切默认配置;
4. Viterna-Corrigan 深失速气动极线全域外推 (大攻角连续性与鲁棒性);
5. 单次评估耗时严格低于 1.0 秒红线。
"""

import time
import unittest
import numpy as np

from src.blade_geometry import (
    BASELINE_PITCH_DEG,
    BEM_NUM_SECTIONS,
    get_baseline_blade_params,
    interpolate_blade_span,
)
from src.evaluator import WindTurbineBladeEvaluator


class TestHighFidelityEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = WindTurbineBladeEvaluator()
        self.x0 = self.evaluator.baseline_x0

    def test_evaluator_configuration_and_locking(self):
        """测试 NREL Phase VI 物理配置对齐与默认参数"""
        self.assertEqual(self.evaluator.n_bem_stations, BEM_NUM_SECTIONS)
        self.assertEqual(self.evaluator.n_bem_stations, 27)
        self.assertAlmostEqual(self.evaluator.wind_speed, 7.0)
        self.assertAlmostEqual(self.evaluator.rpm, 72.0)
        self.assertAlmostEqual(self.evaluator.air_density, 1.225)
        self.assertAlmostEqual(self.evaluator.viscosity, 1.789e-5)
        self.assertAlmostEqual(self.evaluator.pitch_angle_deg, BASELINE_PITCH_DEG)
        self.assertAlmostEqual(self.evaluator.tilt_angle_deg, 0.0)
        self.assertAlmostEqual(self.evaluator.precone_angle_deg, 0.0)
        self.assertAlmostEqual(self.evaluator.wind_shear_exponent, 0.0)
        self.assertFalse(self.evaluator.enable_roughness)

    def test_baseline_physics_metrics(self):
        """测试 NREL Phase VI 基准叶片在空气流场下的物理指标输出"""
        det = self.evaluator.evaluate_detailed(self.x0)
        self.assertTrue(det["feasible"])
        self.assertFalse(det["is_fatal"])

        # 检查全展向截面维度为 27
        self.assertEqual(len(det["span_radii"]), 27)
        self.assertEqual(len(det["span_chord"]), 27)
        self.assertEqual(len(det["span_twist_deg"]), 27)
        self.assertEqual(len(det["span_alpha_deg"]), 27)
        self.assertEqual(len(det["span_effective_pitch_deg"]), 27)

        # 检查功率、转矩与推力处于 NREL Phase VI (7 m/s, 72 rpm) 物理区间 (~6.07 kW, ~805 Nm, ~1.21 kN)
        self.assertGreater(det["power_kW"], 5.0)
        self.assertLess(det["power_kW"], 7.5)
        self.assertGreater(det["torque_Nm"], 650.0)
        self.assertLess(det["torque_Nm"], 1000.0)
        self.assertGreater(det["thrust_kN"], 0.8)
        self.assertLess(det["thrust_kN"], self.evaluator.max_allowable_thrust_kN)

        # 检查风能利用系数与推力系数
        self.assertGreater(det["Cp"], 0.30)
        self.assertLess(det["Cp"], 0.48)
        self.assertGreater(det["Ct"], 0.35)
        self.assertLess(det["Ct"], 0.80)

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

        # 验证阻力严格为正且在大攻角 (90 deg) 收敛至 Viterna cd_max = 1.11 + 0.018*AR (=1.362 for AR=14)
        self.assertTrue(np.all(cd_extrap > 0.0))
        self.assertAlmostEqual(cd_extrap[-1], 1.362, places=3)

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


if __name__ == "__main__":
    unittest.main()
