"""多参数化模型库单元测试 (tests/test_parameterizations.py):
验证 BSplineBladeModel (D=114), Stations6BladeModel (D=114) 在 NREL Phase VI 基准下的几何合法性与评估器兼容性。
"""

import unittest
import numpy as np

from src.blade_geometry import BASELINE_PITCH_DEG
from src.evaluator import WindTurbineBladeEvaluator
from src.parameterizations import (
    BSplineBladeModel,
    Stations6BladeModel,
)


class TestBladeParameterizations(unittest.TestCase):

    def test_dimensions_and_bounds(self):
        m6 = Stations6BladeModel()
        self.assertEqual(m6.dim, 120)
        lb6, ub6 = m6.get_bounds()
        self.assertEqual(len(lb6), 120)
        self.assertEqual(len(ub6), 120)
        self.assertTrue(np.all(ub6 >= lb6))

        m_bs = BSplineBladeModel()
        self.assertEqual(m_bs.dim, 114)
        lb_bs, ub_bs = m_bs.get_bounds()
        self.assertEqual(len(lb_bs), 114)
        self.assertEqual(len(ub_bs), 114)
        self.assertTrue(np.all(ub_bs >= lb_bs))

    def test_baseline_evaluations_feasible(self):
        for model_cls in [Stations6BladeModel, BSplineBladeModel]:
            model = model_cls()
            ev = WindTurbineBladeEvaluator(
                wind_speed=7.0, rpm=72.0, pitch_angle_deg=BASELINE_PITCH_DEG,
                param_model=model,
            )
            det = ev.evaluate_detailed(model.get_x0())
            self.assertTrue(det["feasible"], f"{model.name} baseline should be feasible")
            self.assertGreater(det["power_kW"], 5.0)
            self.assertLess(det["thrust_kN"], 2.0)
            self.assertEqual(len(det["span_chord"]), 27)
            self.assertEqual(len(det["span_twist_deg"]), 27)

    def test_kaya_twist_compatibility(self):
        """验证 Kaya (2019) Phase VI 典型根/尖扭角偏移在 B-Spline 约束空间内保持可行且无负分越界"""
        m_bs = BSplineBladeModel()
        ev = WindTurbineBladeEvaluator(param_model=m_bs)
        x_kaya = m_bs.get_x0()
        # 根部微减扭角 (~ -4 deg) + 中段保持 + 尖部微调
        x_kaya[108:114] = np.array([-4.0, -2.5, -1.0, 0.0, 0.5, 1.0])
        is_fatal, _, _ = m_bs.validate_geometry(x_kaya)
        self.assertFalse(is_fatal)
        self.assertGreater(ev.evaluate(x_kaya), 0.0)

    def test_bspline_smoothness(self):
        m_bs = BSplineBladeModel()
        span = m_bs.to_span_data(m_bs.get_x0())
        d2_c = np.diff(span["chord"], 2)
        d2_tw = np.diff(span["twist_deg"], 2)
        # 气动段二阶差分方差极低，保证连续光滑
        self.assertLess(float(np.var(d2_c)), 0.01)
        self.assertLess(float(np.var(d2_tw)), 0.15)


if __name__ == "__main__":
    unittest.main()
