"""多参数化模型库单元测试 (tests/test_parameterizations.py):
验证 BSplineBladeModel (D=63), Stations3BladeModel (D=60) 在 NREL Phase VI 31 全余弦截面基准下的几何合法性与评估器兼容性。
"""

import unittest
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.blade_geometry import BASELINE_PITCH_DEG, BEM_NUM_SECTIONS
from src.evaluator import WindTurbineBladeEvaluator
from src.parameterizations import (
    BSplineBladeModel,
    Stations3BladeModel,
)


class TestBladeParameterizations(unittest.TestCase):

    def test_dimensions_and_bounds(self):
        m3 = Stations3BladeModel()
        self.assertEqual(m3.dim, 60)
        lb3, ub3 = m3.get_bounds()
        self.assertEqual(len(lb3), 60)
        self.assertEqual(len(ub3), 60)
        self.assertTrue(np.all(ub3 >= lb3))

        m_bs = BSplineBladeModel()
        self.assertEqual(m_bs.dim, 63)
        lb_bs, ub_bs = m_bs.get_bounds()
        self.assertEqual(len(lb_bs), 63)
        self.assertEqual(len(ub_bs), 63)
        self.assertTrue(np.all(ub_bs >= lb_bs))

    def test_baseline_evaluations_feasible(self):
        for model_cls in [Stations3BladeModel, BSplineBladeModel]:
            model = model_cls()
            ev = WindTurbineBladeEvaluator(
                wind_speed=7.0, rpm=72.0, pitch_angle_deg=BASELINE_PITCH_DEG,
                param_model=model,
            )
            det = ev.evaluate_detailed(model.get_x0())
            self.assertTrue(det["feasible"], f"{model.name} baseline should be feasible")
            self.assertGreater(det["power_kW"], 5.0)
            self.assertLess(det["thrust_kN"], 2.0)
            self.assertEqual(len(det["span_chord"]), BEM_NUM_SECTIONS)
            self.assertEqual(len(det["span_twist_deg"]), BEM_NUM_SECTIONS)

    def test_kaya_twist_compatibility(self):
        """验证 Kaya (2019) Phase VI 典型根/尖扭角偏移在 B-Spline 约束空间内保持可行且无负分越界"""
        m_bs = BSplineBladeModel()
        ev = WindTurbineBladeEvaluator(param_model=m_bs)
        x_kaya = m_bs.get_x0()
        # 51 CST + 6 chord + 6 twist -> twist slice is [57:63]
        x_kaya[57:63] = np.array([-4.0, -2.5, -1.0, 0.0, 0.5, 1.0])
        is_fatal, _, _ = m_bs.validate_geometry(x_kaya)
        self.assertFalse(is_fatal)
        self.assertGreater(ev.evaluate(x_kaya), 0.0)

    def test_bspline_smoothness(self):
        m_bs = BSplineBladeModel()
        span = m_bs.to_span_data(m_bs.get_x0())
        r_unif = np.linspace(span["r"][0], span["r"][-1], 27)
        c_unif = PchipInterpolator(span["r"], span["chord"])(r_unif)
        tw_unif = PchipInterpolator(span["r"], span["twist_deg"])(r_unif)
        d2_c = np.diff(c_unif, 2)
        d2_tw = np.diff(tw_unif, 2)
        self.assertLess(float(np.var(d2_c)), 1e-6)
        self.assertLess(float(np.var(d2_tw)), 0.15)


if __name__ == "__main__":
    unittest.main()
