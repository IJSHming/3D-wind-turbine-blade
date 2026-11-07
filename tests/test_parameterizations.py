"""多参数化模型库单元测试 (tests/test_parameterizations.py):
验证 BSplineBladeModel (D=132), Stations29BladeModel (D=580), Stations6BladeModel (D=120) 的几何合法性与评估器兼容性。
"""

import unittest
import numpy as np

from src.evaluator import WindTurbineBladeEvaluator
from src.parameterizations import (
    BSplineBladeModel,
    Stations6BladeModel,
    Stations29BladeModel,
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
        self.assertEqual(m_bs.dim, 132)
        lb_bs, ub_bs = m_bs.get_bounds()
        self.assertEqual(len(lb_bs), 132)
        self.assertEqual(len(ub_bs), 132)
        self.assertTrue(np.all(ub_bs >= lb_bs))

        m29 = Stations29BladeModel()
        self.assertEqual(m29.dim, 580)
        lb29, ub29 = m29.get_bounds()
        self.assertEqual(len(lb29), 580)
        self.assertEqual(len(ub29), 580)
        self.assertTrue(np.all(ub29 >= lb29))

    def test_baseline_evaluations_feasible(self):
        for model_cls in [Stations6BladeModel, BSplineBladeModel, Stations29BladeModel]:
            model = model_cls()
            ev = WindTurbineBladeEvaluator(
                wind_speed=11.0, rpm=6.77094, pitch_angle_deg=2.42859,
                param_model=model,
            )
            det = ev.evaluate_detailed(model.get_x0())
            self.assertTrue(det["feasible"], f"{model.name} baseline should be feasible")
            self.assertGreater(det["power_kW"], 21000.0)
            self.assertLess(det["thrust_kN"], 3600.0)
            self.assertEqual(len(det["span_chord"]), 58)
            self.assertEqual(len(det["span_twist_deg"]), 58)

    def test_bspline_smoothness(self):
        m_bs = BSplineBladeModel()
        span = m_bs.to_span_data(m_bs.get_x0())
        d2_c = np.diff(span["chord"], 2)
        d2_tw = np.diff(span["twist_deg"], 2)
        self.assertLess(float(np.var(d2_c)), 0.01)
        self.assertLess(float(np.var(d2_tw)), 0.05)


if __name__ == "__main__":
    unittest.main()
