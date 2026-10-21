"""自动化单元与集成测试套件 (tests/test_benchmark_pipeline.py):
检验叶片几何参数化、综合评估器、BEM + NeuralFoil 求解、MALIBO 架构与图表文件的完整性。
"""

import os
from pathlib import Path
import unittest
import numpy as np
import torch

from src.blade_geometry import (
    DIM,
    NUM_STATIONS,
    ROTOR_RADIUS,
    HUB_RADIUS,
    get_baseline_blade_params,
    get_blade_bounds,
    unpack_blade_params,
    interpolate_blade_span,
    loft_blade_3d,
)
from src.evaluator import WindTurbineBladeEvaluator
from src.malibo.config import ResFFNConfig
from src.malibo.models import MALIBOMetaModel


class TestBladeGeometry(unittest.TestCase):
    def test_dimensions_and_baseline(self):
        x0 = get_baseline_blade_params()
        self.assertEqual(len(x0), 120)
        self.assertEqual(DIM, 120)
        self.assertEqual(NUM_STATIONS, 6)

        stations = unpack_blade_params(x0)
        self.assertEqual(len(stations), 6)

        lb, ub = get_blade_bounds(x0)
        self.assertEqual(len(lb), 120)
        self.assertEqual(len(ub), 120)
        self.assertTrue(np.all(lb <= x0))
        self.assertTrue(np.all(x0 <= ub))

    def test_span_interpolation_and_lofting(self):
        x0 = get_baseline_blade_params()
        span_d = interpolate_blade_span(x0, n_stations=18)
        self.assertEqual(len(span_d["r"]), 18)
        self.assertEqual(span_d["kulfan_thetas"].shape, (18, 18))

        X, Y, Z = loft_blade_3d(x0, n_span=10, n_chord=20)
        self.assertEqual(X.shape, (10, 40))
        self.assertEqual(Y.shape, (10, 40))
        self.assertEqual(Z.shape, (10, 40))


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = WindTurbineBladeEvaluator(wind_speed=11.0, rpm=7.061)
        self.x0 = get_baseline_blade_params()

    def test_baseline_evaluation(self):
        det = self.evaluator.evaluate_detailed(self.x0)
        self.assertTrue(det["feasible"])
        self.assertFalse(det["is_fatal"])
        # Power should be around 25.59 MW
        self.assertGreater(det["power_kW"], 25000.0)
        self.assertLess(det["power_kW"], 26000.0)
        # Thrust should be safely below allowable limit
        self.assertLess(det["thrust_kN"], self.evaluator.max_allowable_thrust_kN)

    def test_fatal_geometry_penalty(self):
        # Create an inverted/self-crossing airfoil vector
        x_bad = self.x0.copy()
        # Flip station 1 upper and lower weights drastically
        x_bad[0:8] = -1.5
        x_bad[8:16] = 1.5
        is_fatal, fatal_pen, _ = self.evaluator.validate_geometry(x_bad)
        self.assertTrue(is_fatal)
        self.assertGreater(fatal_pen, 10000.0)


class TestMALIBOModel(unittest.TestCase):
    def test_forward_pass(self):
        cfg = ResFFNConfig(in_dim=120, feature_dim=64)
        model = MALIBOMetaModel(num_tasks=21, config=cfg)
        x_dummy = torch.rand(4, 120)
        z_dummy = torch.randn(64)
        out = model(x_dummy, z=z_dummy)
        self.assertEqual(out.shape, (4,))


class TestGeneratedFigures(unittest.TestCase):
    def test_all_figures_exist_and_non_empty(self):
        fig_names = [
            "blade_convergence_curves.png",
            "blade_convergence_curves.pdf",
            "blade_convergence_curves_100nfe.png",
            "blade_convergence_curves_100nfe.pdf",
            "blade_bar_comparison.png",
            "blade_bar_comparison.pdf",
            "blade_boxplot_comparison.png",
            "blade_boxplot_comparison.pdf",
            "blade_chord_twist_distributions.png",
            "blade_chord_twist_distributions.pdf",
            "blade_station_airfoils.png",
            "blade_station_airfoils.pdf",
            "blade_3d_rendering.png",
            "blade_3d_rendering.pdf",
        ]
        for name in fig_names:
            path = Path("figures") / name
            self.assertTrue(path.exists(), f"Figure missing: {path}")
            self.assertGreater(path.stat().st_size, 1000, f"Figure empty: {path}")


if __name__ == "__main__":
    unittest.main()
