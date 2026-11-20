"""自动化单元与集成测试套件 (tests/test_benchmark_pipeline.py):
检验 3 站位 (D=63) 叶片几何参数化、31 全余弦截面评估器、严格原论文对齐 MALIBO 架构与图表文件的完整性。
"""

from pathlib import Path
import unittest
import numpy as np
import torch

from src.blade_geometry import (
    BASELINE_PITCH_DEG,
    BEM_NUM_SECTIONS,
    BLADE_ROOT_RADIUS,
    DIM,
    NUM_BLADES,
    NUM_STATIONS,
    PITCH_AXIS_FRAC,
    ROTOR_RADIUS,
    STATION_INDICES,
    get_baseline_blade_params,
    get_blade_bounds,
    unpack_blade_params,
    interpolate_blade_span,
    loft_blade_3d,
)
from src.evaluator import WindTurbineBladeEvaluator
from src.malibo.config import ResFFNConfig
from src.malibo.losses import LFBOLoss, LatentSpaceRegularizer
from src.malibo.models import MALIBOMetaModel
from src.malibo.optimizer import MALIBOBladeOptimizer


class TestBladeGeometry(unittest.TestCase):
    def test_dimensions_and_baseline(self):
        self.assertEqual(NUM_BLADES, 2)
        self.assertEqual(BEM_NUM_SECTIONS, 31)
        self.assertEqual(list(STATION_INDICES), [0, 15, 30])
        self.assertAlmostEqual(ROTOR_RADIUS, 5.029, places=4)
        self.assertAlmostEqual(BLADE_ROOT_RADIUS, 1.2575, places=4)
        self.assertAlmostEqual(PITCH_AXIS_FRAC, 0.30, places=3)
        self.assertAlmostEqual(BASELINE_PITCH_DEG, 4.815, places=3)

        x0 = get_baseline_blade_params()
        self.assertEqual(len(x0), 60)
        self.assertEqual(DIM, 60)
        self.assertEqual(NUM_STATIONS, 3)

        stations = unpack_blade_params(x0)
        self.assertEqual(len(stations), 3)
        self.assertEqual(len(stations[0]["theta"]), 18)

        lb, ub = get_blade_bounds(x0)
        self.assertEqual(len(lb), 60)
        self.assertEqual(len(ub), 60)
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
        self.evaluator = WindTurbineBladeEvaluator(
            wind_speed=7.0,
            rpm=72.0,
            pitch_angle_deg=BASELINE_PITCH_DEG,
        )
        self.x0 = self.evaluator.baseline_x0

    def test_baseline_evaluation(self):
        det = self.evaluator.evaluate_detailed(self.x0)
        self.assertTrue(det["feasible"])
        self.assertTrue(det["is_feasible"])
        self.assertFalse(det["is_fatal"])
        # NREL Phase VI Sequence H at 7.0 m/s, 72 rpm: ~6.088 kW (~807 N*m torque)
        self.assertGreater(det["power_kW"], 5.0)
        self.assertLess(det["power_kW"], 7.5)
        self.assertGreater(det["torque_Nm"], 650.0)
        self.assertLess(det["torque_Nm"], 1000.0)
        self.assertLess(det["thrust_kN"], self.evaluator.max_allowable_thrust_kN)
        self.assertGreater(self.evaluator.evaluate(self.x0), 0.0)

    def test_fatal_and_infeasible_gating(self):
        x_bad = self.x0.copy()
        x_bad[0:8] = -1.5
        x_bad[8:16] = 1.5
        is_fatal, fatal_pen, _ = self.evaluator.validate_geometry(x_bad)
        self.assertTrue(is_fatal)
        self.assertGreater(fatal_pen, 1.0)
        self.assertLess(self.evaluator.evaluate(x_bad), 0.0)


class TestMALIBOPaperAligned(unittest.TestCase):
    def test_architecture_losses_and_no_trust_region(self):
        # Strictly verify Pan et al. (ICML 2024, Appendix G): 4x64 ResFFN, d=50, no LayerNorm, no TR
        cfg = ResFFNConfig(in_dim=63, hidden_dim=64, num_hidden_layers=4, feature_dim=50)
        model = MALIBOMetaModel(num_tasks=100, config=cfg)
        self.assertFalse(hasattr(model.phi_net, "feat_norm"))

        x_dummy = torch.rand(4, 63)
        z_dummy = torch.randn(50)
        out = model(x_dummy, z=z_dummy)
        self.assertEqual(out.shape, (4,))

        lfbo = LFBOLoss()
        loss_val = lfbo(out, torch.tensor([0.5, 0.2, 0.0, 0.0]))
        self.assertTrue(torch.isfinite(loss_val))

        reg = LatentSpaceRegularizer(feature_dim=50)
        r_tot, ks_l, cov_l = reg(model.task_embeddings)
        self.assertTrue(torch.isfinite(r_tot) and r_tot.item() >= 0.0)

        ev = WindTurbineBladeEvaluator(wind_speed=7.0, rpm=72.0, pitch_angle_deg=BASELINE_PITCH_DEG)
        opt = MALIBOBladeOptimizer(meta_model=model, evaluator=ev, max_evals=20, seed=42)
        self.assertFalse(hasattr(opt, "tr_len"))
        self.assertFalse(hasattr(opt, "prob_perturb"))
        self.assertEqual(opt.n_candidates, 5120)
        self.assertAlmostEqual(opt.gamma, 1.0 / 3.0, places=6)


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
            "blade_3d_rendering.html",
        ]
        for name in fig_names:
            path = Path("figures") / name
            self.assertTrue(path.exists(), f"Figure missing: {path}")
            self.assertGreater(path.stat().st_size, 1000, f"Figure empty: {path}")


if __name__ == "__main__":
    unittest.main()
