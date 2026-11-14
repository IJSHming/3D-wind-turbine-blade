# NREL Phase VI 3D Wind Turbine Blade Aerodynamic Optimization: Research Report

**Turbine Model**: NREL Phase VI (Sequence H) Upwind Horizontal-Axis Wind Turbine (`NREL/TP-500-29955`, Hand et al., 2001; Kaya, 2019)  
- Rotor Diameter $D = 10.058\,\text{m}$ (Rotor Radius $R = 5.029\,\text{m}$), Number of Blades $B = 2$  
- Hub Flange Radius $r_{\text{flange}} = 0.5083\,\text{m}$, Circular Root Shank $0.5083 \sim 0.8835\,\text{m}$, Aerodynamic S809 Lifting Blade Segment $r \in [1.2575\,\text{m}\,(0.25R), 5.0290\,\text{m}\,(1.00R)]$ ($L = 3.7715\,\text{m}$)  
- Wind Tunnel Hub Center Height $H = 12.192\,\text{m}$ (NASA Ames $80 \times 120\,\text{ft}$ test section); Pitch/Twist Center Axis located at $30\%$ chord ($x/c = 0.30$)  
**Operating Medium**: Standard Dry Air ($\rho = 1.225\,\text{kg/m}^3$, Dynamic Viscosity $\mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$)  
**Baseline Design**: Official NREL S809 airfoil (66-pt Selig coordinates fitted with 8-order Kulfan CST at $t_{\text{TE}} = 0.0$, $C_L\,\text{MAE} = 8.40 \times 10^{-5}$, $C_D\,\text{MAE} = 1.00 \times 10^{-5}$, $t/c = 21.07\%$) lofted with the TP-500-29955 27-station spanwise chord ($0.7370 \to 0.3560\,\text{m}$) and twist ($20.040^\circ \to -1.815^\circ$) schedule (`data/baseline_nrel_phase6/baseline_27_sections.npz`).  
**Baseline Operating Condition (Sequence H @ 7 m/s)**:
- Free-stream Wind Speed: $V_\infty = 7.00\,\text{m/s}$ (uniform inflow, $\alpha_{\text{shear}} = 0.0$, clean smooth surface)
- Rotor Speed: $\Omega = 72.00\,\text{rpm}$ ($\omega = 7.53982\,\text{rad/s}$, Tip-Speed Ratio $\lambda = 5.417$, Tip Speed $37.92\,\text{m/s}$)
- Collective Blade Pitch Angle: $\theta_{p,75\%} = 4.815^\circ$ (referenced to zero-twist $0.75R$ station; equivalent to $3.0^\circ$ tip pitch per Hand et al. 2001 & Kaya 2019), Shaft Tilt $0.0^\circ$, Precone $0.0^\circ$
- Baseline Aerodynamic Performance: Power **$6.0712\,\text{kW}$**, Rotor Torque **$805.22\,\text{N}\cdot\text{m}$**, $C_p = 0.3637$, Axial Thrust **$1.2102\,\text{kN}$** ($1210.2\,\text{N}$), $C_t = 0.5075$ (cf. Kaya 2019 URANS CFD baseline torque $784\,\text{N}\cdot\text{m}$ and thrust $1307\,\text{N}$)
**Aerodynamic Evaluator**: Vectorized BEM + NeuralFoil (`model_size='xxxlarge'`, 61-angle batch across all 27 spanwise sections) with bounded Snel 3D rotational stall delay, dynamic $\pm 20^\circ$ polar-matched Viterna post-stall extrapolation, Shen/QBlade tip/root loss, and strict negative infeasible score gating ($< 0.0$). Single evaluation wall-clock time is $\approx 0.10\,\text{s}$.  
**Design Parameter Space**: $D = 114$ (`BSplineBladeModel`: 6 control stations $\times$ 17 closed-TE Kulfan CST = 102 + 6th-order cubic B-spline $\Delta c_{\text{ctrl}}$ = 6 + 6th-order cubic B-spline $\Delta \beta_{\text{ctrl}}$ = 6).  
**Evaluation Budget**: $\text{NFE} = 300$ per run across 10 independent seeds (`[42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]`), aligned $n_{\text{init}} = 15$, structural thrust cap $T \le 1.15\,T_{\text{baseline}} = 1.3917\,\text{kN}$.

---

## 1. Benchmark Performance Summary (10 Independent Seeds, NFE = 300)

| Optimization Method | Final Power (kW) | Rotor Torque (N·m) | Power Coefficient $C_p$ | Mean Gain (%) | Peak Seed (kW / Gain) | Rotor Thrust (kN) | Feasible | Runtime / Seed (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (NREL Phase VI)** | $6.0712$ | $805.22$ | $0.3637$ | $+0.00\%$ | $6.0712$ ($+0.00\%$) | $1.2102$ | Yes | -- |
| **TuRBO-1** | $6.5449 \pm 0.0638$ | $868.05 \pm 8.46$ | $0.3921 \pm 0.0038$ | $+7.80\%$ | $6.6274$ (`8888`, $+9.16\%$) | $1.3910 \pm 0.0007$ | $10/10$ | $\mathbf{79.44 \pm 20.65}$ |
| **MALIBO** | $6.5766 \pm 0.0240$ | $872.25 \pm 3.18$ | $0.3940 \pm 0.0014$ | $+8.32\%$ | $6.6143$ (`777`, $+8.95\%$) | $1.3908 \pm 0.0009$ | $10/10$ | $110.29 \pm 2.11$ |
| **LLM-Meta-TuRBO** | $\mathbf{6.6130 \pm 0.0343}$ | $\mathbf{877.08 \pm 4.55}$ | $\mathbf{0.3962 \pm 0.0021}$ | $\mathbf{+8.92\%}$ | $\mathbf{6.6660}$ (`256`, $\mathbf{+9.80\%}$) | $1.3911 \pm 0.0009$ | $10/10$ | $158.01 \pm 10.97$ |

*Every single run across all three methods (30/30 seeds total) finished strictly feasible with zero constraint violation (`max_penalty = 0.000000`) and strictly within the thrust cap ($1.3917\,\text{kN}$).*

### Convergence Milestones across Optimization Budget (Mean $\pm$ Std in kW)

| Evaluation Step (NFE) | TuRBO-1 (kW) | LLM-Meta-TuRBO (kW) | MALIBO (kW) | Key Algorithmic Observations |
| :---: | :---: | :---: | :---: | :--- |
| **Step 1 (Baseline)** | $6.0712 \pm 0.0000$ | $6.0712 \pm 0.0000$ | $6.0712 \pm 0.0000$ | Identical NREL Phase VI starting blade across all methods. |
| **Step 15 ($n_{\text{init}}$)** | $6.1620 \pm 0.0221$ | $6.1620 \pm 0.0221$ | $6.1620 \pm 0.0221$ | Bitwise identical unit-space $[0, 1]^D$ symmetric initialization ($\pm 0.04$). |
| **Step 30** | $6.2039 \pm 0.0499$ | $6.3568 \pm 0.1069$ | $\mathbf{6.3752 \pm 0.0815}$ | **MALIBO ($+5.01\%$)** and **LLM-Meta-TuRBO ($+4.70\%$)** surge immediately past TuRBO-1 ($+2.19\%$). |
| **Step 50** | $6.3610 \pm 0.0982$ | $\mathbf{6.5444 \pm 0.0468}$ | $6.4533 \pm 0.0454$ | **LLM-Meta-TuRBO reaches $+7.79\%$ by NFE=50**, already matching TuRBO-1's 300-NFE final score! |
| **Step 100** | $6.5242 \pm 0.0651$ | $\mathbf{6.5963 \pm 0.0454}$ | $6.5150 \pm 0.0316$ | LLM-Meta-TuRBO reaches $+8.65\%$ while maintaining half the variance of early TuRBO. |
| **Step 150** | $6.5335 \pm 0.0642$ | $\mathbf{6.6042 \pm 0.0420}$ | $6.5488 \pm 0.0313$ | MALIBO overtakes TuRBO-1 on mean power and keeps narrowing variance. |
| **Step 200** | $6.5384 \pm 0.0667$ | $\mathbf{6.6097 \pm 0.0358}$ | $6.5608 \pm 0.0250$ | Both meta-methods establish clear, low-variance superiority over standalone TuRBO-1. |
| **Step 300 (Final)** | $6.5449 \pm 0.0638$ | $\mathbf{6.6130 \pm 0.0343}$ | $6.5766 \pm 0.0240$ | **LLM-Meta-TuRBO leads peak power ($+8.92\%$)**; **MALIBO achieves lowest cross-seed std ($0.024\,\text{kW}$, $+8.32\%$)**. |

---

## 2. Physical & Algorithmic Insights on NREL Phase VI

1. **Why LLM-Meta-TuRBO Excelled ($+8.92\%$ Mean, $+9.80\%$ Peak) via Real `Qwen/Qwen3.8-27B-FP8` Reasoning**:
   - After aligned $n_{\text{init}} = 15$, local GP surrogate fitting in 114 dimensions tends to stall early when navigating the active thrust ceiling ($T = 1.3917\,\text{kN}$).
   - Every stagnation event triggered a real HTTP call to the local SGLang `Qwen/Qwen3.8-27B-FP8` endpoint (6–7 interventions per 300-NFE seed, zero heuristic fallbacks, zero oracle leaks).
   - Inspecting `intervention_history` confirms that `Qwen3.8-27B-FP8` repeatedly diagnosed stagnation near the thrust margin and prescribed **positive mid-span camber shifts** (`camber_shift_mid = +0.02`), slight mid/tip thickness thinning (`thickness_shift_mid/tip = -0.01`), and gentle mid-span twist adjustments, injected non-destructively into 30% of the Sobol/random candidate pool without overwriting `base_L`.
   - As seen directly in `figures/blade_station_airfoils.png` and `figures/blade_spanwise_angles.png`, the optimized `LLM-Meta-TuRBO` blade evolved clean positive mid-span camber at Stations 3 & 4 ($r = 2.71\,\text{m}, 3.58\,\text{m}$), shifting the zero-lift angle negatively so the mid-span operates at lower local angles of attack ($\alpha \approx 4.5^\circ \sim 7.0^\circ$ vs $7^\circ \sim 9^\circ$) with higher $C_L/C_D$ and greater torque under the exact same $1.391\,\text{kN}$ thrust envelope.
2. **Why MALIBO Showed Rapid Early Warm-Up & Tightest Consistency ($\sigma = 0.024\,\text{kW}$)**:
   - MALIBO was meta-trained on 100 air-medium tasks ($V \in [4.50, 9.50]\,\text{m/s} \setminus \{7.00\}$, scaled $\Omega \approx (72/7)V \times [0.94..1.06]$) collected independently with TuRBO-1.
   - Fixing the covariance KL prior in `trainer.py` prevented task-embedding collapse ($z_t$ learned healthy variance `std = 0.7406` matching the Laplace $\mathcal{N}(0, I)$ inference prior, best Val Loss `0.3082`), and clamping severe negative penalty outliers preserved clean LFBO utility gradients.
   - Consequently, MALIBO jumped to $6.3752\,\text{kW}$ ($+5.01\%$) by Step 30 — the fastest initial jump of any method — and achieved the tightest cross-seed repeatability across all 10 seeds (`std = 0.0240 kW`, nearly $3\times$ tighter than `TuRBO-1` at `0.0638 kW`).
3. **Physical Alignment with Kaya (2019)**:
   - Under NREL Phase VI Sequence H ($7\,\text{m/s}$, $72\,\text{rpm}$), Kaya (2019) reported $\approx 10\%$ torque gain from SVR-based twist optimization alone (starting from CFD torque $784\,\text{N}\cdot\text{m}$).
   - With an explicit $1.15\times$ structural thrust ceiling ($1.3917\,\text{kN}$), closed trailing edges ($t_{\text{TE}} = 0.0$), and strict spanwise monotonicity/smoothness constraints, `LLM-Meta-TuRBO` reached $+8.92\%$ mean ($+9.80\%$ best seed: $884.10\,\text{N}\cdot\text{m}$ torque), demonstrating clean, realistic, literature-consistent aerodynamic gains ready for high-fidelity CFD verification.

---

## 3. Code Audit Fixes Completed During Migration

During the migration from the previous baseline and review of co-authored modules:
1. **Strict Feasible-Domain Isolation**: Fixed `evaluator.evaluate()` and `BaseBladeOptimizer.evaluate_initial()` so infeasible designs strictly return negative scores ($\le -0.25$) and can never beat feasible candidates; aligned `is_feasible` / `total_penalty` keys across optimizers.
2. **LLM-Meta-TuRBO Clean Control**: Removed `base_L` state corruption, replaced distorted `(1..8)/8` lower CST deformation with smooth symmetric `sin` camber/thickness bells, replaced all-5000 candidate shift overwriting with a 30% directional candidate slice, and fixed `max_t_rel` prompt state lookup.
3. **MALIBO Mathematical Calibration**: Fixed the task-embedding prior collapse in `trainer.py` (diagonal KL + off-diagonal decorrelation), fixed the `mean` vs `sum` likelihood scale mismatch against the Laplace Hessian in `adaptation.py`, propagated seeds and outlier clamping into `residual_booster.py` / `dataset.py` / `optimizer.py`, and aligned $[0, 1]^D$ symmetric initialization across all 3 methods.
4. **NREL Phase VI Alignment**: Removed empty AeroSandbox `s809` fallback, built exact Kulfan S809 baseline + 27-section TP-500-29955 dataset, updated `0.30c` pitch/loft axis across BEM and watertight STL export, and updated test assertions accordingly.

---

## 4. Verification & Reproducibility Summary

- **Unit Tests**: `python -m unittest discover -s tests -p "test_*.py" -v` (14 tests covering NREL Phase VI geometry, parameters, S809 bounds, Kaya twist compatibility, `evaluator` physics/speed `< 0.10 s`, MALIBO forward pass, and exported figure assets).
- **Publication Figures & Watertight STLs**: All 7 static figures (PNG + PDF), 4 interactive 3D HTML renderings, and 4 watertight closed 2-manifold STL files generated via `plot_results.py` (`cfd_models/README.md`).
