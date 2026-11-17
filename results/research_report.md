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
**Design Parameter Space**: $D = 114$ (`BSplineBladeModel`: 6 independent spanwise control stations $\times$ 17 closed-TE Kulfan CST = 102 + 6th-order cubic B-spline $\Delta c_{\text{ctrl}}$ = 6 + 6th-order cubic B-spline $\Delta \beta_{\text{ctrl}}$ = 6). Retaining 6 independent stations allows root structural thickness ($21\%$) and outboard/mid-span camber & thinness to co-evolve cleanly rather than forcing a conflicting single 2D section across the entire span.  
**Evaluation Budget**: $\text{NFE} = 300$ per run across 10 independent seeds (`[42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]`), aligned $n_{\text{init}} = 15$, structural thrust cap $T \le 1.15\,T_{\text{baseline}} = 1.3917\,\text{kN}$.

---

## 1. Benchmark Performance Summary (10 Independent Seeds, NFE = 300)

| Optimization Method | Final Power (kW) | Rotor Torque (N·m) | Power Coefficient $C_p$ | Mean Gain (%) | Peak Seed (kW / Gain) | Rotor Thrust (kN) | Feasible | Runtime / Seed (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (NREL Phase VI)** | $6.0712$ | $805.22$ | $0.3637$ | $+0.00\%$ | $6.0712$ ($+0.00\%$) | $1.2102$ | Yes | -- |
| **TuRBO-1** | $6.5449 \pm 0.0638$ | $868.05 \pm 8.46$ | $0.3921 \pm 0.0038$ | $+7.80\%$ | $6.6274$ (`8888`, $+9.16\%$) | $1.3910 \pm 0.0007$ | $10/10$ | $\mathbf{79.40 \pm 20.94}$ |
| **LLM-Meta-TuRBO** | $6.6031 \pm 0.0358$ | $875.77 \pm 4.75$ | $0.3956 \pm 0.0021$ | $+8.76\%$ | $6.6552$ (`9999`, $+9.62\%$) | $1.3908 \pm 0.0013$ | $10/10$ | $141.01 \pm 18.44$ |
| **MALIBO** | $\mathbf{6.6255 \pm 0.0216}$ | $\mathbf{878.73 \pm 2.87}$ | $\mathbf{0.3969 \pm 0.0013}$ | $\mathbf{+9.13\%}$ | $\mathbf{6.6628}$ (`777`, $\mathbf{+9.74\%}$) | $1.3914 \pm 0.0003$ | $10/10$ | $148.11 \pm 2.20$ |

*Every single run across all three methods (30/30 seeds total) finished strictly feasible with zero constraint violation (`max_penalty = 0.000000`) and strictly within the thrust cap ($1.3917\,\text{kN}$).*

### Convergence Milestones across Optimization Budget (Mean $\pm$ Std in kW)

| Evaluation Step (NFE) | TuRBO-1 (kW) | LLM-Meta-TuRBO (kW) | MALIBO (kW) | Key Algorithmic Observations |
| :---: | :---: | :---: | :---: | :--- |
| **Step 1 (Baseline)** | $6.0712 \pm 0.0000$ | $6.0712 \pm 0.0000$ | $6.0712 \pm 0.0000$ | Identical NREL Phase VI starting blade across all methods. |
| **Step 15 ($n_{\text{init}}$)** | $6.1620 \pm 0.0221$ | $6.1620 \pm 0.0221$ | $6.1620 \pm 0.0221$ | Bitwise identical unit-space $[0, 1]^D$ symmetric initialization ($\pm 0.04$). |
| **Step 30** | $6.2039 \pm 0.0499$ | $6.3227 \pm 0.1197$ | $\mathbf{6.3703 \pm 0.1382}$ | **MALIBO ($+4.93\%$)** and **LLM-Meta-TuRBO ($+4.14\%$)** jump well ahead of TuRBO-1 ($+2.19\%$). |
| **Step 50** | $6.3610 \pm 0.0982$ | $6.4304 \pm 0.1152$ | $\mathbf{6.4570 \pm 0.0769}$ | Both meta-methods maintain strong early leads with lower variance than TuRBO-1. |
| **Step 75** | $6.4732 \pm 0.0728$ | $\mathbf{6.5371 \pm 0.0591}$ | $6.5060 \pm 0.0656$ | Smooth stagnation-adaptive LLM interventions pull ahead mid-phase; MALIBO keeps clear margin over TuRBO-1. |
| **Step 100** | $6.5242 \pm 0.0651$ | $\mathbf{6.5552 \pm 0.0519}$ | $6.5370 \pm 0.0635$ | **Zero crossover dip**: Enhanced MALIBO (`6.5370 kW`) remains strictly above TuRBO-1 (`6.5242 kW`) at NFE=100! |
| **Step 150** | $6.5335 \pm 0.0642$ | $6.5758 \pm 0.0412$ | $\mathbf{6.5877 \pm 0.0295}$ | MALIBO overtakes LLM-Meta-TuRBO with narrow variance (`0.0295 kW`) as $\gamma_t$ anneals onto the $+9\%$ frontier. |
| **Step 200** | $6.5384 \pm 0.0667$ | $6.5887 \pm 0.0443$ | $\mathbf{6.6085 \pm 0.0255}$ | MALIBO breaks $6.60\,\text{kW}$ mean while TuRBO-1 plateaus around $6.54\,\text{kW}$. |
| **Step 300 (Final)** | $6.5449 \pm 0.0638$ | $6.6031 \pm 0.0358$ | $\mathbf{6.6255 \pm 0.0216}$ | **MALIBO ranks #1 in both mean ($+9.13\%$) and consistency ($\sigma = 0.0216\,\text{kW}$, $3\times$ tighter than TuRBO-1)**; LLM-Meta-TuRBO cleanly places #2 ($+8.76\%$). |

---

## 2. Physical & Algorithmic Insights on NREL Phase VI

1. **MALIBO Enhancements strictly within Pan et al. (2024) eliminated the 100-NFE Crossover and Boosted Gain to $+9.13\%$**:
   - **Multi-Pitch & Multi-Start Deep Trajectory Meta-Dataset (20,000 samples)**: Expanded offline generation to 200 NFE per task across 20 wind speeds ($\setminus \{7.0\,\text{m/s}\}$), 5 TSR levels, and 6 blade pitch angles ($\setminus \{4.815^\circ\}$), with 60% diverse feasible starting geometries and pre-BEM fatal geometry screening (`~75%-91%` feasibility).
   - **Feature Stability & Exact Eq. (10) Classifier Residual Boosting**: Added `LayerNorm` to `ResFFN` output (`||phi|| ~ 6.12`, bounding out-of-distribution `MeanHead` outputs) + backbone `weight_decay = 1e-4`, and restored the Pan et al. (2024, Eq. 10) `GradientBoostingClassifier(init=MALIBOInitialEstimator)` with stratified validation early stopping on continuous `decision_function` margins.
   - **114D Hybrid Trust-Region Alignment**: Adjusted `tr_len_init = 0.80, tr_len_restart = 0.60, failure_tolerance = 6, prob_perturb = 0.18` (from legacy 18D airfoil defaults `0.40/5`) and replaced uniform sampling with scrambled `SobolEngine` + Step-16 multi-scale warm-start pool.
   - Note for manuscript clarity: Original Pan et al. (2024) maximizes acquisition via global 5,120 random sampling on low/moderate-dimensional tasks; adding the adaptive trust region + coordinate perturbation mask is our `Meta-CFD` / 114D continuous-geometry extension.
   - **Result**: Eliminated premature TR collapse at NFE 50–100, raised mean power from `6.5766` to `6.6255 kW` (`+9.13%`), reached peak `6.6628 kW` (`883.68 Nm`), and reduced cross-seed standard deviation to just `0.0216 kW`.
2. **Stagnation-Adaptive LLM-Meta-TuRBO ($+8.76\%$, 3–8 interventions/seed)**:
   - Removed fixed mid/late milestones `(35, 65, 105, 140)` in favor of pure `stagnation_counter >= 10` triggers (plus an explicit `post_init` review at Step 16), eliminating artificial synchronous staircases at Step 35 while cutting standard deviation nearly in half vs TuRBO-1 (`0.0358` vs `0.0638 kW`, peak Seed 9999 `6.6552 kW` with just 3 LLM calls `[16, 99, 174]`).
3. **Physical Alignment with Kaya (2019)**:
   - Under NREL Phase VI Sequence H ($7\,\text{m/s}$, $72\,\text{rpm}$), Kaya (2019) reported $\approx 10\%$ torque gain from SVR-based twist optimization alone (starting from CFD torque $784\,\text{N}\cdot\text{m}$).
   - Respecting an explicit $1.15\times$ structural thrust cap ($1.3917\,\text{kN}$), closed trailing edges ($t_{\text{TE}} = 0.0$), and strict spanwise monotonicity/smoothness constraints, MALIBO (`+9.13%`, up to `883.68 Nm`) and LLM-Meta-TuRBO (`+8.76%`, up to `882.67 Nm`) demonstrate realistic, literature-consistent aerodynamic torque and power improvements ready for CFD validation.

---

## 3. Verification & Reproducibility Summary

- **Unit Tests**: `python -m unittest discover -s tests -p "test_*.py" -v` (14 tests covering NREL Phase VI geometry, S809 bounds, Kaya twist compatibility, `evaluator` physics/speed `< 0.14 s`, MALIBO forward pass, and exported figure assets).
- **Publication Figures & Watertight STLs**: All 7 static figures (PNG + PDF), 4 interactive 3D HTML renderings, and 4 watertight closed 2-manifold STL files generated via `plot_results.py` (`cfd_models/README.md`).
