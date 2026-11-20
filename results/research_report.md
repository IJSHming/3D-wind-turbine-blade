# NREL Phase VI 3D Wind Turbine Blade Aerodynamic Optimization: Research Report

**Turbine Model**: NREL Phase VI (Sequence H) Upwind Horizontal-Axis Wind Turbine (`NREL/TP-500-29955`, Hand et al., 2001; Kaya, 2019)  
- Rotor Diameter $D = 10.058\,\text{m}$ (Rotor Radius $R = 5.029\,\text{m}$), Number of Blades $B = 2$  
- Hub Flange Radius $r_{\text{flange}} = 0.5083\,\text{m}$, Circular Root Shank $0.5083 \sim 0.8835\,\text{m}$, Aerodynamic S809 Lifting Blade Segment $r \in [1.2575\,\text{m}\,(0.25R), 5.0290\,\text{m}\,(1.00R)]$ ($L = 3.7715\,\text{m}$)  
- Wind Tunnel Hub Center Height $H = 12.192\,\text{m}$ (NASA Ames $80 \times 120\,\text{ft}$ test section); Pitch/Twist Center Axis located at $30\%$ chord ($x/c = 0.30$)  
**Operating Medium**: Standard Dry Air ($\rho = 1.225\,\text{kg/m}^3$, Dynamic Viscosity $\mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$)  
**Baseline Design**: Official NREL S809 airfoil (66-pt Selig coordinates fitted with 8-order Kulfan CST at $t_{\text{TE}} = 0.0$, $C_L\,\text{MAE} = 8.40 \times 10^{-5}$, $C_D\,\text{MAE} = 1.00 \times 10^{-5}$, $t/c = 21.07\%$) lofted onto a **31-section Full-Cosine clustered spanwise grid** (`data/baseline_nrel_phase6/baseline_31_sections.npz`) with official TP-500-29955 chord ($0.7370 \to 0.3560\,\text{m}$) and twist ($20.040^\circ \to -1.815^\circ$) distributions.  
**Baseline Operating Condition (Sequence H @ 7 m/s)**:
- Free-stream Wind Speed: $V_\infty = 7.00\,\text{m/s}$ (uniform inflow, $\alpha_{\text{shear}} = 0.0$, clean smooth surface)
- Rotor Speed: $\Omega = 72.00\,\text{rpm}$ ($\omega = 7.53982\,\text{rad/s}$, Tip-Speed Ratio $\lambda = 5.417$, Tip Speed $37.92\,\text{m/s}$)
- Collective Blade Pitch Angle: $\theta_{p,75\%} = 4.815^\circ$ (referenced to zero-twist $0.75R$ station; equivalent to $3.0^\circ$ tip pitch per Hand et al. 2001 & Kaya 2019), Shaft Tilt $0.0^\circ$, Precone $0.0^\circ$
- Baseline Aerodynamic Performance (31 Full-Cosine BEM): Power **$6.0884\,\text{kW}$**, Rotor Torque **$807.50\,\text{N}\cdot\text{m}$**, $C_p = 0.3647$, Axial Thrust **$1.2134\,\text{kN}$** ($1213.4\,\text{N}$), $C_t = 0.5088$ (compared to 201-node reference solution `6.0963 kW`, quadrature error drops $3\times$ from `0.411%` on legacy 27 uniform nodes to **`0.139%`** on 31 full-cosine nodes)
**Aerodynamic Evaluator**: Vectorized BEM + NeuralFoil (`model_size='xxxlarge'`, 61-angle batch across all 31 cosine sections) with bounded Snel 3D rotational stall delay, dynamic $\pm 20^\circ$ polar-matched Viterna post-stall extrapolation, Shen/QBlade tip/root loss, and strict negative infeasible score gating ($< 0.0$). Single evaluation wall-clock time is $\approx 0.11\,\text{s}$.  
**Design Parameter Space ($D = 63$)**: `BSplineBladeModel` with **3 airfoil control stations** placed symmetrically at the head/mid/tail of the pure S809 lifting span (`Root r=1.2575 m`, `Mid r=3.14325 m`, `Tip r=5.0290 m`; indices `[0, 15, 30]`) $\times$ 17 closed-TE Kulfan CST = 51 variables interpolated spanwise via **monotonicity-preserving $C^1$ cubic PCHIP** (eliminating 3D side-view surface undulations) + 6th-order cubic B-spline $\Delta c_{\text{ctrl}}$ (6) + 6th-order cubic B-spline $\Delta \beta_{\text{ctrl}}$ (6).  
**Evaluation Budget**: $\text{NFE} = 300$ per run across 10 independent seeds (`[42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]`), aligned $n_{\text{init}} = 15$, structural thrust cap $T \le 1.15\,T_{\text{baseline}} = 1.3954\,\text{kN}$.

---

## 1. Benchmark Performance Summary (10 Independent Seeds, NFE = 300)

| Optimization Method | Final Power (kW) | Rotor Torque (N·m) | Power Coefficient $C_p$ | Mean Gain (%) | Peak Seed (kW / Gain) | Rotor Thrust (kN) | Feasible | Runtime / Seed (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (NREL Phase VI)** | $6.0884$ | $807.50$ | $0.3647$ | $+0.00\%$ | $6.0884$ ($+0.00\%$) | $1.2134$ | Yes | -- |
| **TuRBO-1** | $6.5920 \pm 0.0463$ | $874.29 \pm 6.14$ | $0.3949 \pm 0.0028$ | $+8.27\%$ | $6.6526$ (`42`, $+9.27\%$) | $1.3938 \pm 0.0018$ | $10/10$ | $\mathbf{70.87 \pm 1.18}$ |
| **LLM-Meta-TuRBO** | $\mathbf{6.6664 \pm 0.0196}$ | $\mathbf{884.16 \pm 2.60}$ | $\mathbf{0.3994 \pm 0.0012}$ | $\mathbf{+9.49\%}$ | $\mathbf{6.6955}$ (`1024`, $\mathbf{+9.97\%}$) | $1.3948 \pm 0.0003$ | $10/10$ | $157.86 \pm 13.19$ |
| **MALIBO (Paper-Exact, Global)** | $6.1773 \pm 0.0280$ | $819.29 \pm 3.72$ | $0.3701 \pm 0.0017$ | $+1.46\%$ | $6.2276$ (`2024`, $+2.29\%$) | $1.2417 \pm 0.0116$ | $10/10$ | $111.00 \pm 3.99$ |

*Every single run across all three methods (30/30 seeds total) finished strictly feasible with zero constraint violation (`max_penalty = 0.000000`) and strictly within the thrust cap ($1.3954\,\text{kN}$).*

### Paired Statistical Significance Tests (10 Seeds)

| Comparison | Mean Difference (kW) | Win Rate | Paired $t$-test ($p$) | Wilcoxon Signed-Rank ($p$) | Cohen's $d$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LLM-Meta-TuRBO vs. TuRBO-1** | $+0.0744$ | $8/10$ | $t = 4.556$ ($p = 0.00137$) | $p = 0.00977$ | $1.44$ |
| **LLM-Meta-TuRBO vs. MALIBO** | $+0.4891$ | $10/10$ | $t = 43.491$ ($p < 10^{-5}$) | $p = 0.00195$ | $13.75$ |
| **TuRBO-1 vs. MALIBO** | $+0.4146$ | $10/10$ | $t = 31.640$ ($p < 10^{-5}$) | $p = 0.00195$ | $10.01$ |

### Convergence Milestones across Optimization Budget (Mean $\pm$ Std in kW)

| Evaluation Step (NFE) | TuRBO-1 (kW) | LLM-Meta-TuRBO (kW) | MALIBO (kW) | Key Algorithmic Observations |
| :---: | :---: | :---: | :---: | :--- |
| **Step 1 (Baseline)** | $6.0884 \pm 0.0000$ | $6.0884 \pm 0.0000$ | $6.0884 \pm 0.0000$ | Identical NREL Phase VI starting blade across all methods. |
| **Step 15 ($n_{\text{init}}$)** | $6.1773 \pm 0.0280$ | $6.1773 \pm 0.0280$ | $6.1773 \pm 0.0280$ | Bitwise identical unit-space $[0, 1]^{63}$ symmetric initialization ($\pm 0.04$). |
| **Step 30** | $6.2741 \pm 0.0896$ | $\mathbf{6.4311 \pm 0.1550}$ | $6.1773 \pm 0.0280$ | Post-init LLM aerodynamic meta-guidance accelerates early ascent ($+5.63\%$ vs $+3.05\%$). |
| **Step 50** | $6.4400 \pm 0.0742$ | $\mathbf{6.5760 \pm 0.0771}$ | $6.1773 \pm 0.0280$ | `LLM-Meta-TuRBO` already approaches `TuRBO-1`'s final 300-NFE mean. |
| **Step 75** | $6.5228 \pm 0.0710$ | $\mathbf{6.6242 \pm 0.0334}$ | $6.1773 \pm 0.0280$ | `LLM-Meta-TuRBO` crosses `6.62 kW` with half the cross-seed variance of `TuRBO-1`. |
| **Step 100** | $6.5376 \pm 0.0694$ | $\mathbf{6.6434 \pm 0.0263}$ | $6.1773 \pm 0.0280$ | At 100 NFE, `LLM-Meta-TuRBO` (`6.6434 kW`) already exceeds `TuRBO-1`'s 300-NFE final mean (`6.5920 kW`) — $>3\times$ sample efficiency. |
| **Step 150** | $6.5722 \pm 0.0487$ | $\mathbf{6.6554 \pm 0.0237}$ | $6.1773 \pm 0.0280$ | Steady thrust-bounded fine-tuning; `LLM-Meta-TuRBO` tightens std to `0.0237 kW`. |
| **Step 200** | $6.5887 \pm 0.0464$ | $\mathbf{6.6633 \pm 0.0176}$ | $6.1773 \pm 0.0280$ | Near-complete convergence for both trust-region methods. |
| **Step 300 (Final)** | $6.5920 \pm 0.0463$ | $\mathbf{6.6664 \pm 0.0196}$ | $6.1773 \pm 0.0280$ | **`LLM-Meta-TuRBO` ranks #1 in both mean ($+9.49\%$) and consistency ($\sigma = 0.0196\,\text{kW}$)**; peak Seed 1024 reaches `6.6955 kW` (`+9.97%`, `Cp=0.4011`). |

---

## 2. Physical, Geometric & Algorithmic Insights

1. **Impact of 3-Station Parameterization ($D=114 \to 63$) + $C^1$ PCHIP Lofting + 31 Full-Cosine BEM**:
   - **Zero 3D Side-View Surface Undulations**: Reducing independent airfoil control stations from 6 to 3 (`Root / Mid / Tip`) and upgrading spanwise CST interpolation from piecewise-linear (`np.interp`) to monotonicity-preserving cubic `PchipInterpolator` eliminates spanwise ridges and surface ripples completely (verified visually in `figures/blade_3d_sideview.png` and watertight STLs).
   - **Higher Optimization Gain from Lower Dimensionality**: Cutting $D$ nearly in half (`114 -> 63`) raised `LLM-Meta-TuRBO` mean gain from `+8.76%` (`6.6031 kW`) to **`+9.49%` (`6.6664 kW`, peak `+9.97%` / `888.02 Nm`)** and `TuRBO-1` from `+7.80%` (`6.5449 kW`) to **`+8.27%` (`6.5920 kW`, peak `+9.27%`)**, while cutting `LLM-Meta-TuRBO` cross-seed std from `0.0358` to `0.0196 kW`.
   - **3x Higher BEM Quadrature Accuracy**: Full-cosine 31-section clustering matches the 201-node reference power within `0.139%` (vs `0.411%` for legacy 27 uniform sections) at virtually identical runtime (`~107 ms/eval`).
2. **Peak Blade Aerodynamic & Structural Quality (`LLM-Meta-TuRBO`, Seed 1024, `6.6955 kW`, `+9.97%`)**:
   - **Root ($r=1.2575\,\text{m}$)**: $c = 0.8025\,\text{m}$, $\beta = 17.223^\circ$ ($-2.82^\circ$ root detwist consistent with Kaya 2019), $t/c = 20.23\%$ (preserving root structural stiffness).
   - **Mid-Span ($r=3.1433\,\text{m}$)**: $c = 0.5710\,\text{m}$, $\beta = 1.208^\circ$, $t/c = 19.61\%$ with smoothly elevated camber optimizing sectional $C_L/C_D$ at $\alpha \approx 6.0^\circ\text{--}7.5^\circ$.
   - **Tip ($r=5.0290\,\text{m}$)**: $c = 0.3546\,\text{m}$, $\beta = -1.628^\circ$, $t/c = 21.30\%$.
   - **Compact Structural Volume**: Watertight STL volume is $0.146985\,\text{m}^3$ (only $+1.6\%$ over baseline $0.144600\,\text{m}^3$), whereas `TuRBO-1` (Seed 42, $6.6526\,\text{kW}$) inflated volume by $+21.2\%$ ($0.175260\,\text{m}^3$) via thicker mid/tip sections ($23.36\% / 22.24\%$).
3. **Strict Paper-Exact `MALIBO` (Pan et al., ICML 2024) — Diagnosing the High-D Constrained Manifold Bottleneck**:
   - Meta-training faithfully followed Appendix G (`4x64` ResFFN, `ELU`, no LayerNorm, `d=50`, `MeanHead=50`, exact Eq. 2 `LFBOLoss`, exact Eq. 4 / Appendix C rank-based KS + Covariance prior regularizer, Adam `1e-3`, `ExponentialLR(0.999)`), converging cleanly to `Best Val Loss = 0.0272` with task embeddings strictly distributed as $\mathcal{N}(0, I_{50})$ (`z_t std = 1.0057`, `|mean| = 0.0122`).
   - Online adaptation strictly implemented Eq. (6)-(10) (L-BFGS MAP, exact Laplace precision matrix, pure Thompson sampling at fixed $\gamma=1/3$, and 70/30 early-stopped LFBO dual-label residual GBDT) with **Appendix G global 5,120 uniform random search over $[0, 1]^{63}$ (NO trust region, NO sparse coordinate perturbation mask)**.
   - **Key Scientific Finding for Future Journal Work**: Because simultaneous geometric constraints (airfoil smoothness + monotonic chord/twist + curvature bounds) restrict valid 3D blades to a thin feasible manifold of relative volume $< 10^{-4}$ inside $[0, 1]^{63}$, **0 out of 5,120** uniform global random samples are geometrically feasible (MALIBO's meta-classifier accurately scores all top candidates negatively at `logit ~ -3.4`, yet has zero feasible global candidates to select). Consequently, vanilla MALIBO stalls at the shared `n_init = 15` baseline-neighborhood gain (`6.1773 kW`, `+1.46%`). This provides a rigorous, paper-exact conference baseline and directly motivates future journal extensions of MALIBO incorporating trust-region / constrained-manifold acquisition maximization for high-dimensional aerodynamic shape optimization.

---

## 3. Verification & Reproducibility Summary

- **Unit Tests**: `python -m unittest discover -s tests -p "test_*.py" -v` (14 tests covering 3-station `D=63`/`D=60` geometry, 31 full-cosine BEM accuracy & `< 0.11 s` speed, Kaya twist compatibility, strict paper-aligned MALIBO architecture/losses/no-TR verification, and generated figures).
- **Publication Figures & Watertight STLs**: All 8 static figures (PNG + PDF, including `figures/blade_3d_sideview.png`), 4 interactive 3D HTML renderings, and 4 watertight closed 2-manifold STL files generated via `plot_results.py` (`cfd_models/README.md`).
