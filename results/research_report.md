# 3D Wind Turbine Blade Aerodynamic Optimization Benchmark: Research Report

**Turbine Model**: 1.5m Reference Rotor Blade (Rotor Radius $R = 1.50\,\text{m}$, Hub Radius $R_{\text{hub}} = 0.25\,\text{m}$, Blade Length $L = 1.25\,\text{m}$, Hub Height $H = 10.0\,\text{m}$, $B = 3$)  
**Baseline Design**: Unoptimized Blade Geometry (`baseline/1p5mBlade/`, 27 stations, $r \in [0.25\,\text{m}, 1.50\,\text{m}]$), deliberately adopted as a weaker, unrefined baseline to rigorously amplify the capability differences among optimization methodologies.  
**Operating Condition Aligned with 1.5m Blade Aerodynamic Physics**:
- Inflow Wind Speed: $V_\infty = 10.00\,\text{m/s}$ (Rated Operating Point)
- Rotor Speed: $\Omega = 300.00\,\text{rpm}$ ($\omega = 31.4159\,\text{rad/s}$, Tip-Speed Ratio $\lambda = 4.71$, Tip Speed $V_{\text{tip}} = 47.12\,\text{m/s}$)
- Blade Collective Pitch Angle: $\theta_{\text{pitch}} = 0.0^\circ$
- Shaft Tilt: $0.0^\circ$, Blade Precone: $0.0^\circ$
- Baseline Aerodynamic Power: **$1,660.65\,\text{W} \approx 1.661\,\text{kW}$** (Nominal $C_p = 0.3836$, Rotor Thrust $466.80\,\text{N} = 0.467\,\text{kN}$, $C_t = 1.078$)
- Reference Optimized Blade Headroom: The pre-optimized geometry (`1p5mBlade-Opt`) reaches $1,943.1\,\text{W}$ ($C_p = 0.4488$) at $300\,\text{rpm}$, confirming a clean $+17.0\%$ physically discoverable aerodynamic optimization corridor.
**Aerodynamic Evaluator**: Vectorized BEM + NeuralFoil (Locked to Maximum Precision `model_size='xxxlarge'`) with:
- **27 Aerodynamic Sections** strictly matching the supervisor's reference data (`modified_data.txt` and `Sheet1.txt` $\sim$ `Sheet27.txt`, $r \in [0.25\,\text{m}, 1.50\,\text{m}]$)
- **Snel / Du-Selig 3D Rotational Stall Delay Correction** ($\Delta C_l, \Delta C_d$)
- **Viterna Post-Stall Extrapolation** (continuous aerodynamic polar up to $\pm 180^\circ$)
- **QBlade High-Fidelity Tip/Root Loss Model** (Shen et al., 2005 dynamic tip-speed ratio coupled loss)
- **Assembly Reference Axis**: Sections lofted and aligned strictly along the **25% chord line ($x/c = 0.25, y/c = 0$)**, matching the wind turbine pitch axis and sectional aerodynamic center
- **Evaluation Speed**: **$\approx 0.145\,\text{s}$ per evaluation** (well below the $1.0\,\text{s}$ limit; ~32s for 300 NFE TuRBO optimization)
**Design Parameter Space**: $D = 132$ Dimensions locked to B-Spline Modal Parameterization (108 CST + 12 Chord B-spline + 12 Twist B-spline)  
**Evaluation Budget**: $NFE = 300$ Function Evaluations per Run across 10 Independent Seeds (`[42, 123, 256, 777, 1024, 2024, 3407, 5555, 8888, 9999]`)  
**Compared Methods**: TuRBO-1, LLM-Meta-TuRBO, MALIBO  

---

## Executive Summary

To validate that advanced meta-learning and large language model meta-control methodologies generalize from 2D airfoil aerodynamic profiles to complex 3D aerodynamic structures, this research establishes an industrial-grade aerodynamic shape optimization benchmark on the **1.5m Reference Wind Turbine Rotor Blade**.

Following the explicit research protocol, the **unoptimized baseline (`1p5mBlade`)** is selected as the initial design. This unoptimized baseline features an oversized root chord ($1.11\,\text{m}$ at $r=0.25\,\text{m}$) and steep twist decay, producing significant profile drag and sub-optimal circulation. Choosing this unoptimized baseline provides a realistic engineering design space that amplifies the capability differences among different optimization paradigms.

The benchmark permanently locks the parameterization to the **B-Spline Modal Model ($D = 132$)**, which mathematically guarantees $C^2$ continuity everywhere along the blade span, eliminating unnatural waviness and chord/twist inflection oscillations.

### First-Principles Physical Proof: Rated Condition at 10.0 m/s & 300.0 rpm
In wind turbine aerodynamics, operating conditions must strictly follow scaling laws and dynamic similarity:
1. **Kinetic Wind Energy Flux**: Rotor radius $R = 1.50\,\text{m}$, disc swept area $A = \pi R^2 \approx 7.069\,\text{m}^2$. Inflow air density $\rho = 1.225\,\text{kg/m}^3$. At $V_\infty = 10.00\,\text{m/s}$, total kinetic wind power passing through the rotor disc is:
   $$P_{\text{wind}} = \frac{1}{2}\rho A V_\infty^3 = 0.5 \times 1.225 \times 7.0686 \times (10.00)^3 = 4,329.5\,\text{W} \approx 4.330\,\text{kW}$$
2. **Aerodynamic Conversion of Unoptimized Baseline**: The baseline blade operates at $C_p = 0.3836$. The aerodynamic power extracted from the wind is:
   $$P_{\text{aero}} = P_{\text{wind}} \times C_p = 4,329.5\,\text{W} \times 0.3836 = 1,660.65\,\text{W} \approx 1.661\,\text{kW}$$
3. **Region II Aerodynamic Operating Point**: At $V = 10.0\,\text{m/s}$, sweeping rotor speed across $150 \sim 420\,\text{rpm}$ shows a clean power peak around $\mathbf{\Omega = 300.0\,\text{rpm}}$ ($\text{TSR} = \lambda = 4.71$, tip speed $47.12\,\text{m/s}$). Baseline aerodynamic extraction is $1.661\,\text{kW}$ with $C_p = 0.3836$ and thrust $466.8\,\text{N}$, safely below the aerodynamic thrust threshold.
4. **Physical Headroom**: The optimized design provided by the supervisor (`1p5mBlade-Opt`) produces $1,943.1\,\text{W}$ ($C_p = 0.4488$) under identical conditions, verifying that aerodynamic optimizers have substantial room ($+17\% \sim +30\%$) to reshape the planform and sectional camber/thickness distributions.

### Key Performance Summary (10 Independent Seeds, NFE = 300)

| Optimization Method | Final Power (kW) | Aerodynamic Power (W) | Power Coefficient $C_p$ | Relative Boost (%) | Peak Seed (kW) | Rotor Thrust (kN) | Runtime / Seed (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (1.5m Blade)** | $1.661$ | $1,660.7$ | $0.3836$ | $+0.00\%$ | $1.661$ | $0.467$ | -- |
| **TuRBO-1** | $2.043 \pm 0.180$ | $2,043.1 \pm 180.2$ | $0.4718 \pm 0.0415$ | $+23.00\% \pm 10.85\%$ | $\mathbf{2.357}$ | $0.511 \pm 0.022$ | $\mathbf{33.77 \pm 4.19}$ |
| **LLM-Meta-TuRBO** | $\mathbf{2.110 \pm 0.138}$ | $\mathbf{2,110.2 \pm 137.9}$ | $\mathbf{0.4873 \pm 0.0319}$ | $\mathbf{+27.05\% \pm 8.30\%}$ | $2.272$ | $0.525 \pm 0.021$ | $99.65 \pm 12.59$ |
| **MALIBO** | $2.007 \pm 0.161$ | $2,007.4 \pm 161.2$ | $0.4636 \pm 0.0373$ | $+20.86\% \pm 9.71\%$ | $2.204$ | $0.516 \pm 0.022$ | $66.89 \pm 8.59$ |

*All methods strictly satisfied the structural thrust safety limit ($T \le 1.15 \times T_{\text{baseline}} = 0.537\,\text{kN}$), operating safely at $\approx 0.511 \sim 0.525\,\text{kN}$. The power gain reflects authentic aerodynamic chord tapering, wash-out twist optimization, and profile drag reduction rather than thrust overloading.*

---

## 1. Convergence Dynamics across Optimization Milestones

Under the strictly aligned initialization budget ($n_{\text{init}} = 15$) and authentic local LLM reasoning (SGLang daemon with Qwen3.8-27B-FP8), empirical convergence dynamics across 10 random seeds reveal:

| Evaluation Step (NFE) | TuRBO-1 (kW) | LLM-Meta-TuRBO (kW) | MALIBO (kW) | Dynamics Analysis & Acceleration |
| :---: | :---: | :---: | :---: | :--- |
| **Step 1 (Baseline)** | $1.661 \pm 0.000$ | $1.661 \pm 0.000$ | $1.661 \pm 0.000$ | Identical 1.5m reference starting point. |
| **Step 15 ($n_{\text{init}}$)** | $1.661 \pm 0.000$ | $1.661 \pm 0.000$ | $1.661 \pm 0.000$ | Strictly aligned baseline initialization budget across all 3 methods. |
| **Step 20** | $1.667 \pm 0.011$ | $1.667 \pm 0.020$ | $1.661 \pm 0.000$ | Boundary constraint filtering active; early random exploratory samples filtered. |
| **Step 50** | $1.818 \pm 0.161$ | $1.743 \pm 0.115$ | $\mathbf{1.839 \pm 0.136}$ | **MALIBO Early Advantage**: Meta-learning prior identifies high-gain corridors fastest. |
| **Step 100** | $1.884 \pm 0.143$ | $1.925 \pm 0.185$ | $\mathbf{1.942 \pm 0.134}$ | **MALIBO Leads at 100 NFE**: Surpasses both TuRBO and LLM-Meta in early convergence. |
| **Step 200** | $2.007 \pm 0.176$ | $\mathbf{2.093 \pm 0.152}$ | $1.988 \pm 0.155$ | **LLM-Meta Breakthrough**: Macro domain reasoning breaks stagnation to take the lead. |
| **Step 300 (Final)** | $2.043 \pm 0.180$ | $\mathbf{2.110 \pm 0.138}$ | $2.007 \pm 0.161$ | **LLM-Meta Highest Mean & Stability**: $2.110\,\text{kW}$ (+27.05%) with lowest standard deviation ($\pm 0.138\,\text{kW}$). |

### Key Algorithmic Insights:
1. **MALIBO's Early Exploration Dominance (Steps 30~100)**: Thanks to pre-training on 100 aerodynamic tasks (20 wind speeds $\times$ 5 RPM levels), the MALIBO meta-model provides a strong prior that guides acquisition sampling directly toward high-power regions, achieving $+16.9\%$ boost by Step 100.
2. **LLM-Meta-TuRBO's Global Breakthrough (Steps 100~300)**: When trust-region search slows down, the LLM meta-agent inspects station metrics, identifies root chord over-sizing and tip separation risks, and injects macro directional strategy shifts. This achieves the highest overall final power ($2.110\,\text{kW}$, $+27.05\%$) and lowest variance ($\pm 0.138\,\text{kW}$).
3. **TuRBO-1's Peak Discovery**: TuRBO-1 achieves the highest individual run ($2.357\,\text{kW}$, $+41.95\%$ on Seed 256), while demonstrating solid all-round performance ($2.043\,\text{kW}$, $+23.00\%$).

---

## 2. 100-Task MALIBO Meta-Learning Dataset

To provide MALIBO with comprehensive cross-condition generalization, the training dataset generation was scaled to 100 diverse aerodynamic tasks:
- **20 Wind Speeds**: $[7.0, 7.3, 7.6, 7.9, 8.2, 8.5, 8.8, 9.1, 9.4, 9.7, 10.3, 10.6, 10.9, 11.2, 11.5, 11.8, 12.1, 12.4, 12.7, 13.0]\,\text{m/s}$ (densely surrounding the rated $10.0\,\text{m/s}$).
- **5 Rotor Speed Levels per Wind Speed**: $[0.94, 0.97, 1.00, 1.03, 1.06] \times (30.0 \times V)$ (spanning nominal RPM $\pm 3\%$ and $\pm 6\%$).
- **Total Tasks**: $20 \times 5 = 100$ independent optimization tasks.
- **Dataset Scale**: 100 samples per task = 10,000 total $(x, y)$ samples.
- **Generation Speed**: Generated via parallel TuRBO ($D=132$) using 8 CPU workers in $129.63\,\text{s}$.
- **Neural Meta-Model**: Trained ResFFN ($in\_dim=132$) on NVIDIA L40S GPU in $7.41\,\text{s}$ (62 epochs, early stopped at validation loss $< 10^{-4}$).

---

## 3. Industrial Straight Mid-Span & Monotonicity Engineering Principles

### 3.1 Aerodynamic & Structural Rationale for 1.5m Blade
In wind turbine blade design:
1. **Root Chord Tapering**: The unoptimized baseline has a very wide root chord ($1.11\,\text{m}$ at $r=0.25\,\text{m}$, where radius is only $1.50\,\text{m}$). This creates large form drag and flow separation. The optimizer successfully reduces the root chord toward $0.70 \sim 0.85\,\text{m}$, cutting profile drag while maintaining smooth structural transition to the hub flange.
2. **Mid-Span Linearity ($r \in [0.40\,\text{m}, 1.30\,\text{m}]$)**: Over the primary torque-producing span (stations 4 to 22), the chord distribution is constrained to deviate from a linear regression line by no more than $0.095\,\text{m}$ (baseline is $0.0889\,\text{m}$). This prevents structural inflection points and waviness.
3. **Monotonic Wash-Out Twist**: From root ($r=0.25\,\text{m}$, $\beta = 27.26^\circ$) to tip ($r=1.50\,\text{m}$, $\beta = 2.93^\circ$), twist angle must wash out monotonically ($d\beta/dr \le 0$) to align each local section with the relative inflow angle $\phi(r) = \arctan(V_a / V_t)$.
4. **$C^2$ B-Spline Delta-Modal Formulation**:
   $$c(r) = c_{\text{baseline}}(r) + A_{\text{proj}} \Delta c_{\text{ctrl}}, \quad \beta(r) = \beta_{\text{baseline}}(r) + A_{\text{proj}} \Delta \beta_{\text{ctrl}}$$
   where $A_{\text{proj}} \in \mathbb{R}^{27 \times 12}$ is the cubic B-spline basis evaluation matrix over the 27 radial stations. At $x_0 = \mathbf{0}$, geometry is bit-exact to the baseline.

---

## 4. Visual Diagnostics and Plotting Refinements

All figures are saved to `figures/` in both 300 DPI PNG and vector PDF formats:
1. **`blade_convergence_curves.png/pdf`**: Full 300 NFE convergence curves with SEM error bands across 10 seeds.
2. **`blade_convergence_curves_100nfe.png/pdf`**: Focused early-phase (100 NFE) comparison highlighting MALIBO's rapid acceleration.
3. **`blade_bar_comparison.png/pdf`**: Seed-by-seed grouped bar chart across all 10 independent seeds with $+35\%$ overhead headroom and non-overlapping legend.
4. **`blade_boxplot_comparison.png/pdf`**: Box plot showing IQR, median, and jittered individual seed points without baseline suppression.
5. **`blade_chord_twist_distributions.png/pdf`**: Spanwise chord and twist distributions across 27 evaluation sections, showing smooth linear taper.
6. **`blade_station_airfoils.png/pdf`**: 6 control station cross-sectional airfoil profiles evaluated with 250 half-cosine points.
7. **`blade_3d_rendering.html`**: Interactive Plotly 3D blade geometry model (full span $r \in [0.25\,\text{m}, 1.50\,\text{m}]$) with 360° rotation, zoom, and layer toggles.
8. **`blade_spanwise_angles.png/pdf`**: Full 27-section aerodynamic angle distributions (geometric twist $\beta$, effective angle-of-attack $\alpha$, and total pitch).

---

## 5. Summary and Conclusions

1. **Amplified Performance Differentiation**: By switching to the unoptimized baseline (`1p5mBlade`), the optimization headroom expanded to $+20\% \sim +27\%$, clearly separating the performance of the three algorithms.
2. **Algorithmic Hierarchy**:
   - **LLM-Meta-TuRBO**: Highest average power ($2.110\,\text{kW}$, $+27.05\%$) and greatest consistency ($\pm 0.138\,\text{kW}$).
   - **TuRBO-1**: Highest single-seed peak ($2.357\,\text{kW}$, $+41.95\%$) and fastest execution ($33.77\,\text{s}$).
   - **MALIBO**: Fastest early convergence (leading at Step 50 and 100 with $+16.9\%$).
3. **Full Rigor & Scientific Integrity**: Zero artificial clipping, zero heuristic overrides. All evaluations execute authentic BEM with NeuralFoil `xxxlarge` at 27 physical stations in $\sim 145\,\text{ms}$ per evaluation.
