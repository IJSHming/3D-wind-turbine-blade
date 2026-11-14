# 3D Blade Geometry Models for CFD Aerodynamic Validation (NREL Phase VI)

This directory contains high-fidelity, 100% watertight (closed 2-manifold) 3D STL surface meshes of the **NREL Phase VI (Sequence H)** lifting blade segment for RANS/URANS CFD validation.

## Coordinate System Definition
- **Z-axis**: Spanwise rotor radius $r \in [1.2575\,\text{m}, 5.0290\,\text{m}]$ (Aerodynamic root at $Z = 1.2575\,\text{m}$ ($0.25R$), Tip at $Z = 5.0290\,\text{m}$, net lifting span $L = 3.7715\,\text{m}$). Note: Hub flange sits at $r = 0.5083\,\text{m}$ and circular root shank extends from $r = 0.5083\,\text{m}$ to $r = 0.8835\,\text{m}$ with root transition to $r = 1.2575\,\text{m}$.
- **X-axis**: Chordwise direction aligned with the 30% chord pitch axis center $(x/c = 0.30)$ per NREL/TP-500-29955.
- **Y-axis**: Airfoil thickness normal direction.
- **Operating Condition (Sequence H)**: Air inflow $V_\infty = 7.00\,\text{m/s}$, rated rotating speed $\Omega = 72.0\,\text{rpm}$ (Tip Speed Ratio $\lambda = 5.42$), blade pitch $\theta_{p,75\%} = 4.815^\circ$ ($3.0^\circ$ tip pitch), air density $\rho = 1.225\,\text{kg/m}^3$, dynamic viscosity $\mu = 1.789 \times 10^{-5}\,\text{Pa}\cdot\text{s}$.

## Model Inventory & Selection Method
The models correspond to the **Optimal (Peak-performing) Random Seed** across the 10-seed 300-NFE benchmark:

| File Name | Design Description | Selected Seed | Power (kW) | Watertight | Volume ($m^3$) | Surface Area ($m^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `blade_baseline.stl` | Baseline Reference (NREL Phase VI S809) | `N/A` | 6.071 | True | 0.144602 | 4.3494 |
| `blade_turbo.stl` | TuRBO-1 Optimal | `8888` | 6.627 | True | 0.156052 | 4.4890 |
| `blade_llm_meta_turbo.stl` | LLM-Meta-TuRBO Optimal | `256` | 6.666 | True | 0.151436 | 4.5108 |
| `blade_malibo.stl` | MALIBO Optimal | `777` | 6.614 | True | 0.153582 | 4.4874 |

## CFD Meshing Recommendations
- **Volume Mesh Generation**: Since all meshes are fully watertight, surface wrappers (e.g. snappyHexMesh `surfaceFeatures`, Fluent Meshing Fault-tolerant Meshing, STAR-CCM+ Surface Wrapper) can directly ingest these STLs.
- **Prism Layer Extrusion**: With $Re \approx 8 \times 10^5$ in air ($V_{\mathrm{rel}} \approx 26\,\text{m/s}, c \approx 0.48\,\text{m}$), to achieve $y^+ \le 1$ for SST $k$-$\omega$ turbulence modeling, initial first cell wall thickness should be $y_1 \approx 2.5 \times 10^{-5}\,\text{m}$ ($0.025\,\text{mm}$).
- **Trailing Edge**: All geometries have sharp closed trailing edges ($t_{\text{TE}} = 0.0$), eliminating base drag and non-physical recirculation pockets.
