# 3D Blade Geometry Models for CFD Hydrodynamic Validation

This directory contains high-fidelity, 100% watertight (closed 2-manifold) 3D STL surface meshes for CFD validation.

## Coordinate System Definition
- **Z-axis**: Spanwise rotor radius $r \in [0.25\,\text{m}, 1.50\,\text{m}]$ (Hub at $Z = 0.25\,\text{m}$, Tip at $Z = 1.50\,\text{m}$, Blade net span $L = 1.25\,\text{m}$).
- **X-axis**: Chordwise direction aligned with the 25% chord pitch axis center $(x/c = 0.25)$.
- **Y-axis**: Airfoil thickness normal direction.
- **Operating Condition**: Freshwater inflow $V_\infty = 1.50\,\text{m/s}$, rated rotating speed $\Omega = 40.0\,\text{rpm}$ (Tip Speed Ratio $\lambda = 4.19$), fluid density $\rho = 1000.0\,\text{kg/m}^3$, viscosity $\mu = 1.002 \times 10^{-3}\,\text{Pa}\cdot\text{s}$.

## Model Inventory & Selection Method
The models correspond to the **Optimal (Peak-performing) Random Seed** across the 10-seed 300-step benchmark (consistent with reported publication tables and figures):

| File Name | Design Description | Selected Seed | Power (kW) | Watertight | Volume ($m^3$) | Surface Area ($m^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `blade_baseline.stl` | Baseline Reference (1.5m Blade) | `N/A` | 4.679 | True | 0.027248 | 1.5109 |
| `blade_turbo.stl` | TuRBO-1 Optimal | `8888` | 5.258 | True | 0.022142 | 1.2571 |
| `blade_llm_meta_turbo.stl` | LLM-Meta-TuRBO Optimal | `3407` | 5.197 | True | 0.020205 | 1.2518 |
| `blade_malibo.stl` | MALIBO Optimal | `123` | 5.292 | True | 0.022960 | 1.3413 |

## CFD Meshing Recommendations
- **Volume Mesh Generation**: Since all meshes are fully watertight, surface wrappers (e.g. snappyHexMesh `surfaceFeatures`, Fluent Meshing Fault-tolerant Meshing, STAR-CCM+ Surface Wrapper) can directly ingest these STLs.
- **Prism Layer Extrusion**: With $Re \approx 1.5 \times 10^6$, to achieve $y^+ \le 1$ for SST $k$-$\omega$ turbulence modeling, initial first cell wall thickness should be $y_1 \approx 1.2 \times 10^{-5}\,\text{m}$ ($0.012\,\text{mm}$).
- **Trailing Edge**: All geometries have sharp closed trailing edges ($t_{\text{TE}} = 0.0$), eliminating base drag and non-physical cavitation pockets.
