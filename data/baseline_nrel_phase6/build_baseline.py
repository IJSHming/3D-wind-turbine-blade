"""Generate NREL Phase VI baseline 27-section NPZ from official Table A-1 and S809 airfoil data."""

from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import aerosandbox as asb


def build_nrel_phase6_baseline() -> Path:
    """Build and persist NREL Phase VI 27-section aerodynamic blade baseline."""
    data_dir = Path(__file__).resolve().parent
    s809_path = data_dir / "s809.dat"
    table_path = data_dir / "phase6_geometry_table.txt"
    out_path = data_dir / "baseline_27_sections.npz"

    # Load NREL S809 coordinates (skip header line)
    coords = np.loadtxt(s809_path, skiprows=1, dtype=np.float64)
    af_s809 = asb.Airfoil(name="NREL_S809", coordinates=coords)
    kulfan_s809 = af_s809.to_kulfan_airfoil(n_weights_per_side=8)

    theta_s809 = np.concatenate([
        kulfan_s809.upper_weights,
        kulfan_s809.lower_weights,
        [float(kulfan_s809.leading_edge_weight), 0.0],
    ]).astype(np.float64)

    # Parse S809 aerodynamic blade stations (r in [1.2575, 5.0290])
    radii_raw, twists_raw = [], []
    with open(table_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if parts[4] == "S809":
                radii_raw.append(float(parts[0]))
                twists_raw.append(float(parts[3]))

    r_knots = np.array(radii_raw, dtype=np.float64)
    tw_knots = np.array(twists_raw, dtype=np.float64)

    r_root = 1.2575
    r_tip = 5.0290
    n_sections = 27
    radii = np.linspace(r_root, r_tip, n_sections, dtype=np.float64)
    blspn = radii - r_root
    sweep = np.zeros(n_sections, dtype=np.float64)

    # Exact linear taper from Giguere & Selig (1999) / Hand et al. (2001): 0.737 m -> 0.356 m
    chord = 0.737 + (0.356 - 0.737) * (radii - r_root) / (r_tip - r_root)

    # Smooth monotonic twist interpolation calibrated to Table A-1 knots
    twist = PchipInterpolator(r_knots, tw_knots)(radii).astype(np.float64)

    af_ids = np.arange(1, n_sections + 1, dtype=np.int64)
    kulfan_thetas = np.tile(theta_s809, (n_sections, 1)).astype(np.float64)

    np.savez(
        out_path,
        blspn=blspn,
        sweep=sweep,
        twist=twist,
        chord=chord,
        af_ids=af_ids,
        kulfan_thetas=kulfan_thetas,
        radii=radii,
    )
    print(f"Saved NREL Phase VI 27-section baseline to: {out_path}")
    print(f"  Aerodynamic span: r in [{radii[0]:.4f}, {radii[-1]:.4f}] m (L={blspn[-1]:.4f} m)")
    print(f"  Chord range: [{chord[0]:.4f}, {chord[-1]:.4f}] m | Twist range: [{twist[0]:.3f}, {twist[-1]:.3f}] deg")
    print(f"  S809 Kulfan max_thickness = {kulfan_s809.to_airfoil().max_thickness():.5f}, TE_thickness = 0.0")
    return out_path


if __name__ == "__main__":
    build_nrel_phase6_baseline()
