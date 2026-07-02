"""Simulation with fracture: 8-hour creep test.

Produces:
1. ``_output/displacement_frac_8h.png`` — displacement magnitude map at 8 h
2. ``_output/strain_eyy_frac.png`` — ε_yy(t) vs experimental and 1D data
3. ``_output/strain_frac.npz`` — raw strain history

Usage::

    python run_simulation_fracture.py
"""

import os
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

from src.viscoelastic_porepy import (
    ViscoelasticSolidConstants,
    ViscoelasticMomentumBalanceFracture,
    setup_publication_style,
    save_strain_history,
    EXPERIMENTAL_DATA_T,
    EXPERIMENTAL_DATA_EPS,
    SIM_1D_T,
    SIM_1D_EPS,
)

os.makedirs("_output", exist_ok=True)

# =====================================================================
# Material constants
# =====================================================================
NU = 0.3
E1 = 2_143_000_000.0          # 2143 MPa
E2 = 584_000_000.0            # 584 MPa
ETA = 180000000.0 * (60.0 * 60.0)  # Pa·s

DT = 1.0 * pp.SECOND
FINAL_TIME = 8.0 * pp.HOUR

# =====================================================================
# Run
# =====================================================================
if __name__ == "__main__":
    setup_publication_style()

    solid = ViscoelasticSolidConstants(
        shear_modulus=E1 / (2.0 * (1.0 + NU)),
        shear_modulus2=E2 / (2.0 * (1.0 + NU)),
        lame_lambda=E1 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        lame_lambda2=E2 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        viscosity=ETA,
        fracture_normal_stiffness=200_000_000_000.0,   # 200 GPa
        fracture_tangential_stiffness=100_000_000_000.0,  # 100 GPa
    )

    time_manager = pp.TimeManager(
        schedule=[0.0, FINAL_TIME],
        dt_init=DT,
        dt_min_max=(DT, FINAL_TIME),
    )

    params = {
        "material_constants": {
            "solid": solid,
            "fluid": pp.FluidComponent(),
        },
        "time_manager": time_manager,
        "cell_size": 0.005,
        "grid_type": "simplex",
        "snapshot_times": [FINAL_TIME],
    }

    print("=" * 60)
    print("  Viscoelastic creep test — WITH FRACTURE")
    print(f"  dt = {DT:.1f} s, T = {FINAL_TIME/3600:.1f} h")
    print(f"  E1 = {E1/1e6:.0f} MPa, E2 = {E2/1e6:.0f} MPa")
    print(f"  η  = {ETA:.4e} Pa·s")
    tau = ETA / (E2 / (2.0 * (1.0 + NU)))
    print(f"  τ  = η/μ₂ = {tau:.2e} s = {tau/3600:.2f} h")
    print(f"  Fracture: diagonal (0.04, 0.04) → (0.06, 0.06)")
    print("=" * 60)

    model = ViscoelasticMomentumBalanceFracture(params)
    pp.run_time_dependent_model(model)
    print("Simulation complete.")

    # --- Save strain history --------------------------------------------------
    save_strain_history("_output/strain_frac.npz", model.strain_history)
    print("Saved _output/strain_frac.npz")

    # --- Displacement map at 8 h ----------------------------------------------
    snap = model._displacement_snapshots.get(FINAL_TIME)
    if snap is not None:
        model.plot_displacement_map(
            snap["u"],
            title="|u| at t = 8 h (with fracture)",
            filepath="_output/displacement_frac_8h.png",
        )

    # --- ε_yy(t) plot ---------------------------------------------------------
    if len(model.strain_history["times"]) > 0:
        t = np.array(model.strain_history["times"])
        eyy_u = np.array(model.strain_history["eyy_u"])

        # 1D simulation curve (smooth interpolation)
        interp_1d = PchipInterpolator(SIM_1D_T, SIM_1D_EPS)
        t_1d = np.linspace(0, 8, 200)
        eps_1d = interp_1d(t_1d)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            EXPERIMENTAL_DATA_T,
            EXPERIMENTAL_DATA_EPS,
            "bD",
            markersize=6,
            label="Experimental data",
        )
        ax.plot(
            t_1d,
            eps_1d,
            "k--",
            linewidth=1.5,
            label="1D simulation (article)",
        )
        ax.plot(
            t,
            np.abs(eyy_u) * 100,
            "r-",
            linewidth=1.5,
            label="2D simulation (PorePy, fracture)",
        )
        ax.set_xlabel("Time (h)", fontsize=13)
        ax.set_ylabel("Strain (%)", fontsize=13)
        ax.set_xlim(0, 8)
        ax.set_ylim(0.09, 0.15)
        ax.legend(
            framealpha=1.0,
            edgecolor="black",
            fancybox=False,
            loc="lower right",
        )
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(
            width=1.5, direction="in", top=True, right=True
        )
        fig.tight_layout()
        fig.savefig("_output/strain_eyy_frac.png", dpi=300)
        plt.close(fig)
        print("Saved _output/strain_eyy_frac.png")

    print("Done.")
