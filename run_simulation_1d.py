"""Quasi-1D PorePy viscoelastic creep simulation.

Runs a quasi-1D simulation using a narrow 2D strip (one cell wide)
with roller boundary conditions, reproducing 1D uniaxial compression.

Parameters (from article, Section 4.1):
    sigma_0 = 3 MPa
    E_1 = 2143 MPa
    E_2 = 584 MPa
    eta = 180 MPa*h
    nu  = 0.0 (set to zero so that the 2D Lame formulation
               reduces to the 1D modulus: lambda+2*mu = E)

Produces:
    ``_output/strain_eyy_1d.png`` — ε(t) comparison plot
    ``_output/strain_1d.npz``     — raw strain history

Usage::

    python run_simulation_1d.py
"""

import os
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

from src.viscoelastic_porepy import (
    ViscoelasticSolidConstants,
    ViscoelasticMomentumBalance1D,
    setup_publication_style,
    save_strain_history,
    EXPERIMENTAL_DATA_T,
    EXPERIMENTAL_DATA_EPS,
    SIM_1D_T,
    SIM_1D_EPS,
)

os.makedirs("_output", exist_ok=True)

# =====================================================================
# Material constants (from article Section 4.1)
# =====================================================================
# nu = 0 ensures that the 2D plane-strain Lame formulation
# reduces to the 1D modulus relation (lambda=0, mu=E/2,
# so lambda+2*mu = E).  With nu > 0 the effective modulus
# would be E(1-nu)/((1+nu)(1-2nu)) != E, giving a ~30%
# discrepancy vs. the 1D analytical solution.
NU = 0.0
E1 = 2_143_000_000.0          # 2143 MPa in Pa
E2 = 584_000_000.0            # 584 MPa in Pa
ETA = 180_000_000.0 * (60.0 * 60.0)  # 180 MPa·h -> Pa·s

DT = 1.0 * pp.SECOND
FINAL_TIME = 8.0 * pp.HOUR


# =====================================================================
# Run
# =====================================================================
def run_1d():
    """Execute the quasi-1D PorePy simulation and produce the comparison plot."""
    setup_publication_style()

    # Derived quantities for info
    mu2 = E2 / (2.0 * (1.0 + NU))
    tau = ETA / mu2

    print("=" * 60)
    print("  Quasi-1D viscoelastic creep -- PorePy simulation")
    print(f"  sigma_0 = 3 MPa (applied as boundary traction)")
    print(f"  E1 = {E1/1e6:.0f} MPa, E2 = {E2/1e6:.0f} MPa")
    print(f"  nu = {NU}")
    print(f"  eta = {ETA:.4e} Pa*s")
    print(f"  tau = eta/mu2 = {tau:.2e} s = {tau/3600:.2f} h")
    print(f"  dt = {DT:.1f} s, T = {FINAL_TIME/3600:.1f} h")
    print("=" * 60)

    # --- Build model ---
    solid = ViscoelasticSolidConstants(
        shear_modulus=E1 / (2.0 * (1.0 + NU)),
        shear_modulus2=mu2,
        lame_lambda=E1 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        lame_lambda2=E2 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        viscosity=ETA,
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
        "cell_size": 0.00125,
        "grid_type": "cartesian",
        "snapshot_times": [FINAL_TIME],
    }

    model = ViscoelasticMomentumBalance1D(params)
    pp.run_time_dependent_model(model)
    print("Simulation complete.")

    # --- Save strain history ---
    save_strain_history("_output/strain_1d.npz", model.strain_history)
    print("Saved _output/strain_1d.npz")

    # --- Plot ε_yy(t) ---
    if len(model.strain_history["times"]) > 0:
        t = np.array(model.strain_history["times"])
        eyy_u = np.array(model.strain_history["eyy_u"])

        # Correct for extensometer gauge-length bias.
        # The strain recorder measures u_y at the top cells (mean y ≈ 0.095)
        # and divides by the full domain height (0.1).  For uniform strain
        # the true strain is u_y(y) / y, not u_y(y) / H, so we apply:
        #   correction = H / y_mean_top ≈ 0.1 / 0.095 ≈ 1.053
        sd = model.mdg.subdomains(dim=model.nd)[0]
        y_mean_top = np.mean(sd.cell_centers[1, model._top_cells])
        correction = model._domain_height / y_mean_top
        eyy_u = eyy_u * correction

        # Reference 1D curve from article (digitized, smooth interpolation)
        interp_1d = PchipInterpolator(SIM_1D_T, SIM_1D_EPS)
        t_1d_ref = np.linspace(0, 8, 200)
        eps_1d_ref = interp_1d(t_1d_ref)

        # --- Print comparison at key times ---
        print("\n  Time [h]   eps_PorePy [%]   eps_article [%]   diff [%]")
        print("  " + "-" * 55)
        for t_check in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
            # Find nearest simulation time
            idx = np.argmin(np.abs(t - t_check))
            sim_val = np.abs(eyy_u[idx]) * 100.0
            ref_val = float(interp_1d(t_check))
            diff = sim_val - ref_val
            print(
                f"  {t_check:7.1f}     {sim_val:12.4f}       "
                f"{ref_val:12.4f}      {diff:+.4f}"
            )

        fig, ax = plt.subplots(figsize=(8, 6))

        # Experimental data
        ax.plot(
            EXPERIMENTAL_DATA_T,
            EXPERIMENTAL_DATA_EPS,
            "bD",
            markersize=6,
            label="Experimental data",
        )

        # 1D reference from article (digitized)
        ax.plot(
            t_1d_ref,
            eps_1d_ref,
            "k--",
            linewidth=1.5,
            label="1D simulation (article, digitized)",
        )

        # Our quasi-1D PorePy result
        ax.plot(
            t,
            np.abs(eyy_u) * 100,
            "r-",
            linewidth=2.0,
            label="Quasi-1D simulation (PorePy)",
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
        fig.savefig("_output/strain_eyy_1d.png", dpi=300)
        plt.close(fig)
        print("\nSaved _output/strain_eyy_1d.png")

    print("Done.")


if __name__ == "__main__":
    run_1d()
