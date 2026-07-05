"""1D analytical creep analysis based on J-body model (Lv et al., 2019).

Implements equation (3) from the article — the steady-state creep solution
of the J-body (Zener / Standard Linear Solid) model under constant
uniaxial compression σ₀ = 3 MPa, without damage (D(t) = 0).

    ε(t) = σ₀/E₁ · [1 − E₂/(E₁+E₂) · exp(−t/τ)]

where τ = η·(E₁+E₂) / (E₁·E₂)

Parameters (from article, Section 4.1):
    σ₀ = 3 MPa
    E₁ = 2143 MPa
    E₂ = 584 MPa
    η  = 180 MPa·h

Produces:
    ``_output/strain_eyy_1d.png`` — ε(t) comparison plot

Usage::

    python run_simulation_1d.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

from src.viscoelastic_porepy import (
    setup_publication_style,
    EXPERIMENTAL_DATA_T,
    EXPERIMENTAL_DATA_EPS,
    SIM_1D_T,
    SIM_1D_EPS,
)

os.makedirs("_output", exist_ok=True)

# =====================================================================
# Material constants (from article Section 4.1, Table in text)
# =====================================================================
SIGMA_0 = 3.0       # MPa — applied constant stress
E1 = 2143.0          # MPa — elastic modulus of spring E₁
E2 = 584.0           # MPa — elastic modulus of spring E₂ (Maxwell branch)
ETA = 180.0          # MPa·h — viscosity of dashpot η

# Derived
TAU = ETA * (E1 + E2) / (E1 * E2)  # relaxation time [h]


def j_body_creep_1d(t, sigma0, e1, e2, eta):
    """Analytical 1D creep strain from J-body model (article eq. 3).

    Parameters
    ----------
    t : array_like
        Time [h].
    sigma0 : float
        Applied constant stress [MPa].
    e1, e2 : float
        Elastic moduli [MPa].
    eta : float
        Viscosity [MPa·h].

    Returns
    -------
    eps : ndarray
        Strain (dimensionless).
    """
    tau = eta * (e1 + e2) / (e1 * e2)
    eps = (sigma0 / e1) * (1.0 - e2 / (e1 + e2) * np.exp(-t / tau))
    return eps


# =====================================================================
# Run
# =====================================================================
def run_1d():
    """Execute the 1D analytical simulation and produce the comparison plot."""
    setup_publication_style()

    print("=" * 60)
    print("  1D analytical creep -- J-body model (Lv et al., 2019)")
    print(f"  sigma_0 = {SIGMA_0:.1f} MPa")
    print(f"  E1 = {E1:.0f} MPa, E2 = {E2:.0f} MPa")
    print(f"  eta = {ETA:.0f} MPa*h")
    print(f"  tau = eta*(E1+E2)/(E1*E2) = {TAU:.4f} h")
    print("=" * 60)

    # --- Compute 1D analytical strain ---
    t_ana = np.linspace(0, 8, 500)
    eps_ana = j_body_creep_1d(t_ana, SIGMA_0, E1, E2, ETA)

    # Convert to percent
    eps_ana_pct = eps_ana * 100.0

    # --- Reference 1D curve from article (digitized, smooth interpolation) ---
    interp_1d = PchipInterpolator(SIM_1D_T, SIM_1D_EPS)
    t_1d_ref = np.linspace(0, 8, 200)
    eps_1d_ref = interp_1d(t_1d_ref)

    # --- Print comparison at key times ---
    print("\n  Time [h]   eps_1D_calc [%]  eps_1D_article [%]  diff [%]")
    print("  " + "-" * 55)
    for t_check in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        calc = j_body_creep_1d(t_check, SIGMA_0, E1, E2, ETA) * 100.0
        ref = float(interp_1d(t_check))
        diff = calc - ref
        print(f"  {t_check:7.1f}     {calc:12.4f}       {ref:12.4f}      {diff:+.4f}")

    # --- Plot ---
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

    # Our 1D analytical result
    ax.plot(
        t_ana,
        eps_ana_pct,
        "r-",
        linewidth=2.0,
        label="1D analytical (eq. 3, this work)",
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
