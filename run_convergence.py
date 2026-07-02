"""Convergence test: automated mesh/dt sweep with L2 error computation.

Runs the viscoelastic creep simulation for each combination of
cell_size ∈ {0.1, 0.05, 0.025, 0.0125} and dt ∈ {8, 4, 2, 1} s,
records the displacement field at t = 100 s, computes the relative L2
error against the finest-mesh solution, generates:

1. ``_output/convergence_results.json`` — error table
2. ``_output/convergence.png`` — log-log convergence plot
3. ``_output/displacement_coarsest_100s.png`` — map for the coarsest grid
4. ``_output/displacement_finest_100s.png`` — map for the finest grid

Usage::

    python run_convergence.py
"""

import os
import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.interpolate import NearestNDInterpolator

from src.viscoelastic_porepy import (
    ViscoelasticSolidConstants,
    ViscoelasticMomentumBalance,
    setup_publication_style,
    save_convergence_results,
)

os.makedirs("_output", exist_ok=True)

# =====================================================================
# Material constants (updated viscosity)
# =====================================================================
NU = 0.0
E1 = 22575700000.0          # 22575.7 MPa
E2 = 11000000000.0            # 11000.0 MPa
ETA = 11_000_000_000.0 * (45.454545 * 24.0 * 60.0 * 60.0)  # Pa·s


def make_solid(cell_size: float) -> ViscoelasticSolidConstants:
    """Create solid constants (same material for all meshes)."""
    return ViscoelasticSolidConstants(
        shear_modulus=E1 / (2.0 * (1.0 + NU)),
        shear_modulus2=E2 / (2.0 * (1.0 + NU)),
        lame_lambda=E1 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        lame_lambda2=E2 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        viscosity=ETA,
    )


# =====================================================================
# Mesh / dt configurations
# =====================================================================
CELL_SIZES = [0.1, 0.05, 0.025, 0.0125]
DT_VALUES = [8.0, 4.0, 2.0, 1.0]
T_FINAL = 100.0  # seconds — snapshot time


def run_single(cell_size: float, dt: float):
    """Run one simulation and return (model, sd) at t = T_FINAL."""
    print(f"\n{'='*60}")
    print(f"  cell_size = {cell_size}, dt = {dt} s")
    print(f"{'='*60}")

    solid = make_solid(cell_size)
    time_manager = pp.TimeManager(
        schedule=[0.0, T_FINAL],
        dt_init=dt,
        dt_min_max=(dt, T_FINAL),
    )
    params = {
        "material_constants": {
            "solid": solid,
            "fluid": pp.FluidComponent(),
        },
        "time_manager": time_manager,
        "cell_size": cell_size,
        "grid_type": "cartesian",
        "snapshot_times": [T_FINAL],
    }
    model = ViscoelasticMomentumBalance(params)
    pp.run_time_dependent_model(model)
    return model


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    setup_publication_style()

    # --- Run all configurations -----------------------------------------------
    models = {}
    for cs, dt in zip(CELL_SIZES, DT_VALUES):
        model = run_single(cs, dt)
        models[(cs, dt)] = model

    # --- Reference: finest mesh -----------------------------------------------
    finest_key = (CELL_SIZES[-1], DT_VALUES[-1])
    ref_model = models[finest_key]
    ref_sd = ref_model.mdg.subdomains(dim=ref_model.nd)[0]
    ref_u = ref_model._displacement_snapshots[T_FINAL]["u"]
    ref_u_2d = ref_u.reshape(ref_model.nd, -1, order="F")
    ref_cc = ref_sd.cell_centers[:ref_model.nd, :].T  # (N, 2)

    # Build interpolators from finest mesh
    interp_ux = NearestNDInterpolator(ref_cc, ref_u_2d[0, :])
    interp_uy = NearestNDInterpolator(ref_cc, ref_u_2d[1, :])

    # --- Compute errors -------------------------------------------------------
    results = []
    for cs, dt in zip(CELL_SIZES, DT_VALUES):
        model = models[(cs, dt)]
        sd = model.mdg.subdomains(dim=model.nd)[0]
        u = model._displacement_snapshots[T_FINAL]["u"]
        u_2d = u.reshape(model.nd, -1, order="F")
        cc = sd.cell_centers[:model.nd, :].T

        # Interpolate reference to this mesh's cell centers
        ux_ref = interp_ux(cc)
        uy_ref = interp_uy(cc)

        # Weighted L2 error
        diff_sq = (u_2d[0, :] - ux_ref) ** 2 + (u_2d[1, :] - uy_ref) ** 2
        ref_sq = ux_ref**2 + uy_ref**2
        vol = sd.cell_volumes

        error_abs = np.sqrt(np.sum(diff_sq * vol))
        ref_norm = np.sqrt(np.sum(ref_sq * vol))
        error_rel = error_abs / ref_norm if ref_norm > 0 else 0.0

        Nx = int(round(0.1 / cs))
        results.append({
            "cell_size": cs,
            "dt": dt,
            "Nx": Nx,
            "error_abs": error_abs,
            "error_rel": error_rel,
        })
        print(
            f"  cs={cs:.4f}  dt={dt:.1f}  Nx={Nx:>4d}  "
            f"rel_L2={error_rel:.4e}"
        )

    # Save results
    save_convergence_results("_output/convergence_results.json", results)
    print("\nSaved _output/convergence_results.json")

    # --- Print convergence table ----------------------------------------------
    print(f"\n{'Level':<8}{'Nx':>6}{'dt':>8}{'rel L2':>14}{'order':>8}")
    print("-" * 44)
    for i, r in enumerate(results):
        if i == 0 or r["error_rel"] == 0 or results[i - 1]["error_rel"] == 0:
            order_str = "--"
        else:
            order_str = f"{np.log2(results[i-1]['error_rel'] / r['error_rel']):.2f}"
        print(
            f"{i:<8}{r['Nx']:>6}{r['dt']:>8.1f}"
            f"{r['error_rel']:>14.4e}{order_str:>8}"
        )

    # --- Convergence plot -----------------------------------------------------
    # Exclude finest (error = 0 by definition)
    plot_data = [r for r in results if r["error_rel"] > 0]
    if len(plot_data) >= 2:
        Nx_arr = np.array([r["Nx"] for r in plot_data])
        Nt_arr = 1.0 / np.array([r["dt"] for r in plot_data])
        resolution = (Nx_arr**2 * Nt_arr) ** 0.25
        errors = np.array([r["error_rel"] for r in plot_data])

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.loglog(
            resolution,
            errors,
            "o-",
            color="#1f77b4",
            markerfacecolor="white",
            markeredgewidth=1.4,
            markeredgecolor="#1f77b4",
            zorder=5,
            label=(
                r"$\|\mathbf{u} - \mathbf{u}_h\|_{L^2}"
                r" / \|\mathbf{u}\|_{L^2}$"
            ),
        )

        # Reference slope triangle (order 2)
        if len(resolution) >= 2:
            x1, x2 = resolution[-2], resolution[-1]
            y_base = errors[-2] * 2.0
            y_top = y_base * (x1 / x2) ** 2
            tri_x = [x1, x2, x2, x1]
            tri_y = [y_base, y_base, y_top, y_base]
            ax.fill(
                tri_x, tri_y,
                color="#d3d3d3", edgecolor="0.3", linewidth=0.8, zorder=3,
            )
            ax.text(
                np.sqrt(x1 * x2),
                (y_base * y_top) ** 0.5 * 0.85,
                "2",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                color="0.2",
                zorder=6,
            )

        ax.set_xlabel(r"$(N_x^{\,2}\cdot N_t)^{1/4}$")
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.grid(True, which="major", ls="-", lw=0.5, alpha=0.35)
        ax.grid(True, which="minor", ls=":", lw=0.3, alpha=0.25)
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.7",
            framealpha=0.95,
        )
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        fig.tight_layout()
        fig.savefig("_output/convergence.png", dpi=300)
        plt.close(fig)
        print("Saved _output/convergence.png")

    # --- Displacement maps at t = 100 s (coarsest and finest) -----------------
    for label, key in [("coarsest", (CELL_SIZES[0], DT_VALUES[0])),
                        ("finest", finest_key)]:
        model = models[key]
        snap = model._displacement_snapshots.get(T_FINAL)
        if snap is not None:
            filepath = f"_output/displacement_{label}_100s.png"
            model.plot_displacement_map(
                snap["u"],
                title=f"|u| at t = 100 s ({label} mesh, h = {key[0]})",
                filepath=filepath,
            )

    print("\nDone.")
