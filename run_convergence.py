"""Convergence test: automated mesh/dt sweep with L2 error computation.

Runs the viscoelastic creep simulation for each combination of
cell_size and dt, records the displacement field at t = 100 s,
computes the relative L2 error against the finest-mesh solution,
generates:

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
from config import (
    CONV_NU as NU,
    CONV_E1 as E1,
    CONV_E2 as E2,
    CONV_ETA as ETA,
    CONV_T_FINAL as T_FINAL,
    CONV_CELL_SIZES as CELL_SIZES,
    CONV_DT_VALUES as DT_VALUES,
)


# =====================================================================
# Helper functions
# =====================================================================
def make_solid() -> ViscoelasticSolidConstants:
    """Create solid constants (same material for all meshes)."""
    return ViscoelasticSolidConstants(
        shear_modulus=E1 / (2.0 * (1.0 + NU)),
        shear_modulus2=E2 / (2.0 * (1.0 + NU)),
        lame_lambda=E1 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        lame_lambda2=E2 * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        viscosity=ETA,
    )


class ConvergenceMMSModel(ViscoelasticMomentumBalance):
    """MMS Model for convergence test. Overrides BCs and body force."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryConditionVectorial(
            sd,
            domain_sides.south + domain_sides.north + domain_sides.east + domain_sides.west,
            "dir"
        )

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        cc = bg.cell_centers
        A_MMS = 1.0e-3
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        t = self.time_manager.time
        L = 0.1  # domain size in config/geometry is 0.1 m

        ux = A_MMS * np.sin(np.pi * cc[0] / L) * np.sin(np.pi * cc[1] / L) * (1.0 - np.exp(-b_MMS * t))
        uy = A_MMS * np.sin(np.pi * cc[0] / L) * np.sin(np.pi * cc[1] / L) * (1.0 - np.exp(-b_MMS * t))

        data = np.zeros((self.nd, bg.num_cells))
        data[0, :] = ux
        data[1, :] = uy
        return data.ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def _compute_body_force_values(self, subdomains: list[pp.Grid]) -> np.ndarray:
        vals = []
        A_MMS = 1.0e-3
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        t = self.time_manager.time
        beta = self.solid.shear_modulus2 / self.solid.viscosity

        # For NU=0.0
        shear_modulus = self.solid.shear_modulus
        shear_modulus2 = self.solid.shear_modulus2
        lame_lambda = self.solid.lame_lambda
        lame_lambda2 = self.solid.lame_lambda2
        k2 = (2.0 * shear_modulus2 + 3.0 * lame_lambda2) / 3.0

        L = 0.1
        Lx, Ly = L, L
        kx = np.pi / Lx
        ky = np.pi / Ly
        Ax, Ay = A_MMS, A_MMS

        T1 = 1.0 - np.exp(-b_MMS * t)
        T2 = (b_MMS / (beta - b_MMS)) * (np.exp(-b_MMS * t) - np.exp(-beta * t))

        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))
            if sd.dim == 2:
                cc = sd.cell_centers
                x, y = cc[0], cc[1]
                sx = np.sin(kx * x)
                cx = np.cos(kx * x)
                sy = np.sin(ky * y)
                cy = np.cos(ky * y)

                term_x_E = ( (lame_lambda + 2*shear_modulus)*kx**2*Ax + shear_modulus*ky**2*Ax ) * sx * sy - ( (lame_lambda + shear_modulus)*kx*ky*Ay ) * cx * cy
                term_x_E_vis = ( (k2 + 4.0/3.0*shear_modulus2)*kx**2*Ax + shear_modulus2*ky**2*Ax ) * sx * sy - ( (k2 + 1.0/3.0*shear_modulus2)*kx*ky*Ay ) * cx * cy

                term_y_E = ( (lame_lambda + 2*shear_modulus)*ky**2*Ay + shear_modulus*kx**2*Ay ) * sx * sy - ( (lame_lambda + shear_modulus)*kx*ky*Ax ) * cx * cy
                term_y_E_vis = ( (lame_lambda2 + 2*shear_modulus2)*ky**2*Ay + shear_modulus2*kx**2*Ay ) * sx * sy - ( (lame_lambda2 + shear_modulus2)*kx*ky*Ax ) * cx * cy

                force_x = term_x_E * T1 + term_x_E_vis * T2
                force_y = term_y_E * T1 + term_y_E_vis * T2

                data[:, 0] = force_x * sd.cell_volumes
                data[:, 1] = force_y * sd.cell_volumes

            vals.append(data.ravel())
        return np.concatenate(vals)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        if hasattr(self, '_bf_subdomains'):
            new_vals = self._compute_body_force_values(self._bf_subdomains)
            offset = 0
            for sd in self._bf_subdomains:
                sd_data = self.mdg.subdomain_data(sd)
                n = sd.num_cells * self.nd
                pp.set_solution_values("body_force", new_vals[offset:offset+n], sd_data, iterate_index=0)
                pp.set_solution_values("body_force", new_vals[offset:offset+n], sd_data, time_step_index=0)
                offset += n


def run_single(cell_size: float, dt: float) -> ConvergenceMMSModel:
    """Run one simulation and return the model at t = T_FINAL."""
    print(f"\n{'='*60}")
    print(f"  cell_size = {cell_size}, dt = {dt} s")
    print(f"{'='*60}")

    solid = make_solid()
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
    model = ConvergenceMMSModel(params)
    pp.run_time_dependent_model(model)
    return model


def run_all_configurations() -> dict:
    """Run simulations for all (cell_size, dt) pairs.

    Returns
    -------
    dict
        Mapping (cell_size, dt) → model.
    """
    models = {}
    for cs, dt in zip(CELL_SIZES, DT_VALUES):
        model = run_single(cs, dt)
        models[(cs, dt)] = model
    return models


def compute_errors(models: dict) -> list[dict]:
    """Compute relative L2 errors against the exact MMS analytical solution.

    Parameters
    ----------
    models : dict
        Mapping (cell_size, dt) → model, as returned by run_all_configurations.

    Returns
    -------
    list[dict]
        List of result dicts with keys: cell_size, dt, Nx, error_abs, error_rel.
    """
    results = []
    for cs, dt in zip(CELL_SIZES, DT_VALUES):
        model = models[(cs, dt)]
        sd = model.mdg.subdomains(dim=model.nd)[0]
        u_vec = model._displacement_snapshots[T_FINAL]["u"]
        u_2d = u_vec.reshape(model.nd, -1, order="F")
        ux_num = u_2d[0, :]
        uy_num = u_2d[1, :]

        # Exact MMS solution at T_FINAL
        cc = sd.cell_centers
        A_MMS = 1.0e-3
        b_MMS = 0.5 * (model.solid.shear_modulus2 / model.solid.viscosity)
        L = 0.1
        ux_mms = A_MMS * np.sin(np.pi * cc[0] / L) * np.sin(np.pi * cc[1] / L) * (1.0 - np.exp(-b_MMS * T_FINAL))
        uy_mms = A_MMS * np.sin(np.pi * cc[0] / L) * np.sin(np.pi * cc[1] / L) * (1.0 - np.exp(-b_MMS * T_FINAL))

        error_x = ux_num - ux_mms
        error_y = uy_num - uy_mms

        diff_sq = error_x**2 + error_y**2
        ref_sq = ux_mms**2 + uy_mms**2
        vol = sd.cell_volumes

        error_abs = np.sqrt(np.sum(diff_sq * vol))
        ref_norm = np.sqrt(np.sum(ref_sq * vol))
        error_rel = error_abs / ref_norm if ref_norm > 0 else 0.0

        Nx = int(round(L / cs))
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

    return results


def print_convergence_table(results: list[dict]) -> None:
    """Print a formatted convergence table with convergence orders."""
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


def plot_convergence(results: list[dict], filepath: str) -> None:
    """Generate and save a log-log convergence plot.

    Parameters
    ----------
    results : list[dict]
        Convergence results (from compute_errors).
    filepath : str
        Output image path.
    """
    # Exclude finest (error = 0 by definition)
    plot_data = [r for r in results if r["error_rel"] > 0]
    if len(plot_data) < 2:
        print("Not enough data points for convergence plot.")
        return

    Nx_arr = np.array([r["Nx"] for r in plot_data])
    Nt_arr = 1.0 / np.array([r["dt"] for r in plot_data])
    resolution = (Nx_arr**2 * Nt_arr) ** 0.25
    errors = np.array([r["error_rel"] for r in plot_data])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(
        resolution,
        errors,
        "o-",
        color="black",
        markerfacecolor="white",
        markeredgewidth=1.4,
        markeredgecolor="black",
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
    
    # Enable minor tick labels on the x-axis, using plain numbers instead of 10^x
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    formatter = ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    
    # Keep the y-axis minor ticks clean (as they can be too dense)
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Saved {filepath}")


def plot_displacement_maps(models: dict) -> None:
    """Generate displacement magnitude maps for coarsest and finest meshes.

    Parameters
    ----------
    models : dict
        Mapping (cell_size, dt) → model.
    """
    finest_key = (CELL_SIZES[-1], DT_VALUES[-1])
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


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    """Run the full convergence study."""
    os.makedirs("_output", exist_ok=True)
    setup_publication_style()

    # 1. Run all configurations
    models = run_all_configurations()

    # 2. Compute errors against finest mesh
    results = compute_errors(models)

    # 3. Save results
    save_convergence_results("_output/convergence_results.json", results)
    print("\nSaved _output/convergence_results.json")

    # 4. Print convergence table
    print_convergence_table(results)

    # 5. Convergence plot
    plot_convergence(results, "_output/convergence.png")

    # 6. Displacement maps (coarsest and finest)
    plot_displacement_maps(models)

    print("\nDone.")


if __name__ == "__main__":
    main()
