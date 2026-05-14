#!/usr/bin/env python
"""Run the viscoelastic MMS verification simulation.

Usage:
    python scripts/run_simulation.py

Configuration is set via constants at the top of this file.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import porepy as pp

# Ensure the package is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from viscoelastic_porepy.material import ViscoelasticSolidConstants
from viscoelastic_porepy.model import ViscoelasticMomentumBalance

# =============================================================================
# Simulation parameters
# =============================================================================
DT = 1.0 * pp.SECOND       # Time step [s]  (alternatives: 2, 4, 8)
FINAL_TIME = 100.0 * pp.SECOND
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "_output" / "plots"

# Material constants (Generalized Maxwell model)
SOLID = ViscoelasticSolidConstants(
    # E₁ = 22575.7 MPa  →  μ₁ = E₁/2 (ν=0)
    shear_modulus=22575700000.0 / 2.0,
    lame_lambda=0.0,
    # E₂ = 11000.0 MPa  →  μ₂ = E₂/2 (ν=0)
    shear_modulus2=11000000000.0 / 2.0,
    lame_lambda2=0.0,
    # η = E₁ × τ_relax,  τ_relax = 45.454545 days
    viscosity=22575700000.0 * (45.454545 * 24.0 * 60.0 * 60.0),
)


# =============================================================================
# ShowCase subclass with convergence diagnostics
# =============================================================================
class ShowCase(ViscoelasticMomentumBalance):
    """Model subclass that prints convergence diagnostics."""

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        if self.time_manager.time_index == 0:
            tau = self.solid.viscosity / self.solid.shear_modulus2 / 60.0
            print(f"--- Theoretical relaxation time: {tau:.2f} min ---")

        if len(self.mdg.subdomains(dim=self.nd)) == 0:
            return

        sd = self.mdg.subdomains(dim=self.nd)[0]
        cc = sd.cell_centers
        t_now = self.time_manager.time

        # Numerical solution
        u_vec = np.array(
            self.equation_system.evaluate(
                self.displacement(self.mdg.subdomains())
            )
        ).ravel()
        u_reshaped = u_vec.reshape(self.nd, -1, order="F")
        ux_num = u_reshaped[0, :]
        uy_num = u_reshaped[1, :]

        # Analytical MMS solution
        A_MMS = 1.0e-3
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        spatial = np.sin(np.pi * cc[0] / 0.8) * np.sin(np.pi * cc[1] / 0.8)
        temporal = 1.0 - np.exp(-b_MMS * t_now)
        ux_mms = A_MMS * spatial * temporal
        uy_mms = A_MMS * spatial * temporal

        # L2 error norms
        error_x = ux_num - ux_mms
        error_y = uy_num - uy_mms
        abs_L2 = np.sqrt(
            np.sum((error_x**2 + error_y**2) * sd.cell_volumes)
        )
        norm_mms = np.sqrt(
            np.sum((ux_mms**2 + uy_mms**2) * sd.cell_volumes)
        )
        rel_L2 = abs_L2 / norm_mms if norm_mms > 0 else 0.0

        # Periodic logging
        if self.time_manager.time_index % 100 == 0:
            days = self.time_manager.time / pp.DAY
            print(f"\n{'=' * 60}")
            print(f"  t = {days:.2f} days")
            print(f"  Absolute L2 error = {abs_L2:.6e} m")
            print(f"  Relative L2 error = {rel_L2:.6e}")
            print(f"{'=' * 60}\n")

        # Plot at scheduled times
        sched = self.params.get("plot_schedule", [])
        if sched and self.time_manager.time >= sched[0]:
            sched.pop(0)
            mins = int(self.time_manager.time / 60.0)

            if not hasattr(self, "_vmax_u"):
                u_all = self.equation_system.evaluate(
                    self.displacement(self.mdg.subdomains())
                )
                u2_all = self.equation_system.evaluate(
                    self.displacement2(self.mdg.subdomains())
                )
                u_mag = np.linalg.norm(
                    u_all.reshape(self.nd, -1, order="F"), axis=0
                )
                u2_mag = np.linalg.norm(
                    u2_all.reshape(self.nd, -1, order="F"), axis=0
                )
                self._vmax_u = np.max(u_mag) * 2.5
                self._vmax_u2 = np.max(u2_mag)

            for var_name, label, vmax in [
                (self.displacement_variable, "u", self._vmax_u),
                (self.displacement2_variable, "u2", self._vmax_u2),
            ]:
                for sd_i, sd_data in self.mdg.subdomains(return_data=True):
                    vals = pp.get_solution_values(
                        name=var_name, data=sd_data, time_step_index=0
                    )
                    mag = np.linalg.norm(
                        vals.reshape(self.nd, -1, order="F"), axis=0
                    )
                    plt.close("all")
                    pp.plot_grid(
                        sd_i,
                        cell_value=mag,
                        title=f"{label} at {mins} min",
                        if_plot=False,
                        color_map_limits=[0.0, vmax],
                        plot_2d=True,
                    )
                    out_path = OUTPUT_DIR / f"displacement_{label}_{mins}.png"
                    plt.savefig(str(out_path), dpi=200)


# =============================================================================
# Main entry point
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_manager = pp.TimeManager(
        schedule=[0.0, FINAL_TIME],
        dt_init=DT,
        dt_min_max=(0.0 * pp.MINUTE, FINAL_TIME),
    )

    model_params = {
        "material_constants": {
            "solid": SOLID,
            "fluid": pp.FluidComponent(),
        },
        "time_manager": time_manager,
        "plot_schedule": [
            pp.MINUTE * float(i) for i in range(0, 301, 50)
        ],
    }

    model = ShowCase(model_params)
    pp.run_time_dependent_model(model)
    print("Done.")


if __name__ == "__main__":
    main()
