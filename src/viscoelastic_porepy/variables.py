"""Variables and rate equation for the viscous branch u2.

Implements the Grünwald-Letnikov discretization of the fractional
derivative: D^α u₂ + β·u₂ − D^α u = 0.
When α = 1, this reduces to the classical backward Euler scheme.
"""

import numpy as np
import porepy as pp
from typing import cast

Scalar = pp.ad.Scalar


class VariablesU2:
    """Variables u2 (viscous displacement) and interface_u2."""

    displacement2_variable: str
    interface_displacement2_variable: str

    def create_variables(self) -> None:
        super().create_variables()
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.displacement2_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "m"},
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.interface_displacement2_variable,
            interfaces=self.mdg.interfaces(dim=self.nd - 1, codim=1),
            tags={"si_units": "m"},
        )

    def displacement2(self, domains) -> pp.ad.Operator:
        if len(domains) == 0 or all(
            isinstance(g, pp.BoundaryGrid) for g in domains
        ):
            return self.create_boundary_operator(
                name=self.displacement2_variable, domains=domains
            )
        return self.equation_system.md_variable(
            self.displacement2_variable, cast(list[pp.Grid], domains)
        )

    def interface_displacement2(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Variable:
        return self.equation_system.md_variable(
            self.interface_displacement2_variable, interfaces
        )


class RateEquation:
    """Grünwald-Letnikov discretization of D^α u₂ + β·u₂ − D^α u = 0.

    Maintains displacement histories (history_u, history_u2) and computes
    GL weight sums at each time step. Supports fractional order α ∈ (0, 1].
    """

    def compute_gl_weights(self, alpha: float, n: int) -> np.ndarray:
        """Compute Grünwald-Letnikov weights w_0 … w_n for order α."""
        w = np.zeros(n + 1)
        w[0] = 1.0
        for k in range(1, n + 1):
            w[k] = w[k - 1] * (1.0 - (alpha + 1.0) / k)
        return w

    def _update_gl_sums(self) -> None:
        """Compute GL history sums and store in data dictionary."""
        n = self.time_manager.time_index
        w_alpha = self.compute_gl_weights(self.solid.alpha, n)

        if len(self.history_u) == 0:
            # Initial conditions are zero
            for sd in self.mdg.subdomains(dim=self.nd):
                self.history_u.append(np.zeros(sd.num_cells * self.nd))
                self.history_u2.append(np.zeros(sd.num_cells * self.nd))

        if n == 0:
            for sd in self.mdg.subdomains(dim=self.nd):
                sd_data = self.mdg.subdomain_data(sd)
                gl_zero = np.zeros(sd.num_cells * self.nd)
                pp.set_solution_values(
                    "gl_u", gl_zero, sd_data, time_step_index=0
                )
                pp.set_solution_values(
                    "gl_u2", gl_zero, sd_data, time_step_index=0
                )
                pp.set_solution_values(
                    "gl_u", gl_zero, sd_data, iterate_index=0
                )
                pp.set_solution_values(
                    "gl_u2", gl_zero, sd_data, iterate_index=0
                )
            return

        gl_u = np.zeros_like(self.history_u[0])
        gl_u2 = np.zeros_like(self.history_u2[0])

        for k in range(1, n + 1):
            gl_u += w_alpha[k] * self.history_u[n - k]
            gl_u2 += w_alpha[k] * self.history_u2[n - k]

        offset = 0
        for sd in self.mdg.subdomains(dim=self.nd):
            sd_data = self.mdg.subdomain_data(sd)
            num_dofs = sd.num_cells * self.nd
            chunk_u = gl_u[offset : offset + num_dofs]
            chunk_u2 = gl_u2[offset : offset + num_dofs]
            pp.set_solution_values(
                "gl_u", chunk_u, sd_data, time_step_index=0
            )
            pp.set_solution_values(
                "gl_u2", chunk_u2, sd_data, time_step_index=0
            )
            pp.set_solution_values(
                "gl_u", chunk_u, sd_data, iterate_index=0
            )
            pp.set_solution_values(
                "gl_u2", chunk_u2, sd_data, iterate_index=0
            )
            offset += num_dofs

    def set_equations(self) -> None:
        super().set_equations()
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        u = self.displacement(matrix_subdomains)
        u2 = self.displacement2(matrix_subdomains)
        beta = self.beta(matrix_subdomains)

        # D^α u₂ + β·u₂ − D^α u = 0
        gl_u = pp.ad.TimeDependentDenseArray("gl_u", matrix_subdomains)
        gl_u2 = pp.ad.TimeDependentDenseArray("gl_u2", matrix_subdomains)
        dt_alpha = self.ad_time_step ** self.solid.alpha

        D_alpha_u2 = (u2 + gl_u2) / dt_alpha
        D_alpha_u = (u + gl_u) / dt_alpha

        eq = D_alpha_u2 + beta * u2 - D_alpha_u
        eq.set_name("rate_equation")
        self.equation_system.set_equation(
            eq, matrix_subdomains, {"cells": self.nd}
        )

        # Interface equation: u₂ jump across fracture = 0
        interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)
        if len(interfaces) > 0:
            u2_intf = self.interface_displacement2(interfaces)
            eq_intf = Scalar(1) * u2_intf
            eq_intf.set_name("u2_interface_zero")
            self.equation_system.set_equation(
                eq_intf, interfaces, {"cells": self.nd}
            )
