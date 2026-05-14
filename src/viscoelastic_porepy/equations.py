"""Rate equation for the viscoelastic Maxwell model.

Implements the ODE coupling u and u₂:
    du₂/dt + β·u₂ − du/dt = 0
"""

import porepy as pp


class RateEquation:
    """Discretized rate equation coupling the elastic and viscous branches.

    Uses implicit Euler time-stepping via ``pp.ad.dt``.
    """

    def set_equations(self) -> None:
        """Add the rate equation to the equation system (after base equations)."""
        super().set_equations()

        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        u = self.displacement(matrix_subdomains)
        u2 = self.displacement2(matrix_subdomains)
        beta = self.beta(matrix_subdomains)

        # du₂/dt + β·u₂ − du/dt = 0
        eq = (
            pp.ad.dt(u2, self.time_manager.dt)
            + beta * u2
            - pp.ad.dt(u, self.time_manager.dt)
        )
        eq.set_name("rate_equation")
        self.equation_system.set_equation(
            eq, matrix_subdomains, {"cells": self.nd}
        )
