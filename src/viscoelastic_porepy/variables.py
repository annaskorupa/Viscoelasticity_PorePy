"""Variables for the viscous displacement component u₂."""

from typing import cast

import porepy as pp


class VariablesU2:
    """Creates and provides access to u₂ and interface u₂ variables.

    Variable names are set in
    :class:`~viscoelastic_porepy.solution_strategy.SolutionStrategyU2`.
    """

    displacement2_variable: str
    interface_displacement2_variable: str

    def create_variables(self) -> None:
        """Register u₂ and interface_u₂ in the equation system.

        Calls ``super().create_variables()`` first so that base
        MomentumBalance variables (u, interface_u) are created before u₂.
        """
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

    def displacement2(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Access the u₂ displacement variable or boundary operator."""
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
        """Access the interface u₂ displacement variable."""
        return self.equation_system.md_variable(
            self.interface_displacement2_variable, interfaces
        )
