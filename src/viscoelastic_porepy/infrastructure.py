"""Infrastructure mixins: initial conditions, solution strategy, body force.

These mixins provide the supporting infrastructure needed by the
ViscoelasticMomentumBalance model but are not part of the core physics.
"""

import numpy as np
import porepy as pp
from typing import Optional


class InitialConditionsU2:
    """Zero initial conditions for u₂ and body force arrays."""

    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()
        for sd in self.mdg.subdomains(dim=self.nd):
            self.equation_system.set_variable_values(
                np.zeros(sd.num_cells * self.nd),
                [self.displacement2([sd])],
                iterate_index=0,
            )
            # Initialize body_force in data dict for TimeDependentDenseArray
            sd_data = self.mdg.subdomain_data(sd)
            bf_zeros = np.zeros(sd.num_cells * self.nd)
            pp.set_solution_values(
                "body_force", bf_zeros, sd_data, iterate_index=0
            )
            pp.set_solution_values(
                "body_force", bf_zeros, sd_data, time_step_index=0
            )
        for intf in self.mdg.interfaces(dim=self.nd - 1, codim=1):
            self.equation_system.set_variable_values(
                np.zeros(intf.num_cells * self.nd),
                [self.interface_displacement2([intf])],
                iterate_index=0,
            )


class SolutionStrategyU2:
    """MPSA discretization setup for u₂."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self.displacement2_variable = "u2"
        self.interface_displacement2_variable = "u2_interface"
        self.stress2_keyword = "mechanics2"

    def update_discretization_parameters(self) -> None:
        super().update_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    data,
                    self.stress2_keyword,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor2(sd),
                    },
                )


class BodyForceMixin:
    """Zero body force (no gravity, no MMS source term)."""

    def _compute_body_force_values(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        """Return zero body force for all cells."""
        vals = []
        for sd in subdomains:
            vals.append(np.zeros(sd.num_cells * self.nd))
        return np.concatenate(vals)

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force as a TimeDependentDenseArray (reads from data dict)."""
        self._bf_subdomains = subdomains
        return pp.ad.TimeDependentDenseArray("body_force", subdomains)
