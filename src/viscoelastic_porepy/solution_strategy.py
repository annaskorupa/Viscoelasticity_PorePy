"""Solution strategy additions for the viscous branch u₂."""

from typing import Optional

import porepy as pp


class SolutionStrategyU2:
    """Defines variable names and discretization setup for u₂.

    Sets the MPSA discretization keyword ``"mechanics2"`` to avoid
    collision with the elastic branch ``"mechanics"``.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self.displacement2_variable: str = "u2"
        self.interface_displacement2_variable: str = "u2_interface"
        self.stress2_keyword: str = "mechanics2"

    def update_discretization_parameters(self) -> None:
        """Register stiffness tensor and BCs for the u₂ MPSA discretization."""
        super().update_discretization_parameters()

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    sd,
                    data,
                    self.stress2_keyword,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor2(sd),
                    },
                )
