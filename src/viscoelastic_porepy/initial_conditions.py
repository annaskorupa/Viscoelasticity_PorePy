"""Initial conditions for the viscous displacement u₂."""

import numpy as np
import porepy as pp


class InitialConditionsU2:
    """Zero initial conditions for u₂ and interface u₂.

    Also initializes the ``body_force`` data array required by
    :class:`~porepy.ad.TimeDependentDenseArray`.
    """

    def set_initial_values_primary_variables(self) -> None:
        """Set zero initial values for u₂, interface u₂, and body force."""
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains(dim=self.nd):
            # Zero initial displacement u₂
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
