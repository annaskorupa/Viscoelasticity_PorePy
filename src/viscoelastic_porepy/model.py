"""Final assembled model combining all viscoelastic mixins."""

import numpy as np
import porepy as pp

from viscoelastic_porepy.geometry import GeometryMixin
from viscoelastic_porepy.boundary_conditions import BoundaryConditionsMixin
from viscoelastic_porepy.body_force import BodyForceMixin
from viscoelastic_porepy.equations import RateEquation
from viscoelastic_porepy.variables import VariablesU2
from viscoelastic_porepy.constitutive import ConstitutiveLawsU2
from viscoelastic_porepy.initial_conditions import InitialConditionsU2
from viscoelastic_porepy.solution_strategy import SolutionStrategyU2


class ViscoelasticMomentumBalance(
    GeometryMixin,
    BoundaryConditionsMixin,
    BodyForceMixin,
    RateEquation,
    VariablesU2,
    ConstitutiveLawsU2,
    InitialConditionsU2,
    SolutionStrategyU2,
    pp.MomentumBalance,
):
    """Momentum balance model with viscoelastic extension (u + u₂).

    MRO order ensures u₂ mixins are applied before the base
    ``pp.MomentumBalance``, so that ``set_equations()``,
    ``create_variables()``, and ``__init__()`` chains work correctly.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.stress2_keyword = "mechanics2"

    def before_nonlinear_loop(self) -> None:
        """Update body force values in the data dictionary each time step."""
        super().before_nonlinear_loop()
        if not hasattr(self, "_bf_subdomains"):
            return

        new_vals = self._compute_body_force_values(self._bf_subdomains)
        offset = 0
        for sd in self._bf_subdomains:
            sd_data = self.mdg.subdomain_data(sd)
            n = sd.num_cells * self.nd
            pp.set_solution_values(
                "body_force", new_vals[offset : offset + n],
                sd_data, iterate_index=0,
            )
            pp.set_solution_values(
                "body_force", new_vals[offset : offset + n],
                sd_data, time_step_index=0,
            )
            offset += n

    def stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Total stress: sum of elastic and Maxwell stress branches."""
        if all(
            isinstance(d, pp.Grid) and d.dim == self.nd for d in domains
        ):
            return (
                self.mechanical_stress(domains)
                + self.mechanical_stress2(domains)
            )
        return super().stress(domains)

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()
        self.update_boundary_condition(
            self.stress2_keyword, self.bc_values_stress
        )

    def update_boundary_values_primary_variables(self) -> None:
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(
            self.displacement2_variable, self.bc_values_displacement2
        )

    def bc_values_displacement2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero Dirichlet BC for u₂ on all boundaries."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")
