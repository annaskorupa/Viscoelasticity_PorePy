"""Viscoelastic extension of PorePy's MomentumBalance model.

This script adds a second displacement variable (u2) and its associated
momentum balance equation to the existing PorePy MomentumBalance framework.
The goal is to eventually replace Hooke's law with a viscoelastic constitutive
law, where the total displacement is decomposed into an elastic part (u1,
handled by the base MomentumBalance) and a viscous part (u2, added here).

Architecture follows the PorePy mixin pattern, modeled after
TpsaMomentumBalanceMixin in porepy.models.momentum_balance.

Each section is organized by responsibility:
    1. Material constants (ViscoelasticSolidConstants)
    2. Geometry (GeometryMixin)
    3. Elastic moduli for the viscous part (ViscousElasticModuli)
    4. Stress constitutive law for u2 (MechanicalStressU2)
    5. Combined constitutive laws (ConstitutiveLawsU2)
    6. Variables for u2 (VariablesU2)
    7. Equations for u2 (EquationsU2)
    8. Boundary conditions (BoundaryConditionsMixin)
    9. Initial conditions for u2 (InitialConditionsU2)
    10. Solution strategy for u2 (SolutionStrategyU2)
    11. Body force (BodyForceMixin)
    12. Combined viscoelastic mixin (ViscoelasticMixin)
    13. Final model class + run script
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import porepy as pp

from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Sequence, cast

from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models import constitutive_laws

Scalar = pp.ad.Scalar


# =============================================================================
# 1. Material Constants
# =============================================================================
# FIX #1: SolidConstants is a frozen dataclass without shear_modulus2 / lame_lambda2.
# We must create a subclass that declares these extra fields with their SI units.
# This follows the same pattern as FractureDamageSolidConstants in PorePy.
@dataclass(kw_only=True, eq=False)
class ViscoelasticSolidConstants(pp.SolidConstants):
    """Extended solid constants with additional Lamé parameters for the viscous part.

    The viscous component of the viscoelastic model requires its own set of
    elastic moduli (lame_lambda2, shear_modulus2), which are independent of the
    elastic part (lame_lambda, shear_modulus) used in the base MomentumBalance.
    """

    # Deep copy the parent SI_units dict and add new entries.
    SI_units: ClassVar[dict[str, str]] = dict(**pp.SolidConstants.SI_units)
    SI_units.update(
        {
            "lame_lambda2": "Pa",
            "shear_modulus2": "Pa",
        }
    )

    lame_lambda2: float = 1.0
    """Lamé's first parameter for the viscous part [Pa]."""

    shear_modulus2: float = 1.0
    """Shear modulus for the viscous part [Pa]."""


# =============================================================================
# 2. Geometry
# =============================================================================
class GeometryMixin:
    """Defines a 2D square domain with simplex mesh.

    Override set_domain, grid_type, and meshing_arguments to customize the
    computational domain geometry.
    """

    units: pp.Units

    def set_domain(self) -> None:
        """Define a 2D square domain with side length 2."""
        size = self.units.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def grid_type(self) -> str:
        """Use simplex grid (required for non-axis-aligned fractures)."""
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Set the mesh cell size."""
        cell_size = self.units.convert_units(0.25, "m")
        return {"cell_size": cell_size}


# =============================================================================
# 3. Elastic Moduli for the Viscous Part
# =============================================================================
class ViscousElasticModuli:
    """Elastic moduli (Lamé parameters) for the viscous displacement component u2.

    Provides shear_modulus2, lame_lambda2, youngs_modulus2, bulk_modulus2, and
    stiffness_tensor2. All values are read from self.solid, which must be an
    instance of ViscoelasticSolidConstants.

    This is analogous to porepy.models.constitutive_laws.ElasticModuli but for
    the second set of Lamé parameters.
    """

    # Type hint: self.solid must have the extended fields.
    solid: ViscoelasticSolidConstants

    def shear_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus for the viscous part [Pa]."""
        return Scalar(self.solid.shear_modulus2, "shear_modulus2")

    def lame_lambda2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lamé's first parameter for the viscous part [Pa]."""
        return Scalar(self.solid.lame_lambda2, "lame_lambda2")

    def youngs_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus for the viscous part [Pa]."""
        # FIX #2: Fixed typo – was "lame_lambd2", corrected to "lame_lambda2".
        val = (
            self.solid.shear_modulus2
            * (3 * self.solid.lame_lambda2 + 2 * self.solid.shear_modulus2)
            / (self.solid.lame_lambda2 + self.solid.shear_modulus2)
        )
        return Scalar(val, "youngs_modulus2")

    def bulk_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Bulk modulus for the viscous part [Pa]."""
        val = self.solid.lame_lambda2 + 2 * self.solid.shear_modulus2 / 3
        return Scalar(val, "bulk_modulus2")

    def stiffness_tensor2(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Fourth-order stiffness tensor for the viscous part [Pa].

        Parameters:
            subdomain: Subdomain where the tensor is defined.

        Returns:
            Cell-wise FourthOrderTensor built from lame_lambda2 and shear_modulus2.
        """
        lmbda2 = self.solid.lame_lambda2 * np.ones(subdomain.num_cells)
        mu2 = self.solid.shear_modulus2 * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu2, lmbda2)


# =============================================================================
# 4. Mechanical Stress Constitutive Law for u2
# =============================================================================
# FIX #7: Removed inheritance from pp.PorePyModel. This class is a pure mixin;
# pp.PorePyModel is already provided by pp.MomentumBalance in the final MRO.
class MechanicalStressU2:
    """MPSA-discretized mechanical stress for the viscous displacement u2.

    Mirrors LinearElasticMechanicalStress from PorePy but operates on
    displacement2 / interface_displacement2 and uses stress2_keyword for
    discretization parameter lookup.
    """

    # --- Protocol attributes (provided by other mixins) ---
    stress2_keyword: str
    """Discretization keyword for u2 stress. Set in SolutionStrategyU2."""

    displacement2: Callable[
        [pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable
    ]
    """Displacement u2 variable. Provided by VariablesU2."""

    interface_displacement2: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Interface displacement u2. Provided by VariablesU2."""

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction. Provided by ContactTractionVariable."""

    characteristic_contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Characteristic traction. Provided by constitutive laws."""

    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryCondition]
    """BC type for mechanics. Provided by BoundaryConditionsMixin."""

    def mechanical_stress2(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Compute the MPSA-discretized mechanical stress for u2.

        Parameters:
            domains: List of subdomains or boundary grids (co-dimension 0).

        Returns:
            Ad operator for the mechanical stress on grid faces.
        """
        # Handle boundary grids.
        if len(domains) == 0 or all(
            isinstance(d, pp.BoundaryGrid) for d in domains
        ):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.stress2_keyword, domains=domains
            )

        # Validate: must be all Grid, not a mix.
        if not all(isinstance(g, pp.Grid) for g in domains):
            raise ValueError("Argument 'domains' is a mix of Grid and BoundaryGrid.")
        domains = cast(list[pp.Grid], domains)

        for sd in domains:
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of co-dimension 0.")

        # MPSA discretization for u2.
        discr = self._stress_discretization2(domains)

        interfaces = self.subdomains_to_interfaces(domains, [1])
        boundary_operator = self._combine_boundary_operators_stress2(domains)
        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)

        stress2 = (
            discr.stress() @ self.displacement2(domains)
            + discr.bound_stress() @ boundary_operator
            + discr.bound_stress()
            @ proj.mortar_to_primary_avg()
            @ self.interface_displacement2(interfaces)
        )
        stress2.set_name("mechanical_stress2")
        return stress2

    def _combine_boundary_operators_stress2(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Combine Dirichlet/Neumann/Robin boundary operators for u2 stress.

        Parameters:
            subdomains: Subdomains whose boundary operators are combined.

        Returns:
            Combined boundary operator.
        """
        # FIX: Use a distinct bc_values name to avoid collision with u1.
        return self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.displacement2,
            neumann_operator=self.mechanical_stress2,
            robin_operator=self.mechanical_stress2,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics2",  # FIX: unique name for u2 BC values
        )

    def fracture_stress2(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Fracture stress for u2 on interfaces [Pa].

        Parameters:
            interfaces: Fracture-matrix interfaces.

        Returns:
            Operator for the fracture stress.
        """
        for interface in interfaces:
            if interface.dim != self.nd - 1:
                raise ValueError("Interface must be of co-dimension 1.")

        subdomains = self.interfaces_to_subdomains(interfaces)
        fracture_subdomains = [sd for sd in subdomains if sd.dim == self.nd - 1]

        subdomain_projection = pp.ad.SubdomainProjections(subdomains, self.nd)
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )

        nondim_traction = (
            mortar_projection.sign_of_mortar_sides()
            @ mortar_projection.secondary_to_mortar_int()
            @ subdomain_projection.cell_prolongation(fracture_subdomains)
            @ self.local_coordinates(fracture_subdomains).transpose()
            @ self.contact_traction(fracture_subdomains)
        )
        traction = nondim_traction * self.characteristic_contact_traction(
            fracture_subdomains
        )
        traction.set_name("mechanical_fracture_stress2")
        return traction

    def _stress_discretization2(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpsaAd:
        """MPSA discretization for u2 stress.

        Parameters:
            subdomains: Subdomains where the stress is defined.

        Returns:
            MpsaAd discretization operator keyed by stress2_keyword.
        """
        return pp.ad.MpsaAd(self.stress2_keyword, subdomains)


# =============================================================================
# 5. Combined Constitutive Laws for u2
# =============================================================================
class ConstitutiveLawsU2(
    constitutive_laws.ZeroGravityForce,
    ViscousElasticModuli,
    MechanicalStressU2,
    constitutive_laws.ConstantSolidDensity,
):
    """Bundle of constitutive laws needed for the u2 momentum balance.

    Provides:
        - stress2(): the total stress for u2 (delegates to mechanical_stress2)
        - fracture_stress2(): fracture stress for u2
        - Zero gravity force
        - Constant solid density
        - Elastic moduli for the viscous part
    """

    def stress2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Total stress operator for u2.

        Parameters:
            domains: Subdomains where the stress is defined.

        Returns:
            Stress operator for the u2 momentum balance.
        """
        return self.mechanical_stress2(domains)


# =============================================================================
# 6. Variables for u2
# =============================================================================
# FIX MRO: Do NOT inherit from pp.VariableMixin here. The VariableMixin is
# already provided by MomentumBalance in the final MRO. Inheriting it here
# causes C3 linearization to push VariablesU2 AFTER VariablesMomentumBalance,
# which means create_variables() would never be reached. This follows the
# same pattern as VariablesThreeFieldMomentumBalance in PorePy.
class VariablesU2:
    """Variables for the viscous displacement component.

    Creates:
        - displacement u2 on matrix subdomains
        - interface displacement u2 on fracture-matrix interfaces

    Variable names are set in SolutionStrategyU2.__init__.
    """

    displacement2_variable: str
    """Name of u2 variable on subdomains. Set in SolutionStrategyU2."""

    interface_displacement2_variable: str
    """Name of u2 variable on interfaces. Set in SolutionStrategyU2."""

    def create_variables(self) -> None:
        """Register u2 and interface_u2 variables in the equation system.

        Calls super().create_variables() first so that the base MomentumBalance
        variables (u, interface_u, contact_traction) are created before u2.
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
        """Access the u2 displacement variable or boundary operator.

        Parameters:
            domains: Subdomains or boundary grids.

        Returns:
            MixedDimensionalVariable for u2.
        """
        if len(domains) == 0 or all(
            isinstance(grid, pp.BoundaryGrid) for grid in domains
        ):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.displacement2_variable, domains=domains
            )
        if not all(isinstance(grid, pp.Grid) for grid in domains):
            raise ValueError("Mixed subdomain and boundary grids.")
        domains = cast(list[pp.Grid], domains)

        if not all(grid.dim == self.nd for grid in domains):
            raise ValueError("Displacement u2 only defined on dimension nd.")

        return self.equation_system.md_variable(
            self.displacement2_variable, domains
        )

    def interface_displacement2(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Variable:
        """Access the interface u2 displacement variable.

        Parameters:
            interfaces: Fracture-matrix interfaces (dim = nd-1).

        Returns:
            MixedDimensionalVariable for interface u2.
        """
        if not all(intf.dim == self.nd - 1 for intf in interfaces):
            raise ValueError("Interface u2 only defined on dim nd-1.")

        return self.equation_system.md_variable(
            self.interface_displacement2_variable, interfaces
        )


# =============================================================================
# 7. Equations for u2
# =============================================================================
class EquationsU2:
    """Momentum balance equations for the viscous displacement u2.

    Adds:
        - momentum_balance_equation2 on matrix subdomains
        - interface_force_balance_equation2 on fracture-matrix interfaces

    Follows the same pattern as MomentumBalanceEquations in PorePy;
    set_equations() calls super() first (to set u1 equations), then appends u2.
    """

    # --- Protocol attributes (provided by other mixins) ---
    stress2: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Total stress for u2. Provided by ConstitutiveLawsU2."""

    fracture_stress2: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Fracture stress for u2. Provided by MechanicalStressU2."""

    gravity_force: Callable[
        [list[pp.Grid] | list[pp.MortarGrid], str], pp.ad.Operator
    ]
    """Gravity force. Provided by ZeroGravityForce or GravityForce."""

    def set_equations(self) -> None:
        """Set momentum balance equations for u2 (after u1 via super).

        Registers:
            - momentum_balance_equation2 on matrix subdomains
            - interface_force_balance_equation2 on fracture-matrix interfaces
        """
        super().set_equations()

        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)

        matrix_eq = self._momentum_balance_equation2(matrix_subdomains)
        self.equation_system.set_equation(
            matrix_eq, matrix_subdomains, {"cells": self.nd}
        )

        intf_eq = self._interface_force_balance_equation2(interfaces)
        self.equation_system.set_equation(
            intf_eq, interfaces, {"cells": self.nd}
        )

    def _momentum_balance_equation2(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Momentum balance equation for u2 in the matrix.

        Parameters:
            subdomains: Matrix subdomains.

        Returns:
            Operator for the u2 momentum balance.
        """
        accumulation = self._inertia_u2(subdomains)
        # Negative sign by convention (positive tensile stress).
        stress2 = pp.ad.Scalar(-1) * self.stress2(subdomains)
        body_force = self._body_force_u2(subdomains)

        equation = self.balance_equation(
            subdomains, accumulation, stress2, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation2")
        return equation

    def _inertia_u2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Inertial term for u2 (zero by default).

        Override in subclass to add inertia effects.
        """
        return pp.ad.Scalar(0)

    def _body_force_u2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force for u2 equation.

        Parameters:
            subdomains: Matrix subdomains.

        Returns:
            Volume-integrated body force operator.
        """
        return self.volume_integral(
            self.gravity_force(subdomains, "solid"), subdomains, dim=self.nd
        )

    def _interface_force_balance_equation2(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Force balance equation for u2 at fracture-matrix interfaces.

        Parameters:
            interfaces: Fracture-matrix interfaces.

        Returns:
            Operator for the interface force balance.
        """
        for interface in interfaces:
            if interface.dim != self.nd - 1:
                raise ValueError("Interface must be a fracture-matrix interface.")

        subdomains = self.interfaces_to_subdomains(interfaces)
        matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        proj = pp.ad.SubdomainProjections(subdomains, self.nd)

        # Stress from the matrix side projected onto the mortar grid.
        contact_from_primary = (
            mortar_projection.primary_to_mortar_int()
            @ proj.face_prolongation(matrix_subdomains)
            @ self.internal_boundary_normal_to_outwards(
                matrix_subdomains, dim=self.nd
            )
            @ self.stress2(matrix_subdomains)
        )

        # Traction from the fracture (contact) side.
        traction_from_secondary = self.fracture_stress2(interfaces)

        force_balance = (
            contact_from_primary
            + self.volume_integral(
                traction_from_secondary, interfaces, dim=self.nd
            )
        )
        force_balance.set_name("interface_force_balance_equation2")
        return force_balance


# =============================================================================
# 8. Boundary Conditions
# =============================================================================
class BoundaryConditionsMixin:
    """Boundary conditions for the mechanics problem.

    - Dirichlet on west and east boundaries.
    - Neumann (default) on north and south boundaries with stress values.
    """

    units: pp.Units

    def bc_type_mechanics(
        self, sd: pp.Grid
    ) -> pp.BoundaryConditionVectorial:
        """Assign Dirichlet BC on west/east, Neumann on north/south."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, domain_sides.west + domain_sides.east, "dir"
        )
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Stress BC values: 5 Pa on north/south for both x and y components."""
        values = np.ones((self.nd, bg.num_cells))
        domain_sides = self.domain_boundary_sides(bg)

        stress_val = self.units.convert_units(5, "Pa")
        values[0][domain_sides.north + domain_sides.south] *= stress_val
        values[1][domain_sides.north + domain_sides.south] *= stress_val

        return values.ravel("F")


# =============================================================================
# 9. Initial Conditions for u2
# =============================================================================
# FIX MRO: Do NOT inherit from pp.InitialConditionMixin here (same reason as
# VariablesU2 above). InitialConditionMixin is already in MRO via MomentumBalance.
class InitialConditionsU2:
    """Initial values for the viscous displacement u2 and interface u2.

    Sets zero initial conditions for both variables. Override ic_values_*
    methods to customize.
    """

    displacement2: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    interface_displacement2: Callable[[list[pp.MortarGrid]], pp.ad.Operator]

    def set_initial_values_primary_variables(self) -> None:
        """Set zero initial values for u2 and interface u2.

        Calls super() first so that the base MomentumBalance initial conditions
        (u, contact_traction, interface_u) are set beforehand.
        """
        super().set_initial_values_primary_variables()

        # Set u2 initial values on matrix subdomains.
        for sd in self.mdg.subdomains():
            if sd.dim == self.nd:
                self.equation_system.set_variable_values(
                    self.ic_values_displacement2(sd),
                    [cast(pp.ad.Variable, self.displacement2([sd]))],
                    iterate_index=0,
                )

        # Set interface u2 initial values.
        for intf in self.mdg.interfaces(dim=self.nd - 1, codim=1):
            self.equation_system.set_variable_values(
                self.ic_values_interface_displacement2(intf),
                [cast(pp.ad.Variable, self.interface_displacement2([intf]))],
                iterate_index=0,
            )

    def ic_values_displacement2(self, sd: pp.Grid) -> np.ndarray:
        """Initial values for u2 (zero). Override to customize."""
        return np.zeros(sd.num_cells * self.nd)

    def ic_values_interface_displacement2(
        self, intf: pp.MortarGrid
    ) -> np.ndarray:
        """Initial values for interface u2 (zero). Override to customize."""
        return np.zeros(intf.num_cells * self.nd)


# =============================================================================
# 10. Solution Strategy for u2
# =============================================================================
# FIX #5: Renamed from SolutionStrategyMomentumBalance to avoid collision
# with PorePy's own class of the same name.
# FIX MRO: Do NOT inherit from pp.SolutionStrategy here. It's already provided
# by MomentumBalance. Inheriting here breaks the MRO for __init__ chain.
class SolutionStrategyU2:
    """Solution strategy additions for the u2 (viscous) component.

    Defines:
        - Variable names for u2
        - Discretization keyword for u2 stress
        - Update of discretization parameters (stiffness tensor, BCs)
    """

    # FIX #8: Declare the correct stiffness tensor type hint for u2.
    stiffness_tensor2: Callable[[pp.Grid], pp.FourthOrderTensor]
    """Returns the stiffness tensor for u2. Provided by ViscousElasticModuli."""

    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryConditionVectorial]
    """BC type for mechanics. Provided by BoundaryConditionsMixin."""

    def __init__(self, params: Optional[dict] = None) -> None:
        # Chain to the next __init__ in MRO (SolutionStrategyMomentumBalance).
        super().__init__(params)  # type: ignore[call-arg]

        # Variable names for the viscous displacement.
        self.displacement2_variable: str = "u2"
        """Name of the u2 displacement variable on subdomains."""

        self.interface_displacement2_variable: str = "u2_interface"
        """Name of the u2 displacement variable on interfaces."""

        # FIX #6: Use a UNIQUE keyword "mechanics2" (not "mechanics") to avoid
        # collision with the u1 stress discretization keyword.
        self.stress2_keyword: str = "mechanics2"
        """Discretization keyword for u2 stress. Must differ from u1's 'mechanics'."""

    def update_discretization_parameters(self) -> None:
        """Update stiffness tensor and BC type for the u2 MPSA discretization.

        This registers the viscous stiffness tensor (stiffness_tensor2) and BC
        type under the stress2_keyword ("mechanics2") so that the MPSA
        discretization for u2 picks up the correct parameters.
        """
        super().update_discretization_parameters()

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                # FIX #3: Use the new initialize_data signature: (data, keyword, params).
                # The old signature (sd, data, keyword, params) is deprecated.
                # FIX #4: Use stiffness_tensor2 (not stiffness_tensor) for u2.
                pp.initialize_data(
                    data,
                    self.stress2_keyword,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor2(sd),
                    },
                )

    def _is_nonlinear_problem(self) -> bool:
        """Check if the problem is nonlinear (fractures present)."""
        return self.mdg.dim_min() < self.nd


# =============================================================================
# 11. Body Force
# =============================================================================
class BodyForceMixin:
    """Custom body force applied to a central region of the 2D domain.

    Applies gravitational body force (density * -9.8 m/s^2 in y-direction) to
    cells whose centers are within [0.3, 0.7] x [0.3, 0.7].
    """

    nd: int
    units: pp.Units
    solid: ViscoelasticSolidConstants

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute volume-integrated body force on selected cells."""
        vals = []
        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))

            if sd.dim == 2:
                cc = sd.cell_centers
                # Select cells in the central [0.3, 0.7] x [0.3, 0.7] region.
                mask = (
                    (cc[0] > (0.3 / self.units.m))
                    & (cc[0] < (0.7 / self.units.m))
                    & (cc[1] > (0.3 / self.units.m))
                    & (cc[1] < (0.7 / self.units.m))
                )

                acceleration = self.units.convert_units(-9.8, "m * s^-2")
                force = self.solid.density * acceleration
                data[mask, 1] = force * sd.cell_volumes[mask]

            vals.append(data)

        return pp.ad.DenseArray(np.concatenate(vals).ravel(), "body_force")


# NOTE: ViscoelasticMixin wrapper removed. All u2 mixins are listed directly
# in the final class to ensure correct C3 linearization. When bundled in an
# intermediate class, Python may push base-less mixins to the END of the MRO,
# making their set_equations / create_variables / __init__ unreachable.


# =============================================================================
# 13. Final Model Class
# =============================================================================
class ViscoelasticMomentumBalance(
    GeometryMixin,
    BoundaryConditionsMixin,
    BodyForceMixin,
    # --- u2 mixins: must come BEFORE pp.MomentumBalance ---
    # Equations must be first so set_equations() is reached before
    # MomentumBalanceEquations terminates the super() chain.
    EquationsU2,
    # Variables must be before VariablesMomentumBalance.
    VariablesU2,
    # Constitutive laws for u2 stress.
    ConstitutiveLawsU2,
    # Initial conditions for u2 – before InitialConditionsMomentumBalance.
    InitialConditionsU2,
    # Solution strategy for u2 – before SolutionStrategyMomentumBalance.
    SolutionStrategyU2,
    # --- Base PorePy model (provides u1 + all infrastructure) ---
    pp.MomentumBalance,
):
    """Momentum balance model with viscoelastic extension (u1 + u2).

    MRO order rationale:
        1. GeometryMixin – overrides domain/mesh settings
        2. BoundaryConditionsMixin – overrides BC type and values
        3. BodyForceMixin – overrides body force calculation
        4. ViscoelasticMixin – adds u2 variables, equations, constitutive laws,
           initial conditions, and solution strategy
        5. pp.MomentumBalance – base model providing u1 and all PorePy
           infrastructure
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # FIX #6 (reinforced): Ensure stress2_keyword is set to "mechanics2"
        # even after all __init__ calls in the MRO.
        self.stress2_keyword: str = "mechanics2"

    def update_all_boundary_conditions(self) -> None:
        """Register boundary conditions for both u1 and u2 stress keywords."""
        super().update_all_boundary_conditions()
        # Register BC values under the u2 stress keyword.
        self.update_boundary_condition(
            self.stress2_keyword, self.bc_values_stress
        )

    def update_boundary_values_primary_variables(self) -> None:
        """Register u2 displacement boundary values in addition to u1.

        FIX: Without this, BoundaryGrid data lacks 'u2' in iterate_solutions,
        causing KeyError during equation assembly when the AD parser evaluates
        displacement2(boundary_grids) via create_boundary_operator.
        """
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(
            self.displacement2_variable, self.bc_values_displacement2
        )

    def bc_values_displacement2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Displacement u2 values for Dirichlet BC (default: zero).

        Override to prescribe non-zero u2 Dirichlet boundary conditions.
        """
        return np.zeros((self.nd, bg.num_cells)).ravel("F")


# =============================================================================
# Run the model
# =============================================================================
if __name__ == "__main__":
    # FIX #1: Use ViscoelasticSolidConstants instead of plain SolidConstants.
    fluid_constants = pp.FluidComponent(viscosity=0.1, density=0.2)
    solid_constants = ViscoelasticSolidConstants(
        permeability=0.5,
        porosity=0.25,
        # Parameters for the viscous part – adjust to your material.
        lame_lambda2=0.5,
        shear_modulus2=0.3,
    )
    material_constants = {"fluid": fluid_constants, "solid": solid_constants}
    model_params = {"material_constants": material_constants}

    model = ViscoelasticMomentumBalance(model_params)
    pp.run_time_dependent_model(model)

    # Visualization
    pp.plot_grid(
        model.mdg,
        cell_value=model.displacement2_variable,
        rgb=[1, 1, 1],
        figsize=(10, 8),
        linewidth=0.3,
        title="Viscous displacement u2",
        plot_2d=True,
    )

    import matplotlib.pyplot as plt

    plt.show()
    plt.savefig("displacement_u2.png", dpi=300)

    print("Done – viscoelastic MomentumBalance completed successfully.")
