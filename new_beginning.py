import porepy as pp
import numpy as np

from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.momentum_balance import MomentumBalance
from porepy.models.fluid_mass_balance import SinglePhaseFlow

from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)

Scalar = pp.ad.Scalar

import logging
from typing import Callable, Optional, Sequence, cast

from porepy.models import contact_mechanics
from porepy.models.abstract_equations import VariableMixin, EquationMixin, BalanceEquation

from porepy.models import constitutive_laws

class ModifiedGeometry:

    units: pp.Units
    solid: pp.SolidConstants


    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.units.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    # def set_fractures(self) -> None:
    #     """Setting a diagonal fracture"""
    #     frac_1_points = self.units.convert_units(
    #         np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
    #     )
    #     frac_1 = pp.LineFracture(frac_1_points)
    #     self._fractures = [frac_1]

    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a diagonal fracture we cannot use a cartesian grid.
        Cartesian grid is the default grid type, and we therefore override this method to assign simplex instead.

        """
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        cell_size = self.units.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args
    

class ModyfiedEquation:
        """Class for momentum balance equations for u2."""

        stress2: Callable[[list[pp.Grid]], pp.ad.Operator]
        """Stress on the grid faces. Provided by a suitable mixin class that specifies the
        physical laws governing the stress.

        """
        fracture_stress: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
        """Stress on the fracture faces. Provided by a suitable mixin class that specifies
        the physical laws governing the stress, see for instance
        :class:`~porepy.models.constitutive_laws.LinearElasticMechanicalStress` or
        :class:`~porepy.models.constitutive_laws.PressureStress`.

        """
        gravity_force: Callable[[list[pp.Grid] | list[pp.MortarGrid], str], pp.ad.Operator]
        """Gravity force. Normally provided by a mixin instance of
        :class:`~porepy.models.constitutive_laws.GravityForce` or
        :class:`~porepy.models.constitutive_laws.ZeroGravityForce`.

        """

        def set_equations(self) -> None:
            """Set equations for the subdomains and interfaces for u2.

            The following equations are set:
                - Momentum balance in the matrix.
                - Force balance between fracture interfaces.
                - Deformation constraints for fractures, split into normal and tangential
                part.

            See individual equation methods for details.

            """
            super().set_equations()
            matrix_subdomains = self.mdg.subdomains(dim=self.nd)
            interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)
            matrix_eq = self.momentum_balance_equation2(matrix_subdomains)
            intf_eq = self.interface_force_balance_equation2(interfaces)
            self.equation_system.set_equation(
                matrix_eq, matrix_subdomains, {"cells": self.nd}
            )
            self.equation_system.set_equation(intf_eq, interfaces, {"cells": self.nd})

        def momentum_balance_equation2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Momentum balance equation in the matrix.

            Inertial term is not included.

            Parameters:
                subdomains: List of subdomains where the force balance is defined. Only
                    known usage is for the matrix domain(s).

            Returns:
                Operator for the force balance equation in the matrix.

            """
            accumulation = self.inertia(subdomains)
            # By the convention of positive tensile stress, the balance equation is
            # acceleration - stress = body_force. The balance_equation method will *add* the
            # surface term (stress), so we need to multiply by -1.
            stress2 = pp.ad.Scalar(-1) * self.stress2(subdomains)
            body_force = self.body_force(subdomains)

            equation = self.balance_equation(
                subdomains, accumulation, stress2, body_force, dim=self.nd
            )
            equation.set_name("momentum_balance_equation2")
            return equation

        def inertia(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Inertial term [m^2/s].

            Added here for completeness, but not used in the current implementation. Be
            aware that the elasticity discretization has employed herein has, as far as we
            know, never been used to solve a problem with inertia. Thus, if inertia is added
            in a submodel, proceed with caution. In addition to overriding this method, it
            would also be necessary to add the inertial term to the balance equation
            :meth:`momentum_balance_equation`.

            Parameters:
                subdomains: List of subdomains where the inertial term is defined.

            Returns:
                Operator for the inertial term.

            """
            return pp.ad.Scalar(0)

        def interface_force_balance_equation2(
            self,
            interfaces: list[pp.MortarGrid],
        ) -> pp.ad.Operator:
            """Momentum balance equation at matrix-fracture interfaces.

            Parameters:
                interfaces: Fracture-matrix interfaces.

            Returns:
                Operator representing the force balance equation.

            Raises:
                ValueError: If an interface is not a fracture-matrix interface.

            """
            # Check that the interface is a fracture-matrix interface.
            for interface in interfaces:
                if interface.dim != self.nd - 1:
                    raise ValueError("Interface must be a fracture-matrix interface.")

            subdomains = self.interfaces_to_subdomains(interfaces)
            # Split into matrix and fractures. Sort on dimension to allow for multiple
            # matrix domains. Otherwise, we could have picked the first element.
            matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

            # Geometry related
            mortar_projection = pp.ad.MortarProjections(
                self.mdg, subdomains, interfaces, self.nd
            )
            proj = pp.ad.SubdomainProjections(subdomains, self.nd)

            # Contact traction from primary grid and mortar displacements (via primary
            # grid). Spelled out for clarity:
            #   1) The sign of the stress on the matrix subdomain is corrected so that all
            #      stress components point outwards from the matrix (or inwards, EK is not
            #      completely sure, but the point is the consistency).
            #   2) The stress is prolonged from the matrix subdomains to all subdomains seen
            #      by the mortar grid (that is, the matrix and the fracture).
            #   3) The stress is projected to the mortar grid.
            contact_from_primary_mortar = (
                mortar_projection.primary_to_mortar_int()
                @ proj.face_prolongation(matrix_subdomains)
                @ self.internal_boundary_normal_to_outwards(matrix_subdomains, dim=self.nd)
                @ self.stress2(matrix_subdomains)
            )
            # Traction from the actual contact force.
            traction_from_secondary = self.fracture_stress(interfaces)
            # The force balance equation. Note that the force from the fracture is a
            # traction, not a stress, and must be scaled with the area of the interface.
            # This is not the case for the force from the matrix, which is a stress.
            force_balance_eq: pp.ad.Operator = (
                contact_from_primary_mortar
                + self.volume_integral(traction_from_secondary, interfaces, dim=self.nd)
            )
            force_balance_eq.set_name("interface_force_balance_equation2")
            return force_balance_eq

        def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Body force integrated over the subdomain cells.

            Parameters:
                subdomains: List of subdomains where the body force is defined.

            Returns:
                Operator for the body force [kg*m*s^-2].

            """
            return self.volume_integral(
                self.gravity_force(subdomains, "solid"), subdomains, dim=self.nd
            )

class ViscousPartVariable:
    """Linear elastic properties of a viscous part.

    Includes "primary" stiffness parameters (lame_lambda, shear_modulus) and "secondary"
    parameters (bulk_modulus, lame_mu, poisson_ratio). The latter are computed from the
    former. Also provides a method for computing the stiffness matrix as a
    FourthOrderTensor.
    """

    def shear_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise shear modulus operator. The value is picked from the solid
            constants.

        """
        return Scalar(self.solid.shear_modulus2, "shear_modulus2")

    def lame_lambda2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lame's first parameter [Pa].

        Parameters:
            subdomains: List of subdomains where the shear modulus is defined.

        Returns:
            Cell-wise Lame's first parameter operator. The value is picked from the
                solid constants.

        """
        return Scalar(self.solid.lame_lambda2, "lame_lambda2")

    def youngs_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Young's modulus [Pa].

        Parameters:
            subdomains: List of subdomains where the Young's modulus is defined.

        Returns:
            Cell-wise Young's modulus in Pascal. The value is picked from the solid
                constants.

        """
        val = (
            self.solid.shear_modulus2
            * (3 * self.solid.lame_lambd2 + 2 * self.solid.shear_modulus2)
            / (self.solid.lame_lambda2 + self.solid.shear_modulus2)
        )
        return Scalar(val, "youngs_modulus2")

    def bulk_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Bulk modulus [Pa]."""
        val = self.solid.lame_lambda2 + 2 * self.solid.shear_modulus2 / 3
        return Scalar(val, "bulk_modulus2")

    def stiffness_tensor2(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """
        lmbda2 = self.solid.lame_lambda2 * np.ones(subdomain.num_cells)
        mu2 = self.solid.shear_modulus2 * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu2, lmbda2)
    
class NewLinearElasticMechanicalStress (pp.PorePyModel):
    """Linear elastic stress tensor for u2.

    To be used in mechanical problems, e.g. force balance.

    """

    stress2_keyword: str
    """Keyword used to identify the stress discretization. Normally set by a mixin
    instance of
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    displacement2: Callable[[pp.SubdomainsOrBoundaries], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    interface_displacement2: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Displacement variable on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction variable. Normally defined in a mixin instance of
    :class:`~porepy.models.contact_mechanics.ContactTractionVariable`.

    """
    characteristic_contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Characteristic contact traction. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.CharacteristicTractionFromDisplacement` or
    :class:`~porepy.models.constitutive_laws.CharacteristicDisplacementFromTraction`.

    """

    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the momentum problem.
    Normally provided by a mixin instance of
    :class:`~porepy.models.momentum_balance.BoundaryConditionsMomentumBalance`.

    """

    def mechanical_stress2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress.

        .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

        Parameters:
            grids: List of subdomains or boundary grids. If subdomains, should be of
                co-dimension 0.

        Raises:
            ValueError: If any grid is not of co-dimension 0.
            ValueError: If any the method is called with a mixture of subdomains and
                boundary grids.

        Returns:
            Ad operator representing the mechanical stress on the faces of the grids.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.stress2_keyword, domains=domains
            )

        # Check that the subdomains are grids.
        if not all([isinstance(g, pp.Grid) for g in domains]):
            raise ValueError(
                """Argument subdomains a mixture of grids and boundary grids."""
            )
        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense).
        domains = cast(list[pp.Grid], domains)

        for sd in domains:
            # The mechanical stress is only defined on subdomains of co-dimension 0.
            if sd.dim != self.nd:
                raise ValueError("Subdomain must be of co-dimension 0.")

        # No need to facilitate changing of stress discretization, only one is available
        # at the moment.
        discr = self.stress_discretization2(domains)
        # Fractures in the domain.
        interfaces = self.subdomains_to_interfaces(domains, [1])

        # Boundary conditions on external boundaries
        boundary_operator = self.combine_boundary_operators_mechanical_stress2(domains)
        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)
        # The stress in the subdomanis is the sum of the stress in the subdomain, the
        # stress on the external boundaries, and the stress on the interfaces. The
        # latter is found by projecting the displacement on the interfaces to the
        # subdomains, and let these act as Dirichlet boundary conditions on the
        # subdomains.
        stress2 = (
            discr.stress2() @ self.displacement2(domains)
            + discr.bound_stress2() @ boundary_operator
            + discr.bound_stress2()
            @ proj.mortar_to_primary_avg()
            @ self.interface_displacement2(interfaces)
        )
        stress2.set_name("mechanical_stress2")
        return stress2

    def combine_boundary_operators_mechanical_stress2(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Combine mechanical stress boundary operators.

        Note that the default Robin operator is the same as that of Neumann. Override
        this method to define and assign another boundary operator of your choice. The
        new operator should then be passed as an argument to the
        _combine_boundary_operators method, just like self.mechanical_stress is passed
        to robin_operator in the default model.

        Parameters:
            subdomains: List of the subdomains whose boundary operators are to be
                combined.

        Returns:
            The combined mechanical stress boundary operator.

        """
        op = self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.displacement2,
            neumann_operator=self.mechanical_stress2,
            robin_operator=self.mechanical_stress2,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics",
        )
        return op

    def fracture_stress(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Fracture stress on interfaces [Pa].

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Fracture stress operator.

        Raises:
            ValueError: If any interface is not of co-dimension 1.

        """
        for interface in interfaces:
            if any([interface.dim != self.nd - 1]):
                raise ValueError("Interface must be of co-dimension 1.")

        # Subdomains of the interfaces
        subdomains = self.interfaces_to_subdomains(interfaces)
        # Isolate the fracture subdomains
        fracture_subdomains = [sd for sd in subdomains if sd.dim == self.nd - 1]
        # Projection between all subdomains of the interfaces
        subdomain_projection = pp.ad.SubdomainProjections(subdomains, self.nd)
        # Projection between the subdomains and the interfaces
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        # Spelled out, the stress on the interface is found by mapping the contact
        # traction (a primary variable) from local to global coordinates (note the
        # transpose), prolonging the traction from the fracture subdomains to all
        # subdomains (the domain of definition for the mortar projections), projecting
        # to the interface, and switching the sign of the traction depending on the sign
        # of the mortar sides.
        nondim_traction = (
            mortar_projection.sign_of_mortar_sides()
            @ mortar_projection.secondary_to_mortar_int()
            @ subdomain_projection.cell_prolongation(fracture_subdomains)
            @ self.local_coordinates(fracture_subdomains).transpose()
            @ self.contact_traction(fracture_subdomains)
        )
        # Rescale to physical units from the scaled contact traction.
        traction = nondim_traction * self.characteristic_contact_traction(
            fracture_subdomains
        )
        traction.set_name("mechanical_fracture_stress")
        return traction

    def stress_discretization2(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.BiotAd | pp.ad.MpsaAd:
        """Discretization of the stress tensor.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Discretization operator for the stress tensor.

        """
        return pp.ad.MpsaAd(self.stress2_keyword, subdomains)


class ModyfiedConstitutiveLawsMomentumBalance(
    constitutive_laws.ZeroGravityForce,
    ViscousPartVariable,
    NewLinearElasticMechanicalStress,
    constitutive_laws.ConstantSolidDensity,
):
    """Class for constitutive equations for momentum balance equations."""

    def stress2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress2(domains)

class CreateVariable(pp.VariableMixin):
    """Variables for mixed-dimensional deformation.

    The variables are:
        - Displacement in matrix
        - Displacement on fracture-matrix interfaces

    """

    displacement2_variable: str
    """Name of the primary variable representing the displacement in subdomains.
    Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    interface_displacement2_variable: str
    """Name of the primary variable representing the displacement on an interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """

    def create_variables(self) -> None:
        """Introduces the following variables into the system:

        1. Displacement u2 variable on all subdomains.
        2. Displacement u2 variable on all interfaces with codimension 1.
        3. Contact traction variable on all fracture subdomains.

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
 
    def displacement2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Displacement u2 in the matrix.

        Parameters:
            domains: List of subdomains or interface grids where the displacement is
                defined. Should be the matrix subdomains.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the subdomains is not equal to the ambient
                dimension of the problem.
            ValueError: If the method is called on a mixture of grids and boundary
                grids

        """
        if len(domains) == 0 or all(
            isinstance(grid, pp.BoundaryGrid) for grid in domains
        ):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.displacement2_variable, domains=domains
            )
        # Check that the subdomains are grids
        if not all(isinstance(grid, pp.Grid) for grid in domains):
            raise ValueError(
                "Method called on a mixture of subdomain and boundary grids."
            )
        # Now we can cast to Grid
        domains = cast(list[pp.Grid], domains)

        if not all([grid.dim == self.nd for grid in domains]):
            raise ValueError(
                "Displacement is only defined in subdomains of dimension nd."
            )

        return self.equation_system.md_variable(self.displacement2_variable, domains)

    def interface_displacement2(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Variable:
        """Displacement on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interface grids where the displacement is defined.
                Should be between the matrix and fractures.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the interfaces is not equal to the ambient
                dimension of the problem minus one.

        """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError(
                "Interface displacement is only defined on interfaces of dimension "
                "nd - 1."
            )

        return self.equation_system.md_variable(
            self.interface_displacement2_variable, interfaces
        )
        




class ModifiedBoundaryConditions:

    units: pp.Units

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Set boundary condition type for the problem."""
        domain_sides = self.domain_boundary_sides(sd)

        # Set the type of west and east boundaries to Dirichlet. North and south are
        # Neumann by default.
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.west + domain_sides.east, "dir")
        return bc

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting stress boundary condition values at north and south boundaries.

        Specifically, we assign different values for the x- and y-component of the
        boundary value vector.

        """
        values = np.ones((self.nd, bg.num_cells))
        domain_sides = self.domain_boundary_sides(bg)

        # Assigning x-component values
        values[0][domain_sides.north + domain_sides.south] *= self.units.convert_units(5, "Pa")

        # Assigning y-component values
        values[1][domain_sides.north + domain_sides.south] *= self.units.convert_units(5, "Pa")

        return values.ravel("F")

    # def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
         
    #      """Setting displacement boundary condition values.


    #      """
    #     # # Fetch the time of the current time-step
    #      # t = self.time_manager.time

    #      values = np.zeros((self.nd, bg.num_cells))
    #      domain_sides = self.domain_boundary_sides(bg)

    #      # Assign a time dependent value to the x-component of the western boundary
    #      values[0][domain_sides.west] += self.units.convert_units(1.0, "m")
    #      values[0][domain_sides.east] += self.units.convert_units(1.0, "m")
    #      values[1][domain_sides.west] += self.units.convert_units(1.0, "m")
    #      values[1][domain_sides.east] += self.units.convert_units(1.0, "m")

    #     # The convention for flattening nd-arrays of vector values in PorePy is by using
    #     # the Fortran-style ordering (chosen by string "F" when giving a call to ravel).
    #     # That is, the first index changes the fastest and the last index changes
    #     # slowest.
    #      return values.ravel("F")


class PressureSourceBC:
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet boundary condition to the north boundary and Neumann
        everywhere else.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.north, "dir")
        return bc

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Assign fracture source."""
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        values = []
        src_value: float = self.units.convert_units(0.1, "kg * m^-3 * s^-1")
        for sd in subdomains:
            if sd.dim == self.mdg.dim_max():
                values.append(np.zeros(sd.num_cells))
            else:
                values.append(np.ones(sd.num_cells) * src_value)

        external_sources = pp.wrap_as_dense_ad_array(np.concatenate(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


class BodyForceMixin:
    nd: int
    """Ambient dimension."""

    units: pp.Units

    solid: pp.SolidConstants

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        units = self.units
        vals = []
        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))

            # We add the source only to the 2D domain and not the fracture.
            if sd.dim == 2:
                # Selecting central cells
                cell_centers = sd.cell_centers
                indices = (
                    (cell_centers[0] > (0.3 / units.m))
                    & (cell_centers[0] < (0.7 / units.m))
                    & (cell_centers[1] > (0.3 / units.m))
                    & (cell_centers[1] < (0.7 / units.m))
                )

                acceleration = self.units.convert_units(-9.8, "m * s^-2")
                force = self.solid.density * acceleration
                data[indices, 1] = force * sd.cell_volumes[indices]

            vals.append(data)
        return pp.ad.DenseArray(np.concatenate(vals).ravel(), "body_force")



class MomentumBalanceGeometryBC(
    ModifiedGeometry,
    ModyfiedEquation,
    ModyfiedConstitutiveLawsMomentumBalance,
    CreateVariable,
    #SquareDomainOrthogonalFractures,
    ModifiedBoundaryConditions,
    #PressureSourceBC,
    BodyForceMixin,
    pp.MomentumBalance
    ):
    pass

model = MomentumBalanceGeometryBC()
pp.run_time_dependent_model(model)
pp.plot_grid(
    model.mdg,
    cell_value =model.displacement_variable,
    rgb=[1, 1, 1],
    figsize=(10, 8),
    linewidth=0.3,
    title="displacement",
    plot_2d=True
)

import matplotlib.pyplot as plt
plt.show()
plt.savefig("MyProject/displacement26.png", dpi=300)

print("Hello")