import porepy as pp
import numpy as np

from porepy.applications.md_grids.domains import nd_cube_domain

from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)

from typing import Callable, Optional, Sequence, cast


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
    

class NewVariables(pp.momentum_balance.VariablesMomentumBalance):
    """Variables for mixed-dimensional deformation.

    The variables are:
        - Displacement 2 in matrix
        - Displacement 2 on fracture-matrix interfaces

    """

    #=============================================

    displacement_variable2: str = "displacement_2"#! TRZEBA ZAINICJOWAĆ WARTOŚĆ TEJ ZMIENNEJ. PRZY URUCHAMIANIU PROGRAMU displacement_variable2 JEST PUSTE. NA TEN MOMENT ZADEKLAROWANY JEST WYŁĄCZNIE TYP. Dodałem stringa, aby ruszył proces inicjowania klasycznego. Wartość jest inicjowana w metodzie create_variable

    #=============================================
    """Name of the primary variable representing the displacement in subdomains.
    Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    #=============================================
    interface_displacement_variable2: str = "interface_displacement_2" #! TRZEBA ZAINICJOWAĆ WARTOŚĆ TEJ ZMIENNEJ. PRZY URUCHAMIANIU PROGRAMU displacement_variable2 JEST PUSTE. NA TEN MOMENT ZADEKLAROWANY JEST WYŁĄCZNIE TYP. Tu tak samo
    #=============================================
    """Name of the primary variable representing the displacement on an interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """

    def create_variables(self) -> None:
        """Introduces the following variables into the system:

        1. Displacement variable 2on all subdomains.
        2. Displacement variable 2 on all interfaces with codimension 1.
        3. Contact traction variable on all fracture subdomains.

        """

        super().create_variables()

        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.displacement_variable2, #! Poprawiłem typo z displacement2_variable na displacement_variable2
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "m"},
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.interface_displacement_variable2, #! Poprawiłem typo z interface_displacement2_variable na interface_displacement_variable2
            interfaces=self.mdg.interfaces(dim=self.nd - 1, codim=1),
            tags={"si_units": "m"},
        )

    def displacement2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Displacement in the matrix.

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
                name=self.displacement_variable, domains=domains
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

        return self.equation_system.md_variable(self.displacement_variable, domains)

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
        values[0, domain_sides.north] *= self.units.convert_units(4.5, "Pa") * bg.cell_volumes[domain_sides.north]
        values[0, domain_sides.south] *= self.units.convert_units(4.5, "Pa") * bg.cell_volumes[domain_sides.south]

        # Assigning y-component values
        values[1, domain_sides.north] *= self.units.convert_units(0.5, "Pa") * bg.cell_volumes[domain_sides.north]
        values[1, domain_sides.south] *= self.units.convert_units(0.5, "Pa") * bg.cell_volumes[domain_sides.south]

        return values.ravel("F")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Setting displacement boundary condition values.

        This method returns an array of boundary condition values with the value 5 for
        western boundaries and ones for the eastern boundary.

        """

        values = np.zeros((self.nd, bg.num_cells))
        domain_sides = self.domain_boundary_sides(bg)

        # Assign a time dependent value to the x-component of the western boundary
        values[0, domain_sides.west] += self.units.convert_units(5.0 , "m")
        values[0, domain_sides.east] += self.units.convert_units(5.0, "m")

        # The convention for flattening nd-arrays of vector values in PorePy is by using
        # the Fortran-style ordering (chosen by string "F" when giving a call to ravel).
        # That is, the first index changes the fastest and the last index changes
        # slowest.
        return values.ravel("F")

class MomentumBalanceGeometryBC(
    BodyForceMixin,
    ModifiedGeometry,
    NewVariables,
    #SquareDomainOrthogonalFractures,
    ModifiedBoundaryConditions,
    pp.MomentumBalance
):
    pass

fluid_constants = pp.FluidComponent(viscosity=0.1, density=0.2)
solid_constants = pp.SolidConstants(permeability=0.5, porosity=0.25)

print("fluid_constants", fluid_constants)
print("solid_constants", solid_constants)

newVariablesObj = NewVariables()
print("newVariablesObj", newVariablesObj.displacement_variable2)

material_constants = {"fluid": fluid_constants, "solid": solid_constants}
model_params = {"material_constants": material_constants, "linear_solver": "scipy_sparse"}

model = MomentumBalanceGeometryBC(model_params)
pp.run_time_dependent_model(model)


# model = MomentumBalanceGeometryBC()
# pp.run_time_dependent_model(model)
pp.plot_grid(
    model.mdg,
    #vector_value=model.displacement_variable,
    cell_value =model.displacement_variable,
    rgb=[1, 1, 1],
    figsize=(10, 8),
    linewidth=0.3,
    title="Displacement",
    plot_2d=True
)
#u = model.equation_system.get_time_series(model.displacement_variable)
#print(u)

import matplotlib.pyplot as plt
plt.show()
plt.savefig("MyProject/displacementxx.png", dpi=300)

print("Hello")

# --- Odczyt nowej i starej zmiennej ---
#u_values = model.equation_system.get_variable_values(model.displacement_variable)
#u2_values = model.equation_system.get_variable_values(model.u2_variable)

#print("u:", u_values)
#print("u2:", u2_values)