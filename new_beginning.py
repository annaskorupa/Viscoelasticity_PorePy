import porepy as pp
import numpy as np

from porepy.applications.md_grids.domains import nd_cube_domain
#\from porepy.models.momentum_balance import MomentumBalance
from porepy.models.fluid_mass_balance import SinglePhaseFlow

from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)

Scalar = pp.ad.Scalar

class ModifiedGeometry:
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
    pass



class CreateVariable:

    def create_variables(self) -> None:
        """Introduces the following variables into the system:

        1. u2
        2. E2 ->?

        """
        super().create_variables()

        self.equation_system.create_variables(
            self.displacement2_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "m"},
        )
        

class ElaticPartVariable:
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