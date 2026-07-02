"""Boundary conditions for the viscoelastic model.

Setup:
- South face: Dirichlet u = 0 (fixed base)
- North face: Neumann σ_y = sigma_load (compressive load, default −3 MPa)
- East/West faces: Neumann σ = 0 (traction-free)

For fracture sub-domains (dim < nd), all boundary faces are Neumann.
"""

import numpy as np
import porepy as pp


class BoundaryConditionsMixin:
    """Mechanical boundary conditions for the 2D creep test."""

    units: pp.Units

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        if sd.dim == self.nd:
            domain_sides = self.domain_boundary_sides(sd)
            all_external_faces = (
                domain_sides.north
                + domain_sides.south
                + domain_sides.east
                + domain_sides.west
            )
            bc = pp.BoundaryConditionVectorial(
                sd, all_external_faces, "neu"
            )
            # South = Dirichlet (fixed)
            bc.is_dir[:, domain_sides.south] = True
            bc.is_neu[:, domain_sides.south] = False
            return bc
        else:
            bound_faces = sd.tags.get(
                "boundary_faces", np.array([], dtype=int)
            )
            return pp.BoundaryConditionVectorial(sd, bound_faces, "neu")

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero Dirichlet values on the south face."""
        if bg.parent.dim == self.nd:
            values = np.zeros((self.nd, bg.num_cells))
            domain_sides = self.domain_boundary_sides(bg)
            values[0, domain_sides.south] = 0.0
            values[1, domain_sides.south] = 0.0
            return values.ravel("F")
        else:
            return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compressive load on the north face (σ_y = sigma_load)."""
        if bg.parent.dim == self.nd:
            values = np.zeros((self.nd, bg.num_cells))
            domain_sides = self.domain_boundary_sides(bg)

            sigma_load = self.params.get("sigma_load", -3_000_000.0)

            values[0, domain_sides.north] = 0.0
            values[1, domain_sides.north] = (
                self.units.convert_units(sigma_load, "Pa")
                * bg.cell_volumes[domain_sides.north]
            )
            return values.ravel("F")
        else:
            return np.zeros((self.nd, bg.num_cells)).ravel("F")
