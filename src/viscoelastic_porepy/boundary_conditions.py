"""Boundary conditions for the MMS verification problem.

All boundaries are Dirichlet with analytically prescribed displacement
from the Manufactured Solution.
"""

import numpy as np
import porepy as pp


class BoundaryConditionsMixin:
    """MMS boundary conditions: Dirichlet on all sides.

    Displacement values are computed from the analytical MMS solution:
        u(x, t) = A · sin(πx/L) · sin(πy/L) · (1 − exp(−b·t))
    """

    units: pp.Units

    def bc_type_mechanics(
        self, sd: pp.Grid
    ) -> pp.BoundaryConditionVectorial:
        """Assign Dirichlet BC on all boundaries."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            (
                domain_sides.west
                + domain_sides.east
                + domain_sides.north
                + domain_sides.south
            ),
            "dir",
        )
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Compute MMS displacement BC values at boundary faces.

        Parameters:
            bg: Boundary grid.

        Returns:
            Flattened displacement values ``[ux_0, uy_0, ux_1, uy_1, ...]``.
        """
        cc = bg.cell_centers
        A_MMS = 1.0e-3
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        t = self.time_manager.time
        L = 0.8  # domain length [m]

        spatial = np.sin(np.pi * cc[0] / L) * np.sin(np.pi * cc[1] / L)
        temporal = 1.0 - np.exp(-b_MMS * t)

        ux = A_MMS * spatial * temporal
        uy = A_MMS * spatial * temporal

        data = np.zeros((self.nd, bg.num_cells))
        data[0, :] = ux
        data[1, :] = uy
        return data.ravel()

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero traction on Neumann faces (not used with full Dirichlet)."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")
