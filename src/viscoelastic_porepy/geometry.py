"""Geometry mixins for 2D viscoelastic simulations.

Provides two mixins:
- GeometryMixin: 2D square domain (0.1 m × 0.1 m) without fractures.
- FractureGeometryMixin: same domain with a diagonal fracture.

Grid type and cell size are configurable via model params.
"""

import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain


class GeometryMixin:
    """2D square domain (0.1 m × 0.1 m) without fractures.

    Params
    ------
    grid_type : str
        "cartesian" (default) or "simplex".
    cell_size : float
        Approximate cell diameter [m]. Default 0.00125.
    """

    units: pp.Units

    def set_domain(self) -> None:
        size = self.units.convert_units(0.1, "m")
        self._domain = nd_cube_domain(2, size)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        return {"cell_size": self.params.get("cell_size", 0.00125)}


class FractureGeometryMixin(GeometryMixin):
    """2D square domain with a diagonal fracture from (0.04, 0.04) to (0.06, 0.06).

    Default grid type is "simplex" (required for fracture meshing).
    """

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        return {"cell_size": self.params.get("cell_size", 0.005)}

    def set_fractures(self) -> None:
        """Set a diagonal fracture."""
        frac_1_points = self.units.convert_units(
            np.array([[0.04, 0.06], [0.04, 0.06]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]
