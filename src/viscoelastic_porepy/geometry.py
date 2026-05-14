"""Geometry mixin for the 2D computational domain."""

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain


class GeometryMixin:
    """2D square domain with configurable mesh.

    The domain side length is 0.8 m by default. Grid type and cell size
    can be overridden via ``self.params``.
    """

    units: pp.Units

    def set_domain(self) -> None:
        """Define a 2D square domain with side length 0.8 m."""
        size = self.units.convert_units(0.8, "m")
        self._domain = nd_cube_domain(2, size)

    def grid_type(self) -> str:
        """Return the grid type (default: cartesian)."""
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        """Return meshing arguments with configurable cell size."""
        return {"cell_size": self.params.get("cell_size", 0.125)}
