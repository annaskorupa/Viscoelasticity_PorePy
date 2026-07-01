"""Viscoelastic momentum balance models.

Provides two model classes:
- ViscoelasticMomentumBalance: 2D domain without fractures
- ViscoelasticMomentumBalanceFracture: 2D domain with a diagonal fracture

Both inherit from a common ViscoelasticModelMixin that implements:
- Combined stress σ = σ₁(u) + σ₂(u₂)
- Extensometer strain recording
- Displacement snapshot saving at requested times
- GL history management via before_nonlinear_loop
"""

import numpy as np
import porepy as pp
import matplotlib.pyplot as plt

from .geometry import GeometryMixin, FractureGeometryMixin
from .constitutive import ConstitutiveLawsU2
from .variables import VariablesU2, RateEquation
from .boundary_conditions import BoundaryConditionsMixin
from .infrastructure import (
    InitialConditionsU2,
    SolutionStrategyU2,
    BodyForceMixin,
)


class ViscoelasticModelMixin:
    """Common behaviour shared by all viscoelastic model variants.

    Must appear in the MRO **after** all other mixins but **before**
    ``pp.MomentumBalance`` so that ``super()`` calls chain correctly.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.stress2_keyword = "mechanics2"
        self.history_u: list[np.ndarray] = []
        self.history_u2: list[np.ndarray] = []
        self.strain_history: dict[str, list] = {
            "times": [],
            "exx_u": [],
            "eyy_u": [],
            "exy_u": [],
            "exx_u2": [],
            "eyy_u2": [],
            "exy_u2": [],
        }
        self._top_cells: np.ndarray | None = None
        self._domain_height: float = 0.1  # domain is 0.1 m tall
        self._displacement_snapshots: dict[float, dict] = {}

    # ------------------------------------------------------------------
    # Time-stepping hooks
    # ------------------------------------------------------------------
    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        sds = self.mdg.subdomains(dim=self.nd)
        u_val = np.array(
            self.equation_system.evaluate(self.displacement(sds))
        ).ravel()
        u2_val = np.array(
            self.equation_system.evaluate(self.displacement2(sds))
        ).ravel()
        self.history_u.append(u_val)
        self.history_u2.append(u2_val)

        # Record extensometer strain
        self._record_strain(u_val, u2_val)

        # Save displacement snapshots at requested times
        self._save_snapshot_if_needed(u_val, u2_val)

        # Progress logging
        if self.time_manager.time_index % 500 == 0:
            print(
                f"  t = {self.time_manager.time:.1f} s "
                f"({self.time_manager.time / 3600:.3f} h), "
                f"step {self.time_manager.time_index}"
            )

    def before_nonlinear_loop(self) -> None:
        self._update_gl_sums()
        super().before_nonlinear_loop()
        if hasattr(self, "_bf_subdomains"):
            new_vals = self._compute_body_force_values(self._bf_subdomains)
            offset = 0
            for sd in self._bf_subdomains:
                sd_data = self.mdg.subdomain_data(sd)
                n = sd.num_cells * self.nd
                pp.set_solution_values(
                    "body_force",
                    new_vals[offset : offset + n],
                    sd_data,
                    iterate_index=0,
                )
                pp.set_solution_values(
                    "body_force",
                    new_vals[offset : offset + n],
                    sd_data,
                    time_step_index=0,
                )
                offset += n

    # ------------------------------------------------------------------
    # Stress (sum of both branches)
    # ------------------------------------------------------------------
    def stress(self, domains) -> pp.ad.Operator:
        """Total stress σ = σ₁(u) + σ₂(u₂)."""
        if all(
            isinstance(d, pp.Grid) and d.dim == self.nd for d in domains
        ):
            return (
                self.mechanical_stress(domains)
                + self.mechanical_stress2(domains)
            )
        return super().stress(domains)

    # ------------------------------------------------------------------
    # Boundary condition updates for the u₂ branch
    # ------------------------------------------------------------------
    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()
        self.update_boundary_condition(
            self.stress2_keyword, self.bc_values_stress2
        )

    def update_boundary_values_primary_variables(self) -> None:
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(
            self.displacement2_variable, self.bc_values_displacement2
        )

    def bc_values_displacement2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero Dirichlet for u₂ (same BCs as u)."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def bc_values_stress2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero Neumann for u₂ — load is applied only through u."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

    # ------------------------------------------------------------------
    # Strain recording (extensometer approach)
    # ------------------------------------------------------------------
    def _record_strain(self, u_val: np.ndarray, u2_val: np.ndarray) -> None:
        """Record average εyy (extensometer: mean uy_top / height)."""
        sd = self.mdg.subdomains(dim=self.nd)[0]

        if self._top_cells is None:
            y_coords = sd.cell_centers[1, :]
            y_threshold = 0.09  # top 10% of domain
            self._top_cells = np.where(y_coords > y_threshold)[0]
            if len(self._top_cells) == 0:
                y_threshold = 0.08
                self._top_cells = np.where(y_coords > y_threshold)[0]
            if len(self._top_cells) == 0:
                # Very coarse grid: pick the topmost 25% of cells
                n_top = max(1, sd.num_cells // 4)
                sorted_idx = np.argsort(y_coords)[-n_top:]
                self._top_cells = sorted_idx
                y_threshold = y_coords[sorted_idx[0]]
            print(
                f"--- Strain measurement: extensometer ---\n"
                f"    Domain height: {self._domain_height:.4f} m\n"
                f"    Top cells (y > {y_threshold:.4f}): "
                f"{len(self._top_cells)}\n"
                f"    Mean y of top cells: "
                f"{np.mean(y_coords[self._top_cells]):.4f}"
            )

        u_2d = u_val.reshape(self.nd, -1, order="F")
        u2_2d = u2_val.reshape(self.nd, -1, order="F")

        self.strain_history["times"].append(
            self.time_manager.time / 3600.0
        )
        self.strain_history["eyy_u"].append(
            np.mean(u_2d[1, self._top_cells]) / self._domain_height
        )
        self.strain_history["exx_u"].append(
            np.mean(u_2d[0, self._top_cells]) / self._domain_height
        )
        self.strain_history["exy_u"].append(0.0)
        self.strain_history["eyy_u2"].append(
            np.mean(u2_2d[1, self._top_cells]) / self._domain_height
        )
        self.strain_history["exx_u2"].append(
            np.mean(u2_2d[0, self._top_cells]) / self._domain_height
        )
        self.strain_history["exy_u2"].append(0.0)

    # ------------------------------------------------------------------
    # Displacement snapshots
    # ------------------------------------------------------------------
    def _save_snapshot_if_needed(
        self, u_val: np.ndarray, u2_val: np.ndarray
    ) -> None:
        """Save displacement snapshot if current time matches a request."""
        snapshot_times = self.params.get("snapshot_times", [])
        current_time = self.time_manager.time
        for st in snapshot_times:
            if (
                abs(current_time - st) < self.time_manager.dt / 2.0
                and st not in self._displacement_snapshots
            ):
                self._displacement_snapshots[st] = {
                    "u": u_val.copy(),
                    "u2": u2_val.copy(),
                }
                print(f"  [Snapshot] Saved displacement at t = {st:.1f} s")

    # ------------------------------------------------------------------
    # Displacement map plotting
    # ------------------------------------------------------------------
    def plot_displacement_map(
        self,
        u_vals: np.ndarray,
        title: str = "Displacement",
        filepath: str | None = None,
        vmax: float | None = None,
    ) -> None:
        """Plot a displacement magnitude map on the 2D grid.

        Parameters
        ----------
        u_vals : np.ndarray
            Displacement vector (F-order).
        title : str
            Plot title.
        filepath : str or None
            If given, save figure to this path.
        vmax : float or None
            Maximum value for the colour scale. Auto if None.
        """
        sd = self.mdg.subdomains(dim=self.nd)[0]
        mag = np.linalg.norm(
            u_vals.reshape(self.nd, -1, order="F"), axis=0
        )
        if vmax is None:
            vmax = np.max(mag) if np.max(mag) > 0 else 1.0

        plt.close("all")
        pp.plot_grid(
            sd,
            cell_value=mag,
            title=title,
            if_plot=False,
            color_map_limits=[0.0, vmax],
            plot_2d=True,
        )
        fig = plt.gcf()
        fig.axes[-1].set_ylabel("u [m]")
        if filepath:
            plt.savefig(filepath, dpi=300)
            print(f"  Saved {filepath}")
        plt.close("all")


# ======================================================================
# Concrete model classes
# ======================================================================


class ViscoelasticMomentumBalance(
    GeometryMixin,
    BoundaryConditionsMixin,
    BodyForceMixin,
    RateEquation,
    VariablesU2,
    ConstitutiveLawsU2,
    InitialConditionsU2,
    SolutionStrategyU2,
    ViscoelasticModelMixin,
    pp.MomentumBalance,
):
    """Viscoelastic momentum balance on a plain 2D domain (no fractures)."""

    pass


class ViscoelasticMomentumBalanceFracture(
    FractureGeometryMixin,
    BoundaryConditionsMixin,
    BodyForceMixin,
    RateEquation,
    VariablesU2,
    ConstitutiveLawsU2,
    InitialConditionsU2,
    SolutionStrategyU2,
    ViscoelasticModelMixin,
    pp.MomentumBalance,
):
    """Viscoelastic momentum balance with a diagonal fracture."""

    pass
