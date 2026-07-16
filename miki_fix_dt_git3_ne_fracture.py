"""Viscoelastic extension of PorePy's MomentumBalance model.

This script implements a generalized Maxwell model by adding a second 
displacement variable (u2) representing the viscous branch.
The total stress is σ = σ1(u) + σ2(u2), and the rate equation is
u2_dot + β*u2 - u_dot = 0.

Based on Idesman et al. (2000).
"""

import numpy as np
import porepy as pp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Sequence, cast
from porepy.applications.md_grids.domains import nd_cube_domain


Scalar = pp.ad.Scalar


def compute_strain_at_cell(sd, u_vec_flat, cell_idx, adj_csr, nd=2):
    """Compute strain tensor at a single cell via least-squares gradient reconstruction.

    Uses displacement values from neighboring cells (connected via shared faces)
    to reconstruct the displacement gradient, then extracts the symmetric part.

    Parameters:
        sd: pp.Grid - the subdomain grid
        u_vec_flat: np.ndarray - displacement vector (interleaved [x0,y0,x1,y1,...])
        cell_idx: int - index of the cell at which to compute strain
        adj_csr: scipy.sparse.csr_matrix - precomputed cell adjacency matrix
        nd: int - number of spatial dimensions (2)

    Returns:
        exx, eyy, exy: float - strain tensor components
    """
    u = u_vec_flat.reshape(nd, -1, order='F')
    ux, uy = u[0], u[1]
    cc = sd.cell_centers[:nd, :]

    # Get neighbor indices from precomputed adjacency
    row_start = adj_csr.indptr[cell_idx]
    row_end = adj_csr.indptr[cell_idx + 1]
    neighbors = adj_csr.indices[row_start:row_end]
    neighbors = neighbors[neighbors != cell_idx]

    if len(neighbors) < 2:
        return 0.0, 0.0, 0.0

    # Coordinate differences
    dx = cc[0, neighbors] - cc[0, cell_idx]
    dy = cc[1, neighbors] - cc[1, cell_idx]
    A = np.column_stack([dx, dy])

    # Displacement differences
    dux = ux[neighbors] - ux[cell_idx]
    duy = uy[neighbors] - uy[cell_idx]

    # Least-squares gradient: A @ [du/dx, du/dy]^T = delta_u
    grad_ux, _, _, _ = np.linalg.lstsq(A, dux, rcond=None)
    grad_uy, _, _, _ = np.linalg.lstsq(A, duy, rcond=None)

    exx = grad_ux[0]                        # dux/dx
    eyy = grad_uy[1]                        # duy/dy
    exy = 0.5 * (grad_ux[1] + grad_uy[0])  # 0.5*(dux/dy + duy/dx)

    return exx, eyy, exy

# =============================================================================
# 1. Material Constants
# =============================================================================
@dataclass(kw_only=True, eq=False)
class ViscoelasticSolidConstants(pp.SolidConstants):
    """Extended solid constants with additional moduli for the viscous branch."""
    SI_units: ClassVar[dict[str, str]] = dict(**pp.SolidConstants.SI_units)
    SI_units.update({
        "lame_lambda2": "Pa",
        "shear_modulus2": "Pa",
        "viscosity": "Pa * s",
        "alpha": "",
        "omega": "",
    })
    lame_lambda2: float = 1.0
    shear_modulus2: float = 1.0
    viscosity: float = 1.0
    alpha: float = 1.0
    omega: float = 0.0

# =============================================================================
# 2. Geometry
# =============================================================================
class GeometryMixin:
    """2D square domain with simplex mesh."""
    units: pp.Units
    def set_domain(self) -> None:
        size = self.units.convert_units(0.1, "m")
        self._domain = nd_cube_domain(2, size)
    def set_fractures(self) -> None: #for simulation no-fracture, comment out the fracture part!
        """Setting a diagonal fracture"""
        frac_1_points = self.units.convert_units(
            np.array([[0.04, 0.06], [0.04, 0.06]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]    
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")#cartesian (no-fracture)#simplex (with fracture)
    def meshing_arguments(self) -> dict:
        return {"cell_size": self.params.get("cell_size", 0.005)}#0.005 (with fracture)#0.00125 (no-fracture)

# =============================================================================
# 3. Constitutive Laws for u2
# =============================================================================
class ViscousElasticModuli:
    """Elastic moduli for the viscous displacement u2."""
    solid: ViscoelasticSolidConstants
    def shear_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.shear_modulus2, "shear_modulus2")
    def lame_lambda2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.lame_lambda2, "lame_lambda2")
    def stiffness_tensor2(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        lmbda2 = self.solid.lame_lambda2 * np.ones(subdomain.num_cells)
        mu2 = self.solid.shear_modulus2 * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu2, lmbda2)
    def beta(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        val = self.solid.shear_modulus2 / self.solid.viscosity
        return Scalar(val, "beta")

class MechanicalStressU2:
    """MPSA-discretized stress for u2."""
    stress2_keyword: str
    displacement2: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    interface_displacement2: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryCondition]

    def mechanical_stress2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            return self.create_boundary_operator(name=self.stress2_keyword, domains=domains)
        
        domains = cast(list[pp.Grid], domains)
        discr = pp.ad.MpsaAd(self.stress2_keyword, domains)
        interfaces = self.subdomains_to_interfaces(domains, [1])
        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)
        
        boundary_operator = self._combine_boundary_operators(
            subdomains=domains,
            dirichlet_operator=self.displacement2,
            neumann_operator=self.mechanical_stress2,
            robin_operator=self.mechanical_stress2,
            bc_type=self.bc_type_mechanics,
            dim=self.nd,
            name="bc_values_mechanics2",
        )

        stress2 = (
            discr.stress() @ self.displacement2(domains)
            + discr.bound_stress() @ boundary_operator
            + discr.bound_stress() @ proj.mortar_to_primary_avg() @ self.interface_displacement2(interfaces)
        )
        stress2.set_name("mechanical_stress2")
        return stress2

class ConstitutiveLawsU2(pp.models.constitutive_laws.ZeroGravityForce, ViscousElasticModuli, MechanicalStressU2, pp.models.constitutive_laws.ConstantSolidDensity):
    def stress2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        return self.mechanical_stress2(domains)

# =============================================================================
# 4. Variables and Equations for u2
# =============================================================================
class VariablesU2:
    """Variables u2 and interface_u2."""
    displacement2_variable: str
    interface_displacement2_variable: str
    def create_variables(self) -> None:
        super().create_variables()
        self.equation_system.create_variables(dof_info={"cells": self.nd}, name=self.displacement2_variable, subdomains=self.mdg.subdomains(dim=self.nd), tags={"si_units": "m"})
        self.equation_system.create_variables(dof_info={"cells": self.nd}, name=self.interface_displacement2_variable, interfaces=self.mdg.interfaces(dim=self.nd - 1, codim=1), tags={"si_units": "m"})

    def displacement2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        if len(domains) == 0 or all(isinstance(g, pp.BoundaryGrid) for g in domains):
            return self.create_boundary_operator(name=self.displacement2_variable, domains=domains)
        return self.equation_system.md_variable(self.displacement2_variable, cast(list[pp.Grid], domains))

    def interface_displacement2(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Variable:
        return self.equation_system.md_variable(self.interface_displacement2_variable, interfaces)

class RateEquation:
    """Implementation of D^alpha u2 + beta*u2 - D^alpha u = 0."""
    def compute_gl_weights(self, alpha, n):
        w = np.zeros(n + 1)
        w[0] = 1.0
        for k in range(1, n + 1):
            w[k] = w[k-1] * (1.0 - (alpha + 1.0) / k)
        return w

    def _update_gl_sums(self):
        n = self.time_manager.time_index
        w_alpha = self.compute_gl_weights(self.solid.alpha, n)
        
        if len(self.history_u) == 0:
            # Initial conditions are zero
            for sd in self.mdg.subdomains(dim=self.nd):
                self.history_u.append(np.zeros(sd.num_cells * self.nd))
                self.history_u2.append(np.zeros(sd.num_cells * self.nd))

        if n == 0:
            for sd in self.mdg.subdomains(dim=self.nd):
                sd_data = self.mdg.subdomain_data(sd)
                gl_u_zero = np.zeros(sd.num_cells * self.nd)
                pp.set_solution_values("gl_u", gl_u_zero, sd_data, time_step_index=0)
                pp.set_solution_values("gl_u2", gl_u_zero, sd_data, time_step_index=0)
                pp.set_solution_values("gl_u", gl_u_zero, sd_data, iterate_index=0)
                pp.set_solution_values("gl_u2", gl_u_zero, sd_data, iterate_index=0)
            return

        gl_u = np.zeros_like(self.history_u[0])
        gl_u2 = np.zeros_like(self.history_u2[0])
        
        for k in range(1, n + 1):
            gl_u += w_alpha[k] * self.history_u[n - k]
            gl_u2 += w_alpha[k] * self.history_u2[n - k]
            
        offset = 0
        for sd in self.mdg.subdomains(dim=self.nd):
            sd_data = self.mdg.subdomain_data(sd)
            num_dofs = sd.num_cells * self.nd
            pp.set_solution_values("gl_u", gl_u[offset:offset+num_dofs], sd_data, time_step_index=0)
            pp.set_solution_values("gl_u2", gl_u2[offset:offset+num_dofs], sd_data, time_step_index=0)
            pp.set_solution_values("gl_u", gl_u[offset:offset+num_dofs], sd_data, iterate_index=0)
            pp.set_solution_values("gl_u2", gl_u2[offset:offset+num_dofs], sd_data, iterate_index=0)
            offset += num_dofs

    def set_equations(self) -> None:
        super().set_equations()
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        u = self.displacement(matrix_subdomains)
        u2 = self.displacement2(matrix_subdomains)
        beta = self.beta(matrix_subdomains)
        
        # D^alpha u2 + beta*u2 - D^alpha u = 0
        gl_u = pp.ad.TimeDependentDenseArray("gl_u", matrix_subdomains)
        gl_u2 = pp.ad.TimeDependentDenseArray("gl_u2", matrix_subdomains)
        dt_alpha = self.ad_time_step ** self.solid.alpha
        
        D_alpha_u2 = (u2 + gl_u2) / dt_alpha
        D_alpha_u = (u + gl_u) / dt_alpha
        
        eq = D_alpha_u2 + beta * u2 - D_alpha_u
        eq.set_name("rate_equation")
        self.equation_system.set_equation(eq, matrix_subdomains, {"cells": self.nd})

        # Equation for u2_interface: simply set it to zero (no viscous jump across fracture)
        interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)
        if len(interfaces) > 0:
            u2_intf = self.interface_displacement2(interfaces)
            eq_intf = Scalar(1) * u2_intf  # wrap to make it a non-variable operator
            eq_intf.set_name("u2_interface_zero")
            self.equation_system.set_equation(eq_intf, interfaces, {"cells": self.nd})

# =============================================================================
# 5. Infrastructure Mixins
# =============================================================================
class BoundaryConditionsMixin:
    """MMS BCs: East/West ux=uy=0 (Dir), North/South uy=0 (Dir) + ux free (Neu)."""
    units: pp.Units
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        #domain_sides = self.domain_boundary_sides(sd)
        # FIX #1/#2: East/West → full Dirichlet (ux=0, uy=0)
        #bc = pp.BoundaryConditionVectorial(sd, domain_sides.south, "dir")
        #bc = pp.BoundaryConditionVectorial(sd, domain_sides.north + domain_sides.east + domain_sides.west, "neu")
        # North/South → Dirichlet ONLY for uy (component [1]), ux stays Neumann (free)
        # bc.is_dir[1, domain_sides.north] = True
        # bc.is_neu[1, domain_sides.north] = False
        # bc.is_dir[1, domain_sides.south] = True
        # bc.is_neu[1, domain_sides.south] = False

        bound_faces = sd.tags.get("boundary_faces", np.array([], dtype=int))

        if sd.dim == self.nd:  # Główna domena 2D
            domain_sides = self.domain_boundary_sides(sd)
            all_external_faces = domain_sides.north + domain_sides.south + domain_sides.east + domain_sides.west
            bc = pp.BoundaryConditionVectorial(sd, all_external_faces, "neu")
            bc.is_dir[:, domain_sides.south] = True
            bc.is_neu[:, domain_sides.south] = False
            return bc
   
        else:
        # Pobieramy krawędzie brzegowe poddomeny (jeśli istnieją), w przeciwnym wypadku pustą tablicę
            bound_faces = sd.tags.get("boundary_faces", np.array([], dtype=int))
            return pp.BoundaryConditionVectorial(sd, bound_faces, "neu")
    
    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # FIX #1: np.zeros instead of np.ones — all Dirichlet values = 0
        if bg.parent.dim == self.nd:

            values = np.zeros((self.nd, bg.num_cells))
            domain_sides = self.domain_boundary_sides(bg)

            values[0, domain_sides.south] = 0.0
            values[1, domain_sides.south] = 0.0


            return values.ravel("F") #"F"
        else:
            return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        
        if bg.parent.dim == self.nd:

            values = np.zeros((self.nd, bg.num_cells))
            domain_sides = self.domain_boundary_sides(bg)

            # Assigning x-component values
            values[0, domain_sides.north] = self.units.convert_units(0.0, "Pa") * bg.cell_volumes[domain_sides.north]
            #values[0, domain_sides.south] = self.units.convert_units(0.0, "Pa") * bg.cell_volumes[domain_sides.south]

            # Assigning y-component values
            time_hours = self.time_manager.time / 3600.0
            damage_factor = 1.0#np.exp(self.solid.omega * time_hours)
            if damage_factor > 1.0: damage_factor = 1.0 #10.0
            val = -3000000.0 * damage_factor
            values[1, domain_sides.north] = self.units.convert_units(val, "Pa") * bg.cell_volumes[domain_sides.north]
            #values[1, domain_sides.south] = self.units.convert_units(0.0, "Pa") * bg.cell_volumes[domain_sides.south]
            
            return values.ravel("F") #FIX "F" deleted
        else:
            return np.zeros((self.nd, bg.num_cells)).ravel("F")

class InitialConditionsU2:
    """Zero initial conditions for u2."""
    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()
        for sd in self.mdg.subdomains(dim=self.nd):
            self.equation_system.set_variable_values(np.zeros(sd.num_cells * self.nd), [self.displacement2([sd])], iterate_index=0)
            # FIX #8: Initialize body_force in data dict for TimeDependentDenseArray
            sd_data = self.mdg.subdomain_data(sd)
            bf_zeros = np.zeros(sd.num_cells * self.nd)
            pp.set_solution_values("body_force", bf_zeros, sd_data, iterate_index=0)
            pp.set_solution_values("body_force", bf_zeros, sd_data, time_step_index=0)
        for intf in self.mdg.interfaces(dim=self.nd - 1, codim=1):
            self.equation_system.set_variable_values(np.zeros(intf.num_cells * self.nd), [self.interface_displacement2([intf])], iterate_index=0)

class SolutionStrategyU2:
    """MPSA setup for u2."""
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self.displacement2_variable = "u2"
        self.interface_displacement2_variable = "u2_interface"
        self.stress2_keyword = "mechanics2"
    def update_discretization_parameters(self) -> None:
        super().update_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(sd, data, self.stress2_keyword, {"bc": self.bc_type_mechanics(sd), "fourth_order_tensor": self.stiffness_tensor2(sd)})

class BodyForceMixin:
    """MMS body force f(x,t) — no gravity, only fx from manufactured solution."""
    solid: ViscoelasticSolidConstants
    units: pp.Units

    def _compute_body_force_values(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Compute MMS body force values at the current time step."""
        vals = []


        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))
            if sd.dim == 2:

                data[:, 0] = 0.0 * sd.cell_volumes  # fx for all cells
                data[:, 1] = 0.0 * sd.cell_volumes  # fy for all cells
            vals.append(data.ravel()) #FIX2 "F" deleted# FIX #8: Must be F-order [x0,x1... y0,y1...]
        return np.concatenate(vals)

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return body force as TimeDependentDenseArray (reads from data dict)."""
        # FIX #8: Use TimeDependentDenseArray so values update each time step
        self._bf_subdomains = subdomains  # save for later updates
        return pp.ad.TimeDependentDenseArray("body_force", subdomains)

# =============================================================================
# 6. Final Model
# =============================================================================
class ViscoelasticMomentumBalance(GeometryMixin, BoundaryConditionsMixin, BodyForceMixin, RateEquation, VariablesU2, ConstitutiveLawsU2, InitialConditionsU2, SolutionStrategyU2, pp.MomentumBalance):
    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.stress2_keyword = "mechanics2"
        self.history_u = []
        self.history_u2 = []
        # Strain history at monitoring point
        self.strain_history = {
            'times': [],
            'exx_u': [], 'eyy_u': [], 'exy_u': [],
            'exx_u2': [], 'eyy_u2': [], 'exy_u2': [],
        }
        self._adj_csr = None
        self._monitor_cell = None

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        u_val = np.array(self.equation_system.evaluate(self.displacement(self.mdg.subdomains(dim=self.nd)))).ravel()
        u2_val = np.array(self.equation_system.evaluate(self.displacement2(self.mdg.subdomains(dim=self.nd)))).ravel()
        self.history_u.append(u_val)
        self.history_u2.append(u2_val)

    def before_nonlinear_loop(self) -> None:
        """FIX #8: Update body force values and GL sums each time step."""
        self._update_gl_sums()
        super().before_nonlinear_loop()
        if hasattr(self, '_bf_subdomains'):
            new_vals = self._compute_body_force_values(self._bf_subdomains)
            # Write updated values to data dictionary for TimeDependentDenseArray
            offset = 0
            for sd in self._bf_subdomains:
                sd_data = self.mdg.subdomain_data(sd)
                n = sd.num_cells * self.nd
                pp.set_solution_values("body_force", new_vals[offset:offset+n], sd_data, iterate_index=0)
                pp.set_solution_values("body_force", new_vals[offset:offset+n], sd_data, time_step_index=0)
                offset += n
            # Debug: verify body force pipeline
            if self.time_manager.time_index % 100 == 0:
                # 1) Computed values
                print(f"  [DEBUG] t={self.time_manager.time/pp.DAY:.1f}d, max|bf_computed|={np.max(np.abs(new_vals)):.4e}")
                # 2) Read back from data dict
                sd = self._bf_subdomains[0]
                readback = pp.get_solution_values("body_force", self.mdg.subdomain_data(sd), iterate_index=0)
                print(f"  [DEBUG] max|bf_readback|={np.max(np.abs(readback)):.4e}, len={len(readback)}")
                # 3) Check what the AD operator evaluates to
                bf_op = self.body_force(self._bf_subdomains)
                bf_eval = bf_op.value(self.equation_system)
                print(f"  [DEBUG] max|bf_eval|={np.max(np.abs(bf_eval)):.4e}, type={type(bf_op).__name__}")
    
    def stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Sum of elastic and Maxwell stress branches."""
        if all(isinstance(d, pp.Grid) and d.dim == self.nd for d in domains):
            return self.mechanical_stress(domains) + self.mechanical_stress2(domains)
        return super().stress(domains)

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()
        self.update_boundary_condition(self.stress2_keyword, self.bc_values_stress2)
    
    def update_boundary_values_primary_variables(self) -> None:
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(self.displacement2_variable, self.bc_values_displacement2)

    def bc_values_displacement2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

    def bc_values_stress2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero Neumann BC for the viscous branch — load is applied only via u."""
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

# =============================================================================nu = 0.3
    #ps_factor = 0.918  # precisely tuned to hit 0.140% asymptote
    E1 = 2143000000.0 #* ps_factor   # E₁ = 2143 MPa (article) × correction
    E2 = 584000000.0 #* ps_factor    # E₂ = 584 MPa × correction
    eta = 180000000.0 #* ps_factor * 0.35  # η optimized to 35% for the best fit to the experimental knee
# 7. Run Script
# =============================================================================
if __name__ == "__main__":

    dt =  1 * pp.SECOND
    final_time = 8.0 * pp.HOUR
    time_manager = pp.TimeManager(
        schedule=[0.0, final_time],
        dt_init=dt,
        dt_min_max=(0.0 * pp.MINUTE, final_time),
    )
    
    # Plane strain correction: in 2D, εyy = σ*(1-ν²)/E, but article uses 1D: εyy = σ/E
    # Scale E by (1-ν²) so that 2D result matches 1D article values
    nu = 0.0#0.3
    #ps_factor = 0.918  # precisely tuned to hit 0.140% asymptote
    E1 = 2143000000.0 #* ps_factor   # E₁ = 2143 MPa (article) × correction
    E2 = 584000000.0 #* ps_factor    # E₂ = 584 MPa × correction
    eta = 180000000.0 #* ps_factor * 0.35  # η optimized to 35% for the best fit to the experimental knee

    solid_constants = ViscoelasticSolidConstants(
        # λ = E*ν/((1+ν)(1-2ν)), μ = E/(2*(1+ν))
        shear_modulus = E1 / (2.0 * (1.0 + nu)),
        shear_modulus2 = E2 / (2.0 * (1.0 + nu)),
        lame_lambda = E1 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        lame_lambda2 = E2 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        viscosity = eta * (60.0 * 60.0),  # convert MPa·h → Pa·s
        fracture_normal_stiffness = 200000000000.0,
        fracture_tangential_stiffness = 100000000000.0
    )
    
    model_params = {
        "material_constants": {"solid": solid_constants, "fluid": pp.FluidComponent()},
        "time_manager": time_manager,
        "plot_schedule": [pp.MINUTE * float(i) for i in range(0, 481, 60)],
    }

    class ShowCase(ViscoelasticMomentumBalance):
        def after_nonlinear_convergence(self) -> None:
            super().after_nonlinear_convergence()
            if self.time_manager.time_index == 0:
                print(f"--- Theoretical relaxation time: {self.solid.viscosity/self.solid.shear_modulus2/60.0:.2f} min ---")
            
            # --- DIAGNOSTIC: Print ux at center (0.4, 0.4) at end of simulation ---
            current_days = self.time_manager.time / pp.DAY
            # Retrieve numerical displacement at the center
            if len(self.mdg.subdomains(dim=self.nd)) > 0:
                sd = self.mdg.subdomains(dim=self.nd)[0]
                center_coord = np.array([[0.2], [0.2], [0.0]])
                diff = sd.cell_centers - center_coord
                dist = np.linalg.norm(diff, axis=0)
                center_cell = np.argmin(dist)

                # u_vec is interleaved [x0, y0, x1, y1, ...]
                u_vec = np.array(self.equation_system.evaluate(
                    self.displacement(self.mdg.subdomains(dim=self.nd)))).ravel()
                
                # Reshape to (2, N) where row 0 is ux, row 1 is uy
                u_reshaped = u_vec.reshape(self.nd, -1, order='F')
                ux_num = u_reshaped[0, :] #[0, center_cell]
                uy_num = u_reshaped[1, :] #[1, center_cell]

               

                print(f"SIMULATION TIME: {self.time_manager.time}")
                print(f"MAX ux_num: {np.max(np.abs(ux_num))}")
                print(f"MAX uy_num: {np.max(np.abs(uy_num))}")
                print("\n")
                print(print(f"MAX u_num: {np.max(np.abs(u_reshaped))}"))

                # --- Record average specimen strain (extensometer: uy_top / height) ---
                if not hasattr(self, '_top_cells') or self._top_cells is None:
                    # One-time initialization: find cells near the top boundary
                    self._domain_height = 0.1  # domain is 0.1 m tall
                    # Top cells: cells whose centers have y > 0.09 (top 10% of domain)
                    y_coords = sd.cell_centers[1, :]
                    y_threshold = 0.09  # top 10% of domain
                    self._top_cells = np.where(y_coords > y_threshold)[0]
                    if len(self._top_cells) == 0:
                        # Fallback: use top 20% of cells
                        y_threshold = 0.08
                        self._top_cells = np.where(y_coords > y_threshold)[0]
                    print(f"--- Strain measurement: extensometer (uy_top / height) ---")
                    print(f"--- Domain height: {self._domain_height:.4f} m ---")
                    print(f"--- Top cells (y > {y_threshold}): {len(self._top_cells)} ---")
                    print(f"--- Mean y of top cells: {np.mean(y_coords[self._top_cells]):.4f} ---")

                u2_vec = np.array(self.equation_system.evaluate(
                    self.displacement2(self.mdg.subdomains(dim=self.nd)))).ravel()

                # Reshape displacements: row 0 = ux, row 1 = uy
                u_2d = u_vec.reshape(self.nd, -1, order='F')
                u2_2d = u2_vec.reshape(self.nd, -1, order='F')

                # Extensometer: εyy = mean(uy at top cells) / height
                eyy_u_avg = np.mean(u_2d[1, self._top_cells]) / self._domain_height
                exx_u_avg = np.mean(u_2d[0, self._top_cells]) / self._domain_height
                exy_u_avg = 0.0
                # For u2:
                eyy_u2_avg = np.mean(u2_2d[1, self._top_cells]) / self._domain_height
                exx_u2_avg = np.mean(u2_2d[0, self._top_cells]) / self._domain_height
                exy_u2_avg = 0.0

                self.strain_history['times'].append(self.time_manager.time / 3600.0)
                self.strain_history['exx_u'].append(exx_u_avg)
                self.strain_history['eyy_u'].append(eyy_u_avg)
                self.strain_history['exy_u'].append(exy_u_avg)
                self.strain_history['exx_u2'].append(exx_u2_avg)
                self.strain_history['eyy_u2'].append(eyy_u2_avg)
                self.strain_history['exy_u2'].append(exy_u2_avg)

                                    
            
            sched = self.params.get('plot_schedule', [])
            if sched and self.time_manager.time >= sched[0]:
                sched.pop(0)
                mins = int(self.time_manager.time) # / 60.0)
                
                if not hasattr(self, '_vmax_u'):
                    u_all = self.equation_system.evaluate(self.displacement(self.mdg.subdomains(dim=self.nd)))
                    u2_all = self.equation_system.evaluate(self.displacement2(self.mdg.subdomains(dim=self.nd)))
                    u_mag = np.linalg.norm(u_all.reshape(self.nd, -1, order='F'), axis=0)
                    u2_mag = np.linalg.norm(u2_all.reshape(self.nd, -1, order='F'), axis=0)
                    self._vmax_u, self._vmax_u2 = np.max(u_mag) * 1, np.max(u2_mag)
                    print(f"--- Fixed VMAX: u={self._vmax_u:.2f}, u2={self._vmax_u2:.2f} ---")

                for var_name, name, vmax in [(self.displacement_variable, "u", self._vmax_u), (self.displacement2_variable, "u2", self._vmax_u2)]:
                    for sd, sd_data in self.mdg.subdomains(return_data=True, dim=self.nd):
                        # Get magnitude explicitly
                        vals = pp.get_solution_values(name=var_name, data=sd_data, time_step_index=0)
                        mag = np.linalg.norm(vals.reshape(self.nd, -1, order='F'), axis=0)
                        
                        plt.close('all')
                        pp.plot_grid(sd, cell_value=mag, title=f"{name} at {mins} s", if_plot=False, color_map="viridis", color_map_limits=[0.0, vmax], plot_2d=True)
                        fig = plt.gcf()
                        fig.axes[-1].set_ylabel("u [m]")
                        plt.savefig(f"displacement_with_fracture_{name}_{mins}.png", dpi=200) #_ne

                        

    model = ShowCase(model_params)
    pp.run_time_dependent_model(model)
    print("Done.")

    # ==========================================================================
    # 8. Strain vs. Time Plots
    # ==========================================================================
    if len(model.strain_history['times']) > 0:
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "lines.linewidth": 1.5,
            "lines.markersize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

        t = np.array(model.strain_history['times'])
        me = max(1, len(t) // 20)  # marker spacing

        # --- Single-panel: |epsilon_yy| vs time — matching article Figure 3 ---
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        eyy_u = np.array(model.strain_history['eyy_u'])
        eyy_u2 = np.array(model.strain_history['eyy_u2'])

        # ----- Experimental data from article (digitized from Figure 3) -----
        t_exp = np.array([
            0.0, 0.15, 0.30, 0.45, 0.70, 1.00, 1.25, 1.50, 1.80, 2.10,
            2.35, 2.60, 2.85, 3.10, 3.35, 3.60, 3.85, 4.10, 4.35, 4.60,
            4.85, 5.10, 5.35, 5.60, 5.85, 6.10, 6.35, 6.60, 6.85, 7.10, 
            7.35, 7.60
        ])
        e_exp = np.array([
            0.1115, 0.1180, 0.1280, 0.1340, 0.1375, 0.1380, 0.1390, 0.1390, 0.1400, 0.1400,
            0.1395, 0.1405, 0.1405, 0.1400, 0.1390, 0.1395, 0.1390, 0.1390, 0.1395, 0.1395,
            0.1390, 0.1395, 0.1390, 0.1390, 0.1390, 0.1395, 0.1400, 0.1410, 0.1400, 0.1395,
            0.1400, 0.1400
        ])

        # ----- 1D analytical simulation (digitized from article's red curve) -----
        t_1d_art = np.array([
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0
        ])
        e_1d_art = np.array([
            0.1110, 0.1155, 0.1205, 0.1245, 0.1280, 0.1310, 0.1330, 0.1355, 0.1370, 0.1380, 0.1390, 0.1395, 0.1398, 0.1400, 0.1400, 0.1400, 0.1400
        ])
        from scipy.interpolate import PchipInterpolator
        interp_1d = PchipInterpolator(t_1d_art, e_1d_art)
        t_1d = np.linspace(0, 8, 200)
        eps_1d_pct = interp_1d(t_1d)

        # ----- Plot all three -----
        ax1.plot(t_exp, e_exp, 'bD', markersize=6,
                 label='Experimental data')
        ax1.plot(t_1d, eps_1d_pct, 'k--', linewidth=1.5,
                 label='1D simulation (article)')
        ax1.plot(t, np.abs(eyy_u) * 100, 'r-', linewidth=1.5,
                 label='2D simulation (PorePy)')
        ax1.set_xlabel('Time (h)', fontsize=13)
        ax1.set_ylabel('Strain (%)', fontsize=13)
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0.09, 0.15)
        ax1.legend(framealpha=1.0, edgecolor='black', fancybox=False, loc='lower right')
        ax1.grid(True, alpha=0.3)
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
        ax1.tick_params(width=1.5, direction='in', top=True, right=True)
        fig1.tight_layout()
        fig1.savefig('strain_eyy_vs_time_with_frac.png', dpi=300) #with
        plt.close(fig1)
        print("Saved strain_eyy_vs_time_with_frac.png") #with

        # --- Three-panel: all strain components ---
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        components = [
            ('exx', r'$\varepsilon_{xx}$'),
            ('eyy', r'$\varepsilon_{yy}$'),
            ('exy', r'$\varepsilon_{xy}$'),
        ]
        for ax, (comp, label) in zip(axes, components):
            vals_u = np.array(model.strain_history[f'{comp}_u'])
            vals_u2 = np.array(model.strain_history[f'{comp}_u2'])
            ax.plot(t, vals_u * 100, 'b-o', markevery=me, markersize=4,
                    linewidth=1.5, label=r'$\varepsilon(u)$ total')
            ax.plot(t, vals_u2 * 100, 'r-s', markevery=me, markersize=4,
                    linewidth=1.5, label=r'$\varepsilon(u_2)$ viscous')
            ax.set_xlabel(r'$t$ (h)')
            ax.set_ylabel(f'{label} (%)')
            ax.legend(fontsize=10, framealpha=1.0, edgecolor='black', fancybox=False)
            ax.grid(True, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
            ax.tick_params(direction='in', top=True, right=True)
        fig2.suptitle('Strain components at monitoring point', fontsize=14, y=1.02)
        fig2.tight_layout()
        fig2.savefig('strain_components_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("Saved strain_components_vs_time.png")
