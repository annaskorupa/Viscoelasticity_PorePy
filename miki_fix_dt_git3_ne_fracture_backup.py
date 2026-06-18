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
    })
    lame_lambda2: float = 1.0
    shear_modulus2: float = 1.0
    viscosity: float = 1.0

# =============================================================================
# 2. Geometry
# =============================================================================
class GeometryMixin:
    """2D square domain with simplex mesh."""
    units: pp.Units
    def set_domain(self) -> None:
        size = self.units.convert_units(0.1, "m")
        self._domain = nd_cube_domain(2, size)
    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.units.convert_units(
            np.array([[0.04, 0.06], [0.04, 0.06]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]    
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")
    def meshing_arguments(self) -> dict:
        return {"cell_size": self.params.get("cell_size", 0.005)}

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
    """Implementation of u2_dot + beta*u2 - u_dot = 0."""
    def set_equations(self) -> None:
        super().set_equations()
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        u = self.displacement(matrix_subdomains)
        u2 = self.displacement2(matrix_subdomains)
        beta = self.beta(matrix_subdomains)
        
        # d/dt(u2) + beta*u2 - d/dt(u) = 0
        eq = pp.ad.dt(u2, self.ad_time_step) + beta * u2 - pp.ad.dt(u, self.ad_time_step)
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
            values[1, domain_sides.north] = self.units.convert_units(-3000000.0, "Pa") * bg.cell_volumes[domain_sides.north]
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

    def before_nonlinear_loop(self) -> None:
        """FIX #8: Update body force values in data dictionary each time step."""
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

# =============================================================================
# 7. Run Script
# =============================================================================
if __name__ == "__main__":

    dt =  1 * pp.SECOND
    final_time = 5.0 * pp.HOUR
    time_manager = pp.TimeManager(
        schedule=[0.0, final_time],
        dt_init=dt,
        dt_min_max=(0.0 * pp.MINUTE, final_time),
    )
    
    solid_constants = ViscoelasticSolidConstants(
        # E₁ = 2143.0 MPa, E₂ = 584.0 MPa , ν = 0.3 → λ = E*v/((1+v)(1-2v)), μ = E/(2*(1+v))
        shear_modulus = 2143000000.0 / (2.0 * 1.3),     
        shear_modulus2 = 584000000.0 / (2.0 * 1.3),    
        lame_lambda = 2143000000.0 * 0.3 /(1.3 * (1 - 2 * 0.3)),
        lame_lambda2 = 584000000.0 * 0.3 /(1.3 * (1 - 2 * 0.3)),
        viscosity=180000000.0 * (60.0 * 60.0), #180.0 MPa*h,
        fracture_normal_stiffness = 200000000000.0,
        fracture_tangential_stiffness = 100000000000.0
    )
    
    model_params = {
        "material_constants": {"solid": solid_constants, "fluid": pp.FluidComponent()},
        "time_manager": time_manager,
        "plot_schedule": [pp.MINUTE * float(i) for i in range(0, 301, 60)],
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
                        pp.plot_grid(sd, cell_value=mag, title=f"{name} at {mins} s", if_plot=False, color_map_limits=[0.0, vmax], plot_2d=True)
                        plt.savefig(f"displacement_with_fractures_{name}_{mins}.png", dpi=200)

                        

    model = ShowCase(model_params)
    pp.run_time_dependent_model(model)
    print("Done.")
