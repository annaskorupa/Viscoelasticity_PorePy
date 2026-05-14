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
        size = self.units.convert_units(0.8, "m")
        self._domain = nd_cube_domain(2, size)
    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")
    def meshing_arguments(self) -> dict:
        return {"cell_size": self.params.get("cell_size", 0.125)} #other:0.1 0.05, 0.025, 0.0125

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
        eq = pp.ad.dt(u2, self.time_manager.dt) + beta * u2 - pp.ad.dt(u, self.time_manager.dt)
        eq.set_name("rate_equation")
        self.equation_system.set_equation(eq, matrix_subdomains, {"cells": self.nd})

# =============================================================================
# 5. Infrastructure Mixins
# =============================================================================
class BoundaryConditionsMixin:
    """MMS BCs: East/West ux=uy=0 (Dir), North/South uy=0 (Dir) + ux free (Neu)."""
    units: pp.Units
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        domain_sides = self.domain_boundary_sides(sd)
        # FIX #1/#2: East/West → full Dirichlet (ux=0, uy=0)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.west + domain_sides.east + domain_sides.north + domain_sides.south, "dir")
        # North/South → Dirichlet ONLY for uy (component [1]), ux stays Neumann (free)
        #bc.is_dir[1, domain_sides.north] = True
        #bc.is_neu[1, domain_sides.north] = False
        #bc.is_dir[1, domain_sides.south] = True
        #bc.is_neu[1, domain_sides.south] = False
        return bc
    #def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # FIX #1: np.zeros instead of np.ones — all Dirichlet values = 0
        #return np.zeros((self.nd, bg.num_cells)).ravel("F")
    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        cc = bg.cell_centers
        A_MMS = 1.0e-3
        Ax = A_MMS
        Ay = A_MMS
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        t = self.time_manager.time
        L = 0.8  # domain length [m]
        Lx = L
        Ly = L

        ux = Ax * np.sin(np.pi * cc[0] / Lx) * np.sin(np.pi * cc[1] / Ly) * (1.0 - np.exp(-b_MMS * t))
        uy = Ay * np.sin(np.pi * cc[0] / Lx) * np.sin(np.pi * cc[1] / Ly) * (1.0 - np.exp(-b_MMS * t))

        data = np.zeros((self.nd, bg.num_cells))
        data[0, :] = ux
        data[1, :] = uy
        return data.ravel()
    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # FIX #6: Explicit zero traction on Neumann faces (ux on N/S)
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
        A_MMS = 1.0e-3
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        t = self.time_manager.time
        beta = self.solid.shear_modulus2 / self.solid.viscosity
        E1 = 2.0 * self.solid.shear_modulus    # ν=0.0 → E = 2μ
        E2 = 2.0 * self.solid.shear_modulus2
        k = E1 / 3.0 # Bulk modulus for ν=0.0 is K = E/3
        k2 = E2 / 3.0
        shear_modulus = self.solid.shear_modulus
        shear_modulus2 = self.solid.shear_modulus2
        lame_lambda = self.solid.lame_lambda
        lame_lambda2 = self.solid.lame_lambda2
        L = 0.8  # domain length [m]
        Lx = L
        Ly = L
        kx = np.pi / Lx
        ky = np.pi / Ly
        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))
            if sd.dim == 2:
                cc = sd.cell_centers
                sx = np.sin(kx * x)
                cx = np.cos(kx * x)
                sy = np.sin(ky * y)
                cy = np.cos(ky * y)

                sx = np.sin(kx * x)
                cx = np.cos(kx * x)
                sy = np.sin(ky * y)
                cy = np.cos(ky * y)

                T1 = 1.0 - np.exp(-b_MMS * t)
                T2 = (b_MMS / (beta - b_MMS)) * (np.exp(-b_MMS * t) - np.exp(-beta * t))

                term_x_E = (
                    # Grupa sin(kx)*cos(ky)
                    ( (lame_lambda + 2*shear_modulus)*kx**2*Ax + shear_modulus*ky**2*Ax ) * sx * sy 
                    # Grupa cos(kx)*cos(ky) - to jest ta brakująca część!
                    - ( (lame_lambda + shear_modulus)*kx*ky*Ay ) * cx * cy
                )

                term_x_E_vis = (
                                ( (k2 + 4.0/3.0*shear_modulus2)*kx**2*Ax + shear_modulus2*ky**2*Ax ) * sx * sy
                                - ( (k2 + 1.0/3.0*shear_modulus2)*kx*ky*Ay ) * cx * cy
                )

                term_y_E = (
                            #    Grupa sin(kx)*sin(ky) - GŁÓWNY NAPĘD UY
                            ( (lame_lambda + 2*shear_modulus)*ky**2*Ay + shear_modulus*kx**2*Ay ) * sx * sy
                            # Grupa cos(kx)*sin(ky)
                            - ( (lame_lambda + shear_modulus)*kx*ky*Ax ) * cx * cy
                )

                term_y_E_vis = (
                                ( (lame_lambda2 + 2*shear_modulus2)*ky**2*Ay + shear_modulus2*kx**2*Ay ) * sx * sy
                                - ( (lame_lambda2 + shear_modulus2)*kx*ky*Ax ) * cx * cy
                )

                force_x = term_x_E * T1 + term_x_E_vis * T2
                force_y = term_y_E * T1 + term_y_E_vis * T2

                
                # f(x,t) = A*(π/L)²*sin(πx/L)*[E₁*(1-e^{-bt}) + E₂*(b/(b-β))*(e^{-βt}-e^{-bt})]
                #force = A_MMS * (np.pi / L)**2 * np.sin(np.pi * cc[0] / L) * (
                #    E1 * (1.0 - np.exp(-b_MMS * t))
                #    + E2 * (b_MMS / (b_MMS - beta)) * (np.exp(-beta * t) - np.exp(-b_MMS * t))
                #)
                #data[:, 0] = force * sd.cell_volumes  # fx for all cells

                #force_x = (np.pi**2 * (3.0 * Lx * b_MMS * shear_modulus2 * (Ax * Lx * np.sin(np.pi * cc[0] / Lx) 
                #        - Ay * Ly * np.cos(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t)) * np.exp(t * (3.0 * b_MMS + beta)) 
                #        - 3.0 * Lx * shear_modulus * (1 - np.exp(b_MMS * t))*(b_MMS - beta) * (Ax * Lx * np.sin(np.pi * cc[0] / Lx) 
                #        - Ay * Ly * np.cos(np.pi * cc[0] / Lx)) * np.exp(t * (3.0 * b_MMS + 2.0 * beta)) + 2 * Ly * b_MMS * shear_modulus2
                #        * (2.0 * Ax * Ly * np.sin(np.pi * cc[0] /Lx)
                #        + Ay * Lx * np.cos(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t)) * np.exp(t * (3.0 * b_MMS + beta)) 
                #        - Ly * (1 - np.exp(b_MMS * t)) * (b_MMS - beta) * (3.0 * k *(Ax * Ly * np.sin(np.pi * cc[0] / Lx) - Ay * Lx * np.cos(np.pi * cc[0] / Lx))
                #        + 2.0 * shear_modulus * (2.0 * Ax * Ly * np.sin(np.pi * cc[0] / Lx) + Ay * Lx * np.cos(np.pi * cc[0] / Lx))) * np.exp(t * (3.0 * b_MMS + 2.0 * beta)))
                #        * np.exp(-2.0 * t * (2.0 * b_MMS + beta)) * np.cos(np.pi * cc[1] / Ly) / (3.0 * Lx**2 * Ly**2 * (b_MMS - beta))
                #)       

                #force_y = (np.pi**2 * (-2.0 * Lx * b_MMS * shear_modulus2* (Ax * Ly * np.cos(np.pi * cc[0] / Lx) 
                #            - 2.0 * Ay * Lx * np.sin(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t))
                #            * np.exp(t * (3.0 * b_MMS + beta)) - Lx * (1 - np.exp(b_MMS * t)) * (b_MMS - beta)
                #            * (3.0 * k * (Ax * Ly * np.cos(np.pi * cc[0] / Lx) + Ay * Lx * np.sin(np.pi * cc[0] / Lx)) 
                #            - 2.0 * shear_modulus * (Ax * Ly * np.cos(np.pi * cc[0] / Lx) - 2.0 * Ay * Lx *np.sin(np.pi * cc[0] / Lx)))
                #            * np.exp(t * (3.0 * b_MMS + 2.0 * beta)) + 3.0 * Ly * b_MMS * shear_modulus2 * (Ax * Lx * np.cos(np.pi * cc[0] / Lx) 
                #            + Ay * Ly * np.sin(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t)) * np.exp(t * (3.0 * b_MMS + beta)) 
                #            - 3.0 * Ly * shear_modulus * (1 - np.exp(b_MMS * t)) * (b_MMS - beta) * (Ax * Lx * np.cos(np.pi * cc[0] / Lx) 
                #            + Ay * Ly * np.sin(np.pi * cc[0] / Lx)) * np.exp(t * (3.0 * b_MMS + 2.0 * beta))) 
                #            * np.exp(-2.0 * t * (2 * b_MMS + beta)) * np.sin(np.pi * cc[1] / Ly) / (3.0 * Lx**2 * Ly**2 * (b_MMS - beta))

                #)


                
                data[:, 0] = force_x * sd.cell_volumes  # fx for all cells
                data[:, 1] = force_y * sd.cell_volumes  # fy for all cells
            
            vals.append(data.ravel()) #FIX2 "F" deleted # FIX #8: Must be F-order [x0,x1... y0,y1...]
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
        self.update_boundary_condition(self.stress2_keyword, self.bc_values_stress)
    
    def update_boundary_values_primary_variables(self) -> None:
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(self.displacement2_variable, self.bc_values_displacement2)

    def bc_values_displacement2(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.zeros((self.nd, bg.num_cells)).ravel("F")

# =============================================================================
# 7. Run Script
# =============================================================================
if __name__ == "__main__":
    dt = 1.0 * pp.SECOND #other 2, 4, 8
    final_time = 100.0 * pp.SECOND
    time_manager = pp.TimeManager(
        schedule=[0.0, final_time],
        dt_init=dt,
        dt_min_max=(0.0 * pp.MINUTE, final_time),
    )
    
    solid_constants = ViscoelasticSolidConstants(
        # E₁ = 22575.7 MPa, E₂ = 11000.0 MPa (= 11 GPa), ν = 0.0 → μ = E/2.0, λ = 0.0
        shear_modulus=22575700000.0 / 2.0,     # μ₁ = E₁/2.0 = 11,287,850,000.0 Pa
        shear_modulus2=11000000000.0 / 2.0,    # μ₂ = E₂/2.0 = 5,500,000,000.0 Pa
        lame_lambda=0.0,
        lame_lambda2=0.0,
        # FIX #4: η = μ₂ × τ (standard Maxwell: β = μ₂/η = 1.0/τ)
        # τ_relax = 45.454545 days = 3,927,273.0 s
        viscosity= 22575700000.0 * (45.454545 * 24.0 * 60.0 * 60.0)  #(11000000000.0 / 2.0) * (45.454545 * 24.0 * 60.0 * 60.0),
    )
    
    model_params = {
        "material_constants": {"solid": solid_constants, "fluid": pp.FluidComponent()},
        "time_manager": time_manager,
        "plot_schedule": [pp.MINUTE * float(i) for i in range(0, 301, 50)],
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
                center_coord = np.array([[0.395], [0.395], [0.0]])
                diff = sd.cell_centers - center_coord
                dist = np.linalg.norm(diff, axis=0)
                center_cell = np.argmin(dist)

                # u_vec is interleaved [x0, y0, x1, y1, ...]
                u_vec = np.array(self.equation_system.evaluate(
                    self.displacement(self.mdg.subdomains()))).ravel()
                
                # Reshape to (2, N) where row 0 is ux, row 1 is uy
                u_reshaped = u_vec.reshape(self.nd, -1, order='F')
                ux_num = u_reshaped[0, :] #[0, center_cell]
                uy_num = u_reshaped[1, :] #[1, center_cell]

                A_MMS, b_MMS = 1.0e-3, 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
                t_now = self.time_manager.time
                #ux_mms = A_MMS * np.sin(np.pi * 0.4 / 0.8) * (1.0 - np.exp(-b_MMS * t_now))
                cc = sd.cell_centers
                ux_mms = A_MMS * np.sin(np.pi * cc[0] / 0.8) * np.sin(np.pi * cc[1] / 0.8) * (1.0 - np.exp(-b_MMS * t_now))
                uy_mms = A_MMS * np.sin(np.pi * cc[0] / 0.8) * np.sin(np.pi * cc[1] / 0.8) * (1.0 - np.exp(-b_MMS * t_now))

                #Convergence calculation
                error_x = ux_num - ux_mms
                error_y = uy_num - uy_mms

                print(f"CZAS SYMULACJI: {self.time_manager.time}")
                print(f"MAX ux_num: {np.max(np.abs(ux_num))}")
                print(f"MAX ux_mms: {np.max(np.abs(ux_mms))}")
                print(f"DEBUG: diff_x_max = {np.max(np.abs(ux_num - ux_mms))}")
                print(f"DEBUG: diff_y_max = {np.max(np.abs(uy_num - uy_mms))}")

                abs_L2_x = np.sqrt(np.sum(error_x**2 * sd.cell_volumes))
                abs_L2_y = np.sqrt(np.sum(error_y**2 * sd.cell_volumes))
                rel_L2_x = np.sqrt(np.sum(error_x**2 * sd.cell_volumes)) / np.sqrt(np.sum(ux_mms**2 * sd.cell_volumes))
                rel_L2_y = np.sqrt(np.sum(error_y**2 * sd.cell_volumes)) / np.sqrt(np.sum(uy_mms**2 * sd.cell_volumes))

                abs_L2_total = np.sqrt(np.sum((error_x**2 + error_y**2) * sd.cell_volumes))
                rel_L2_total = np.sqrt(np.sum((error_x**2 + error_y**2) * sd.cell_volumes)) / np.sqrt(np.sum((ux_mms**2 + uy_mms**2) * sd.cell_volumes)

                ux_mms_p = A_MMS * np.sin(np.pi * 0.15 / 0.8) * np.sin(np.pi * 0.15 / 0.8) * (1.0 - np.exp(-b_MMS * t_now))
                uy_mms_p = A_MMS * np.sin(np.pi * 0.15 / 0.8) * np.sin(np.pi * 0.15 / 0.8) * (1.0 - np.exp(-b_MMS * t_now))
                
                if self.time_manager.time_index % 100 == 0:
                    print(f"\n{'='*60}")
                    print(f"  t = {current_days:.2f} days | center ({sd.cell_centers[0,center_cell]:.4f}, {sd.cell_centers[1,center_cell]:.4f})")
                    # print(f"  ux_num  = {ux_num:.6e} m")
                    # print(f"  ux_MMS  = {ux_mms_p:.6e} m")
                    # print(f"  uy_num  = {uy_num:.6e} m")
                    # print(f"  uy_MMS  = {uy_mms_p:.6e} m")
                    # print(f"  error x   = {abs(ux_num - ux_mms_p)/abs(ux_mms_p)*100:.6e} %")
                    # print(f"  error y   = {abs(uy_num - uy_mms_p)/abs(uy_mms_p)*100:.6e} %")
                    print(f"  ABsolute L2 error x = {abs_L2_x:.6e} m")
                    print(f"  ABsolute L2 error y = {abs_L2_y:.6e} m")
                    print(f"  ABsolute L2 total error = {abs_L2_total:.6e} m")
                    print(f"  Relative L2 error x = {rel_L2_x:.6e}")
                    print(f"  Relative L2 error y = {rel_L2_y:.6e}")
                    print(f"  Relative L2 total error = {rel_L2_total:.6e}")
                    print("============================================================\n")
            
            sched = self.params.get('plot_schedule', [])
            if sched and self.time_manager.time >= sched[0]:
                sched.pop(0)
                mins = int(self.time_manager.time / 60.0)
                
                if not hasattr(self, '_vmax_u'):
                    u_all = self.equation_system.evaluate(self.displacement(self.mdg.subdomains()))
                    u2_all = self.equation_system.evaluate(self.displacement2(self.mdg.subdomains()))
                    u_mag = np.linalg.norm(u_all.reshape(self.nd, -1, order='F'), axis=0)
                    u2_mag = np.linalg.norm(u2_all.reshape(self.nd, -1, order='F'), axis=0)
                    self._vmax_u, self._vmax_u2 = np.max(u_mag) * 2.5, np.max(u2_mag)
                    print(f"--- Fixed VMAX: u={self._vmax_u:.2f}, u2={self._vmax_u2:.2f} ---")

                for var_name, name, vmax in [(self.displacement_variable, "u", self._vmax_u), (self.displacement2_variable, "u2", self._vmax_u2)]:
                    for sd, sd_data in self.mdg.subdomains(return_data=True):
                        # Get magnitude explicitly
                        vals = pp.get_solution_values(name=var_name, data=sd_data, time_step_index=0)
                        mag = np.linalg.norm(vals.reshape(self.nd, -1, order='F'), axis=0)
                        
                        plt.close('all')
                        pp.plot_grid(sd, cell_value=mag, title=f"{name} at {mins} min", if_plot=False, color_map_limits=[0.0, vmax], plot_2d=True)
                        plt.savefig(f"displacement_{name}_{mins}.png", dpi=200)

    model = ShowCase(model_params)
    pp.run_time_dependent_model(model)
    print("Done.")
