"""Viscoelastic extension of PorePy's MomentumBalance model.

This script implements a generalized Maxwell model by adding a second 
displacement variable (u2) representing the viscous branch.
The total stress is σ = σ1(u) + σ2(u2), and the rate equation is
u2_dot + β*u2 - u_dot = 0.

Based on Idesman et al. (2000).
"""

import numpy as np
import porepy as pp
import sympy as sp
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
        return {"cell_size": self.params.get("cell_size", 0.025)} #other: 0.1, 0.05, 0.025, 0.0125

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
        # bc.is_dir[1, domain_sides.north] = True
        # bc.is_neu[1, domain_sides.north] = False
        # bc.is_dir[1, domain_sides.south] = True
        # bc.is_neu[1, domain_sides.south] = False
        return bc
    # def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
    #     # FIX #1: np.zeros instead of np.ones — all Dirichlet values = 0
    #     return np.zeros((self.nd, bg.num_cells)).ravel("F")
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

        ux = Ax * np.sin(np.pi * cc[0] / Lx) * np.sin(np.pi * cc[1] / Ly) * (1.0 - np.exp(-b_MMS * t)) #t#
        uy = Ay * np.sin(np.pi * cc[0] / Lx) * np.sin(np.pi * cc[1] / Ly) * (1.0 - np.exp(-b_MMS * t)) #t#

        data = np.zeros((self.nd, bg.num_cells))
        data[0, :] = ux
        data[1, :] = uy
        return data.ravel()
    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # FIX #6: Explicit zero traction on Neumann faces (ux on N/S)
        return np.zeros((self.nd, bg.num_cells)).ravel() #FIX "F" deleted

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
        Ax = A_MMS
        Ay = A_MMS
        b_MMS = 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
        t = self.time_manager.time
        beta = self.solid.shear_modulus2 / self.solid.viscosity
        E1 = 2.0 * self.solid.shear_modulus    # ν=0.0 → E = 2μ
        E2 = 2.0 * self.solid.shear_modulus2
        k = 2.0 * self.solid.shear_modulus#E1 / 3.0  # Bulk modulus for ν=0.0 is K = E/3
        k2 = 2.0 * self.solid.shear_modulus2#E2 / 3.0
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
                x = cc[0]
                y = cc[1]

                sx = np.sin(kx * x)
                cx = np.cos(kx * x)
                sy = np.sin(ky * y)
                cy = np.cos(ky * y)

                T1 = 1.0 - np.exp(-b_MMS * t) #t#
                T2 = (b_MMS / (beta - b_MMS)) * (np.exp(-b_MMS * t) - np.exp(-beta * t)) #(1.0 / beta) * (1.0 - np.exp(-beta * t))#

                term_x_E = (
                            # Grupa sin(kx)*cos(ky)
                            ( (lame_lambda + 2*shear_modulus)*kx**2*Ax + shear_modulus*ky**2*Ax ) * sx * sy 
                            # Grupa cos(kx)*cos(ky) - to jest ta brakująca część!
                            - ( (lame_lambda + shear_modulus)*kx*ky*Ay ) * cx * cy
                )

                term_x_E_vis = (
                                ( (lame_lambda2 + 2*shear_modulus2)*kx**2*Ax + shear_modulus2*ky**2*Ax ) * sx * sy
                                - ( (lame_lambda2 + shear_modulus2)*kx*ky*Ay ) * cx * cy
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
                # force = A_MMS * (np.pi / L)**2 * np.sin(np.pi * cc[0] / L) * (
                #    E1 * (1.0 - np.exp(-b_MMS * t))
                #    + E2 * (b_MMS / (b_MMS - beta)) * (np.exp(-beta * t) - np.exp(-b_MMS * t))
                # )

                # force_x = (np.pi**2 * (3.0 * Lx * b_MMS * shear_modulus2 * (Ax * Lx * np.sin(np.pi * cc[0] / Lx) 
                #         - Ay * Ly * np.cos(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t)) * np.exp(t * (3.0 * b_MMS + beta)) 
                #         - 3.0 * Lx * shear_modulus * (1 - np.exp(b_MMS * t))*(b_MMS - beta) * (Ax * Lx * np.sin(np.pi * cc[0] / Lx) 
                #         - Ay * Ly * np.cos(np.pi * cc[0] / Lx)) * np.exp(t * (3.0 * b_MMS + 2.0 * beta)) + 2 * Ly * b_MMS * shear_modulus2
                #         * (2.0 * Ax * Ly * np.sin(np.pi * cc[0] /Lx)
                #         + Ay * Lx * np.cos(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t)) * np.exp(t * (3.0 * b_MMS + beta)) 
                #         - Ly * (1 - np.exp(b_MMS * t)) * (b_MMS - beta) * (3.0 * k *(Ax * Ly * np.sin(np.pi * cc[0] / Lx) - Ay * Lx * np.cos(np.pi * cc[0] / Lx))
                #         + 2.0 * shear_modulus * (2.0 * Ax * Ly * np.sin(np.pi * cc[0] / Lx) + Ay * Lx * np.cos(np.pi * cc[0] / Lx))) * np.exp(t * (3.0 * b_MMS + 2.0 * beta)))
                #         * np.exp(-2.0 * t * (2.0 * b_MMS + beta)) * np.cos(np.pi * cc[1] / Ly) / (3.0 * Lx**2 * Ly**2 * (b_MMS - beta))
                # )       

                # force_y = (np.pi**2 * (-2.0 * Lx * b_MMS * shear_modulus2* (Ax * Ly * np.cos(np.pi * cc[0] / Lx) 
                #             - 2.0 * Ay * Lx * np.sin(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t))
                #             * np.exp(t * (3.0 * b_MMS + beta)) - Lx * (1 - np.exp(b_MMS * t)) * (b_MMS - beta)
                #             * (3.0 * k * (Ax * Ly * np.cos(np.pi * cc[0] / Lx) + Ay * Lx * np.sin(np.pi * cc[0] / Lx)) 
                #             - 2.0 * shear_modulus * (Ax * Ly * np.cos(np.pi * cc[0] / Lx) - 2.0 * Ay * Lx *np.sin(np.pi * cc[0] / Lx)))
                #             * np.exp(t * (3.0 * b_MMS + 2.0 * beta)) + 3.0 * Ly * b_MMS * shear_modulus2 * (Ax * Lx * np.cos(np.pi * cc[0] / Lx) 
                #             + Ay * Ly * np.sin(np.pi * cc[0] / Lx)) * (np.exp(b_MMS * t) - np.exp(beta * t)) * np.exp(t * (3.0 * b_MMS + beta)) 
                #             - 3.0 * Ly * shear_modulus * (1 - np.exp(b_MMS * t)) * (b_MMS - beta) * (Ax * Lx * np.cos(np.pi * cc[0] / Lx) 
                #             + Ay * Ly * np.sin(np.pi * cc[0] / Lx)) * np.exp(t * (3.0 * b_MMS + 2.0 * beta))) 
                #             * np.exp(-2.0 * t * (2 * b_MMS + beta)) * np.sin(np.pi * cc[1] / Ly) / (3.0 * Lx**2 * Ly**2 * (b_MMS - beta))

                # )
                data[:, 0] = force_x * sd.cell_volumes  # fx for all cells
                data[:, 1] = force_y * sd.cell_volumes  # fy for all cells
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

    dt =  2 * pp.SECOND #other: 4 * pp.SECOND, 2 * pp.SECOND, 1 * pp.SECOND
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
        viscosity=11000000000.0 * (45.454545 * 24.0 * 60.0 * 60.0) #(11000000000.0 / 2.0) * (45.454545 * 24.0 * 60.0 * 60.0),
    )
    
    model_params = {
        "material_constants": {"solid": solid_constants, "fluid": pp.FluidComponent()},
        "time_manager": time_manager,
        "plot_schedule": [pp.SECOND * float(i) for i in range(0, 101, 100)],
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
                    self.displacement(self.mdg.subdomains()))).ravel()
                
                # Reshape to (2, N) where row 0 is ux, row 1 is uy
                u_reshaped = u_vec.reshape(self.nd, -1, order='F')
                ux_num = u_reshaped[0, :] #[0, center_cell]
                uy_num = u_reshaped[1, :] #[1, center_cell]

                A_MMS, b_MMS = 1.0e-3, 0.5 * (self.solid.shear_modulus2 / self.solid.viscosity)
                t_now = self.time_manager.time
                cc = sd.cell_centers
                ux_mms = A_MMS * np.sin(np.pi * cc[0] / 0.8) * np.sin(np.pi * cc[1] / 0.8) * (1.0 - np.exp(-b_MMS * t_now)) #t_now#
                uy_mms = A_MMS * np.sin(np.pi * cc[0] / 0.8) * np.sin(np.pi * cc[1] / 0.8) * (1.0 - np.exp(-b_MMS * t_now)) #t_now#

                #Convergence calculation
                error_x = ux_num - ux_mms
                error_y = uy_num - uy_mms

                print(f"CZAS SYMULACJI: {self.time_manager.time}")
                print(f"MAX ux_num: {np.max(np.abs(ux_num))}")
                print(f"MAX ux_mms: {np.max(np.abs(ux_mms))}")
                print(f"DEBUG: diff_x_max = {np.max(np.abs(ux_num - ux_mms))}")
                print(f"DEBUG: diff_y_max = {np.max(np.abs(uy_num - uy_mms))}")
                print("\n")

                abs_L2_x = np.sqrt(np.sum(error_x**2 * sd.cell_volumes))
                abs_L2_y = np.sqrt(np.sum(error_y**2 * sd.cell_volumes))
                rel_L2_x = np.sqrt(np.sum(error_x**2 * sd.cell_volumes)) / np.sqrt(np.sum(ux_mms**2 * sd.cell_volumes))
                rel_L2_y = np.sqrt(np.sum(error_y**2 * sd.cell_volumes)) / np.sqrt(np.sum(uy_mms**2 * sd.cell_volumes))

                abs_L2_total = np.sqrt(np.sum((error_x**2 + error_y**2) * sd.cell_volumes))
                rel_L2_total = np.sqrt(np.sum((error_x**2 + error_y**2) * sd.cell_volumes)) / np.sqrt(np.sum((ux_mms**2 + uy_mms**2) * sd.cell_volumes))

                ux_mms_p = A_MMS * np.sin(np.pi * 0.15 / 0.8) * np.sin(np.pi * 0.15 / 0.8) * (1.0 - np.exp(-b_MMS * t_now)) #t_now#
                uy_mms_p = A_MMS * np.sin(np.pi * 0.15 / 0.8) * np.sin(np.pi * 0.15 / 0.8) * (1.0 - np.exp(-b_MMS * t_now)) #t_now#

                print(f"  ABsolute L2 error x = {abs_L2_x:.6e} m")
                print(f"  ABsolute L2 error y = {abs_L2_y:.6e} m")
                print(f"  ABsolute L2 total error = {abs_L2_total:.6e} m")
                print(f"  Relative L2 error x = {rel_L2_x:.6e}")
                print(f"  Relative L2 error y = {rel_L2_y:.6e}")
                print(f"  Relative L2 total error = {rel_L2_total:.6e}")
                print("============================================================\n")

                if self.time_manager.time_index % 1 == 0:
                    print(f"\n{'='*60}")
                    print(f"  t = {current_days:.2f} days | center ({sd.cell_centers[0,center_cell]:.4f}, {sd.cell_centers[1,center_cell]:.4f})")
                    # print(f"  ux_num  = {ux_num:.6e} m")
                    # print(f"  ux_MMS  = {ux_mms_p:.6e} m")
                    # print(f"  uy_num  = {uy_num:.6e} m")
                    # print(f"  uy_MMS  = {uy_mms_p:.6e} m")
                    # print(f"  error x   = {abs(ux_num - ux_mms_p)/abs(ux_mms_p)*100:.6e} %")
                    # prin=t(f"  error y   = {abs(uy_num - uy_mms_p)/abs(uy_mms_p)*100:.6e} %")
                    print(f"  ABsolute L2 error x = {abs_L2_x:.6e} m")
                    print(f"  ABsolute L2 error y = {abs_L2_y:.6e} m")
                    print(f"  ABsolute L2 total error = {abs_L2_total:.6e} m")
                    print(f"  Relative L2 error x = {rel_L2_x:.6e}")
                    print(f"  Relative L2 error y = {rel_L2_y:.6e}")
                    print(f"  Relative L2 total error = {rel_L2_total:.6e}")
                    print("============================================================\n")
                    print(f"CZAS SYMULACJI: {self.time_manager.time}")
                    print(f"MAX ux_num: {np.max(np.abs(ux_num))}")
                    print(f"MAX ux_mms: {np.max(np.abs(ux_mms))}")
                    print(f"MAX uy_num: {np.max(np.abs(uy_num))}")
                    print(f"MAX uy_mms: {np.max(np.abs(uy_mms))}")
                    print(f"MAX u_num: {np.max(np.abs(u_reshaped))}")
                    print(f"DEBUG: diff_x_max = {np.max(np.abs(ux_num - ux_mms))}")
                    print(f"DEBUG: diff_y_max = {np.max(np.abs(uy_num - uy_mms))}")
                    print("\n")


            # TRACTION CALCULATION!-----
            # numerical traction t = σ·n, where σ is stress tensor and n is unit normal vector        
            # sigma_vec is interleaved [x0, y0, x1, y1, ...]
            sigma_vec = np.array(self.equation_system.evaluate(
                self.stress(self.mdg.subdomains()))).ravel()
                            
            # Reshape to (2, N) where row 0 is ux, row 1 is uy
            traction_num = sigma_vec.reshape((sd.dim, sd.num_faces), order='F') / sd.face_areas
            # sigma_reshaped = sigma_vec.reshape((3, sd.num_cells), order='F')
            # sigmaxx_num = sigma_reshaped[0, :] #[0, center_cell]
            # sigmayy_num = sigma_reshaped[1, :] #[1, center_cell]
            # sigmaxy_num = sigma_reshaped[2, :] #[2, center_cell]

            # cell_faces_map = np.abs(sd.cell_faces)
            # cells_per_face = np.array(cell_faces_map.sum(axis=1)).flatten()

            # sigma_xx_f = (cell_faces_map @ sigmaxx_num) / cells_per_face
            # sigma_yy_f = (cell_faces_map @ sigmayy_num) / cells_per_face
            # sigma_xy_f = (cell_faces_map @ sigmaxy_num) / cells_per_face

            n_unit = sd.face_normals / sd.face_areas
            # # traction = np.einsum('ijf, jf -> if', sigma_vec, n_unit) 
            # # t_x = sigma_vec[0,:] * n_unit[0, :] + sigma_vec[2,:] * n_unit[1, :]
            # # t_y = sigma_vec[2,:] * n_unit[0, :] + sigma_vec[1,:] * n_unit[1, :]
            # # traction = np.vstack([t_x, t_y])
            
            # t_num_x = sigma_xx_f * n_unit[0, :] + sigma_xy_f * n_unit[1, :]
            # t_num_y = sigma_xy_f * n_unit[0, :] + sigma_yy_f * n_unit[1, :]

            # traction_num = np.vstack([t_num_x, t_num_y])
            print(f"MAX traction_num: {np.max(np.abs(traction_num))}")
    

            # # analitical traction t = σ·n, where σ is stress tensor and n is unit normal vector
            # x, y, t = sp.symbols('x y t')

            # b_MMS = 1.0
            
            # ux_mms_sym = A_MMS * sp.sin(sp.pi * x / 0.8) * sp.sin(sp.pi * y / 0.8) * (1.0 - sp.exp(-b_MMS * t)) #t_now#
            # uy_mms_sym = A_MMS * sp.sin(sp.pi * x / 0.8) * sp.sin(sp.pi * y / 0.8) * (1.0 - sp.exp(-b_MMS * t)) #t_now#

            # dux_dx = sp.diff(ux_mms_sym, x)  # d(ux)/dx
            # duy_dy = sp.diff(uy_mms_sym, y)  # d(uy)/dy
            # dux_dy = sp.diff(ux_mms_sym, y)  # d(ux)/dy
            # duy_dx = sp.diff(uy_mms_sym, x)  # d(uy)/dx

            # eps_xx = dux_dx
            # eps_yy = duy_dy
            # eps_xy = 0.5 * (dux_dy + duy_dx)

            # deps_xx_dt = sp.diff(eps_xx, t)
            # deps_yy_dt = sp.diff(eps_yy, t)
            # deps_xy_dt = sp.diff(eps_xy, t)

            # lame_lambda = self.solid.lame_lambda
            # shear_modulus = self.solid.shear_modulus
            # shear_modulus2 = self.solid.shear_modulus2
            # eta = self.solid.viscosity
            # tau = eta / shear_modulus2  # relaxation time
            # #lame_lambda_eff = (2 * lame_lambda * shear_modulus) / (lame_lambda + 2 * shear_modulus)
            # # tr_eps = eps_xx + eps_yy
            # # sig_xx_el = lame_lambda * tr_eps + 2 * shear_modulus * eps_xx
            # # sig_yy_el = lame_lambda * tr_eps + 2 * shear_modulus * eps_yy
            # # sig_xy_el = 2 * shear_modulus * eps_xy
            # tr_eps = eps_xx + eps_yy
            # sig_xx_el = 2 * shear_modulus * eps_xx
            # sig_yy_el = 2 * shear_modulus * eps_yy
            # sig_xy_el = 2 * shear_modulus * eps_xy

            # # Viscous damping/relaxation contribution
            # # tr_deps_dt = deps_xx_dt + deps_yy_dt
            # # sig_xx_visc = lame_lambda * tr_deps_dt + 2 * shear_modulus * deps_xx_dt
            # # sig_yy_visc = lame_lambda * tr_deps_dt + 2 * shear_modulus * deps_yy_dt
            # # sig_xy_visc = 2 * shear_modulus * deps_xy_dt
            # sig_xx_visc = 2 * shear_modulus2 * (deps_xx_dt / (1.0 / tau + b_MMS))
            # sig_yy_visc = 2 * shear_modulus2 * (deps_yy_dt / (1.0 / tau + b_MMS))
            # sig_xy_visc = 2 * shear_modulus2 * (deps_xy_dt / (1.0 / tau + b_MMS))

            # sig_xx = sig_xx_el + sig_xx_visc
            # sig_yy = sig_yy_el + sig_yy_visc
            # sig_xy = sig_xy_el + sig_xy_visc

            # f_mms_x = -(sp.diff(sig_xx, x) + sp.diff(sig_xy, y))
            # f_mms_y = -(sp.diff(sig_xy, x) + sp.diff(sig_yy, y))

            # t_now = self.time_manager.time

            # sig_xx_now = sig_xx.subs(t, t_now)
            # sig_yy_now = sig_yy.subs(t, t_now)
            # sig_xy_now = sig_xy.subs(t, t_now)

            # f_mms_x_now = f_mms_x.subs(t, t_now)
            # f_mms_y_now = f_mms_y.subs(t, t_now)

            
            # self.f_body_x_func = sp.lambdify((x, y), f_mms_x_now, 'numpy')
            # self.f_body_y_func = sp.lambdify((x, y), f_mms_y_now, 'numpy')

            # f_sxx = sp.lambdify((x, y), sig_xx_now, 'numpy')
            # f_syy = sp.lambdify((x, y), sig_yy_now, 'numpy')
            # f_sxy = sp.lambdify((x, y), sig_xy_now, 'numpy')

            # fx, fy = sd.face_centers[0, :], sd.face_centers[1, :]

            # s_xx_f = f_sxx(fx, fy)
            # s_yy_f = f_syy(fx, fy)
            # s_xy_f = f_sxy(fx, fy)

            # t_exact_x = s_xx_f * n_unit[0, :] + s_xy_f * n_unit[1, :]
            # t_exact_y = s_xy_f * n_unit[0, :] + s_yy_f * n_unit[1, :]

            # t_exact = np.vstack([t_exact_x, t_exact_y])

            # analitycal without sympy - gemini
    #         fx, fy = sd.face_centers[0, :], sd.face_centers[1, :]
    #         t_now = self.time_manager.time

    #         L = 0.8

    #         # dux_dx_f = A_MMS * (np.pi / 0.8) * np.cos(np.pi * fx / 0.8) * np.sin(np.pi * fy / 0.8) * spatial_factor
    #         # # du_y / dy
    #         # duy_dy_f = A_MMS * (np.pi / 0.8) * np.sin(np.pi * fx / 0.8) * np.cos(np.pi * fy / 0.8) * spatial_factor
    #         # # du_x / dy
    #         # dux_dy_f = A_MMS * (np.pi / 0.8) * np.sin(np.pi * fx / 0.8) * np.cos(np.pi * fy / 0.8) * spatial_factor
    #         # # du_y / dx
    #         # duy_dx_f = A_MMS * (np.pi / 0.8) * np.cos(np.pi * fx / 0.8) * np.sin(np.pi * fy / 0.8) * spatial_factor

    #         dux_dx_f = A_MMS * (np.pi / L) * np.cos(np.pi * fx / L) * np.sin(np.pi * fy / L)
    #         dux_dy_f = A_MMS * (np.pi / L) * np.sin(np.pi * fx / L) * np.cos(np.pi * fy / L)

    #         duy_dx_f = A_MMS * (np.pi / L) * np.cos(np.pi * fx / L) * np.sin(np.pi * fy / L)
    #         duy_dy_f = A_MMS * (np.pi / L) * np.sin(np.pi * fx / L) * np.cos(np.pi * fy / L)

    #         eps_xx_f = dux_dx_f
    #         eps_yy_f = duy_dy_f
    #         eps_xy_f = 0.5 * (dux_dy_f + duy_dx_f)

            
    #         tr_eps_f = eps_xx_f + eps_yy_f

    #         lame_lambda = self.solid.lame_lambda
    #         G1 = self.solid.shear_modulus
    #         G2 = self.solid.shear_modulus2
    #         tau_rel = self.solid.viscosity / G2

    #         elastic_time_factor = (1.0 - np.exp(-b_MMS * t_now))

    #         if np.isclose(b_MMS * tau_rel, 1.0):
    #             viscous_time_factor = b_MMS * t_now * np.exp(-b_MMS * t_now)
    #         else:
    #             viscous_time_factor = (b_MMS * tau_rel / (b_MMS * tau_rel - 1.0)) * (
    #                 np.exp(-t_now / tau_rel) - np.exp(-b_MMS * t_now)
    # )

    #         s_xx_el = (lame_lambda * tr_eps_f + 2 * G1 * eps_xx_f) * elastic_time_factor
    #         s_yy_el = (lame_lambda * tr_eps_f + 2 * G1 * eps_yy_f) * elastic_time_factor
    #         s_xy_el = (2 * G1 * eps_xy_f) * elastic_time_factor

    #         s_xx_visc = (2 * G2 * eps_xx_f) * viscous_time_factor
    #         s_yy_visc = (2 * G2 * eps_yy_f) * viscous_time_factor
    #         s_xy_visc = (2 * G2 * eps_xy_f) * viscous_time_factor
                        
    #         s_xx_f = s_xx_el + s_xx_visc
    #         s_yy_f = s_yy_el + s_yy_visc
    #         s_xy_f = s_xy_el + s_xy_visc

            #analitical traction without sympy - copilot
            #parameters
            lame_lambda = self.solid.lame_lambda
            G1 = self.solid.shear_modulus
            G2 = self.solid.shear_modulus2
            L= 0.8
            fx, fy = sd.face_centers[0, :], sd.face_centers[1, :]
            t_now = self.time_manager.time
            x = fx
            y = fy
            A_x = A_MMS
            A_y = A_MMS
            b = b_MMS
            Lx = L
            Ly = L
            dt = self.time_manager.dt
            t_now = self.time_manager.time
            beta = self.solid.shear_modulus2 / self.solid.viscosity

            n_steps = int(round(t_now / dt))

            fx, fy = sd.face_centers[0, :], sd.face_centers[1, :]
            t_now = self.time_manager.time

            L = 0.8
            A = A_MMS

            dt = self.time_manager.dt
            beta = G2 / self.solid.viscosity

            # ======================================================
            # MMS displacement time factors
            # ======================================================

            factor_u = 1.0 - np.exp(-b_MMS * t_now)

            # ======================================================
            # Discrete Backward Euler solution for u2
            # ======================================================

            n_steps = int(round(t_now / dt))

            factor_u2 = 0.0

            for k in range(1, n_steps + 1):

                t_k = k * dt
                t_km1 = (k - 1) * dt

                f_k = 1.0 - np.exp(-b_MMS * t_k)
                f_km1 = 1.0 - np.exp(-b_MMS * t_km1)

                factor_u2 = (
                    factor_u2 + (f_k - f_km1)
                ) / (1.0 + beta * dt)

            # ======================================================
            # Spatial derivatives of MMS shape function
            # ======================================================

            dphi_dx = (
                A
                * (np.pi / L)
                * np.cos(np.pi * fx / L)
                * np.sin(np.pi * fy / L)
            )

            dphi_dy = (
                A
                * (np.pi / L)
                * np.sin(np.pi * fx / L)
                * np.cos(np.pi * fy / L)
            )

            # ======================================================
            # Strains epsilon = grad_s(u)
            # ======================================================

            eps_xx = dphi_dx * factor_u
            eps_yy = dphi_dy * factor_u

            eps_xy = 0.5 * (
                dphi_dy * factor_u
                + dphi_dx * factor_u
            )

            # ======================================================
            # Strains epsilon2 = grad_s(u2)
            # ======================================================

            eps2_xx = dphi_dx * factor_u2
            eps2_yy = dphi_dy * factor_u2

            eps2_xy = 0.5 * (
                dphi_dy * factor_u2
                + dphi_dx * factor_u2
            )

            # ======================================================
            # Deviators
            # ======================================================

            tr_eps = eps_xx + eps_yy
            tr_eps2 = eps2_xx + eps2_yy

            eps_dev_xx = eps_xx - tr_eps / 3.0
            eps_dev_yy = eps_yy - tr_eps / 3.0
            eps_dev_xy = eps_xy

            eps2_dev_xx = eps2_xx - tr_eps2 / 3.0
            eps2_dev_yy = eps2_yy - tr_eps2 / 3.0
            eps2_dev_xy = eps2_xy

            # ======================================================
            # Material parameters
            # ======================================================

            G1 = self.solid.shear_modulus
            G2 = self.solid.shear_modulus2
            lam = self.solid.lame_lambda

            K = lam + 2.0 * G1 / 3.0

            # ======================================================
            # Stress tensor
            # ======================================================
            tr_eps = eps_xx + eps_yy

            # sigma_xx = (
            #     2.0 * G1 * eps_dev_xx
            #     + 2.0 * G2 * eps2_dev_xx
            #     + K * tr_eps
            # )

            # sigma_yy = (
            #     2.0 * G1 * eps_dev_yy
            #     + 2.0 * G2 * eps2_dev_yy
            #     + K * tr_eps
            # )

            # sigma_xy = (
            #     2.0 * G1 * eps_dev_xy
            #     + 2.0 * G2 * eps2_dev_xy
            # )

            sigma_xx = lam * tr_eps + 2.0 * G1 * eps_xx + 2.0 * G2 * eps2_xx
            sigma_yy = lam * tr_eps + 2.0 * G1 * eps_yy + 2.0 * G2 * eps2_yy
            sigma_xy = 2.0 * G1 * eps_xy + 2.0 * G2 * eps2_xy



            # MMS strains
            # eps_xx = A_MMS * (np.pi / L) * np.cos(np.pi * fx / L) * np.sin(np.pi * fy / L)
            # eps_yy = A_MMS * (np.pi / L) * np.sin(np.pi * fx / L) * np.cos(np.pi * fy / L)
            # eps_xy = 0.5 * (
            #     A_MMS * (np.pi / L) * np.sin(np.pi * fx / L) * np.cos(np.pi * fy / L)
            #     +
            #     A_MMS * (np.pi / L) * np.cos(np.pi * fx / L) * np.sin(np.pi * fy / L)
            # )

            

            # # time factors
            # elastic_factor = 1.0 - np.exp(-b_MMS * t_now)

            # p = G2 / self.solid.viscosity  # = 1/tau

            # if np.isclose(p, b_MMS):
            #     viscous_factor = b_MMS * t_now * np.exp(-b_MMS * t_now)
            # else:
            #     viscous_factor = (
            #         b_MMS / (p - b_MMS)
            #         * (np.exp(-b_MMS * t_now) - np.exp(-p * t_now))
            #     )

            # # total strain
            # tr_eps = (eps_xx + eps_yy) * elastic_factor

            # # deviatoric strain
            # eps_dev_xx = eps_xx - (eps_xx + eps_yy) / 3.0
            # eps_dev_yy = eps_yy - (eps_xx + eps_yy) / 3.0
            # eps_dev_xy = eps_xy

            # # deviatoric viscous strain
            # eps2_dev_xx = eps_dev_xx * viscous_factor
            # eps2_dev_yy = eps_dev_yy * viscous_factor
            # eps2_dev_xy = eps_dev_xy * viscous_factor

            # # elastic deviatoric strain
            # eps_dev_xx *= elastic_factor
            # eps_dev_yy *= elastic_factor
            # eps_dev_xy *= elastic_factor

            # # bulk modulus
            # K = lame_lambda + 2.0 * G1 / 3.0

            # # stresses
            # sigma_xx = (
            #     2.0 * G1 * eps_dev_xx
            #     + 2.0 * G2 * eps2_dev_xx
            #     + K * tr_eps
            # )

            # sigma_yy = (
            #     2.0 * G1 * eps_dev_yy
            #     + 2.0 * G2 * eps2_dev_yy
            #     + K * tr_eps
            # )

            # sigma_xy = (
            #     2.0 * G1 * eps_dev_xy
            #     + 2.0 * G2 * eps2_dev_xy
            # )

            t_exact_x = sigma_xx * n_unit[0, :] + sigma_xy * n_unit[1, :]
            t_exact_y = sigma_xy * n_unit[0, :] + sigma_yy * n_unit[1, :]

            t_exact = np.vstack([t_exact_x, t_exact_y])

            print(f"MAX traction_mms: {np.max(np.abs(t_exact))}")
            print("\n")

            error_s = t_exact - traction_num

            idx = np.argmax(
            np.linalg.norm(error_s, axis=0)
            )
            print("face =", idx)
            print("num =", traction_num[:, idx])
            print("exact =", t_exact[:, idx])
            print("normal =", n_unit[:, idx])
            print("face center =", fx[idx], fy[idx])

            cosang = np.sum(
                traction_num * t_exact,
                axis=0
            ) / (
                np.linalg.norm(traction_num, axis=0)
                * np.linalg.norm(t_exact, axis=0)
                + 1e-16
            )

            print("mean cos =", np.mean(cosang))
            print("min cos =", np.min(cosang))
            print("sum =", np.sum(cosang < 0))
            print("num_faces =", sd.num_faces)
            print("neg_cos =", np.sum(cosang < 0))

            bf = sd.get_boundary_faces()

            print("num boundary =", len(bf))
            print(idx in bf)
            cosang_b = cosang[bf]

            print(np.mean(cosang_b))
            print(np.min(cosang_b))

            ratio = (
                np.linalg.norm(traction_num[:, bf], axis=0)
                /
                (np.linalg.norm(t_exact[:, bf], axis=0) + 1e-15)
            )

            print("np.mean(ratio) =", np.mean(ratio))
            print("np.min(ratio) =", np.min(ratio))
            print("np.max(ratio) =", np.max(ratio))

 
            bf = sd.get_boundary_faces()

            error_b = error_s[:, bf]
            tex_b = t_exact[:, bf]

            abs_b = np.sqrt(
                np.sum(
                    np.sum(error_b**2, axis=0)
                    * sd.face_areas[bf]
                )
            )

            norm_b = np.sqrt(
                np.sum(
                    np.sum(tex_b**2, axis=0)
                    * sd.face_areas[bf]
                )
            )

            rel_b = abs_b / norm_b

            print("Boundary rel L2 =", rel_b)
            print("\n")

            err_x = np.sqrt(
                np.sum(error_s[0,:]**2 * sd.face_areas)
            )

            err_y = np.sqrt(
                np.sum(error_s[1,:]**2 * sd.face_areas)
            )

            print("diagnostic: err_x =", err_x)
            print("diagnostic: err_y =", err_y)
            print(sd.face_normals.shape)
            print(sd.dim)
            print(sigma_vec.shape)
            print("\n")

            area = sd.face_areas[idx]

            print("area =", area)
            print("traction =", traction_num[:, idx])

            force = sigma_vec.reshape((sd.dim, sd.num_faces), order="F")[:, idx]

            print("force =", force)
            print("force/area =", force / area)

            
            print(f"CZAS SYMULACJI: {self.time_manager.time}")
            print(f"DEBUG: diff_s_max = {np.max(np.abs(error_s))}")
            print("\n")
            
            abs_L2_s = np.sqrt(np.sum(np.sum(error_s**2, axis=0) * sd.face_areas))
            norm_t_exact_L2 = np.sqrt(np.sum(np.sum(t_exact**2, axis=0) * sd.face_areas))
            rel_L2_s = abs_L2_s / norm_t_exact_L2
            
            print(f"  ABsolute L2 total error for sigma = {abs_L2_s:.6e}")
            print(f"  Relative L2 total error for sigma = {rel_L2_s:.6e}")
            print("============================================================\n")
            
            if self.time_manager.time_index % 1 == 0:
                print(f"\n{'='*60}")
                print(f"  t = {current_days:.2f} days | center ({sd.cell_centers[0,center_cell]:.4f}, {sd.cell_centers[1,center_cell]:.4f})")
                print(f"  ABsolute L2 error x = {abs_L2_x:.6e} m")
                print(f"  ABsolute L2 error y = {abs_L2_y:.6e} m")
                print(f"  ABsolute L2 total error = {abs_L2_total:.6e} m")
                print(f"  Relative L2 error x = {rel_L2_x:.6e}")
                print(f"  Relative L2 error y = {rel_L2_y:.6e}")
                print(f"  Relative L2 total error = {rel_L2_total:.6e}")
                print("============================================================\n")
                print(f"CZAS SYMULACJI: {self.time_manager.time}")
                print(f"MAX ux_num: {np.max(np.abs(ux_num))}")
                print(f"MAX ux_mms: {np.max(np.abs(ux_mms))}")
                print(f"MAX uy_num: {np.max(np.abs(uy_num))}")
                print(f"MAX uy_mms: {np.max(np.abs(uy_mms))}")
                print(f"MAX u_num: {np.max(np.abs(u_reshaped))}")
                print(f"DEBUG: diff_x_max = {np.max(np.abs(ux_num - ux_mms))}")
                print(f"DEBUG: diff_y_max = {np.max(np.abs(uy_num - uy_mms))}")
                print("\n")



            sched = self.params.get('plot_schedule', [])
            if sched and self.time_manager.time >= sched[0]:
                sched.pop(0)
                mins = int(self.time_manager.time) # / 60.0)
                
                if not hasattr(self, '_vmax_u'):
                    u_all = self.equation_system.evaluate(self.displacement(self.mdg.subdomains()))
                    u2_all = self.equation_system.evaluate(self.displacement2(self.mdg.subdomains()))
                    u_mag = np.linalg.norm(u_all.reshape(self.nd, -1, order='F'), axis=0)
                    u2_mag = np.linalg.norm(u2_all.reshape(self.nd, -1, order='F'), axis=0)
                    self._vmax_u, self._vmax_u2 = 6.9e-9, np.max(u2_mag)#np.max(u_mag) *100
                    print(f"--- Fixed VMAX: u={self._vmax_u:.2f}, u2={self._vmax_u2:.2f} ---")


                for var_name, name, vmax in [(self.displacement_variable, "u", self._vmax_u), (self.displacement2_variable, "u2", self._vmax_u2)]:
                    for sd, sd_data in self.mdg.subdomains(return_data=True):
                        # Get magnitude explicitly
                        vals = pp.get_solution_values(name=var_name, data=sd_data, time_step_index=0)
                        mag = np.linalg.norm(vals.reshape(self.nd, -1, order='F'), axis=0)
                        
                        plt.close('all')
                        pp.plot_grid(sd, cell_value=mag, title=f"{name} at {mins} s", if_plot=False, color_map_limits=[0.0, vmax], color_map="viridis", plot_2d=True)
                        fig = plt.gcf()
                        fig.axes[-1].set_ylabel("u [m]")
                        plt.savefig(f"displacement_{name}_{mins}.png", dpi=200)

                        

    model = ShowCase(model_params)
    pp.run_time_dependent_model(model)
    print("Done.")
