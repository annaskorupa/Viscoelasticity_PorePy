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
        return {"cell_size": self.params.get("cell_size", 0.01)}

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
    """Dirichlet on North/South uy=0, Dirichlet on East/West ux = uy = 0."""
    units: pp.Units
    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryConditionVectorial(sd, domain_sides.west + domain_sides.east + domain_sides.north + domain_sides.south, "dir")
    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.ones((self.nd, bg.num_cells))
        domain_sides = self.domain_boundary_sides(bg)
        displacement_val = self.units.convert_units(0.0, "m")
        values[:, domain_sides.east] *= displacement_val
        values[:, domain_sides.west] *= displacement_val
        values[1:, domain_sides.north] *= displacement_val
        values[1:, domain_sides.south] *= displacement_val
        return values.ravel("F")
    # def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
    #     values = np.ones((self.nd, bg.num_cells))
    #     domain_sides = self.domain_boundary_sides(bg)
    #     stress_val = self.units.convert_units(5000000, "Pa")
    #     stress_val_0 = self.units.convert_units(0, "Pa")
    #     values[:, domain_sides.east] *= stress_val
    #     values[:, domain_sides.south] *= stress_val_0
    #     return values.ravel("F")

class InitialConditionsU2:
    """Zero initial conditions for u2."""
    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()
        for sd in self.mdg.subdomains(dim=self.nd):
            self.equation_system.set_variable_values(np.zeros(sd.num_cells * self.nd), [self.displacement2([sd])], iterate_index=0)
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
    """Body force in a central region. -> for MMS gravity force will be naglected, only fx is calculated."""
    solid: ViscoelasticSolidConstants
    units: pp.Units
    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        vals = []
        A_MMS = 0.0003
        b_MMS = 1.0 * 10**(-7)
        t = self.time_manager.time
        beta = self.solid.shear_modulus2 / self.solid.viscosity
        for sd in subdomains:
            data = np.zeros((sd.num_cells, self.nd))
            if sd.dim == 2:
                cc = sd.cell_centers
                mask = (cc[0] > 0.3/self.units.m) & (cc[0] < 0.7/self.units.m) & (cc[1] > 0.3/self.units.m) & (cc[1] < 0.7/self.units.m)
                #force = self.solid.density * self.units.convert_units(-9.8, "m * s^-2")
                force = self.units.convert_units(A_MMS * (np.pi / 0.8)**2 * np.sin(np.pi * cc[0, mask]) * (22575700000 * (1 - np.exp(-b_MMS * t)) + 11000000000 * (b_MMS/(b_MMS - beta)) * (np.exp(-beta * t) - np.exp(-b_MMS * t))), "N")
                data[mask, 0] = force * sd.cell_volumes[mask]
            vals.append(data)
        return pp.ad.DenseArray(np.concatenate(vals).ravel(), "body_force")

# =============================================================================
# 6. Final Model
# =============================================================================
class ViscoelasticMomentumBalance(GeometryMixin, BoundaryConditionsMixin, BodyForceMixin, RateEquation, VariablesU2, ConstitutiveLawsU2, InitialConditionsU2, SolutionStrategyU2, pp.MomentumBalance):
    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.stress2_keyword = "mechanics2"
    
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
    time_manager = pp.TimeManager(
        schedule=[0, 450 * pp.DAY],
        dt_init=10 * pp.MINUTE,
        dt_min_max=(1 * pp.MINUTE, 1 * pp.DAY),
        iter_max=450,
    )
    
    solid_constants = ViscoelasticSolidConstants(
        # mi_1 = 33 500 MPa, mi_2 = 1500 MPa -> DOI: 10.1016/S0045-7825(99)00463-6
        # E_1 = 22575700000 Pa, E_2 = 11000000 Pa, Poisson_nu = 0.0 -> https://doi.org/10.1016/j.apm.2006.04.006
        shear_modulus=22575700000/2, shear_modulus2=11000000000/2, 
        lame_lambda=0.0, lame_lambda2=0.0, 
        viscosity=22575700000*(45.454545*24*60*60)# bo  eta  = 2*shear_modulus*fi from fig 7 and eq. (10), # tau = 1 hour
        #density=0.2, permeability=0.5, porosity=0.25
    )
    
    model_params = {
        "material_constants": {"solid": solid_constants, "fluid": pp.FluidComponent()},
        "time_manager": time_manager,
        "plot_schedule": [pp.MINUTE * i for i in range(0, 301, 50)],
    }

    class ShowCase(ViscoelasticMomentumBalance):
        def after_nonlinear_convergence(self) -> None:
            super().after_nonlinear_convergence()
            if self.time_manager.time_index == 0:
                print(f"--- Theoretical relaxation time: {self.solid.viscosity/self.solid.shear_modulus2/60:.2f} min ---")
            
            sched = self.params.get('plot_schedule', [])
            if sched and self.time_manager.time >= sched[0]:
                sched.pop(0)
                mins = int(self.time_manager.time / 60)
                
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
                        pp.plot_grid(sd, cell_value=mag, title=f"{name} at {mins} min", if_plot=False, color_map_limits=[0, vmax], plot_2d=True)
                        plt.savefig(f"displacement_{name}_{mins}.png", dpi=200)

    model = ShowCase(model_params)
    pp.run_time_dependent_model(model)
    print("Done.")
