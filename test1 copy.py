import porepy as pp
import numpy as np

class ExtraMatrixVariableMixin:
    """ Adding variable u2 """

    def create_variables(self):
        super().create_variables()

        # adding u2
        self.u2_variable = "u2"

        # dimensions 2D/3D
        nd = self.nd

        self.equation_system.create_variables(
            dof_info={"cells": nd},
            name=self.u2_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "m"},
        )

    def u2(self, domains):
        return self.equation_system.md_variable(self.u2_variable, domains)
    
    #     # Boundary conditions - ?
    # def bc_type_u2(self, sd):
    #     # like u - ?
    #     return self.bc_type_mechanics(sd)

    # def bc_values_u2(self, bg):
    #     # like u - ?
    #     return self.bc_values_displacement(bg)

    # def update_all_boundary_conditions(self):
    #     super().update_all_boundary_conditions()

    #     self.update_boundary_condition(self.u2_variable, self.bc_values_u2)

    # Initial condition
    # def ic_values_u2(self, sd):
    #     # like u - ?
    #     return self.ic_values_displacement(sd)

    # def set_initial_values_primary_variables(self):
    #     super().set_initial_values_primary_variables()

    #     # initial value
    #     for sd in self.mdg.subdomains(dim=self.nd):
    #         self.equation_system.set_variable_values(
    #             self.ic_values_u2(sd),
    #             [self.equation_system.md_variable(self.u2_variable, [sd])],
    #             iterate_index=0,
    #        )

#class ExtraStrainMixin:
# """ New variable for strain2 """

#    def strain_u2(self, domains):
#        u2 = self.u2(domains)
#        grad_u2 = pp.ad.Gradient(domains)(u2)
#        return pp.ad.SymmetricGradient(domains)(grad_u2)

    
class ExtraConstitutiveLawMixin:

    def stiffness_tensor_u2(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        # it is random values, in the future it should be in different place
        lam = 1e9   # lambda_2
        mu = 5e8    # mu_2
        return pp.FourthOrderTensor(
            lame_lambda=lam,
            lame_mu=mu,
            dim=self.nd
        )
    
    
    def stress_u2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """σ₂ = C₂ : ε(u₂)"""

        stress_ops = []

        for sd in subdomains:
            if sd.dim != self.nd:
                continue

            # strain operator (this is PUBLIC API)
            eps_u2 = self.sym_grad(self.u2([sd]))

            # material tensor
            C2 = self.stiffness_tensor_u2(sd)

            # Hooke law operator provided by PorePy
            stress = self.hooke_law(eps_u2, C2)

            stress_ops.append(stress)

        return pp.ad.sum_operator_list(stress_ops)

#    def stress_u2(self, domains):
#        """σ2 = C2 : eps(u2)."""
 #       eps = self.strain_u2(domains)
#
#       lam, mu = self.stiffness_tensor_u2(domains[0])
#
#        I = pp.ad.IdentityTensor(self.nd)
#        trace_eps = pp.ad.Trace()(eps)
#
#        return lam * trace_eps * I + 2 * mu * eps #ask about dev eps!!!

    # def mechanical_stress_u2(self, subdomains):
    #     stress_ops = []

    #     for sd in subdomains:
    #         if sd.dim != self.nd:
    #             continue

    #         data = self.mdg.subdomain_data(sd)

    #         # tensor - Hooke law for C2
    #         C2 = self.stiffness_tensor_u2(sd)

    #         # strain operator (MPSA) - ?
    #         B = data["mechanics"]["strain_displacement_operator"]

    #         # u2
    #         u2 = self.u2([sd])

    #         # eps(u2)
    #         strain_u2 = B @ u2

    #         # σ2 = C2 : eps
    #         stress_u2 = C2 @ strain_u2

    #         stress_ops.append(stress_u2)

    #     return pp.ad.sum_operator_list(stress_ops)

class ExtraMomentumEquationMixin:

    def set_equations(self):
        super().set_equations()

        subdomains = self.mdg.subdomains(dim=self.nd)

        # standardowe ∇·σ(u)
        stress_u = self.stress(subdomains)

        # nowe ∇·σ₂(u₂)
        stress_u2 = self.stress_u2(subdomains)

        eq = self.balance_equation(
            subdomains=subdomains,
            accumulation=pp.ad.Scalar(0),
            flux=pp.ad.Scalar(-1) * (stress_u + stress_u2),
            source=pp.ad.Scalar(0),
            dim=self.nd,
        )

        eq.set_name("momentum_balance_u_u2")

        self.equation_system.set_equation(
            eq, subdomains, {"cells": self.nd}
        )
   
class ExtraEvolutionEquationMixin:

    beta: float = 1.0

    def set_equations(self):
        super().set_equations()

        subdomains = self.mdg.subdomains(dim=self.nd)

        u = self.displacement(subdomains)
        u2 = self.u2(subdomains)

        du_dt = self.ad_time_derivative(u) #it doesn't exist!!!
        du2_dt = self.ad_time_derivative(u2) #it doesn't exist!!!

        eq = self.beta * u + du_dt - du2_dt
        eq.set_name("u_u2_evolution")

        self.equation_system.set_equation(
            eq,
            subdomains,
            {"cells": self.nd}
        )

    

class ExtraMomentumModel(pp.momentum_balance.MomentumBalance):
    
    def equations(self):
        # eqs = {} # dictionary with equations

        # # two lists with two equations
        # momentum_list = []
        # evolution_list = []

        eqs = super().equations()
        subdomains = self.mdg.subdomains(dim=self.nd)
        # AD operator for u2
        for g in self.md.grids:
            if g.dim == self.nd:
                # given step in time (unknow)
                u = self.ad_variable_scalar("u", [g])
                u2 = self.ad_variable_scalar("u2", [g])
                
                # previos step in time (know)
                # implicit Euler
                u_old = self.initial_condition_scalar("u", [g])
                u2_old = self.initial_condition_scalar("u2", [g])
                
                dt = self.time_step()
                beta = 1.0 
                
                # First equation
                # div(E2 : eps(u2)) + div(E1 : eps(u)) = 0
                stress_discr = self.get_discretization(g, "u")
                stress_discr2 = self.get_discretization(g, "u2")  
                #momentum_list.append(stress_discr2 @ u2 + stress_discr @ u)
                eq1 = stress_discr2 @ u2 + stress_discr @ u
                eq1.set_name("momentum_balance1")
                
                # Second equation (Implicit Euler)
                # (beta + d/dt) * u2 - (d/dt) * u = 0
                # d/dt u \approx (u - u_old) / dt
                
                # (beta + d/dt) * u2
                term_u2 = (beta + 1/dt) * u2 - (1/dt) * u2_old
                
                # (d/dt) * u
                term_u = - (1/dt) * u + (1/dt) * u_old
                
                #evolution_list.append(term_u2 + term_u)
                eq2 = term_u + term_u2
                eq2.set_name("momentum_balance2")

                # conect the equtions
                #eqs["momentum"] = pp.ad.Expression(momentum_list, self.md.grids)
                #eqs["evolution"] = pp.ad.Expression(evolution_list, self.md.grids)
                # combined_eq = pp.ad.concatenate([eq1, eq2])
                # momentum_list.append(combined_eq)
                eqs = eq1 + eq2
                print(eqs)

        return eqs#{"momentum": pp.ad.Expression(momentum_list, self.md.grids)

    #discretization - ?
    def assign_discretizations(self):
        super().assign_discretizations()
        for g, d in self.md.grid_data_iter():
            # MPSA for u and u2
            # definition of \nabla \cdot (E : \nabla^s u)
            mpsa = pp.MPSA(self.mechanics_parameter_key)
            self.set_discretization(g, "u", mpsa)
            self.set_discretization(g, "u2", mpsa)
            
    # initial conditions
    def set_initial_condition(self):

        # Initialization
        super().set_initial_condition()
        
        for g, d in self.md.grid_data_iter():
            # DoF
            dof = self.nd * g.num_cells
            
            # Initial values -> 0
            u_init = np.zeros(dof)
            u2_init = np.zeros(dof)
            
           
            # dictionary STATE under the appropriate keys
            d[pp.STATE]["u"] = u_init
            d[pp.STATE]["u2"] = u2_init

# old...
#     def momentum_balance_equation_u2(self, subdomains):
#         """∇·σ2(u2) + f2 = 0."""

#         stress2 = self.stress_op(subdomains)
#         div_stress2 = pp.ad.Divergence(subdomains)(stress2)

#         # jeśli nie masz dodatkowych sił – 0
#         f2 = pp.ad.Scalar(0)

#         return div_stress2 + f2

#     def set_equations(self):
#         super().set_equations()

#         subdomains = self.mdg.subdomains(dim=self.nd)

#         eq_u2 = self.momentum_balance_equation_u2(subdomains)

#         self.equation_system.add_equation(
#             lhs=eq_u2,
#             variable=self.u2(subdomains),
#             name="momentum_balance_u2",
#         )
    
    
# class ExtraMomentumEquationMixin(ExtraMomentumMixin):
#     def momentum_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
#         # Pobierz standardowe równanie (z u)
#         eq = super().momentum_balance_equation(subdomains)
#         eq2 =super().momentum_balance_equation_u2(subdomains)


#         return eq + eq2

class MyModel(
    ExtraMatrixVariableMixin,
    ExtraConstitutiveLawMixin,
    ExtraMomentumEquationMixin,
    ExtraEvolutionEquationMixin,
    #ExtraMomentumModel,
    pp.MomentumBalance,
    #pp.Poromechanics
):
    pass

model = MyModel()
pp.run_time_dependent_model(model)

# Pobranie wartości nowej zmiennej
#u2_values = model.equation_system.get_variable_values(model.u2_variable)

#print("Wartości u2:", u2_values)

# Jeśli to macierz 2x2:
#num_cells = model.mdg.subdomains()[0].num_cells
#u2_matrix = u2_values.reshape((num_cells, 2, 2), order="F")

#print("u2 jako macierz 2×2 na każdą komórkę:")
#print(u2_matrix)

