"""Constitutive laws for the viscous displacement branch u2.

Provides MPSA-discretized stress and elastic moduli for the second
displacement variable u2 in the generalized Maxwell model.
"""

import numpy as np
import porepy as pp
from typing import Callable, cast

Scalar = pp.ad.Scalar


class ViscousElasticModuli:
    """Elastic moduli for the viscous displacement u2."""

    def shear_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.shear_modulus2, "shear_modulus2")

    def lame_lambda2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return Scalar(self.solid.lame_lambda2, "lame_lambda2")

    def stiffness_tensor2(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        lmbda2 = self.solid.lame_lambda2 * np.ones(subdomain.num_cells)
        mu2 = self.solid.shear_modulus2 * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu2, lmbda2)

    def beta(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Relaxation rate β = μ₂ / η."""
        val = self.solid.shear_modulus2 / self.solid.viscosity
        return Scalar(val, "beta")


class MechanicalStressU2:
    """MPSA-discretized stress for u2."""

    stress2_keyword: str
    displacement2: Callable
    interface_displacement2: Callable
    bc_type_mechanics: Callable

    def mechanical_stress2(self, domains) -> pp.ad.Operator:
        if len(domains) == 0 or all(
            isinstance(d, pp.BoundaryGrid) for d in domains
        ):
            return self.create_boundary_operator(
                name=self.stress2_keyword, domains=domains
            )

        domains = cast(list[pp.Grid], domains)
        discr = pp.ad.MpsaAd(self.stress2_keyword, domains)
        interfaces = self.subdomains_to_interfaces(domains, [1])
        proj = pp.ad.MortarProjections(
            self.mdg, domains, interfaces, dim=self.nd
        )

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
            + discr.bound_stress()
            @ proj.mortar_to_primary_avg()
            @ self.interface_displacement2(interfaces)
        )
        stress2.set_name("mechanical_stress2")
        return stress2


class ConstitutiveLawsU2(
    pp.models.constitutive_laws.ZeroGravityForce,
    ViscousElasticModuli,
    MechanicalStressU2,
    pp.models.constitutive_laws.ConstantSolidDensity,
):
    """Combined constitutive laws for the viscous branch."""

    def stress2(self, domains) -> pp.ad.Operator:
        return self.mechanical_stress2(domains)
