"""Constitutive laws for the viscous displacement component u₂.

Provides elastic moduli, MPSA-discretized stress, and the combined
constitutive law bundle for the Maxwell branch.
"""

from typing import Callable, cast

import numpy as np
import porepy as pp

from viscoelastic_porepy.material import ViscoelasticSolidConstants

Scalar = pp.ad.Scalar


# ---------------------------------------------------------------------------
# Elastic moduli for the viscous branch
# ---------------------------------------------------------------------------
class ViscousElasticModuli:
    """Elastic moduli (Lamé parameters) for the viscous displacement u₂.

    All values are read from ``self.solid``, which must be an instance
    of :class:`ViscoelasticSolidConstants`.
    """

    solid: ViscoelasticSolidConstants

    def shear_modulus2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear modulus for the viscous part [Pa]."""
        return Scalar(self.solid.shear_modulus2, "shear_modulus2")

    def lame_lambda2(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Lamé's first parameter for the viscous part [Pa]."""
        return Scalar(self.solid.lame_lambda2, "lame_lambda2")

    def stiffness_tensor2(self, subdomain: pp.Grid) -> pp.FourthOrderTensor:
        """Fourth-order stiffness tensor for the viscous part [Pa]."""
        lmbda2 = self.solid.lame_lambda2 * np.ones(subdomain.num_cells)
        mu2 = self.solid.shear_modulus2 * np.ones(subdomain.num_cells)
        return pp.FourthOrderTensor(mu2, lmbda2)

    def beta(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Relaxation rate β = μ₂ / η [1/s]."""
        val = self.solid.shear_modulus2 / self.solid.viscosity
        return Scalar(val, "beta")


# ---------------------------------------------------------------------------
# MPSA stress discretization for u₂
# ---------------------------------------------------------------------------
class MechanicalStressU2:
    """MPSA-discretized mechanical stress for the viscous displacement u₂."""

    stress2_keyword: str
    displacement2: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    interface_displacement2: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryCondition]

    def mechanical_stress2(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Compute the MPSA-discretized mechanical stress for u₂.

        Parameters:
            domains: Subdomains or boundary grids.

        Returns:
            AD operator for the mechanical stress on grid faces.
        """
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


# ---------------------------------------------------------------------------
# Combined constitutive laws for u₂
# ---------------------------------------------------------------------------
class ConstitutiveLawsU2(
    pp.models.constitutive_laws.ZeroGravityForce,
    ViscousElasticModuli,
    MechanicalStressU2,
    pp.models.constitutive_laws.ConstantSolidDensity,
):
    """Bundle of constitutive laws for the u₂ momentum balance."""

    def stress2(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Total stress operator for u₂ (delegates to mechanical_stress2)."""
        return self.mechanical_stress2(domains)
