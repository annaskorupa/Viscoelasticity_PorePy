"""Viscoelastic material constants for PorePy models.

Extends PorePy's SolidConstants with parameters for the generalized Maxwell model:
two elastic branches (E1, E2) and a viscous dashpot (η), with optional fractional
derivative order (alpha) for Grünwald-Letnikov time discretization.

Based on Idesman et al. (2000).
"""

import porepy as pp
from dataclasses import dataclass
from typing import ClassVar


@dataclass(kw_only=True, eq=False)
class ViscoelasticSolidConstants(pp.SolidConstants):
    """Extended solid constants with additional moduli for the viscous branch.

    Parameters
    ----------
    lame_lambda2 : float
        Second Lamé parameter for the viscous branch [Pa].
    shear_modulus2 : float
        Shear modulus for the viscous branch [Pa].
    viscosity : float
        Viscosity of the dashpot [Pa·s].
    alpha : float
        Fractional derivative order (0 < alpha <= 1). Default 1.0 (classical).
    omega : float
        Damage growth rate [1/h]. Default 0.0 (no damage).
    """

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
