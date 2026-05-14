"""Material constants for the viscoelastic model.

Extends PorePy's SolidConstants with additional Lamé parameters
and viscosity for the Maxwell (viscous) branch.
"""

from dataclasses import dataclass
from typing import ClassVar

import porepy as pp


@dataclass(kw_only=True, eq=False)
class ViscoelasticSolidConstants(pp.SolidConstants):
    """Extended solid constants with moduli for the viscous branch.

    Attributes:
        lame_lambda2: Lamé's first parameter for the viscous part [Pa].
        shear_modulus2: Shear modulus for the viscous part [Pa].
        viscosity: Dynamic viscosity of the dashpot [Pa·s].
    """

    SI_units: ClassVar[dict[str, str]] = dict(**pp.SolidConstants.SI_units)
    SI_units.update({
        "lame_lambda2": "Pa",
        "shear_modulus2": "Pa",
        "viscosity": "Pa * s",
    })

    lame_lambda2: float = 1.0
    shear_modulus2: float = 1.0
    viscosity: float = 1.0
