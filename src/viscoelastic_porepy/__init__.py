"""Viscoelastic extension of PorePy's MomentumBalance model.

This package implements a generalized Maxwell model by adding a second
displacement variable (u2) representing the viscous branch.
The total stress is σ = σ₁(u) + σ₂(u₂), and the rate equation is
u₂_dot + β·u₂ - u_dot = 0.

Based on Idesman et al. (2000).
"""

from viscoelastic_porepy.material import ViscoelasticSolidConstants
from viscoelastic_porepy.model import ViscoelasticMomentumBalance

__all__ = [
    "ViscoelasticSolidConstants",
    "ViscoelasticMomentumBalance",
]
