"""Viscoelastic PorePy — modular package for generalized Maxwell models.

Usage::

    from src.viscoelastic_porepy import (
        ViscoelasticSolidConstants,
        ViscoelasticMomentumBalance,
        ViscoelasticMomentumBalanceFracture,
    )
"""

from .constants import ViscoelasticSolidConstants
from .model import (
    ViscoelasticModelMixin,
    ViscoelasticMomentumBalance,
    ViscoelasticMomentumBalanceFracture,
)
from .utils import (
    compute_strain_at_cell,
    setup_publication_style,
    save_convergence_results,
    load_convergence_results,
    save_strain_history,
    load_strain_history,
    EXPERIMENTAL_DATA_T,
    EXPERIMENTAL_DATA_EPS,
    SIM_1D_T,
    SIM_1D_EPS,
)

__all__ = [
    "ViscoelasticSolidConstants",
    "ViscoelasticModelMixin",
    "ViscoelasticMomentumBalance",
    "ViscoelasticMomentumBalanceFracture",
    "compute_strain_at_cell",
    "setup_publication_style",
    "save_convergence_results",
    "load_convergence_results",
    "save_strain_history",
    "load_strain_history",
    "EXPERIMENTAL_DATA_T",
    "EXPERIMENTAL_DATA_EPS",
    "SIM_1D_T",
    "SIM_1D_EPS",
]
