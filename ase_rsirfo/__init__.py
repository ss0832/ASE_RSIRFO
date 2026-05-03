"""
ase_rsirfo
==========

Restricted-Step Image Rational Function Optimisation (RS-I-RFO) as a
plug-in :class:`ase.optimize.optimize.Optimizer` subclass. Suitable for
energy minimisation (``order=0``) or transition-state search
(``order=1``); higher saddle orders are also supported.

Quick start
-----------
::

    from ase.build import molecule
    from ase.calculators.emt import EMT
    from ase_rsirfo import RSIRFO

    atoms = molecule("H2O")
    atoms.calc = EMT()

    opt = RSIRFO(atoms, hessian="fischer", trajectory="opt.traj")
    opt.run(fmax=0.05)

Author : ss0832
License: GPL-3.0-or-later
"""

from .hessian_updaters import HessianUpdater
from .optimizer import RSIRFO, numerical_hessian_from_forces
from .parameters import D3Parameters

# Model Hessians are an optional component (see NOTICE.md). The core
# optimiser works without them, so we import lazily and silently degrade.
try:
    from .model_hessian import FischerD3ModelHessian, SwartD2ModelHessian
    _MODEL_HESSIAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    FischerD3ModelHessian = None  # type: ignore[assignment]
    SwartD2ModelHessian = None    # type: ignore[assignment]
    _MODEL_HESSIAN_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "ss0832"
__license__ = "GPL-3.0-or-later"

__all__ = [
    "RSIRFO",
    "HessianUpdater",
    "D3Parameters",
    "numerical_hessian_from_forces",
    "FischerD3ModelHessian",
    "SwartD2ModelHessian",
    "__version__",
    "__author__",
    "__license__",
]
