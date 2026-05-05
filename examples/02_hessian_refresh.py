# Copyright (C) 2026 ss0832
# This file is part of ASE_RSIRFO and is licensed under GPL-3.0-or-later.
# See the LICENSE file in the repository root for the full text, or visit
# <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 2: Periodic Hessian refresh strategies
===============================================

Shows the three recomputation modes available in RSIRFO:
  (a) model Hessian rebuilt at current geometry every N steps
  (b) full numerical Hessian every N steps
  (c) user-supplied analytic Hessian via callback

All three options are mutually exclusive — set hessian_recompute_method
to select the one you want.
"""

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from ase_rsirfo import RSIRFO, numerical_hessian_from_forces

# ── shared setup ────────────────────────────────────────────────────────────
def make_atoms():
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.00, 0.00, 0.00],
            [3.80, 0.00, 0.00],
            [1.90, 3.30, 0.00],
            [1.90, 1.10, 3.10],
        ],
    )
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    return atoms


# ── (a) Model Hessian refresh ───────────────────────────────────────────────
print("=== (a) Fischer model Hessian refresh every 5 steps ===")
atoms_a = make_atoms()
opt_a = RSIRFO(
    atoms_a,
    hessian="fischer",
    hessian_recompute_interval=5,
    hessian_recompute_method="model",   # rebuild model at current geometry
    logfile="-",
)
opt_a.run(fmax=1e-4, steps=60)
print(f"  -> converged in {opt_a._iteration} steps\n")


# ── (b) Numerical Hessian refresh ───────────────────────────────────────────
print("=== (b) Numerical Hessian refresh every 5 steps ===")
atoms_b = make_atoms()
opt_b = RSIRFO(
    atoms_b,
    hessian="identity",
    hessian_recompute_interval=5,
    hessian_recompute_method="numerical",
    numerical_hessian_step=0.01,        # Angstrom
    reset_history_on_recompute=True,    # wipe secant history after refresh
    logfile="-",
)
opt_b.run(fmax=1e-4, steps=60)
print(f"  -> converged in {opt_b._iteration} steps\n")


# ── (c) Analytic Hessian via callback ────────────────────────────────────────
print("=== (c) Analytic (here: numerical) Hessian via callback ===")
call_count = [0]

def my_hessian_callback(atoms: Atoms) -> np.ndarray:
    """Return the (3N x 3N) Hessian in eV/Angstrom^2."""
    call_count[0] += 1
    # In production code replace this with your analytic Hessian routine.
    return numerical_hessian_from_forces(atoms, delta=0.005)

atoms_c = make_atoms()
opt_c = RSIRFO(
    atoms_c,
    hessian="identity",
    hessian_recompute_interval=4,
    hessian_recompute_method="callback",
    hessian_callback=my_hessian_callback,
    logfile="-",
)
opt_c.run(fmax=1e-4, steps=60)
print(f"  -> converged in {opt_c._iteration} steps, callback called {call_count[0]} times")
