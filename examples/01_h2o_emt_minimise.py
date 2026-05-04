# Copyright (C) 2026 ss0832
# This file is part of ASE_RSIRFO and is licensed under GPL-3.0-or-later.
# See the LICENSE file in the repository root for the full text, or visit
# <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 1: Water molecule energy minimisation with EMT
======================================================

Demonstrates the simplest usage of RSIRFO: geometry optimisation of
a slightly distorted H2O molecule using ASE's built-in EMT calculator.
"""

from ase.build import molecule
from ase.calculators.emt import EMT

from ase_rsirfo import RSIRFO

# Build a slightly distorted water molecule
atoms = molecule("H2O")
atoms.rattle(stdev=0.05, seed=42)

# Attach a calculator
atoms.calc = EMT()

# Run RS-RFO optimisation
opt = RSIRFO(
    atoms,
    hessian="identity",       # start from the identity Hessian
    hessian_update="auto",    # flowchart-based update selector
    trajectory="h2o_opt.traj",
    logfile="h2o_opt.log",
)
opt.run(fmax=0.05)

print(f"Converged in {opt._iteration} steps")
print(f"Final positions:\n{atoms.positions}")
