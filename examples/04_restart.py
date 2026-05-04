# Copyright (C) 2026 ss0832
# This file is part of ASE_RSIRFO and is licensed under GPL-3.0-or-later.
# See the LICENSE file in the repository root for the full text, or visit
# <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 4: ASE restart / checkpoint
=====================================

Demonstrates how RSIRFO saves and restores its internal state
(Hessian, trust radius, step counter, secant-pair history) across
Python sessions, using the standard ASE restart mechanism.

Run this script once: it will stop after 3 steps.
Run it again: it will resume from the saved checkpoint.
"""

import os
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT

from ase_rsirfo import RSIRFO

RESTART_FILE = "rsirfo_restart.pckl"
TRAJ_FILE    = "restart_example.traj"

atoms = molecule("H2O")
atoms.rattle(stdev=0.05, seed=7)
atoms.calc = EMT()

fresh_start = not os.path.exists(RESTART_FILE)

opt = RSIRFO(
    atoms,
    restart=RESTART_FILE,
    trajectory=TRAJ_FILE,
    append_trajectory=not fresh_start,
    logfile="-",
    hessian="identity",
)

if fresh_start:
    print("=== Fresh start: running 3 steps, then stopping ===")
    opt.run(fmax=0.001, steps=3)
    print(f"Stopped at step {opt._iteration}, TR = {opt.trust_radius:.4f}")
    print("Re-run this script to continue from the checkpoint.")
else:
    print(f"=== Resuming from checkpoint (step {opt._iteration}) ===")
    opt.run(fmax=0.05)
    print(f"Converged in {opt._iteration} total steps")
    os.remove(RESTART_FILE)   # clean up
