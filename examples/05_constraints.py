# Copyright (C) 2026 ss0832
# This file is part of ASE_RSIRFO and is licensed under GPL-3.0-or-later.
# See the LICENSE file in the repository root for the full text, or visit
# <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 5: ASE constraint handling
====================================

Shows that RSIRFO honours ASE constraints attached to ``atoms``. Three
strategies are available via the ``constraint_method`` argument:

* ``'auto'`` (default) — pick based on the constraint type:
    - pure ``FixAtoms`` / ``FixCartesian`` -> ``'none'`` (rely on ASE's
      ``adjust_positions``) plus secant-pair zeroing on fixed DOFs
    - any other constraint type -> ``'freeze'`` diagonal
* ``'subspace'`` — explicit projection of ``H``, ``g`` to the active
  subspace before solving the RFO equations.
* ``'freeze'`` — replace fixed rows/cols of ``H`` with a scaled identity;
  works for any constraint type at the cost of some conditioning.
* ``'none'`` — disable constraint-aware processing entirely.

In all four modes, the secant pair ``(s, y)`` used by the quasi-Newton
update is zeroed on fixed DOFs to prevent Hessian pollution.
"""

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms, FixCartesian

from ase_rsirfo import RSIRFO


# --------------------------------------------------------------------- #
# Example A: FixAtoms — pin one corner of an Ar3 trimer at the origin
# --------------------------------------------------------------------- #
print("=== Example A: FixAtoms (pin atom 0 at the origin) ===")
atoms = Atoms("Ar3", positions=[[0, 0, 0], [4.5, 0, 0], [2.0, 3.5, 0]])
atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
atoms.set_constraint(FixAtoms(indices=[0]))

opt = RSIRFO(
    atoms,
    hessian="identity",
    # constraint_method='auto' (default) is fine here
    logfile="-",
    trajectory="constraints_a.traj",
)
opt.run(fmax=1e-5, steps=80)

print(f"\nConverged in {opt._iteration} steps")
print(f"atom 0 stayed at {atoms.positions[0]}  (must be (0, 0, 0))")
bonds = [
    np.linalg.norm(atoms.positions[i] - atoms.positions[j])
    for i in range(3) for j in range(i)
]
print(f"bond lengths: {[f'{b:.4f}' for b in bonds]}  "
      f"(expected ~{3.40*2**(1/6):.4f} A)")


# --------------------------------------------------------------------- #
# Example B: FixCartesian — fix only the z-component of atom 1
# --------------------------------------------------------------------- #
print("\n=== Example B: FixCartesian (atom 1, z-component only) ===")
atoms_b = Atoms("Ar3", positions=[[0, 0, 0], [4.0, 0, 1.0], [2.0, 3.5, 0.5]])
atoms_b.calc = LennardJones(epsilon=0.0103, sigma=3.40)
atoms_b.set_constraint(FixCartesian(1, mask=(False, False, True)))
initial_z = atoms_b.positions[1, 2]

opt_b = RSIRFO(atoms_b, hessian="identity", logfile=None)
opt_b.run(fmax=1e-3, steps=80)

print(f"atom 1 z stayed at {atoms_b.positions[1, 2]:.6f} "
      f"(initial: {initial_z:.6f})")


# --------------------------------------------------------------------- #
# Example C: Compare the four constraint_method options side-by-side
# --------------------------------------------------------------------- #
print("\n=== Example C: side-by-side comparison ===")
print(f"{'method':<10} {'iters':<6} {'fmax':<10} fixed_atom_moved")

for method in ["auto", "subspace", "freeze", "none"]:
    a = Atoms("Ar3", positions=[[0, 0, 0], [4.5, 0, 0], [2.0, 3.5, 0]])
    a.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    a.set_constraint(FixAtoms(indices=[0]))
    o = RSIRFO(a, hessian="identity",
               constraint_method=method, logfile=None)
    o.run(fmax=1e-5, steps=80)
    moved = float(np.linalg.norm(a.positions[0]))
    print(f"{method:<10} {o._iteration:<6} "
          f"{float(np.max(np.abs(a.get_forces()))):<10.2e} {moved:.2e}")
