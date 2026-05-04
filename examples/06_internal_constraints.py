# Copyright (C) 2026 ss0832
# This file is part of ASE_RSIRFO and is licensed under GPL-3.0-or-later.
# See the LICENSE file in the repository root for the full text, or visit
# <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Example 6: Internal-coordinate constraints (bond / angle / dihedral)
=====================================================================

Demonstrates that RSIRFO honours ASE's internal-coordinate constraints
(``FixBondLength`` for distances, ``FixInternals`` for angles and
dihedrals) and that the trans/rotational projection is automatically
kept active for these cases.

Why T/R projection matters here
-------------------------------
Internal coordinates are invariant under rigid translation/rotation, so
the rigid-body modes of an Atoms object remain genuine zero-energy
directions even when, for example, one bond length is fixed. Without
T/R projection the RFO solver would allocate part of every step to those
zero-eigenvalue directions, giving slow or oscillatory convergence.

For pure ``FixAtoms`` constraints, by contrast, the fixed atoms break
the rigid-body symmetry, so T/R projection must be skipped (it would
delete legitimate DOFs of the moving atoms). When both constraint types
coexist, RSIRFO defers to the atom-fix rule: T/R is skipped because the
fixed atoms already eliminate the rigid-body subspace.
"""

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms, FixBondLength, FixInternals

from ase_rsirfo import RSIRFO


# --------------------------------------------------------------------- #
# Example A: Distance constraint
# --------------------------------------------------------------------- #
print("=== Example A: Fix bond 0-1 at 5.0 A in an Ar3 cluster ===")
atoms = Atoms("Ar3", positions=[[0, 0, 0], [5.0, 0, 0], [2.5, 3.0, 0]])
atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
atoms.set_constraint(FixBondLength(0, 1))
initial_d01 = np.linalg.norm(atoms.positions[1] - atoms.positions[0])

opt = RSIRFO(atoms, hessian="identity", logfile=None)
opt.run(fmax=1e-4, steps=80)

final_d01 = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
final_d02 = np.linalg.norm(atoms.positions[2] - atoms.positions[0])
final_d12 = np.linalg.norm(atoms.positions[2] - atoms.positions[1])
print(f"  bond 0-1: {final_d01:.6f}  (fixed at {initial_d01:.6f})")
print(f"  bond 0-2: {final_d02:.4f}  (free, expected ~3.8164)")
print(f"  bond 1-2: {final_d12:.4f}  (free, expected ~3.8164)")
print(f"  iters: {opt._iteration}")


# --------------------------------------------------------------------- #
# Example B: Angle constraint
# --------------------------------------------------------------------- #
print("\n=== Example B: Fix angle 0-1-2 at 120 deg in an Ar3 cluster ===")
atoms_b = Atoms("Ar3", positions=[
    [0, 0, 0],
    [3.5, 0, 0],
    [3.5 + 3.5*np.cos(np.deg2rad(60)), 3.5*np.sin(np.deg2rad(60)), 0],
])
atoms_b.calc = LennardJones(epsilon=0.0103, sigma=3.40)
atoms_b.set_constraint(FixInternals(angles_deg=[[120.0, [0, 1, 2]]]))


def angle_deg(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


opt_b = RSIRFO(atoms_b, hessian="identity", logfile=None)
opt_b.run(fmax=1e-3, steps=80)

print(f"  angle 0-1-2: "
      f"{angle_deg(*atoms_b.positions[[0,1,2]]):.4f} deg  (fixed at 120)")
print(f"  bond 0-1:    "
      f"{np.linalg.norm(atoms_b.positions[1]-atoms_b.positions[0]):.4f}  (free)")
print(f"  bond 1-2:    "
      f"{np.linalg.norm(atoms_b.positions[2]-atoms_b.positions[1]):.4f}  (free)")
print(f"  iters: {opt_b._iteration}")


# --------------------------------------------------------------------- #
# Example C: Dihedral constraint
# --------------------------------------------------------------------- #
print("\n=== Example C: Fix dihedral 0-1-2-3 at 60 deg in an Ar4 chain ===")
# Bend the 1-2-3 inner angle and offset atom 3 out of plane so the dihedral
# is well-defined (its derivative diverges at exactly planar geometries).
atoms_c = Atoms("Ar4", positions=[
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [3.8 + 3.8 * np.cos(np.deg2rad(70)),
     3.8 * np.sin(np.deg2rad(70)), 0.0],
    [3.8 + 3.8 * np.cos(np.deg2rad(70)) + 3.0,
     3.8 * np.sin(np.deg2rad(70)) + 1.5,
     2.5],
])
atoms_c.calc = LennardJones(epsilon=0.0103, sigma=3.40)
atoms_c.set_constraint(
    FixInternals(dihedrals_deg=[[60.0, [0, 1, 2, 3]]])
)


def dihedral_deg(p):
    b1, b2, b3 = p[1]-p[0], p[2]-p[1], p[3]-p[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m = np.cross(n1, b2/np.linalg.norm(b2))
    x = np.dot(n1, n2); y = np.dot(m, n2)
    return float(np.degrees(np.arctan2(y, x)))


opt_c = RSIRFO(atoms_c, hessian="identity", logfile=None)
opt_c.run(fmax=1e-3, steps=80)

print(f"  dihedral 0-1-2-3: {dihedral_deg(atoms_c.positions):.4f} deg "
      f"(fixed at 60)")
print(f"  iters: {opt_c._iteration}")


# --------------------------------------------------------------------- #
# Example D: Mixed constraints (FixAtoms + FixBondLength)
# --------------------------------------------------------------------- #
print("\n=== Example D: Mixed FixAtoms + FixBondLength ===")
atoms_d = Atoms("Ar4", positions=[[0,0,0], [3.5,0,0], [3.5,3.5,0], [0,3.5,0]])
atoms_d.calc = LennardJones(epsilon=0.0103, sigma=3.40)
atoms_d.set_constraint([FixAtoms(indices=[0]), FixBondLength(2, 3)])

initial_d23 = float(np.linalg.norm(atoms_d.positions[3] - atoms_d.positions[2]))
opt_d = RSIRFO(atoms_d, hessian="identity", logfile=None)
opt_d.run(fmax=1e-3, steps=80)

print(f"  atom 0 stayed at: {atoms_d.positions[0]}")
print(f"  bond 2-3: "
      f"{np.linalg.norm(atoms_d.positions[3]-atoms_d.positions[2]):.6f}  "
      f"(fixed at {initial_d23:.6f})")
print(f"  iters: {opt_d._iteration}")
