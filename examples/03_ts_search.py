"""
Example 3: Transition-state search (order=1, I-RFO)
====================================================

Uses the image-RFO projection to climb along the softest Hessian mode while
minimising along all others. This is the standard eigenvector-following
approach to locate first-order saddle points.

For an Ar3 Lennard-Jones trimer the minimum is an equilateral triangle and
the lowest-energy transition state between two equivalent triangles is the
*collinear* arrangement (atoms equispaced along a line). We start near
that configuration with a tiny perpendicular perturbation to break the
symmetry, then let RS-I-RFO climb the soft bending mode.

Two pieces matter for a robust TS search:

* a chemistry-aware initial Hessian (here ``hessian="fischer"``) so the
  lowest eigenmode reflects the bending coordinate from the start, and
* a small initial trust radius (here ``0.05 Angstrom``).
"""

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from ase_rsirfo import RSIRFO

# ── Initial geometry: three Ar atoms approximately collinear ───────────────
sigma = 3.40
r_lin = 3.95   # collinear LJ TS spacing is slightly longer than triangular
atoms = Atoms(
    "Ar3",
    positions=[
        [-r_lin, 0.00, 0.0],
        [0.00,  0.00, 0.0],
        [+r_lin, 0.00, 0.0],
    ],
)
# Break symmetry by tilting the central atom off-axis a touch
atoms.positions[1, 1] += 0.05

atoms.calc = LennardJones(epsilon=0.0103, sigma=sigma)

# ── Run the TS search (order=1, I-RFO) ─────────────────────────────────────
opt = RSIRFO(
    atoms,
    order=1,                # invert the lowest curvature
    hessian="fischer",      # chemistry-aware initial guess
    # hessian_update is automatically set to "block_bofill" for order >= 1
    trust_radius=0.05,      # conservative initial trust radius
    trust_radius_max=0.3,
    logfile="-",
    trajectory="ts_search.traj",
)
opt.run(fmax=1e-3, steps=80)

# ── Diagnose: count imaginary frequencies at the converged geometry ────────
eigvals = np.linalg.eigvalsh(opt.hessian)
n_imaginary = int(np.sum(eigvals < -1e-3))
print(f"\nConverged in {opt._iteration} steps")
print(f"Hessian eigenvalues (lowest 5): "
      f"{[f'{v:+.4e}' for v in sorted(eigvals.tolist())[:5]]}")
print(f"Number of imaginary frequencies: {n_imaginary}  "
      f"(expected 1 for a TS)")

dists = [
    float(np.linalg.norm(atoms.positions[1] - atoms.positions[0])),
    float(np.linalg.norm(atoms.positions[2] - atoms.positions[1])),
    float(np.linalg.norm(atoms.positions[2] - atoms.positions[0])),
]
print(f"Final distances (1-2, 2-3, 1-3): "
      f"{dists[0]:.4f}, {dists[1]:.4f}, {dists[2]:.4f} A")
