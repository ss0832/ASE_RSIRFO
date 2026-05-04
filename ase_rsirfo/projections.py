# Copyright (C) 2026 ss0832
#
# This file is part of ASE_RSIRFO.
#
# ASE_RSIRFO is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# ASE_RSIRFO is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASE_RSIRFO. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
projections.py
==============

Projection of translation / rotation modes from gradients and Hessians.

For an isolated (non-periodic) molecule there are six rigid-body degrees of
freedom (three translations + three rotations) that have zero force constant
in an exact Hessian. Numerical Hessians often pick up small spurious
eigenvalues along these modes that destabilise the RFO step. We therefore
project them out before solving the RFO equations.

For periodic systems (``atoms.pbc.any() == True``):

* Continuous **rotational** symmetry is broken by the simulation cell, so
  rotational projection is skipped.
* Continuous **translational** symmetry remains (the energy is invariant
  under a uniform shift of all atoms) -- the three translation vectors are
  still projected out.
* If the user lets the cell relax (e.g. through an ASE filter), the extra
  cell degrees of freedom are appended to the Cartesian gradient by ASE
  itself; we leave those untouched.

References
----------
* W. H. Miller, N. C. Handy, J. E. Adams, *J. Chem. Phys.* **72**,
  99 (1980) - vibrational projection technique.
* J. Baker, *J. Comput. Chem.* **7**, 385 (1986) - use of projection in
  RFO-based geometry optimisation.
"""

from __future__ import annotations

import numpy as np


def _gram_schmidt(vectors: np.ndarray, tol: float = 1.0e-10) -> np.ndarray:
    """Return an orthonormal basis from the rows of ``vectors``.

    Linearly dependent rows are dropped (helpful for linear molecules where
    one rotation vector is identically zero).
    """
    basis: list[np.ndarray] = []
    for v in vectors:
        w = v.astype(float, copy=True)
        for b in basis:
            w -= np.dot(v, b) * b
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)
    if not basis:
        return np.zeros((0, vectors.shape[1]))
    return np.vstack(basis)


def build_tr_rot_basis(
    coords: np.ndarray,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> np.ndarray:
    """Return an orthonormal basis ``(K, 3*N)`` of rigid-body modes.

    ``K`` ranges from 0 (both projections disabled) up to 6 (full molecule).
    For linear molecules one rotational mode is null and is dropped.
    Coordinates are mass-unweighted (suitable for the optimiser's Cartesian
    Hessian).
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    n_atoms = coords.shape[0]
    centred = coords - coords.mean(axis=0)

    rows: list[np.ndarray] = []
    if project_translation:
        for axis in range(3):
            v = np.zeros_like(centred)
            v[:, axis] = 1.0
            rows.append(v.flatten())
    if project_rotation and n_atoms >= 2:
        # R_x: (0, -z, y), R_y: (z, 0, -x), R_z: (-y, x, 0)
        rx = np.zeros_like(centred)
        rx[:, 1] = -centred[:, 2]
        rx[:, 2] = centred[:, 1]
        ry = np.zeros_like(centred)
        ry[:, 0] = centred[:, 2]
        ry[:, 2] = -centred[:, 0]
        rz = np.zeros_like(centred)
        rz[:, 0] = -centred[:, 1]
        rz[:, 1] = centred[:, 0]
        rows.extend([rx.flatten(), ry.flatten(), rz.flatten()])

    if not rows:
        return np.zeros((0, 3 * n_atoms))
    return _gram_schmidt(np.vstack(rows))


def project_gradient(
    gradient: np.ndarray,
    coords: np.ndarray,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> np.ndarray:
    """Subtract the rigid-body component of a Cartesian gradient.

    ``gradient`` and ``coords`` are reshaped to ``(N, 3)`` if necessary.
    The returned array shares the input shape.
    """
    g = np.asarray(gradient, dtype=float)
    flat = g.ravel()
    Q = build_tr_rot_basis(
        coords,
        project_translation=project_translation,
        project_rotation=project_rotation,
    )
    if Q.size == 0:
        return g
    flat_proj = flat - Q.T @ (Q @ flat)
    return flat_proj.reshape(g.shape)


def project_hessian(
    hessian: np.ndarray,
    coords: np.ndarray,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> np.ndarray:
    """Project rigid-body modes out of a Cartesian Hessian.

    Applied as ``P^T H P`` with ``P = I - Q^T Q``. The result is symmetrised
    to remove rounding asymmetries.
    """
    H = np.asarray(hessian, dtype=float)
    Q = build_tr_rot_basis(
        coords,
        project_translation=project_translation,
        project_rotation=project_rotation,
    )
    if Q.size == 0:
        return 0.5 * (H + H.T)
    n = H.shape[0]
    P = np.eye(n) - Q.T @ Q
    H_proj = P @ H @ P
    return 0.5 * (H_proj + H_proj.T)


def projection_modes_for_atoms(atoms) -> tuple[bool, bool]:
    """Decide which rigid-body modes to project for an ASE :class:`Atoms`.

    Rules:

    * Non-periodic system  -> project both translation and rotation.
    * Any periodic axis    -> project translation only.
    * Cell-filter wrappers -> the optimiser sees flattened DOFs that include
      strain components; we still pass through the atomic-coordinate part of
      the gradient via this routine, so the rule above suffices.
    """
    if atoms is None:
        return True, True
    pbc = getattr(atoms, "pbc", None)
    if pbc is None:
        return True, True
    pbc = np.asarray(pbc, dtype=bool)
    if pbc.any():
        return True, False
    return True, True
