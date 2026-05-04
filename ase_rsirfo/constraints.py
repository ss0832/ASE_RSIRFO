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
constraints.py
==============

Handling of ASE constraints inside the RS-I-RFO step.

ASE's :class:`~ase.constraints.FixAtoms` and friends are normally applied at
two stages:

* :meth:`~ase.Atoms.get_forces` calls each constraint's ``adjust_forces`` —
  fixed components arrive at the optimiser already zeroed.
* :meth:`~ase.Atoms.set_positions` calls each constraint's
  ``adjust_positions`` — any displacement on a fixed coordinate is silently
  rolled back.

That is enough for a steepest-descent / line-search optimiser. For a
Newton-style optimiser like RS-I-RFO it is **not** enough:

1. The quasi-Newton Hessian update will accumulate spurious information on
   the fixed rows / columns from the artificially-zeroed gradient
   differences, slowly corrupting the search direction in the *active*
   subspace.
2. The RFO secular equation will allocate a fraction of every step to the
   fixed degrees of freedom, only for ``set_positions`` to throw it away —
   so the achieved step is shorter than the trust-radius bound, the model
   ratio becomes biased, and the trust-radius adapter contracts the radius
   indefinitely.

This module provides two equivalent fixes:

* :func:`apply_freeze_diagonal` — overwrite the fixed rows/columns of the
  Hessian with rows/columns of an identity scaled by ``freeze_value`` and
  zero out the matching entries of the gradient. Cheap and keeps the matrix
  shape unchanged; the freeze value must be much larger than the largest
  active eigenvalue so that the corresponding RFO step components are
  numerically zero. Algebraically equivalent to a level shift on the fixed
  subspace.

* :func:`build_active_projector` and :func:`reduce_to_active_subspace` /
  :func:`expand_from_active_subspace` — solve the optimisation problem in
  the unconstrained subspace itself. Numerically robust and exact: the
  fixed coordinates simply do not exist while the step is being computed.

For ``FixAtoms`` and ``FixCartesian`` the active subspace is a strict subset
of the Cartesian basis and the projector is constructible analytically. For
constraints with a continuous projection (``FixedPlane``, ``FixedLine``,
``Hookean``...) we fall back to the diagonal-freeze method, which is always
safe because it works on the gradient ASE has already filtered.

References
----------
* Pulay, *Mol. Phys.* **17**, 197 (1969) — projector formalism for
  constrained Hessians (the so-called *(I − P)* trick).
* Schlegel, *J. Comput. Chem.* **3**, 214 (1982) — discussion of how
  zero-eigenvalue handling matters in geometry optimisation.
* Baker, *J. Comput. Chem.* **7**, 385 (1986) — fixed-coordinate handling
  in Newton-Raphson geometry optimisers.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Constraint discovery
# --------------------------------------------------------------------------- #


#: ASE constraint class names that lock individual Cartesian DOFs.
_CARTESIAN_FIX_CLASSES: tuple[str, ...] = ("FixAtoms", "FixCartesian")

#: ASE constraint class names that lock internal coordinates (distance,
#: angle, dihedral) and require translational/rotational projection of the
#: Hessian to remain physically meaningful.
_INTERNAL_FIX_CLASSES: tuple[str, ...] = (
    "FixBondLength",     # alias for FixBondLengths
    "FixBondLengths",
    "FixInternals",
    "FixedPlane",
    "FixedLine",
    "Hookean",
)


def detect_fixed_dofs(atoms) -> tuple[np.ndarray, bool, bool]:
    """Identify Cartesian degrees of freedom held fixed by an ASE constraint.

    Inspects ``atoms.constraints`` for fully-fixed Cartesian DOFs handled by
    :class:`~ase.constraints.FixAtoms` and :class:`~ase.constraints.FixCartesian`
    (the common analytic cases). Internal-coordinate constraints
    (``FixBondLength``, ``FixInternals``, ``FixedPlane``, ``FixedLine``,
    ``Hookean``) are *flagged* here but not converted to a Cartesian mask:
    those constraints are non-linear in Cartesian coordinates and ASE
    handles them iteratively in ``adjust_forces`` / ``adjust_positions``.

    Parameters
    ----------
    atoms
        :class:`~ase.Atoms` instance whose constraints are to be analysed.

    Returns
    -------
    fixed_mask : np.ndarray of shape (3*N,)
        Boolean mask: ``True`` where the Cartesian DOF is fixed by an
        analytic Cartesian constraint (FixAtoms / FixCartesian).
    has_internal_constraints : bool
        ``True`` if any internal-coordinate / soft constraint is present
        (FixBondLength, FixInternals, FixedPlane, FixedLine, Hookean).
        The optimiser must keep T/R projection enabled in this case so
        that the rigid-body subspace does not interfere with the
        constraint manifold.
    has_other_constraints : bool
        ``True`` if there is any constraint that is neither an analytic
        Cartesian fix nor a recognised internal constraint. The
        diagonal-freeze safety net should be applied for these.
    """
    n_atoms = len(atoms)
    fixed_mask = np.zeros(3 * n_atoms, dtype=bool)
    has_internal = False
    has_other = False

    for c in getattr(atoms, "constraints", []) or []:
        cname = type(c).__name__
        if cname == "FixAtoms":
            indices = np.asarray(c.index)
            if indices.dtype == bool:
                indices = np.where(indices)[0]
            for a in indices:
                fixed_mask[3 * int(a):3 * int(a) + 3] = True
        elif cname == "FixCartesian":
            # FixCartesian(a, mask=...): per-component fix.
            # ASE convention (modern): ``mask`` entries are True where the
            # component is FIXED. Older releases sometimes used the opposite
            # convention; we detect by probing adjust_forces on a small array.
            indices = np.atleast_1d(np.asarray(c.index, dtype=int))
            mask_attr = np.asarray(c.mask, dtype=bool)
            if mask_attr.shape != (3,) and mask_attr.shape[-1] == 3:
                mask_attr = mask_attr.reshape(-1, 3)[0]
            # Probe: a unit force is preserved on free components and zeroed
            # on fixed ones by adjust_forces.
            n_atoms_local = len(atoms)
            probe = np.ones((n_atoms_local, 3), dtype=float)
            try:
                c.adjust_forces(atoms, probe)
                fixed_components = (probe[int(indices[0])] == 0.0)
            except Exception:  # noqa: BLE001
                # Defensive fall-back: assume mask "True == fixed"
                fixed_components = mask_attr
            for a in indices:
                a = int(a)
                for j in range(3):
                    if fixed_components[j]:
                        fixed_mask[3 * a + j] = True
        elif cname in _INTERNAL_FIX_CLASSES:
            # Internal-coordinate constraint: handled by ASE iteratively
            # via adjust_forces / adjust_positions. We just flag it.
            has_internal = True
        else:
            # Unknown constraint type
            has_other = True

    return fixed_mask, has_internal, has_other


# --------------------------------------------------------------------------- #
#  Method 1: diagonal freeze (in-place, shape preserved)
# --------------------------------------------------------------------------- #


def apply_freeze_diagonal(
    hessian: np.ndarray,
    gradient: np.ndarray,
    fixed_mask: np.ndarray,
    freeze_value: float = 1.0e8,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out fixed rows/cols, set diagonal to ``freeze_value``, zero grad.

    This is algebraically equivalent to applying an enormous level shift to
    the constrained subspace. The resulting RFO step components on the
    fixed DOFs are smaller than ``|grad| / freeze_value`` (≈ 0 with the
    default ``freeze_value=1e8``), avoiding singularities while preserving
    the shape of the matrix. Suitable for any constraint type because it
    works on whatever gradient ASE has already produced (i.e. with all the
    constraint-specific filtering already applied).

    Parameters
    ----------
    hessian
        ``(3N, 3N)`` Cartesian Hessian, modified in place.
    gradient
        ``(3N,)`` Cartesian gradient (or ``(N, 3)`` accepted), modified in
        place after ``ravel``.
    fixed_mask
        Boolean mask of fixed DOFs (see :func:`detect_fixed_dofs`).
    freeze_value
        Diagonal value used on the fixed DOFs. Should be (much) larger than
        the largest active Hessian eigenvalue so that the RFO step on the
        fixed subspace is numerically negligible.

    Returns
    -------
    H_out, g_out
        The modified ``(3N, 3N)`` Hessian and ``(3N,)`` gradient arrays
        (same objects as the inputs, returned for chaining convenience).
    """
    if not np.any(fixed_mask):
        return hessian, gradient.ravel()

    H = hessian
    g = gradient.ravel()
    idx = np.where(fixed_mask)[0]
    # Zero rows / columns
    H[idx, :] = 0.0
    H[:, idx] = 0.0
    # Diagonal freeze
    H[idx, idx] = freeze_value
    # Zero gradient on the fixed DOFs (ASE already does this for FixAtoms,
    # but doing it again is harmless and protects against unknown
    # constraint types that may leave residual components).
    g[idx] = 0.0
    return H, g


# --------------------------------------------------------------------------- #
#  Method 2: explicit subspace reduction
# --------------------------------------------------------------------------- #


def build_active_projector(fixed_mask: np.ndarray) -> np.ndarray:
    """Return a ``(K, 3N)`` selector that maps full → active DOFs.

    Where ``K = 3N - sum(fixed_mask)`` is the number of unconstrained DOFs.
    The reverse operation (active → full) is just the transpose, since the
    rows of the selector form an orthonormal basis of the active subspace.
    """
    n_dof = fixed_mask.size
    active_idx = np.where(~fixed_mask)[0]
    P = np.zeros((active_idx.size, n_dof), dtype=float)
    P[np.arange(active_idx.size), active_idx] = 1.0
    return P


def reduce_to_active_subspace(
    hessian: np.ndarray,
    gradient: np.ndarray,
    projector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``H`` and ``g`` to the active subspace.

    Returns
    -------
    H_active : (K, K)
        ``P H P.T``
    g_active : (K,)
        ``P g``
    """
    g_flat = gradient.ravel()
    H_active = projector @ hessian @ projector.T
    g_active = projector @ g_flat
    return H_active, g_active


def expand_from_active_subspace(
    step_active: np.ndarray,
    projector: np.ndarray,
) -> np.ndarray:
    """Expand an active-subspace step back to the full ``(3N,)`` shape.

    Components on the constrained DOFs are exactly zero by construction.
    """
    return projector.T @ step_active
