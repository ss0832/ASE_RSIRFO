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
internal_coords.py
==================

Wilson B-vectors for the four primitive internal coordinates that the
Fischer-Almlöf and Swart-Bickelhaupt model Hessians need:

* bond stretch              :func:`stretch`
* bond angle (bend)         :func:`bend`
* torsion (proper dihedral) :func:`torsion`
* out-of-plane angle        :func:`out_of_plane`

The formulas follow Wilson, Decius & Cross (1955), with numerically robust
implementations adapted from later expositions

* P. Pulay, G. Fogarasi, *J. Chem. Phys.* **96**, 2856 (1992)
* V. Bakken, T. Helgaker, *J. Chem. Phys.* **117**, 9160 (2002)

"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  Wilson B-vectors
# --------------------------------------------------------------------------- #


def stretch(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Bond-stretch coordinate r and its B-vector.

    Parameters
    ----------
    coords
        ``(2, 3)`` array - two atomic positions ``[A, B]``.

    Returns
    -------
    r, B
        ``r`` is the bond length; ``B`` has shape ``(2, 3)`` with
        ``B[0] = dr/dr_A`` and ``B[1] = dr/dr_B``.
    """
    a, b = np.asarray(coords[0]), np.asarray(coords[1])
    diff = a - b
    r = float(np.linalg.norm(diff))
    if r < 1e-12:
        raise ArithmeticError("Two atoms are at the same position.")
    u = diff / r
    return r, np.array([u, -u])


def bend(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Bend (angle) coordinate theta and its B-vector.

    Parameters
    ----------
    coords
        ``(3, 3)`` array - atoms ``[A, J, C]`` where ``J`` is the apex.

    Returns
    -------
    theta, B
        ``theta`` is the angle ``A-J-C`` (radians); ``B`` has shape
        ``(3, 3)`` with rows for atoms A, J, C respectively.
    """
    a, j, c = (np.asarray(coords[i]) for i in range(3))
    u = a - j
    v = c - j
    ru = float(np.linalg.norm(u))
    rv = float(np.linalg.norm(v))
    if ru < 1e-12 or rv < 1e-12:
        raise ArithmeticError("Bond length zero in bend coordinate.")
    uh = u / ru
    vh = v / rv
    cos_theta = np.clip(np.dot(uh, vh), -1.0, 1.0)
    sin_theta = np.sqrt(max(1.0 - cos_theta ** 2, 0.0))
    if sin_theta < 1e-10:
        raise ArithmeticError("Bend angle is collinear; B-vector singular.")
    theta = float(np.arccos(cos_theta))
    # dtheta/dr_A = (cos(theta)*uh - vh) / (ru*sin(theta))
    dA = (cos_theta * uh - vh) / (ru * sin_theta)
    dC = (cos_theta * vh - uh) / (rv * sin_theta)
    dJ = -dA - dC
    return theta, np.array([dA, dJ, dC])


def torsion(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Proper dihedral tau and its B-vector.

    Parameters
    ----------
    coords
        ``(4, 3)`` array - atoms ``[A, B, C, D]``.

    Returns
    -------
    tau, B
        ``tau`` is the dihedral angle (radians) signed by the right-hand rule
        about the B-C bond; ``B`` has shape ``(4, 3)``.

    Notes
    -----
    Implementation follows Bakken & Helgaker (2002), eq. 18-21.
    """
    a, b, c, d = (np.asarray(coords[i]) for i in range(4))
    rab = b - a
    rbc = c - b
    rcd = d - c
    n_bc = np.linalg.norm(rbc)
    if n_bc < 1e-12:
        raise ArithmeticError("Central bond length zero in torsion.")
    bc_hat = rbc / n_bc

    # Project rab and rcd onto the plane perpendicular to bc_hat.
    p = rab - np.dot(rab, bc_hat) * bc_hat
    q = rcd - np.dot(rcd, bc_hat) * bc_hat
    np_p = np.linalg.norm(p)
    np_q = np.linalg.norm(q)
    if np_p < 1e-10 or np_q < 1e-10:
        raise ArithmeticError("Torsion is degenerate (collinear segment).")
    cos_tau = np.clip(np.dot(p, q) / (np_p * np_q), -1.0, 1.0)
    cross = np.cross(p, q)
    sin_tau = np.dot(cross, bc_hat) / (np_p * np_q)
    tau = float(np.arctan2(sin_tau, cos_tau))

    # B-vectors (Bakken & Helgaker, eq. 21 - using vector form).
    n_ab = np.cross(rab, rbc)
    n_cd = np.cross(rbc, rcd)
    sq_ab = np.dot(n_ab, n_ab)
    sq_cd = np.dot(n_cd, n_cd)
    if sq_ab < 1e-20 or sq_cd < 1e-20:
        raise ArithmeticError("Torsion plane normal is zero (degenerate).")

    dA = -n_bc * n_ab / sq_ab
    dD =  n_bc * n_cd / sq_cd
    # Lever-arm contributions on B and C.
    bc_ab = np.dot(rab, rbc) / (n_bc ** 2)
    bc_cd = np.dot(rcd, rbc) / (n_bc ** 2)
    dB = (bc_ab - 1.0) * dA - bc_cd * dD
    dC = (bc_cd - 1.0) * dD - bc_ab * dA

    return tau, np.array([dA, dB, dC, dD])


def out_of_plane(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Out-of-plane angle and its B-vector.

    Parameters
    ----------
    coords
        ``(4, 3)`` array - atoms ``[A, B, C, J]`` where ``J`` is the apex
        atom and ``A``, ``B``, ``C`` are its three reference neighbours.

    Returns
    -------
    theta, B
        ``theta`` is the angle between the J-A bond and the plane spanned
        by J-B and J-C (signed). ``B`` has shape ``(4, 3)``.

    Notes
    -----
    A finite-difference fall-back is used because closed-form derivatives of
    the OOP angle are lengthy; fewer than a percent of optimisation steps
    typically use this coordinate, so the cost is negligible.
    """
    coords = np.asarray(coords, dtype=float)
    eps = 1.0e-5

    def _theta(x: np.ndarray) -> float:
        a, b, c, j = x[0], x[1], x[2], x[3]
        u = a - j
        v = b - j
        w = c - j
        rv = np.linalg.norm(v)
        rw = np.linalg.norm(w)
        ru = np.linalg.norm(u)
        if ru < 1e-12 or rv < 1e-12 or rw < 1e-12:
            raise ArithmeticError("Zero bond in OOP coordinate.")
        nrm = np.cross(v, w)
        n_nrm = np.linalg.norm(nrm)
        if n_nrm < 1e-10:
            raise ArithmeticError("Reference plane is degenerate.")
        # Sin of the OOP angle.
        sin_theta = np.dot(nrm, u) / (n_nrm * ru)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return float(np.arcsin(sin_theta))

    theta = _theta(coords)
    grad = np.zeros((4, 3))
    for i in range(4):
        for k in range(3):
            x_plus = coords.copy()
            x_plus[i, k] += eps
            x_minus = coords.copy()
            x_minus[i, k] -= eps
            grad[i, k] = (_theta(x_plus) - _theta(x_minus)) / (2.0 * eps)
    return theta, grad


