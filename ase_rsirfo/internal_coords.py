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

This module also provides the analytic Cartesian Hessian of a
**Grimme D2 pair-dispersion** term, which the Swart model Hessian uses for
its long-range correction.

References
----------
* Grimme, *J. Comput. Chem.* **27**, 1787 (2006) -- D2 dispersion form.
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


# --------------------------------------------------------------------------- #
#  D2 dispersion second derivatives (Cartesian, pair contribution)
# --------------------------------------------------------------------------- #
#
# The Grimme D2 dispersion energy of a pair (i, j) is
#
#     E_disp = - s6 * C6_ij / R^6 * f_d(R),
#     f_d(R) = 1 / (1 + exp(-d * (R / R_r - 1))),
#
# with R_r = R_VDW_i + R_VDW_j, d = 20 (canonical D2).
#
# Differentiating once gives the pair force, twice gives the pair Hessian.
# The full Cartesian Hessian block ``d^2 E / d r_a d r_b`` for atoms a, b can
# be decomposed into a parallel (radial) part and a perpendicular part:
#
#     H_alpha_beta(R) = h_par(R) * n_alpha n_beta
#                     + h_perp(R) * (delta_alpha_beta - n_alpha n_beta)
#
# where  n = (r_b - r_a) / R,  h_par = d^2 E / dR^2,  h_perp = (1/R) dE/dR.
#
# The two helper functions below evaluate the diagonal (alpha == beta) and
# off-diagonal (alpha != beta) components, matching the layout used by the
# Swart model Hessian.
# --------------------------------------------------------------------------- #


def _d2_radial_derivatives(
    r: float,
    c6: float,
    r_vdw: float,
    s6: float = 1.0,
    d_damp: float = 20.0,
) -> tuple[float, float]:
    """Return ``(dE/dR, d^2 E / dR^2)`` for the D2 pair energy."""
    if r < 1e-10:
        return 0.0, 0.0

    x = d_damp * (r / r_vdw - 1.0)
    fd = 1.0 / (1.0 + np.exp(-x))
    fd_prime = (d_damp / r_vdw) * fd * (1.0 - fd)
    fd_double = (
        (d_damp / r_vdw) ** 2 * fd * (1.0 - fd) * (1.0 - 2.0 * fd)
    )

    e_no_damp = -s6 * c6 / r ** 6
    de_no_damp = 6.0 * s6 * c6 / r ** 7
    dde_no_damp = -42.0 * s6 * c6 / r ** 8

    # E = e_no_damp * fd
    de_dr = de_no_damp * fd + e_no_damp * fd_prime
    dde_dr2 = (
        dde_no_damp * fd
        + 2.0 * de_no_damp * fd_prime
        + e_no_damp * fd_double
    )
    return float(de_dr), float(dde_dr2)


def vdw_isotropic(
    x_a: float,
    y_a: float,
    z_a: float,
    c6: float,
    r_vdw: float,
    s6: float = 1.0,
    d_damp: float = 20.0,
) -> float:
    """Diagonal element ``d^2 E_disp / dx_a^2`` for one D2 pair.

    Arguments are the **components of r_ij** (a is the component along which
    we differentiate twice, the other two are the perpendicular ones).
    """
    r2 = x_a * x_a + y_a * y_a + z_a * z_a
    if r2 < 1e-20:
        return 0.0
    r = float(np.sqrt(r2))
    de, dde = _d2_radial_derivatives(r, c6, r_vdw, s6, d_damp)
    h_par = dde
    h_perp = de / r
    n_a2 = x_a * x_a / r2
    return float(h_par * n_a2 + h_perp * (1.0 - n_a2))


def vdw_anisotropic(
    x_a: float,
    x_b: float,
    x_c: float,
    c6: float,
    r_vdw: float,
    s6: float = 1.0,
    d_damp: float = 20.0,
) -> float:
    """Off-diagonal element ``d^2 E_disp / dx_a dx_b`` for one D2 pair."""
    r2 = x_a * x_a + x_b * x_b + x_c * x_c
    if r2 < 1e-20:
        return 0.0
    r = float(np.sqrt(r2))
    de, dde = _d2_radial_derivatives(r, c6, r_vdw, s6, d_damp)
    h_par = dde
    h_perp = de / r
    return float((h_par - h_perp) * x_a * x_b / r2)


# --------------------------------------------------------------------------- #
#  D3-BJ dispersion second derivatives (Cartesian, pair contribution)
# --------------------------------------------------------------------------- #


def d3_pair_hessian(
    r_vec: np.ndarray,
    c6: float,
    c8: float,
    r0: float,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
) -> np.ndarray:
    """``3 x 3`` Hessian block for one D3-BJ pair.

    Energy form:
    ``E = - sum_{n=6,8} s_n C_n / (R^n + (a1 R0 + a2)^n)``.

    Returns
    -------
    h
        ``(3, 3)`` Hessian block ``d^2 E / d r_a d r_a``. The block for
        atoms (a, b) follows the standard pair Hessian pattern
        ``H_aa = +h, H_bb = +h, H_ab = -h, H_ba = -h``.
    """
    r = float(np.linalg.norm(r_vec))
    if r < 1e-10:
        return np.zeros((3, 3))

    a_const = a1 * r0 + a2

    # Radial first / second derivatives of the energy.
    def part(n: int, c: float, scale: float) -> tuple[float, float]:
        denom = r ** n + a_const ** n
        # E_n = -scale * c / denom
        e_n = -scale * c / denom
        # dE_n/dR
        d_denom = n * r ** (n - 1)
        de = scale * c * d_denom / denom ** 2
        # d^2 E_n / dR^2
        dd_denom = n * (n - 1) * r ** (n - 2)
        dde = (
            scale * c * dd_denom / denom ** 2
            - 2.0 * scale * c * d_denom ** 2 / denom ** 3
        )
        return float(de), float(dde)

    de6, dde6 = part(6, c6, s6)
    de8, dde8 = part(8, c8, s8)
    de_total = de6 + de8
    dde_total = dde6 + dde8

    n_hat = r_vec / r
    proj = np.outer(n_hat, n_hat)
    h_par = dde_total
    h_perp = de_total / r
    return h_par * proj + h_perp * (np.eye(3) - proj)
