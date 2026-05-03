"""
rfo.py
======

Rational Function Optimisation (RFO) step machinery.

The augmented Hessian
::
    A(alpha) = | H/alpha   g/alpha |
               | g^T/alpha     0   |

has eigenvalues that interlace those of ``H``. Banerjee, Adams, Simons &
Shepard (1985) showed that the optimal RFO step is the negative gradient
vector divided by the difference between the lowest (or highest, for image
RFO) eigenvalue of ``A`` and the eigenvalues of ``H``.

For the restricted-step variant (RS-I-RFO) the trust-radius parameter
``alpha`` is varied so that the resulting step lies on the trust-radius
sphere.

References
----------
* A. Banerjee, N. Adams, J. Simons, R. Shepard,
  *J. Phys. Chem.* **89**, 52 (1985) - original RFO paper.
* J. Baker, *J. Comput. Chem.* **7**, 385 (1986) - implementation guidance.
* E. Besalú, J. M. Bofill, *Theor. Chem. Acc.* **100**, 265 (1998) -
  step restriction strategy used here (RS-(I-)RFO).
* A. Heyden, A. T. Bell, F. J. Keil, *J. Chem. Phys.* **123**, 224101
  (2005) - image-RFO (saddle search) projection used in I-RFO.
* J. J. Moré, D. C. Sorensen, *SIAM J. Sci. Stat. Comput.* **4**, 553
  (1983) - safeguarded Newton solver for the secular equation.

Implementation note
-------------------
We solve the secular equation in ``O(N)`` operations (Moré-Sorensen) rather
than diagonalising the augmented Hessian explicitly. The classical
``(N+1) x (N+1)`` eigen-problem reduces to finding the smallest root of
::
    f(lambda) = lambda + sum_i (g_i'^2) / (lambda_i' - lambda)
where ``lambda_i' = eigenvalues_of_H / alpha`` and ``g_i'`` are the
gradient components in the eigenbasis (also divided by ``alpha``).
This implementation follows the ``pysisyphus`` reference very closely
(https://github.com/eljost/pysisyphus, MIT-licensed; we ported the logic
to a self-contained NumPy module under GPL-3.0+).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# --------------------------------------------------------------------------- #
#  Secular-equation (single root) solver
# --------------------------------------------------------------------------- #


def solve_secular_more_sorensen(
    eigvals: np.ndarray,
    grad_components: np.ndarray,
    alpha: float,
    log: callable | None = None,
) -> float:
    """Find the smallest eigenvalue of the augmented Hessian.

    Implements a safeguarded Newton iteration with a Brent fall-back. The
    function ``f`` has poles at every ``lambda_i / alpha`` and one finite
    root strictly below the smallest pole; we hunt for that root.
    """
    if log is None:
        log = lambda *args, **kwargs: None

    eigvals = np.asarray(eigvals, dtype=float)
    grad = np.asarray(grad_components, dtype=float)
    eigvals_p = eigvals / alpha
    grad_sq = (grad / alpha) ** 2
    lambda_min_asymp = float(np.min(eigvals_p))

    g_norm = float(np.linalg.norm(grad))
    initial = lambda_min_asymp - max(2.0 * g_norm, 1e-3)

    def f(l: float) -> float:
        denoms = eigvals_p - l
        safe = np.where(np.abs(denoms) < 1e-30, np.sign(denoms) * 1e-30, denoms)
        safe[safe == 0] = 1e-30
        return float(l + np.sum(grad_sq / safe))

    # Try Brent root-finding first. Build a bracket [a, b] with f(a) < 0 < f(b).
    try:
        margin = max(1e-12, abs(lambda_min_asymp) * 1e-10)
        b = lambda_min_asymp - margin
        f_b = f(b)
        a = initial
        f_a = f(a)
        guard = 10
        while f_a > 0.0 and guard > 0:
            step_back = max(g_norm, 0.1 * abs(a), 1e-8)
            a -= step_back
            f_a = f(a)
            guard -= 1
        if f_a * f_b < 0.0:
            return float(brentq(f, a, b, xtol=1e-10, rtol=1e-10, maxiter=100))
    except Exception:  # noqa: BLE001 - we fall back below.
        pass

    # Newton fall-back.
    lam = initial
    tol = 1e-10 * abs(lambda_min_asymp) + 1e-12
    for _ in range(50):
        denoms = eigvals_p - lam
        safe = np.where(np.abs(denoms) < 1e-30, np.sign(denoms) * 1e-30, denoms)
        safe[safe == 0] = 1e-30
        f_l = lam + float(np.sum(grad_sq / safe))
        if abs(f_l) < tol:
            break
        f_p = 1.0 + float(np.sum(grad_sq / safe ** 2))
        if abs(f_p) < 1e-20:
            break
        lam_new = lam - f_l / f_p
        if lam_new >= lambda_min_asymp:
            lam_new = 0.5 * (lam + lambda_min_asymp)
        lam = lam_new
    return float(lam)


def solve_rfo(
    eigvals: np.ndarray,
    grad_components: np.ndarray,
    alpha: float,
    log: callable | None = None,
) -> tuple[np.ndarray, float]:
    """Solve the RFO equations for a given ``alpha``.

    Returns the step in the eigenbasis of ``H`` and the augmented-Hessian
    eigenvalue ``lambda``.
    """
    lam = solve_secular_more_sorensen(eigvals, grad_components, alpha, log=log)
    denom = (eigvals / alpha) - lam
    safe = np.where(np.abs(denom) < 1e-20, np.sign(denom) * 1e-20, denom)
    safe[safe == 0] = 1e-20
    step = -(grad_components / alpha) / safe
    return step, lam


# --------------------------------------------------------------------------- #
#  Restricted-step search over alpha
# --------------------------------------------------------------------------- #


def restricted_step(
    eigvals: np.ndarray,
    grad_components: np.ndarray,
    trust_radius: float,
    alpha_init: float = 1.0,
    alpha_max: float = 1000.0,
    alpha_step_max: float = 10.0,
    max_micro_cycles: int = 40,
    step_norm_tol: float = 1.0e-3,
    log: callable | None = None,
) -> tuple[np.ndarray, float, float]:
    """Find the alpha that puts the RFO step **on** the trust-radius sphere.

    Combines a Brent root finder over ``alpha`` (preferred) with Newton
    iterations on ``U(alpha) = ||step||^2 - R^2`` as a fall-back.

    Returns
    -------
    step, step_norm, alpha
        The refined step (in eigenbasis), its norm and the converged alpha.
    """
    if log is None:
        log = lambda *args, **kwargs: None

    grad_components = np.asarray(grad_components, dtype=float)

    def step_at(alpha: float) -> np.ndarray:
        step, _ = solve_rfo(eigvals, grad_components, alpha, log=log)
        return step

    def norm_at(alpha: float) -> float:
        return float(np.linalg.norm(step_at(alpha)))

    def objective(alpha: float) -> float:
        return norm_at(alpha) - trust_radius

    # ---------- 1) Brent root finder ----------------------------------------
    alpha_lo = max(alpha_init, 1.0e-6)
    alpha_hi = min(alpha_max, max(50.0 * alpha_lo, 50.0))
    try:
        f_lo = objective(alpha_lo)
        f_hi = objective(alpha_hi)
        if f_lo * f_hi < 0.0:
            alpha = float(
                brentq(objective, alpha_lo, alpha_hi,
                       xtol=1e-6, rtol=1e-6, maxiter=50)
            )
            step = step_at(alpha)
            sn = float(np.linalg.norm(step))
            if abs(sn - trust_radius) < step_norm_tol:
                return step, sn, alpha
            log(f"  Brent gave |s|={sn:.4f}, refining with Newton.")
            alpha_init = alpha
        else:
            log("  No Brent bracket; entering Newton loop.")
    except Exception:
        log("  Brent failed; entering Newton loop.")

    # ---------- 2) Newton iterations on U(alpha) = ||s||^2 - R^2 ------------
    alpha = float(alpha_init)
    best_step: np.ndarray | None = None
    best_diff = float("inf")
    a_left: float | None = None
    a_right: float | None = None

    for _ in range(max_micro_cycles):
        step, lam = solve_rfo(eigvals, grad_components, alpha, log=log)
        sn = float(np.linalg.norm(step))
        diff = abs(sn - trust_radius)
        if diff < best_diff:
            best_step = step.copy()
            best_diff = diff
        u = sn * sn - trust_radius * trust_radius
        if u < 0 and (a_left is None or alpha > a_left):
            a_left = alpha
        elif u > 0 and (a_right is None or alpha < a_right):
            a_right = alpha
        if abs(u) < 1e-8 or diff < step_norm_tol:
            break

        # Derivative of step norm squared with respect to alpha.
        denom = (eigvals / alpha) - lam
        safe = np.where(np.abs(denom) < 1e-20, np.sign(denom) * 1e-20, denom)
        safe[safe == 0] = 1e-20
        # ds_i/dalpha for fixed lambda: (g_i / alpha^2) / safe
        # plus chain through lambda; the chain term cancels at the optimum.
        # We use the textbook approximation that ignores the chain term, which
        # is sufficient for the Newton refinement loop.
        ds_dalpha = (grad_components / alpha ** 2) / safe
        d_norm2 = 2.0 * float(np.dot(step, ds_dalpha))
        if abs(d_norm2) < 1e-10:
            if a_left is not None and a_right is not None:
                alpha = 0.5 * (a_left + a_right)
            else:
                alpha = alpha * (0.5 if u > 0 else 2.0)
        else:
            step_alpha = -u / d_norm2
            step_alpha = np.clip(step_alpha, -alpha_step_max, alpha_step_max)
            alpha_new = alpha + step_alpha
            if a_left is not None and a_right is not None:
                alpha_new = max(min(alpha_new, 0.99 * a_right), 1.01 * a_left)
            alpha = alpha_new
        alpha = float(np.clip(alpha, 1.0e-6, alpha_max))
        if alpha == 1.0e-6 or alpha == alpha_max:
            break

    if best_step is None:
        best_step = step_at(alpha)
    return best_step, float(np.linalg.norm(best_step)), float(alpha)
