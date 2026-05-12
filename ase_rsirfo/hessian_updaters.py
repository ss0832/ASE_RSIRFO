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
hessian_updaters.py
===================

Quasi-Newton Hessian update formulas used by the RS-I-RFO optimiser. All
methods take the **current** Hessian ``B``, a Cartesian displacement ``s``
and a gradient difference ``y = g_new - g_old``, and return the **increment**
``delta_B`` so that the new Hessian is ``B + delta_B``.

The single secant variants are stateless. The "block" (multi-secant)
variants need a short history of ``(s, y)`` pairs and therefore live on a
stateful class.

References
----------
Quasi-Newton Hessian updates
* C. G. Broyden, *J. Inst. Math. Appl.* **6**, 76 (1970) - BFGS.
* R. Fletcher, *Practical Methods of Optimization*, Wiley (1987).
* M. J. D. Powell, *Math. Program.* **14**, 31 (1978) - PSB.
* J. M. Bofill, *J. Comput. Chem.* **15**, 1 (1994) - PSB / SR1 mix.
* J. M. Anglada, J. M. Bofill, *J. Comput. Chem.* **19**, 349 (1998).
* H. B. Schlegel, *Theor. Chem. Acc.* **103**, 294 (2000) - BFGS / SR1 mix.
* J. Nocedal, S. J. Wright, *Numerical Optimization*, Springer (2006).
* P. Bakó, A. G. Császár, *Theor. Chem. Acc.* **135**, 84 (2016) - flowchart
  selection between BFGS / FSB / SR1.

Block (multi-secant) variants
* P. E. Gill, M. W. Leonard, *Math. Program. B* **94**, 519 (2003).
* Schnabel, *Math. Program.* **24**, 245 (1982) - block update theory.
* Berahas, Curtis, Robinson, Zhou, "Quasi-Newton methods for machine
  learning: forget the past, just sample", arXiv:2106.10989.

CFD (compact finite difference) variants
* H. Wu, M. Rahman, J. Wang, U. Louderaj, W. L. Hase, Y. Zhuang,
  *J. Chem. Phys.* **133**, 074101 (2010).
* Y. Zhuang, M. R. Siebert, W. L. Hase, K. G. Kay, M. Ceotto,
  *J. Chem. Theory Comput.* **9**, 54 (2013).

Damping
* Nocedal & Wright, eq. 18.15 (Powell damping); arXiv:2006.08877v3
  (double damping).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

_DENOM = 1.0e-10


def _to_col(v: np.ndarray) -> np.ndarray:
    """Reshape a vector to a column ``(n, 1)``."""
    return np.asarray(v, dtype=float).reshape(-1, 1)


# --------------------------------------------------------------------------- #
#  Single-secant primitives
# --------------------------------------------------------------------------- #


def _bfgs_delta(B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standard BFGS update.

    delta_B = (y y^T) / (y^T s) - (B s)(B s)^T / (s^T B s)
    """
    s = _to_col(s)
    y = _to_col(y)
    n = s.shape[0]
    delta = np.zeros((n, n))
    ys = float((s.T @ y).item())
    if abs(ys) >= _DENOM:
        delta += (y @ y.T) / ys
    Bs = B @ s
    sBs = float((s.T @ Bs).item())
    if abs(sBs) >= _DENOM:
        delta -= (Bs @ Bs.T) / sBs
    return delta


def _sr1_delta(z: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Symmetric Rank-1 update.  ``z = y - B s`` (or scaled).
    delta_B = z z^T / (z^T s)
    """
    z = _to_col(z)
    s = _to_col(s)
    denom = float((z.T @ s).item())
    if abs(denom) < _DENOM:
        return np.zeros((s.shape[0], s.shape[0]))
    return (z @ z.T) / denom


def _psb_delta(B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Powell symmetric Broyden update."""
    s = _to_col(s)
    y = _to_col(y)
    z = y - B @ s
    ss = float((s.T @ s).item())
    if ss < _DENOM:
        return np.zeros_like(B)
    sz = float((s.T @ z).item())
    return (z @ s.T + s @ z.T) / ss - sz * (s @ s.T) / (ss * ss)


def _bofill_phi(z: np.ndarray, s: np.ndarray) -> float:
    """Mixing parameter ``phi`` of the Bofill update (in [0, 1])."""
    z = np.asarray(z).reshape(-1)
    s = np.asarray(s).reshape(-1)
    zs = float(np.dot(z, s))
    zz = float(np.dot(z, z))
    ss = float(np.dot(s, s))
    if zz * ss < _DENOM:
        return 0.0
    return (zs * zs) / (zz * ss)


# --------------------------------------------------------------------------- #
#  Damping primitives
# --------------------------------------------------------------------------- #


def _powell_damping(
    B: np.ndarray, s: np.ndarray, y: np.ndarray, mu: float = 0.2
) -> np.ndarray:
    """Powell damping of ``y`` (Nocedal & Wright eq. 18.15).

    Ensures ``s^T y_tilde >= mu * s^T B s`` so that the BFGS update remains
    positive-definite.
    """
    s = np.asarray(s).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Bs = B @ s
    sBs = float(s @ Bs)
    sy = float(s @ y)
    if sy >= mu * sBs:
        return y
    if abs(sBs - sy) < _DENOM:
        return y
    theta = (1.0 - mu) * sBs / (sBs - sy)
    theta = float(np.clip(theta, 0.0, 1.0))
    return theta * y + (1.0 - theta) * Bs


def _double_damping_step2(
    s: np.ndarray, y: np.ndarray, mu2: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Step 2 of the double-damping procedure (B-independent).

    Equivalent to Powell damping with ``B = I``: ensures ``s^T y_tilde >=
    mu2 * s^T s``.
    """
    s_v = np.asarray(s).reshape(-1)
    y_v = np.asarray(y).reshape(-1)
    sy = float(s_v @ y_v)
    ss = float(s_v @ s_v)
    if sy >= mu2 * ss:
        return s_v, y_v
    denom = ss - sy
    if abs(denom) < _DENOM:
        return s_v, y_v
    theta = (1.0 - mu2) * ss / denom
    theta = float(np.clip(theta, 0.0, 1.0))
    y_til = theta * y_v + (1.0 - theta) * s_v
    return s_v, y_til


# --------------------------------------------------------------------------- #
#  HessianUpdater - single-secant + block (multi-secant) updates
# --------------------------------------------------------------------------- #


class HessianUpdater:
    """Stateful collection of all Hessian update formulas.

    Block (multi-secant) updates require a short history of recent
    ``(s, y)`` pairs; this is held in :pyattr:`S_history` /
    :pyattr:`Y_history` on the instance. Single-secant updates ignore this
    state.
    """

    #: List of method aliases recognised by :meth:`update`.
    METHODS: tuple[str, ...] = (
        "auto", "flowchart",
        "bfgs", "bfgs_dd",
        "sr1",
        "psb",
        "fsb", "fsb_dd",
        "bofill",
        "cfd_fsb", "cfd_fsb_dd",
        "cfd_bofill",
        "pcfd_bofill",
        "msp",
        # Block (multi-secant)
        "block_bfgs", "block_bfgs_dd",
        "block_fsb", "block_fsb_dd", "block_fsb_weighted",
        "block_cfd_fsb", "block_cfd_fsb_dd", "block_cfd_fsb_weighted",
        "block_bofill", "block_bofill_weighted",
        "block_cfd_bofill", "block_cfd_bofill_weighted",
    )

    def __init__(
        self,
        block_size: int = 4,
        max_window: int = 8,
        dd_mu1: float = 0.2,
        dd_mu2: float = 0.2,
    ) -> None:
        self.block_size = int(block_size)
        self.max_window = max(int(max_window), int(block_size))
        self.dd_mu1 = float(dd_mu1)
        self.dd_mu2 = float(dd_mu2)
        self.S_history: list[np.ndarray] = []
        self.Y_history: list[np.ndarray] = []
        self._auto_scaled = False  # First-iteration scaling done?

    # ----- history management -----------------------------------------------
    def reset_history(self) -> None:
        self.S_history.clear()
        self.Y_history.clear()

    def _push_history(self, s: np.ndarray, y: np.ndarray) -> None:
        s_v = np.asarray(s, dtype=float).reshape(-1)
        y_v = np.asarray(y, dtype=float).reshape(-1)
        self.S_history.append(s_v.copy())
        self.Y_history.append(y_v.copy())
        if len(self.S_history) > self.max_window:
            self.S_history.pop(0)
            self.Y_history.pop(0)

    def _assemble_block(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if not self.S_history:
            return None, None
        k = min(self.block_size, len(self.S_history))
        # Block updates require at least two history pairs to be well-posed;
        # otherwise the (S^T B S) and (S^T Y) factors degenerate to scalars and
        # the resulting "block" update is just a noisy single-secant update.
        if k < 2:
            return None, None
        S = np.column_stack(self.S_history[-k:])
        Y = np.column_stack(self.Y_history[-k:])
        return S, Y

    # ----- top-level dispatch -----------------------------------------------
    def update(
        self,
        B: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        method: str = "auto",
    ) -> np.ndarray:
        """Compute the Hessian update increment.

        Parameters
        ----------
        B, s, y
            Current Hessian and the new ``(s, y)`` pair.
        method
            Method name. Accepts any string in :pyattr:`METHODS`.
            Substring matching follows the original prioritised list (most
            specific first), so ``"block_cfd_fsb_dd"`` is matched before
            ``"block_cfd_fsb"``.

        Returns
        -------
        delta_B
            Increment such that the new Hessian is ``B + delta_B``.
        """
        s = np.asarray(s, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        # Push history *after* selecting the method so the current pair is
        # available for the next block update.
        delta = self._dispatch(B, s, y, method)
        self._push_history(s, y)
        return 0.5 * (delta + delta.T)

    # ----- prioritised dispatch table ---------------------------------------
    def _dispatch(
        self,
        B: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        method: str,
    ) -> np.ndarray:
        m = method.lower().strip() if method else "auto"

        # Most-specific keys first.
        priority: list[tuple[str, callable]] = [
            ("flowchart", self._flowchart_update),
            # block_*
            ("block_cfd_fsb_dd",       self._block_cfd_fsb_dd),
            ("block_cfd_fsb_weighted", self._block_cfd_fsb_weighted),
            ("block_cfd_fsb",          self._block_cfd_fsb),
            ("block_cfd_bofill_weighted", self._block_cfd_bofill_weighted),
            ("block_cfd_bofill",       self._block_cfd_bofill),
            ("block_bfgs_dd",          self._block_bfgs_dd),
            ("block_bfgs",             self._block_bfgs),
            ("block_fsb_dd",           self._block_fsb_dd),
            ("block_fsb_weighted",     self._block_fsb_weighted),
            ("block_fsb",              self._block_fsb),
            ("block_bofill_weighted",  self._block_bofill_weighted),
            ("block_bofill",           self._block_bofill),
            # single-secant
            ("bfgs_dd",                self._bfgs_dd),
            ("bfgs",                   self._bfgs),
            ("sr1",                    self._sr1),
            ("psb",                    self._psb),
            ("pcfd_bofill",            self._pcfd_bofill),
            ("cfd_fsb_dd",             self._cfd_fsb_dd),
            ("cfd_fsb",                self._cfd_fsb),
            ("cfd_bofill",             self._cfd_bofill),
            ("fsb_dd",                 self._fsb_dd),
            ("fsb",                    self._fsb),
            ("bofill",                 self._bofill),
            ("msp",                    self._msp),
        ]
        for key, func in priority:
            if key in m:
                return func(B, s, y)
        # 'auto' / unknown -> flowchart
        return self._flowchart_update(B, s, y)

    # ----- flowchart auto selection -----------------------------------------
    def _flowchart_update(
        self, B: np.ndarray, s: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Select between SR1 / BFGS / FSB based on Bakó & Császár 2016."""
        z = y - B @ y  # cf. original implementation - kept verbatim.
        nrm_s = np.linalg.norm(s)
        nrm_z = np.linalg.norm(z)
        nrm_y = np.linalg.norm(y)
        if nrm_s * nrm_z < _DENOM or nrm_s * nrm_y < _DENOM:
            return self._fsb(B, s, y)
        zs = float(np.dot(z, s)) / (nrm_s * nrm_z)
        ys = float(np.dot(y, s)) / (nrm_s * nrm_y)
        if zs < -0.1:
            return self._sr1(B, s, y)
        if ys > 0.1:
            return self._bfgs(B, s, y)
        return self._fsb(B, s, y)

    # ----- single-secant methods --------------------------------------------
    def _bfgs(self, B, s, y):
        return _bfgs_delta(B, s, y)

    def _bfgs_dd(self, B, s, y):
        # Step 2 of double damping (B-independent).
        s_t, y_t = _double_damping_step2(s, y, self.dd_mu2)
        # Step 1 (Powell damping with current B).
        y_t = _powell_damping(B, s_t, y_t, self.dd_mu1)
        return _bfgs_delta(B, s_t, y_t)

    def _sr1(self, B, s, y):
        z = y - B @ s
        return _sr1_delta(z, s)

    def _psb(self, B, s, y):
        return _psb_delta(B, s, y)

    def _fsb(self, B, s, y):
        z = y - B @ s
        d_sr1 = _sr1_delta(z, s)
        d_bfgs = _bfgs_delta(B, s, y)
        phi = np.sqrt(_bofill_phi(z, s))
        return (1.0 - phi) * d_bfgs + phi * d_sr1

    def _fsb_dd(self, B, s, y):
        s_t, y_t = _double_damping_step2(s, y, self.dd_mu2)
        y_t = _powell_damping(B, s_t, y_t, self.dd_mu1)
        return self._fsb(B, s_t, y_t)

    def _bofill(self, B, s, y):
        z = y - B @ s
        d_sr1 = _sr1_delta(z, s)
        d_psb = _psb_delta(B, s, y)
        phi = _bofill_phi(z, s)
        return (1.0 - phi) * d_psb + phi * d_sr1

    def _cfd_fsb(self, B, s, y):
        # CFD variant: scale the SR1 residual by 2.
        z = 2.0 * (y - B @ s)
        d_sr1 = _sr1_delta(z, s)
        d_bfgs = _bfgs_delta(B, s, y)
        phi = np.sqrt(_bofill_phi(z, s))
        return (1.0 - phi) * d_bfgs + phi * d_sr1

    def _cfd_fsb_dd(self, B, s, y):
        s_t, y_t = _double_damping_step2(s, y, self.dd_mu2)
        y_t = _powell_damping(B, s_t, y_t, self.dd_mu1)
        return self._cfd_fsb(B, s_t, y_t)

    def _cfd_bofill(self, B, s, y):
        z = 2.0 * (y - B @ s)
        d_sr1 = _sr1_delta(z, s)
        d_psb = _psb_delta(B, s, y)
        phi = _bofill_phi(z, s)
        return (1.0 - phi) * d_psb + phi * d_sr1

    def _pcfd_bofill(self, B, s, y):
        # Projected CFD-Bofill: project out the SR1 direction first.
        z = 2.0 * (y - B @ s)
        s_v = _to_col(s)
        z_v = _to_col(z)
        zs = float((z_v.T @ s_v).item())
        if abs(zs) < _DENOM:
            return self._cfd_bofill(B, s, y)
        n = s_v.shape[0]
        proj = np.eye(n) - (z_v @ z_v.T) / float((z_v.T @ z_v).item() + _DENOM)
        s_proj = (proj @ s_v).reshape(-1)
        return self._cfd_bofill(B, s_proj, y)

    def _msp(self, B, s, y):
        """Modified Symmetric Powell (Bofill 1994 alternative form)."""
        z = y - B @ s
        s_v = _to_col(s)
        z_v = _to_col(z)
        ss = float((s_v.T @ s_v).item())
        if ss < _DENOM:
            return np.zeros_like(B)
        zs = float((z_v.T @ s_v).item())
        # PSB-like with symmetric scaling (cf. Theochem 591, 35 (2002)).
        return (
            (z_v @ s_v.T + s_v @ z_v.T) / ss
            - 2.0 * zs * (s_v @ s_v.T) / (ss * ss)
        )

    # ----- block (multi-secant) methods -------------------------------------
    _BLOCK_NORM_MAX: float = 1.0e6  # cap on block-update increment norm

    def _safe_inv(self, A: np.ndarray, reg: float = 1e-8) -> np.ndarray:
        """Regularised inverse; fall back to pseudo-inverse on singularity."""
        try:
            cond = np.linalg.cond(A)
            if cond > 1.0 / (reg + 1e-30):
                return np.linalg.pinv(A)
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A + reg * np.eye(A.shape[0]))

    def _block_bfgs_core(self, B, S, Y):
        if S is None:
            return np.zeros_like(B)
        BS = B @ S
        sBs = S.T @ BS
        ys = S.T @ Y
        inv_sBs = self._safe_inv(sBs)
        inv_ys = self._safe_inv(ys)
        result = Y @ inv_ys @ Y.T - BS @ inv_sBs @ BS.T
        if np.linalg.norm(result) > self._BLOCK_NORM_MAX:
            return np.zeros_like(B)
        return result

    def _block_psb_core(self, B, S, Y):
        if S is None:
            return np.zeros_like(B)
        Z = Y - B @ S
        SS = S.T @ S
        inv_SS = self._safe_inv(SS)
        # Generalised PSB: symmetric correction (Schnabel 1982).
        first = Z @ inv_SS @ S.T + S @ inv_SS @ Z.T
        SZ = S.T @ Z
        sym = inv_SS @ (SZ + SZ.T) @ inv_SS
        return first - S @ sym @ S.T

    def _block_sr1_core(self, B, S, Y):
        if S is None:
            return np.zeros_like(B)
        Z = Y - B @ S
        denom = S.T @ Z
        inv = self._safe_inv(denom)
        result = Z @ inv @ Z.T
        # Guard against numerical blow-up (ill-conditioned denom)
        if np.linalg.norm(result) > self._BLOCK_NORM_MAX:
            return np.zeros_like(B)
        return result

    def _block_cfd_sr1_core(self, B, S, Y):
        if S is None:
            return np.zeros_like(B)
        Z = 2.0 * (Y - B @ S)
        denom = S.T @ Z
        inv = self._safe_inv(denom)
        result = Z @ inv @ Z.T
        if np.linalg.norm(result) > self._BLOCK_NORM_MAX:
            return np.zeros_like(B)
        return result

    def _individual_phis(
        self, B, S, Y, *, cfd: bool = False, bofill_logic: bool = False
    ) -> np.ndarray:
        """Per-column mixing weights for weighted block FSB / Bofill."""
        if cfd:
            Z = 2.0 * (Y - B @ S)
        else:
            Z = Y - B @ S
        phis = np.zeros(S.shape[1])
        for col in range(S.shape[1]):
            phis[col] = _bofill_phi(Z[:, col], S[:, col])
            if not bofill_logic:
                phis[col] = np.sqrt(phis[col])
        return phis

    # block BFGS
    def _block_bfgs(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._bfgs(B, s, y)
        return self._block_bfgs_core(B, S, Y)

    def _block_bfgs_dd(self, B, s, y):
        # Apply DD step2 on each (s, y) pair in history.
        S, Y = self._assemble_block()
        if S is None:
            return self._bfgs_dd(B, s, y)
        for i in range(S.shape[1]):
            _, y_t = _double_damping_step2(S[:, i], Y[:, i], self.dd_mu2)
            Y[:, i] = y_t
        return self._block_bfgs_core(B, S, Y)

    # block FSB / Bofill (uniform mixing)
    def _block_fsb(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._fsb(B, s, y)
        d_bfgs = self._block_bfgs_core(B, S, Y)
        d_sr1 = self._block_sr1_core(B, S, Y)
        phi = float(np.mean(self._individual_phis(B, S, Y)))
        return (1.0 - phi) * d_bfgs + phi * d_sr1

    def _block_fsb_dd(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._fsb_dd(B, s, y)
        for i in range(S.shape[1]):
            _, y_t = _double_damping_step2(S[:, i], Y[:, i], self.dd_mu2)
            Y[:, i] = y_t
        d_bfgs = self._block_bfgs_core(B, S, Y)
        d_sr1 = self._block_sr1_core(B, S, Y)
        phi = float(np.mean(self._individual_phis(B, S, Y)))
        return (1.0 - phi) * d_bfgs + phi * d_sr1

    def _block_fsb_weighted(self, B, s, y):
        # Each column uses its own phi -> rebuild SR1 / BFGS column-wise.
        S, Y = self._assemble_block()
        if S is None:
            return self._fsb(B, s, y)
        phis = self._individual_phis(B, S, Y)
        delta = np.zeros_like(B)
        for i in range(S.shape[1]):
            si, yi = S[:, i], Y[:, i]
            zi = yi - B @ si
            d_sr1 = _sr1_delta(zi, si)
            d_bfgs = _bfgs_delta(B, si, yi)
            delta += (1.0 - phis[i]) * d_bfgs + phis[i] * d_sr1
        return delta / S.shape[1]

    def _block_cfd_fsb(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._cfd_fsb(B, s, y)
        d_bfgs = self._block_bfgs_core(B, S, Y)
        d_sr1 = self._block_cfd_sr1_core(B, S, Y)
        phi = float(np.mean(self._individual_phis(B, S, Y, cfd=True)))
        return (1.0 - phi) * d_bfgs + phi * d_sr1

    def _block_cfd_fsb_dd(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._cfd_fsb_dd(B, s, y)
        for i in range(S.shape[1]):
            _, y_t = _double_damping_step2(S[:, i], Y[:, i], self.dd_mu2)
            Y[:, i] = y_t
        d_bfgs = self._block_bfgs_core(B, S, Y)
        d_sr1 = self._block_cfd_sr1_core(B, S, Y)
        phi = float(np.mean(self._individual_phis(B, S, Y, cfd=True)))
        return (1.0 - phi) * d_bfgs + phi * d_sr1

    def _block_cfd_fsb_weighted(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._cfd_fsb(B, s, y)
        phis = self._individual_phis(B, S, Y, cfd=True)
        delta = np.zeros_like(B)
        for i in range(S.shape[1]):
            si, yi = S[:, i], Y[:, i]
            zi = 2.0 * (yi - B @ si)
            d_sr1 = _sr1_delta(zi, si)
            d_bfgs = _bfgs_delta(B, si, yi)
            delta += (1.0 - phis[i]) * d_bfgs + phis[i] * d_sr1
        return delta / S.shape[1]

    def _block_bofill(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._bofill(B, s, y)
        d_psb = self._block_psb_core(B, S, Y)
        d_sr1 = self._block_sr1_core(B, S, Y)
        phi = float(np.mean(self._individual_phis(B, S, Y, bofill_logic=True)))
        return (1.0 - phi) * d_psb + phi * d_sr1

    def _block_bofill_weighted(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._bofill(B, s, y)
        phis = self._individual_phis(B, S, Y, bofill_logic=True)
        delta = np.zeros_like(B)
        for i in range(S.shape[1]):
            si, yi = S[:, i], Y[:, i]
            zi = yi - B @ si
            d_sr1 = _sr1_delta(zi, si)
            d_psb = _psb_delta(B, si, yi)
            delta += (1.0 - phis[i]) * d_psb + phis[i] * d_sr1
        return delta / S.shape[1]

    def _block_cfd_bofill(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._cfd_bofill(B, s, y)
        d_psb = self._block_psb_core(B, S, Y)
        d_sr1 = self._block_cfd_sr1_core(B, S, Y)
        phi = float(np.mean(
            self._individual_phis(B, S, Y, cfd=True, bofill_logic=True)
        ))
        return (1.0 - phi) * d_psb + phi * d_sr1

    def _block_cfd_bofill_weighted(self, B, s, y):
        S, Y = self._assemble_block()
        if S is None:
            return self._cfd_bofill(B, s, y)
        phis = self._individual_phis(B, S, Y, cfd=True, bofill_logic=True)
        delta = np.zeros_like(B)
        for i in range(S.shape[1]):
            si, yi = S[:, i], Y[:, i]
            zi = 2.0 * (yi - B @ si)
            d_sr1 = _sr1_delta(zi, si)
            d_psb = _psb_delta(B, si, yi)
            delta += (1.0 - phis[i]) * d_psb + phis[i] * d_sr1
        return delta / S.shape[1]
