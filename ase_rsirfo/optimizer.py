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
optimizer.py
============

ASE-compatible Restricted-Step Image Rational Function Optimisation
(RS-I-RFO) optimiser. Saddle order is selectable: ``order=0`` minimises,
``order=1`` finds a transition state (one negative eigenvalue), and so on.

Public API
----------
:class:`RSIRFO` is a strict subclass of
``ase.optimize.optimize.Optimizer`` and follows the standard ASE conventions:

* the user passes an :class:`~ase.Atoms` object and a calculator,
* convergence is checked through ``fmax`` (per-atom max force in eV/Angstrom),
* a trajectory file and a log file may be requested,
* a restart pickle persists the Hessian, history and trust radius across
  Python sessions.

Hessian acquisition
-------------------
Three independent things can happen to the Hessian during optimisation:

1. **Initial build** (``hessian=...``)
   Identity (default), Fischer-D3 model, Swart-D2 model, or a user-supplied
   ``np.ndarray`` in eV/Angstrom^2.

   For transition-state searches the recommended practice is to pass an exact
   numerical Hessian computed with :func:`numerical_hessian_from_forces`::

       H0 = numerical_hessian_from_forces(atoms, delta=0.01)
       opt = RSIRFO(atoms, order=1, hessian=H0,
                    hessian_recompute_interval=5,
                    hessian_recompute_method="numerical")

   **Step 0 is never recomputed automatically** — the recompute schedule
   (item 3 below) fires only at ``iteration > 0``.  Therefore the only way
   to start from the exact numerical Hessian is to pass it explicitly via
   ``hessian=``.

2. **Quasi-Newton update** (``hessian_update=...``)
   Applied at every step using the latest ``(s, y)`` pair (and possibly a
   short history for block updates).

3. **Periodic recomputation** (``hessian_recompute_interval=N``)
   Every ``N`` accepted steps (``iteration > 0`` only) the running Hessian
   is **discarded** and replaced by a freshly computed one.  The replacement
   uses one of:

   * ``"model"``     -- rebuild the same model Hessian (fischer / swart) at
     the current geometry. Cheap and keeps the chemistry right; only valid
     when the initial Hessian was a model.
   * ``"numerical"`` -- central-difference Hessian from the calculator's
     forces (eV/Angstrom^2). Works with any ASE calculator.
   * ``"callback"``  -- call a user-provided function returning the
     analytical Hessian (eV/Angstrom^2). Use this for calculators that can
     supply analytical second derivatives.

   When ``hessian_recompute_method`` is ``None`` (default), the choice is
   inferred from the initial Hessian: ``"model"`` for ``fischer``/``swart``,
   ``"callback"`` if a callback is provided, otherwise ``"numerical"``.

   The auto-default for ``hessian_recompute_interval`` depends on the type
   of the initial Hessian:

   * ``"fischer"`` / ``"swart"`` / callback provided → ``50`` (minimisation)
     or ``5`` (saddle search, ``order >= 1``).
   * ``"identity"`` / user ``ndarray`` → ``0`` (off).

   When passing a pre-computed ``ndarray`` as the initial Hessian, set
   ``hessian_recompute_interval`` explicitly; the auto-default is 0 (off).

Algorithm summary
-----------------
1. Compute / read the current Cartesian gradient ``g`` (eV/Angstrom).
2. Possibly recompute the Hessian (every N steps), or quasi-Newton update.
3. Project translations (always) and rotations (when non-PBC) out of
   ``B`` and ``g``.
4. Diagonalise ``B`` and apply image-RFO projection
   ``P = I - 2 * sum_{i < order}(v_i v_i^T)`` to invert the curvature along
   the lowest ``order`` modes (Heyden et al., *J. Chem. Phys.* **123**,
   224101, 2005).
5. Optionally apply a level shift to the smallest eigenvalues.
6. Solve the restricted-step RFO problem in eigenbasis, varying ``alpha``
   until the step lies on the trust-radius sphere (Besalu & Bofill,
   *Theor. Chem. Acc.* **100**, 265, 1998).
7. Accept the step and update the trust radius adaptively (Fletcher 1987
   ratio test, with a curvature-based fall-back near convergence).

References
----------
* A. Banerjee, N. Adams, J. Simons, R. Shepard, *J. Phys. Chem.* **89**,
  52 (1985)
* J. Baker, *J. Comput. Chem.* **7**, 385 (1986)
* J. M. Bofill, *J. Comput. Chem.* **15**, 1 (1994)
* E. Besalu, J. M. Bofill, *Theor. Chem. Acc.* **100**, 265 (1998)
* A. Heyden, A. T. Bell, F. J. Keil, *J. Chem. Phys.* **123**, 224101 (2005)
* J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from ase.optimize.optimize import Optimizer

from .hessian_updaters import HessianUpdater
from .parameters import (
    BOHR_TO_ANGSTROM,
    HARTREE_PER_BOHR2_TO_EV_PER_A2,
)
from .constraints import (
    apply_freeze_diagonal,
    build_active_projector,
    detect_fixed_dofs,
    expand_from_active_subspace,
    reduce_to_active_subspace,
)
from .projections import (
    project_gradient,
    project_hessian,
    projection_modes_for_atoms,
)
from .rfo import restricted_step

# Model Hessians live in a separate module that may be omitted at distribution
# time without breaking the core optimiser. We import lazily inside helpers.
try:
    from .model_hessian import FischerD3ModelHessian, SwartD2ModelHessian
    _MODEL_HESSIAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    FischerD3ModelHessian = None  # type: ignore[assignment]
    SwartD2ModelHessian = None    # type: ignore[assignment]
    _MODEL_HESSIAN_AVAILABLE = False


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _model_hessian_to_ev_per_a2(H_au: np.ndarray) -> np.ndarray:
    """Convert Hartree/Bohr^2 to eV/Angstrom^2."""
    return H_au * HARTREE_PER_BOHR2_TO_EV_PER_A2


def numerical_hessian_from_forces(
    atoms,
    delta: float = 0.01,
    symmetrize: bool = True,
) -> np.ndarray:
    """Central-difference Hessian (eV/Angstrom^2) from an ASE calculator.

    Equivalent to the standard finite-difference recipe used by
    ``ase.vibrations.Vibrations``: H_ij = -(F_i(x_j+delta) - F_i(x_j-delta))
    / (2 * delta). Restores ``atoms.positions`` on exit.

    This is the recommended way to obtain the initial Hessian for a
    transition-state search.  Because :meth:`RSIRFO._maybe_recompute` always
    skips step 0, the periodic recompute schedule alone cannot supply the
    exact Hessian at the very first step; you must pass the result of this
    function explicitly as the ``hessian=`` argument::

        H0 = numerical_hessian_from_forces(atoms, delta=0.01)
        opt = RSIRFO(atoms, order=1, hessian=H0,
                     hessian_recompute_interval=5,
                     hessian_recompute_method="numerical")

    The cost is 6*N* single-point force evaluations (symmetric finite
    differences over all 3*N* Cartesian degrees of freedom).

    Parameters
    ----------
    atoms
        :class:`~ase.Atoms` with an attached calculator that provides forces.
    delta
        Finite-difference step in Angstrom (default 0.01).
    symmetrize
        Symmetrise the result with ``0.5 * (H + H.T)`` (default True).

    Returns
    -------
    H
        ``(3N, 3N)`` Hessian matrix in eV/Angstrom^2.
    """
    pos0 = atoms.get_positions().copy()
    n = len(atoms)
    ndof = 3 * n
    H = np.zeros((ndof, ndof), dtype=float)
    flat0 = pos0.flatten()
    try:
        for j in range(ndof):
            flat = flat0.copy()
            flat[j] = flat0[j] + delta
            atoms.set_positions(flat.reshape(-1, 3))
            f_plus = np.asarray(atoms.get_forces(), dtype=float).flatten()

            flat = flat0.copy()
            flat[j] = flat0[j] - delta
            atoms.set_positions(flat.reshape(-1, 3))
            f_minus = np.asarray(atoms.get_forces(), dtype=float).flatten()

            # Forces = -dE/dx, so the Hessian column is -dF/dx.
            H[:, j] = -(f_plus - f_minus) / (2.0 * delta)
    finally:
        atoms.set_positions(pos0)

    if symmetrize:
        H = 0.5 * (H + H.T)
    return H


# --------------------------------------------------------------------------- #
#  RS-I-RFO optimiser
# --------------------------------------------------------------------------- #


class RSIRFO(Optimizer):
    """Restricted-Step Image Rational Function Optimisation.

    Parameters
    ----------
    atoms
        An :class:`~ase.Atoms` instance with an attached calculator.
    restart
        File path for the optimiser pickle (Hessian, trust radius, history).
    logfile
        Log file name; ``'-'`` means stdout.
    trajectory
        Trajectory file name (``.traj``); ``None`` disables trajectory.
    append_trajectory
        If ``True`` append to an existing trajectory file.

    Keyword-only RS-I-RFO parameters
    --------------------------------
    order
        Saddle order: ``0`` for minima (default), ``1`` for transition
        states, ``n`` for ``n``-th order saddles.
    hessian
        Initial Hessian: ``'identity'`` (default), ``'fischer'``,
        ``'swart'``, or a user-supplied ``(3N, 3N)`` ``np.ndarray`` in
        eV/Angstrom^2.

        For transition-state searches, pass the result of
        :func:`numerical_hessian_from_forces` here to start from the exact
        numerical Hessian at step 0.  The periodic recompute schedule
        (``hessian_recompute_interval``) always skips step 0, so this is
        the only way to ensure the exact curvature is used from the very
        first step.  Remember to also set ``hessian_recompute_interval``
        explicitly, because the auto-default is 0 (off) for ``ndarray``
        inputs.
    fischer_functional
        D3 functional preset for the Fischer-D3 model Hessian
        (default ``'pbe0'``).
    swart_kwargs
        Additional keyword arguments forwarded to
        :class:`~.model_hessian.SwartD2ModelHessian`.
    hessian_update
        Quasi-Newton update method - any string in
        :pyattr:`HessianUpdater.METHODS` (default ``'auto'``).
    block_size, max_window, dd_mu1, dd_mu2
        Block (multi-secant) update parameters - see
        :class:`HessianUpdater`.
    hessian_recompute_interval
        Refresh the Hessian from scratch every ``N`` accepted steps.
        **Step 0 is always skipped** — the initial Hessian is always the
        value passed to ``hessian=``, regardless of this setting.
        Recomputation fires at iterations 1, 2, … when
        ``iteration % N == 0``.

        Default depends on the initial Hessian:

        * ``hessian='identity'`` or user-supplied ``ndarray`` → ``0``
          (off); set this explicitly when passing a pre-computed Hessian
          and periodic refresh is wanted.
        * ``hessian='fischer'`` / ``'swart'`` or ``hessian_callback``
          set → ``50`` (minimisation) or ``5`` (saddle search).

        Pass ``0`` explicitly to disable refresh entirely.
    hessian_recompute_method
        ``"model"``, ``"numerical"``, ``"callback"``, or ``None`` (auto -
        infer from the initial Hessian / callback presence).
    hessian_callback
        Optional ``Callable[[Atoms], np.ndarray]`` returning a Cartesian
        Hessian in eV/Angstrom^2; required for the ``"callback"`` method.
    numerical_hessian_step
        Finite-difference step (Angstrom) for the numerical Hessian.
    reset_history_on_recompute
        If ``True`` (default) wipe the secant-pair history when the Hessian
        is re-evaluated; the previous pairs are inconsistent with the new
        matrix.
    trust_radius
        Initial trust radius (Angstrom). Defaults: 0.3 for minima,
        0.2 for saddle search.
    trust_radius_min, trust_radius_max
        Bounds for the adaptive trust radius (Angstrom).
    alpha0, alpha_max, alpha_step_max, max_micro_cycles
        Restricted-step solver parameters; see :func:`.rfo.restricted_step`.
    use_adaptive_trust_radius
        If ``True`` (default), enlarge / shrink the trust radius based on
        Fletcher's actual-vs-predicted ratio.
    good_step_threshold, poor_step_threshold,
    trust_radius_increase_factor, trust_radius_decrease_factor
        Trust-radius adaptation thresholds and multipliers.
    adaptive_trust_gradient_norm_threshold, max_curvature_factor,
    negative_curvature_safety
        Curvature-based safety net used near convergence.
    use_level_shift, level_shift_value, auto_level_shift,
    condition_number_threshold, small_eigval_thresh
        Level-shifting parameters.
    project_translation, project_rotation
        Forces a particular projection. Default: from ``atoms.pbc``.
    constraint_method
        How to handle ASE constraints attached to ``atoms``. One of:

        * ``'auto'`` (default) -- use exact subspace reduction when only
          ``FixAtoms`` / ``FixCartesian`` constraints are present, fall
          back to ``'freeze'`` for any other constraint type.
        * ``'subspace'`` -- always reduce ``H``, ``g`` to the active
          subspace before solving the RFO step. Exact and numerically
          robust; works only for analytic Cartesian fixes.
        * ``'freeze'`` -- replace fixed rows/cols of ``H`` with a scaled
          identity (``freeze_value``) and zero the matching gradient
          components. Works for any constraint type at the cost of some
          conditioning.
        * ``'none'`` -- rely on ASE's standard ``adjust_forces`` /
          ``adjust_positions`` only. Not recommended for Newton-style
          optimisers as quasi-Newton updates corrupt the Hessian on the
          fixed DOFs over many iterations.
    freeze_value
        Diagonal value for ``constraint_method='freeze'`` (default 1e8).
        Should exceed the largest active Hessian eigenvalue by several
        orders of magnitude.
    """

    def __init__(
        self,
        atoms,
        restart: str | None = None,
        logfile: str | None = "-",
        trajectory: str | None = None,
        append_trajectory: bool = False,
        *,
        # ----- saddle order -------------------------------------------------
        order: int = 0,
        # ----- Hessian initialisation --------------------------------------
        hessian: str | np.ndarray = "identity",
        fischer_functional: str = "pbe0",
        swart_kwargs: dict[str, Any] | None = None,
        # ----- Hessian updates ---------------------------------------------
        hessian_update: str | None = None,
        block_size: int = 4,
        max_window: int = 8,
        dd_mu1: float = 0.2,
        dd_mu2: float = 0.2,
        # ----- verbose output ----------------------------------------------
        verbose: bool = True,
        eigval_log_count: int = 10,
        # ----- Hessian recomputation ---------------------------------------
        hessian_recompute_interval: int | None = None,
        hessian_recompute_method: str | None = None,
        hessian_callback: Callable[[Any], np.ndarray] | None = None,
        numerical_hessian_step: float = 0.01,
        reset_history_on_recompute: bool = True,
        # ----- trust-radius / RFO solver -----------------------------------
        trust_radius: float | None = None,
        trust_radius_min: float = 1.0e-3,
        trust_radius_max: float | None = None,
        alpha0: float = 1.0,
        alpha_max: float = 1.0e3,
        alpha_step_max: float = 10.0,
        max_micro_cycles: int = 40,
        # ----- adaptive trust-radius ---------------------------------------
        use_adaptive_trust_radius: bool = True,
        good_step_threshold: float = 0.75,
        poor_step_threshold: float = 0.25,
        trust_radius_increase_factor: float = 1.4,
        trust_radius_decrease_factor: float = 0.5,
        adaptive_trust_gradient_norm_threshold: float = 1.0e-2,
        max_curvature_factor: float = 5.0,
        negative_curvature_safety: float = 0.5,
        # ----- level shifting ----------------------------------------------
        use_level_shift: bool = True,
        level_shift_value: float = 1.0e-3,
        auto_level_shift: bool = True,
        condition_number_threshold: float = 1.0e8,
        small_eigval_thresh: float = 1.0e-6,
        # ----- projections -------------------------------------------------
        project_translation: bool | None = None,
        project_rotation: bool | None = None,
        # ----- ASE constraint handling -------------------------------------
        constraint_method: str = "auto",
        freeze_value: float = 1.0e8,
        # ----- pass-through ASE kwargs -------------------------------------
        **kwargs,
    ) -> None:

        # =====================================================================
        # Every attribute that may be touched by `self.read()` (the restart
        # loader called from inside `Optimizer.__init__`) MUST be initialised
        # before we call `super().__init__()`.
        # =====================================================================

        # --- saddle order & projections -----------------------------------
        if int(order) < 0:
            raise ValueError("Saddle order must be non-negative.")
        self.order = int(order)

        # --- ASE constraint handling --------------------------------------
        # constraint_method:
        #   'auto'    -> 'subspace' for pure FixAtoms / FixCartesian,
        #                'freeze' as a fall-back for any other constraint type
        #   'subspace' -> always reduce H, g to the active subspace (exact)
        #   'freeze'   -> always overwrite fixed rows/cols with a level shift
        #   'none'     -> rely entirely on ASE's adjust_forces / adjust_positions
        cm = str(constraint_method).lower()
        if cm not in ("auto", "subspace", "freeze", "none"):
            raise ValueError(
                f"Unknown constraint_method {constraint_method!r}. "
                "Use 'auto', 'subspace', 'freeze' or 'none'."
            )
        self.constraint_method = cm
        self.freeze_value = float(freeze_value)
        # Fixed mask is recomputed every step (constraints might be modified
        # between calls), but cache the immutable atom count for sanity checks.
        self._n_atoms_init = len(atoms) if atoms is not None else 0

        auto_t, auto_r = projection_modes_for_atoms(atoms)
        self.project_translation = (
            bool(auto_t) if project_translation is None
            else bool(project_translation)
        )
        self.project_rotation = (
            bool(auto_r) if project_rotation is None
            else bool(project_rotation)
        )

        # --- trust radius -------------------------------------------------
        # Saddle-order-aware defaults (cf. original multioptpy implementation):
        # large maximum trust radius is fine for minima but too generous for
        # TS searches where the eigenvalue spectrum has mixed signs.
        if trust_radius is None:
            trust_radius = 0.3 if self.order == 0 else 0.2
        if trust_radius_max is None:
            trust_radius_max = 0.5 if self.order == 0 else 0.2
        self.trust_radius = float(trust_radius)
        self.trust_radius_min = float(trust_radius_min)
        self.trust_radius_max = float(trust_radius_max)

        # --- RS-RFO parameters --------------------------------------------
        self.alpha0 = float(alpha0)
        self.alpha_max = float(alpha_max)
        self.alpha_step_max = float(alpha_step_max)
        self.max_micro_cycles = int(max_micro_cycles)

        # --- trust-radius adaptation --------------------------------------
        self.use_adaptive_trust_radius = bool(use_adaptive_trust_radius)
        self.good_step_threshold = float(good_step_threshold)
        self.poor_step_threshold = float(poor_step_threshold)
        self.trust_radius_increase_factor = float(trust_radius_increase_factor)
        self.trust_radius_decrease_factor = float(trust_radius_decrease_factor)
        self.adaptive_trust_gradient_norm_threshold = float(
            adaptive_trust_gradient_norm_threshold
        )
        self.max_curvature_factor = float(max_curvature_factor)
        self.negative_curvature_safety = float(negative_curvature_safety)

        # --- level shifting ------------------------------------------------
        self.use_level_shift = bool(use_level_shift)
        self.level_shift_value = float(level_shift_value)
        self.auto_level_shift = bool(auto_level_shift)
        self.condition_number_threshold = float(condition_number_threshold)
        self.small_eigval_thresh = float(small_eigval_thresh)

        # --- Verbose output -----------------------------------------------
        self.verbose = bool(verbose)
        self.eigval_log_count = int(eigval_log_count)

        # --- Hessian updater & initialisation spec ------------------------
        # Default update method depends on saddle order:
        #   order = 0 (minimisation)   -> block_fsb (multi-secant FSB,
        #                                  empirically best for minima)
        #   order >= 1 (saddle search) -> block_bofill (SR1/PSB mix is
        #                                  the standard choice for TS)
        if hessian_update is None:
            hessian_update = "block_fsb" if self.order == 0 else "block_bofill"
        self.hessian_update_method = str(hessian_update)
        self.updater = HessianUpdater(
            block_size=block_size,
            max_window=max_window,
            dd_mu1=dd_mu1,
            dd_mu2=dd_mu2,
        )
        self._hessian_init_spec = hessian
        self._fischer_functional = fischer_functional
        self._swart_kwargs = dict(swart_kwargs or {})

        # --- Hessian recomputation policy ---------------------------------
        # Step 0 is always skipped by _maybe_recompute (see below).  The
        # initial Hessian is therefore always whatever is passed via hessian=.
        # To start a TS search from the exact numerical Hessian, call
        # numerical_hessian_from_forces() before constructing RSIRFO and pass
        # the result as hessian=.  Then set hessian_recompute_interval
        # explicitly, because the auto-default for ndarray inputs is 0 (off).
        #
        # Auto-defaults:
        #   identity / user-ndarray   -> 0 (off)
        #   model ('fischer'/'swart') -> 50 (minimisation) or 5 (TS search)
        #   callback provided         -> same as model (user wants refresh)
        #
        # Pass hessian_recompute_interval=0 to disable refresh entirely.
        if hessian_recompute_interval is None:
            init_is_model = (
                isinstance(hessian, str)
                and hessian.lower() in ("fischer", "swart")
            )
            wants_refresh = init_is_model or (hessian_callback is not None)
            if wants_refresh:
                hessian_recompute_interval = 50 if self.order == 0 else 5
            else:
                hessian_recompute_interval = 0
        if int(hessian_recompute_interval) < 0:
            raise ValueError("hessian_recompute_interval must be >= 0.")
        self.hessian_recompute_interval = int(hessian_recompute_interval)
        self.hessian_recompute_method = (
            None if hessian_recompute_method is None
            else str(hessian_recompute_method).lower()
        )
        if (
            self.hessian_recompute_method is not None
            and self.hessian_recompute_method
            not in ("model", "numerical", "callback")
        ):
            raise ValueError(
                f"Unknown hessian_recompute_method "
                f"{self.hessian_recompute_method!r}. "
                "Use 'model', 'numerical', 'callback' or None (auto)."
            )
        self.hessian_callback = hessian_callback
        self.numerical_hessian_step = float(numerical_hessian_step)
        self.reset_history_on_recompute = bool(reset_history_on_recompute)

        # --- per-step state -----------------------------------------------
        self._hessian: np.ndarray | None = None
        self._prev_positions: np.ndarray | None = None
        self._prev_gradient: np.ndarray | None = None
        self._prev_energy: float | None = None
        self._predicted_energy_change: float | None = None
        self._iteration: int = 0
        # Energy-change history (last 3 pairs) for step-quality assessment
        self._predicted_energy_history: list[float] = []
        self._actual_energy_history: list[float] = []

        # --- ASE base class -----------------------------------------------
        Optimizer.__init__(
            self,
            atoms=atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

    # ----------------------------------------------------- Hessian accessors
    #
    # Three ways to read the optimiser's internal Hessian, in order of
    # increasing post-processing:
    #
    #   * :meth:`get_raw_hessian` -- the bare quasi-Newton matrix as stored.
    #     No projection, no constraint handling. Use this if you want the
    #     full Cartesian curvature signal without any modes removed.
    #
    #   * :meth:`get_hessian` -- one-stop accessor with explicit knobs.
    #     Defaults match what the optimiser uses internally for the RFO
    #     solve at the *current* step (T/R projection from constraint
    #     state, no constraint freeze applied). Override the keyword
    #     arguments to switch on/off any post-processing step.
    #
    #   * the :pyattr:`hessian` property -- backward-compatible shortcut
    #     equivalent to ``get_hessian(project_tr=True)`` with the default
    #     ``project_translation`` / ``project_rotation`` flags from the
    #     constructor. Returns ``None`` before the first step.
    #
    # The setter on the property writes to the raw store (no projection).

    def get_raw_hessian(self) -> "np.ndarray | None":
        """Return the raw Cartesian Hessian without any projection.

        This is the matrix that the quasi-Newton update writes into; it
        contains the full curvature signal including translation /
        rotation modes and any DOFs frozen by ASE constraints. Useful for
        debugging Hessian-update behaviour or for serialising the matrix
        to a file.

        Returns
        -------
        H : np.ndarray of shape (3*N, 3*N), or None
            The stored Hessian, or ``None`` if no step has been taken yet.
        """
        if self._hessian is None:
            return None
        return np.array(self._hessian, copy=True)

    def get_hessian(
        self,
        *,
        project_tr: bool | None = None,
        apply_constraints: bool = False,
    ) -> "np.ndarray | None":
        """Return the optimiser's Hessian with optional post-processing.

        Parameters
        ----------
        project_tr
            If ``True``, remove translation / rotation modes via
            ``P^T H P`` (using the optimiser's stored
            :pyattr:`project_translation` / :pyattr:`project_rotation`
            flags). If ``False``, skip T/R projection. If ``None``
            (default), project T/R only when no fixed atoms are present
            (matching the constraint-aware logic used at step time).
        apply_constraints
            If ``True``, additionally apply the freeze-diagonal treatment
            for ASE constraints attached to ``atoms`` (so that fixed
            Cartesian DOFs have their rows/columns replaced by a scaled
            identity, ``freeze_value``). Default ``False`` to keep the
            output free of artificial values that would distort an
            eigenvalue analysis.

        Returns
        -------
        H : np.ndarray of shape (3*N, 3*N), or None
            The processed Hessian, or ``None`` if no step has been taken.
        """
        if self._hessian is None:
            return None

        H = np.array(self._hessian, copy=True)
        positions = getattr(self.atoms, "get_positions", lambda: None)()

        # --- decide T/R projection ------------------------------------
        if project_tr is None:
            try:
                fixed_mask, _has_internal, _has_other = detect_fixed_dofs(
                    self.atoms
                )
                n_fixed = int(np.sum(fixed_mask))
            except Exception:  # noqa: BLE001
                n_fixed = 0
            do_tr = (n_fixed == 0)
        else:
            do_tr = bool(project_tr)

        if do_tr and positions is not None:
            H = project_hessian(
                H,
                positions,
                project_translation=self.project_translation,
                project_rotation=self.project_rotation,
            )

        # --- optional constraint freeze --------------------------------
        if apply_constraints:
            try:
                fixed_mask, _hi, _ho = detect_fixed_dofs(self.atoms)
                if int(np.sum(fixed_mask)) > 0:
                    g_dummy = np.zeros(H.shape[0])
                    apply_freeze_diagonal(
                        H, g_dummy, fixed_mask,
                        freeze_value=self.freeze_value,
                    )
            except Exception:  # noqa: BLE001
                pass

        return H

    @property
    def hessian(self) -> "np.ndarray | None":
        """Return the T/R-projected Hessian (backward-compatible shortcut).

        For more control, use :meth:`get_hessian` (explicit projection
        flags) or :meth:`get_raw_hessian` (no projection).
        """
        return self.get_hessian(project_tr=True, apply_constraints=False)

    @hessian.setter
    def hessian(self, value: "np.ndarray | None") -> None:
        """Store raw Hessian (no projection applied on write)."""
        self._hessian = None if value is None else np.asarray(value, dtype=float)

    # ----------------------------------------------------------------- meta
    def initialize(self) -> None:
        """Called by :class:`Optimizer` when no restart file exists."""
        return

    # ------------------------------------------------------- restart support
    def todict(self) -> dict[str, Any]:
        """Serialise mutable optimiser state for restart."""
        return {
            "type": "optimization",
            "optimizer": "RSIRFO",
            "iteration": self._iteration,
            "trust_radius": self.trust_radius,
            "hessian": (
                None if self._hessian is None else np.asarray(self._hessian)
            ),
            "prev_positions": (
                None if self._prev_positions is None
                else np.asarray(self._prev_positions)
            ),
            "prev_gradient": (
                None if self._prev_gradient is None
                else np.asarray(self._prev_gradient)
            ),
            "prev_energy": self._prev_energy,
            "S_history": [s.copy() for s in self.updater.S_history],
            "Y_history": [y.copy() for y in self.updater.Y_history],
            "predicted_energy_history": list(self._predicted_energy_history),
            "actual_energy_history": list(self._actual_energy_history),
        }

    def read(self) -> None:  # noqa: D401 - ASE name
        """Read an optimiser pickle written by :meth:`Optimizer.dump`."""
        try:
            data = self.load()
        except (FileNotFoundError, EOFError):
            return
        if not isinstance(data, dict):
            return
        self._iteration = int(data.get("iteration", 0))
        self.trust_radius = float(
            data.get("trust_radius", self.trust_radius)
        )
        h = data.get("hessian", None)
        self._hessian = None if h is None else np.asarray(h, dtype=float)
        pp = data.get("prev_positions", None)
        self._prev_positions = (
            None if pp is None else np.asarray(pp, dtype=float)
        )
        pg = data.get("prev_gradient", None)
        self._prev_gradient = (
            None if pg is None else np.asarray(pg, dtype=float)
        )
        self._prev_energy = data.get("prev_energy", None)
        self.updater.reset_history()
        for s, y in zip(
            data.get("S_history", []), data.get("Y_history", [])
        ):
            self.updater.S_history.append(np.asarray(s, dtype=float))
            self.updater.Y_history.append(np.asarray(y, dtype=float))
        self._predicted_energy_history = list(
            data.get("predicted_energy_history", [])
        )
        self._actual_energy_history = list(
            data.get("actual_energy_history", [])
        )

    # ------------------------------------------------------- initial Hessian
    def _build_model_hessian(self) -> np.ndarray:
        """Build the initial / refresh model Hessian at the current geometry."""
        if not _MODEL_HESSIAN_AVAILABLE:
            raise RuntimeError(
                "Model Hessians (fischer / swart) are not available - "
                "the optional `model_hessian` module is missing."
            )
        spec = self._hessian_init_spec
        if not isinstance(spec, str):
            raise RuntimeError(
                "Cannot build a model Hessian: the initial Hessian was not "
                "specified as 'fischer' or 'swart'."
            )
        kind = spec.lower()
        coord_bohr = (
            np.asarray(self.atoms.get_positions(), dtype=float)
            / BOHR_TO_ANGSTROM
        )
        elements = [str(s) for s in self.atoms.get_chemical_symbols()]
        if kind == "fischer":
            generator = FischerD3ModelHessian(
                d3_functional=self._fischer_functional
            )
        elif kind == "swart":
            generator = SwartD2ModelHessian(**self._swart_kwargs)
        else:
            raise RuntimeError(
                f"Initial Hessian {spec!r} is not a model Hessian; cannot "
                "rebuild via 'model' method."
            )
        H_au = generator.build(coord_bohr, elements)
        return _model_hessian_to_ev_per_a2(H_au)

    def _build_initial_hessian(self) -> np.ndarray:
        spec = self._hessian_init_spec
        n_atoms = len(self.atoms)
        ndof = 3 * n_atoms

        if isinstance(spec, np.ndarray):
            H = np.asarray(spec, dtype=float)
            if H.shape != (ndof, ndof):
                raise ValueError(
                    f"User-provided Hessian has shape {H.shape}, "
                    f"expected ({ndof}, {ndof})."
                )
            return 0.5 * (H + H.T)

        if not isinstance(spec, str):
            raise TypeError(
                f"Unsupported `hessian` argument: {type(spec).__name__}"
            )
        kind = spec.lower()
        if kind == "identity":
            return np.eye(ndof, dtype=float)
        if kind in ("fischer", "swart"):
            return self._build_model_hessian()
        raise ValueError(
            f"Unknown initial Hessian type {spec!r}. "
            "Use 'identity', 'fischer', 'swart' or pass a NumPy array."
        )

    # ---------------------------------------------------- recomputation
    def _resolve_recompute_method(self) -> str:
        """Pick a recomputation method when the user left it as ``None``."""
        if self.hessian_recompute_method is not None:
            return self.hessian_recompute_method
        spec = self._hessian_init_spec
        if isinstance(spec, str) and spec.lower() in ("fischer", "swart"):
            return "model"
        if self.hessian_callback is not None:
            return "callback"
        return "numerical"

    def _recompute_hessian(self) -> np.ndarray:
        """Return a freshly computed Hessian in eV/Angstrom^2."""
        method = self._resolve_recompute_method()
        if method == "model":
            self._log("  [Hessian refresh] rebuilding model Hessian.")
            return self._build_model_hessian()
        if method == "callback":
            if self.hessian_callback is None:
                raise RuntimeError(
                    "hessian_recompute_method='callback' requires "
                    "hessian_callback to be supplied."
                )
            self._log("  [Hessian refresh] calling user callback.")
            H = np.asarray(self.hessian_callback(self.atoms), dtype=float)
            ndof = 3 * len(self.atoms)
            if H.shape != (ndof, ndof):
                raise ValueError(
                    f"Callback returned Hessian shape {H.shape}, "
                    f"expected ({ndof}, {ndof})."
                )
            return 0.5 * (H + H.T)
        # numerical
        self._log(
            "  [Hessian refresh] computing numerical Hessian "
            f"(delta={self.numerical_hessian_step:.4f} A)."
        )
        return numerical_hessian_from_forces(
            self.atoms, delta=self.numerical_hessian_step
        )

    def _maybe_recompute(self) -> bool:
        """Refresh the Hessian if the recompute schedule says so.

        Step 0 (``self._iteration <= 0``) is **always skipped**.  The initial
        Hessian is always whatever was passed to ``hessian=`` at construction
        time.  Recomputation can only happen at iteration 1 or later, when
        ``iteration % hessian_recompute_interval == 0``.

        Returns ``True`` if the Hessian was successfully replaced, ``False``
        otherwise (schedule not due, interval disabled, or recompute failed).
        """
        if (
            self.hessian_recompute_interval <= 0
            or self._iteration <= 0
            or self._iteration % self.hessian_recompute_interval != 0
        ):
            return False
        try:
            self._hessian = self._recompute_hessian()
        except Exception as exc:  # noqa: BLE001
            self._log(f"  [Hessian refresh] FAILED: {exc!r}; keeping previous.")
            return False
        if self.reset_history_on_recompute:
            self.updater.reset_history()
        return True

    # --------------------------------------------------------- step driver
    def step(self, forces: np.ndarray | None = None) -> None:
        """Compute and apply one RS-I-RFO step."""
        atoms = self.atoms
        if forces is None:
            forces = atoms.get_forces()
        positions = atoms.get_positions()
        try:
            energy = atoms.get_potential_energy()
        except Exception:  # noqa: BLE001
            energy = None

        gradient = -np.asarray(forces, dtype=float).flatten()
        positions_flat = np.asarray(positions, dtype=float).flatten()

        # ----- Detect ASE constraints --------------------------------------
        # Re-evaluated every step: the user may add/remove constraints
        # between calls (e.g. inside an ASE Filter wrapper).
        fixed_mask, has_internal_constraints, has_other_constraints = (
            detect_fixed_dofs(atoms)
        )
        n_fixed = int(np.sum(fixed_mask))
        if self.constraint_method == "auto":
            # Decision rules for 'auto':
            #   pure FixAtoms / FixCartesian       -> 'none'
            #     ASE's adjust_positions handles fixed atoms directly; our
            #     secant-pair zeroing protects the Hessian.
            #   internal-coordinate constraint     -> 'none'
            #     ASE iteratively projects forces / positions; T/R stays on
            #     because rigid-body modes do not interfere with such
            #     constraints (see the constraint section below).
            #   any other (unknown) constraint     -> 'freeze' diagonal
            #     conservative safety net.
            if has_other_constraints:
                active_method = "freeze"
            else:
                active_method = "none"
        elif self.constraint_method == "none":
            active_method = "none"
        else:
            active_method = self.constraint_method

        if (n_fixed > 0 or has_internal_constraints
                or has_other_constraints) and self._iteration == 0:
            tags = []
            if n_fixed:
                tags.append(f"{n_fixed} Cartesian DOF(s)")
            if has_internal_constraints:
                tags.append("internal-coord constraint")
            if has_other_constraints:
                tags.append("other constraint(s)")
            self._log(
                f"  [Constraints] detected: {', '.join(tags)}; "
                f"using method='{active_method}'"
            )

        # ----- Hessian initialisation -------------------------------------
        if self._hessian is None:
            self._hessian = self._build_initial_hessian()

        # ----- T/R projection decision (constraint-aware) -----------------
        # Three cases drive whether translational / rotational projection is
        # needed for THIS step:
        #
        # 1. Atom-fix only (FixAtoms / FixCartesian, no internal constraints)
        #    -> SKIP T/R. The fixed atoms already break continuous T/R
        #       symmetry, and projecting them out would discard legitimate
        #       DOFs of the moving atoms.
        #
        # 2. Internal-coordinate constraint only (FixBondLength,
        #    FixInternals, FixedPlane, FixedLine, Hookean, ...) and no fixed
        #    atom -> KEEP T/R. Internal coordinates are invariant under
        #    rigid translation/rotation, so the rigid-body modes still
        #    represent zero-energy directions and must be projected out.
        #    Failing to project would let alpha drift in the rigid-body
        #    subspace and slow convergence dramatically.
        #
        # 3. Mixed (atom-fix AND internal-coord) -> SKIP T/R. Atom fixing
        #    dominates: rigid-body modes are already absent of physical
        #    meaning because some atoms cannot move. Internal-coord
        #    constraints are themselves T/R-invariant, so this is consistent.
        if n_fixed > 0:
            # Cases 1 and 3: atom fixing present
            skip_tr_for_constraints = True
        elif has_internal_constraints:
            # Case 2: internal-coord only -> keep T/R explicitly
            skip_tr_for_constraints = False
        else:
            # No constraints: keep the user's / PBC-derived defaults
            skip_tr_for_constraints = False
        proj_t_eff = self.project_translation and not skip_tr_for_constraints
        proj_r_eff = self.project_rotation and not skip_tr_for_constraints
        g_proj = project_gradient(
            gradient,
            positions,
            project_translation=proj_t_eff,
            project_rotation=proj_r_eff,
        )

        # ----- Periodic refresh OR quasi-Newton update --------------------
        # Uses the raw gradient (not projected) for the secant vectors y and s.
        # T/R contamination that enters self._hessian this way is harmless:
        # the RFO step always reads H_proj = P^T H P (projected each step below),
        # and the public hessian property also applies the projection on read.
        # Using raw y preserves the full curvature signal and matches the
        # original convergence speed; projecting y reduces information content
        # and slows quasi-Newton convergence.
        refreshed = self._maybe_recompute()
        if (
            not refreshed
            and self._iteration > 0
            and self._prev_positions is not None
            and self._prev_gradient is not None
        ):
            s = positions_flat - self._prev_positions
            y = gradient - self._prev_gradient   # raw secant pair
            # Constraint-aware quasi-Newton: zero out the secant components
            # on fixed DOFs so that the Hessian update sees no spurious
            # information on the constrained subspace. This protects the
            # *active* part of the Hessian over many iterations.
            if n_fixed > 0:
                s = s.copy()
                y = y.copy()
                s[fixed_mask] = 0.0
                y[fixed_mask] = 0.0
            disp_norm = float(np.linalg.norm(s))
            grad_diff_norm = float(np.linalg.norm(y))
            sy = float(np.dot(s, y))
            # Quasi-Newton update guards (cf. Nocedal & Wright eq. 6.7):
            # skip when (a) the displacement is numerically zero,
            # (b) the gradient barely changed, or (c) the curvature condition
            # s·y > 0 fails for BFGS-flavoured updates. SR1 / Bofill / PSB do
            # not strictly need s·y > 0, so we still update those even when
            # the condition fails - the dispatcher will pick a robust formula.
            if disp_norm < 1e-10 or grad_diff_norm < 1e-10:
                self._log(
                    f"  [Hessian update] SKIPPED |s|={disp_norm:.2e}, "
                    f"|y|={grad_diff_norm:.2e}"
                )
            else:
                update_method = self.hessian_update_method.lower()
                bfgs_only = "bfgs" in update_method and "block_bfgs" not in update_method
                if bfgs_only and sy <= 0.0:
                    self._log(
                        f"  [Hessian update] SKIPPED (s.y = {sy:.2e} <= 0; "
                        "BFGS would lose positive definiteness)"
                    )
                else:
                    delta = self.updater.update(
                        self._hessian, s, y, method=self.hessian_update_method
                    )
                    self._hessian = self._hessian + delta
                    self._hessian = 0.5 * (self._hessian + self._hessian.T)

        # ----- Adaptive trust radius (Fletcher ratio) ---------------------
        if (
            self.use_adaptive_trust_radius
            and self._iteration > 0
            and energy is not None
            and self._prev_energy is not None
            and self._predicted_energy_change is not None
        ):
            self._update_trust_radius(float(energy - self._prev_energy))

        # ----- T/R projection of Hessian (applied every step, raw H kept) --
        # self._hessian is never modified here; H_proj is a local projected copy
        # used only for the RFO eigensolve this step.
        # g_proj was already computed above before the quasi-Newton update.
        H_proj = project_hessian(
            self._hessian,
            positions,
            project_translation=proj_t_eff,
            project_rotation=proj_r_eff,
        )

        # ----- Apply ASE constraints to the projected H, g ----------------
        # Two equivalent strategies (cf. constraints.py docstring):
        # 1. "subspace" - exact projection P H P^T, P g; the eigensolve runs
        #    on a smaller matrix so the fixed DOFs literally do not exist.
        # 2. "freeze"   - replace fixed rows/cols with a scaled identity;
        #    matrix shape is preserved but the RFO step components on the
        #    fixed DOFs are numerically zero.
        # Either way the secant-pair zeroing above already guards the raw
        # self._hessian against pollution.
        constraint_projector = None  # set when we use subspace reduction
        if n_fixed > 0 and active_method == "subspace":
            constraint_projector = build_active_projector(fixed_mask)
            H_proj, g_proj = reduce_to_active_subspace(
                H_proj, g_proj, constraint_projector
            )
        elif n_fixed > 0 and active_method == "freeze":
            # Make sure we work on copies so self._hessian stays untouched.
            H_proj = H_proj.copy()
            g_proj = g_proj.copy()
            apply_freeze_diagonal(
                H_proj, g_proj, fixed_mask, freeze_value=self.freeze_value
            )

        # ----- Eigendecompose & filter near-zero modes --------------------
        try:
            eigvals_all, eigvecs_all = np.linalg.eigh(H_proj)
        except np.linalg.LinAlgError as exc:
            # Critical fallback: Hessian eigendecomposition failed entirely.
            # Reset to the identity matrix so that the next step is a pure
            # steepest-descent move (Nocedal & Wright sec. 3.3).
            self._log(
                f"  [CRITICAL] Hessian eigendecomposition failed: {exc!r}; "
                "resetting to identity (steepest-descent fallback)"
            )
            ndof = H_proj.shape[0]
            self._hessian = np.eye(ndof)
            self.updater.reset_history()
            eigvals_all, eigvecs_all = np.linalg.eigh(self._hessian)

        # Detect NaN/Inf even when eigh succeeds (rare but happens with
        # extreme conditioning) and trigger the same identity fallback.
        if not (np.all(np.isfinite(eigvals_all))
                and np.all(np.isfinite(eigvecs_all))):
            self._log(
                "  [CRITICAL] NaN/Inf in Hessian eigendecomposition; "
                "resetting to identity (steepest-descent fallback)"
            )
            ndof = H_proj.shape[0]
            self._hessian = np.eye(ndof)
            self.updater.reset_history()
            eigvals_all, eigvecs_all = np.linalg.eigh(self._hessian)

        keep = np.abs(eigvals_all) > self.small_eigval_thresh
        eigvals = eigvals_all[keep]
        eigvecs = eigvecs_all[:, keep]

        if eigvals.size == 0:
            self._log("All Hessian modes filtered out; aborting step.")
            return

        # ----- Verbose: log Hessian eigenvalues ---------------------------
        if self.verbose:
            self._log_eigvals(eigvals_all, eigvals)

        # ----- Level shift ------------------------------------------------
        eigvals = self._apply_level_shift(eigvals)

        # ----- Project gradient into eigenbasis ---------------------------
        g_components = eigvecs.T @ g_proj

        # ----- Image-RFO: apply the reflection P = I - 2 sum v_i v_i^T ----
        # to BOTH the Hessian and the gradient. In the eigenbasis of the
        # original Hessian this is equivalent to flipping the sign of both
        # the eigenvalue AND the gradient component along the imaged modes.
        # Flipping the eigenvalue alone (the common pitfall) reduces I-RFO
        # to ordinary minimisation along that mode - the step ends up going
        # downhill instead of climbing toward the saddle point.
        if self.order > 0:
            n_invert = min(self.order, eigvals.size)
            eigvals = eigvals.copy()
            g_components = g_components.copy()
            eigvals[:n_invert] *= -1.0
            g_components[:n_invert] *= -1.0
            self._log(
                f"  [I-RFO] inverted {n_invert} mode(s); imaged eigvals = "
                + ", ".join(f"{v:+.4e}" for v in eigvals[:n_invert])
            )

        # ----- Initial-alpha early-out ------------------------------------
        # If the unrestricted RFO step at alpha = alpha0 already lies inside
        # the trust radius, use it directly without iterating on alpha. Saves
        # 5-40 secular-equation evaluations per step in well-behaved cases.
        from .rfo import solve_rfo
        initial_step, initial_lam = solve_rfo(
            eigvals, g_components, self.alpha0, log=self._log
        )
        initial_norm = float(np.linalg.norm(initial_step))
        if initial_norm <= self.trust_radius:
            self._log(
                f"  [RFO] initial step at alpha={self.alpha0:.3g} has "
                f"|s|={initial_norm:.4e} <= TR={self.trust_radius:.4e}; "
                "skipping alpha search"
            )
            step_eig = initial_step
            step_norm = initial_norm
            alpha = self.alpha0
        else:
            # ----- Restricted-step search over alpha --------------------------
            step_eig, step_norm, alpha = restricted_step(
                eigvals,
                g_components,
                trust_radius=self.trust_radius,
                alpha_init=self.alpha0,
                alpha_max=self.alpha_max,
                alpha_step_max=self.alpha_step_max,
                max_micro_cycles=self.max_micro_cycles,
                log=self._log,
            )

        # ----- Hard trust-radius cap (safety net) -------------------------
        # The Newton fall-back inside restricted_step can occasionally fail
        # to honour the trust radius - particularly during early iterations
        # of an image-RFO saddle search when the Hessian eigenvalue
        # spectrum has mixed signs. Scaling the step uniformly preserves
        # its direction (which encodes the eigenvector-following choice)
        # while strictly enforcing the TR bound.
        if step_norm > self.trust_radius * 1.001:
            scale = self.trust_radius / step_norm
            step_eig = step_eig * scale
            self._log(
                f"  [TR cap] step rescaled by {scale:.4e} "
                f"(|s| {step_norm:.3e} -> {self.trust_radius:.3e})"
            )
            step_norm = float(np.linalg.norm(step_eig))

        # ----- Back-transform to Cartesian --------------------------------
        delta_x = eigvecs @ step_eig

        # ----- Predict energy change for next-step trust-radius update ---
        # The model E_pred = g.T s + 0.5 s.T H s must be computed in the same
        # space as the step, so this happens BEFORE expanding back to the
        # full Cartesian dimension.
        self._predicted_energy_change = float(
            np.dot(g_proj, delta_x)
            + 0.5 * float(delta_x @ H_proj @ delta_x)
        )

        # ----- Expand step from active subspace back to full Cartesian ----
        # When constraint_method='subspace' was used, delta_x lives in the
        # K-dimensional active subspace; re-inject zeros on the fixed DOFs.
        if constraint_projector is not None:
            delta_x_full = expand_from_active_subspace(
                delta_x, constraint_projector
            )
        else:
            delta_x_full = delta_x

        # ----- NaN safety net on the computed step ------------------------
        if not np.all(np.isfinite(delta_x_full)):
            self._log(
                "  [CRITICAL] Computed step contains NaN/Inf; "
                "falling back to steepest descent within trust radius"
            )
            # Use the full-dimensional gradient for the steepest-descent
            # fall-back; ASE's adjust_forces zeroed the fixed components, and
            # adjust_positions will silently keep them fixed even if we leak
            # a tiny number from rounding.
            delta_x_full = -np.asarray(forces).flatten()
            if n_fixed > 0:
                delta_x_full = delta_x_full.copy()
                delta_x_full[fixed_mask] = 0.0
            norm = float(np.linalg.norm(delta_x_full))
            if norm > 1e-12:
                delta_x_full = delta_x_full * (self.trust_radius / norm)
            else:
                delta_x_full = np.zeros(3 * len(atoms))
            step_norm = float(np.linalg.norm(delta_x_full))

        # ----- Apply step --------------------------------------------------
        new_positions = positions + delta_x_full.reshape(positions.shape)
        atoms.set_positions(new_positions)

        # ----- Update prediction-vs-actual history & quality assessment -
        if self._iteration > 0 and energy is not None and self._prev_energy is not None:
            actual_change = float(energy - self._prev_energy)
            # Stale predicted_energy_change here is the value from the
            # PREVIOUS iteration (used in _update_trust_radius this turn);
            # store both for quality bookkeeping before overwriting.
            if self._predicted_energy_change is not None:
                self._actual_energy_history.append(actual_change)
                if len(self._actual_energy_history) > 3:
                    self._actual_energy_history.pop(0)

        # Push the predicted change for THIS step onto the history
        self._predicted_energy_history.append(self._predicted_energy_change or 0.0)
        if len(self._predicted_energy_history) > 3:
            self._predicted_energy_history.pop(0)

        if self.verbose:
            self._evaluate_step_quality()

        # ----- Persist previous-step state --------------------------------
        self._prev_positions = positions_flat.copy()
        self._prev_gradient = gradient.copy()   # raw: secant pair y = g_t - g_{t-1} uses raw
        self._prev_energy = energy
        self._iteration += 1

        gnorm = float(np.linalg.norm(g_proj))
        self._log(
            f"  RFO step #{self._iteration}: |g|={gnorm:.4e}  "
            f"|s|={step_norm:.4e}  alpha={alpha:.3g}  "
            f"TR={self.trust_radius:.3f}"
            + ("  [refreshed]" if refreshed else "")
        )

        # Persist optimiser state on each step.
        self.dump(self.todict())

    # ---------------------------------------------------- helper routines
    def _update_trust_radius(self, actual_change: float) -> None:
        """Adaptive trust-radius adjustment.

        Uses a 5-tier ratio test (Nocedal & Wright 2006, Fletcher 1987 with
        a fine-grained band structure):

        =====================  ==========================
        ``ratio``              action
        =====================  ==========================
        > 0.75 (excellent)     aggressively increase
        0.50-0.75 (good)       moderately increase
        0.25-0.50 (acceptable) maintain or slow expansion
        0.10-0.25 (poor)       halve
        < 0.10 (very poor)     quarter
        =====================  ==========================

        Below the small-gradient threshold the increase factor is also
        scaled by ``1 / max(|lambda_min|, 0.1)`` (capped at
        ``max_curvature_factor``), so that flat regions get bigger steps and
        steep / negatively-curved regions get smaller ones (the latter via
        ``negative_curvature_safety`` when the lowest eigenvalue is < 0,
        important for stability during saddle search).
        """
        predicted = self._predicted_energy_change or 0.0
        if abs(predicted) < 1e-15:
            return
        ratio = actual_change / predicted
        old_tr = self.trust_radius

        # ----- curvature factor (gradient-conditional) --------------------
        gn = (
            float(np.linalg.norm(self._prev_gradient))
            if self._prev_gradient is not None else float("inf")
        )
        use_adaptive = (
            self.use_adaptive_trust_radius
            and gn < self.adaptive_trust_gradient_norm_threshold
            and self._hessian is not None
        )
        curvature_factor = 1.0
        min_eig = None
        if use_adaptive:
            try:
                eig = np.linalg.eigvalsh(self.hessian)  # T/R-projected
                eig_nonzero = eig[np.abs(eig) > self.small_eigval_thresh]
                if eig_nonzero.size:
                    min_eig = float(eig_nonzero[0])  # smallest signed eigval
                    abs_min = abs(min_eig)
                    if abs_min > 1e-6:
                        curvature_factor = min(
                            self.max_curvature_factor,
                            1.0 / max(abs_min, 0.1),
                        )
                    else:
                        curvature_factor = 1.5
                    # Saddle search: be more conservative on the negative
                    # curvature direction.
                    if self.order > 0 and min_eig < -1e-6:
                        curvature_factor *= self.negative_curvature_safety
            except np.linalg.LinAlgError:
                pass

        # ----- 5-tier ratio test ------------------------------------------
        if ratio > 0.75:
            factor = min(
                1.5 * curvature_factor, self.max_curvature_factor
            )
            self.trust_radius = min(old_tr * factor, self.trust_radius_max)
            band = "excellent"
        elif ratio > 0.50:
            factor = min(1.1 * curvature_factor, 1.5)
            self.trust_radius = min(old_tr * factor, self.trust_radius_max)
            band = "good"
        elif ratio > 0.25:
            if curvature_factor > 1.2:
                self.trust_radius = min(old_tr * 1.05, self.trust_radius_max)
                band = "acceptable (expanding slowly)"
            else:
                band = "acceptable (maintaining)"
        elif ratio > 0.10:
            self.trust_radius = max(
                old_tr * self.trust_radius_decrease_factor,
                self.trust_radius_min,
            )
            band = "poor"
        else:
            # Very poor: aggressive shrink
            self.trust_radius = max(
                old_tr * 0.25, self.trust_radius_min
            )
            band = "very poor"

        self.trust_radius = float(np.clip(
            self.trust_radius,
            self.trust_radius_min,
            self.trust_radius_max,
        ))

        if abs(self.trust_radius - old_tr) > 1e-12:
            min_eig_str = f"{min_eig:+.4e}" if min_eig is not None else "n/a"
            self._log(
                f"  [TR adjust] ratio={ratio:+.3f} ({band}); "
                f"TR {old_tr:.4f} -> {self.trust_radius:.4f}  "
                f"(curvature_factor={curvature_factor:.3f}, "
                f"min_eig={min_eig_str})"
            )

    def _evaluate_step_quality(self) -> None:
        """Aggregate quality assessment over the last 2 prediction pairs.

        Uses both the average ratio (actual / predicted) and a sign-agreement
        check. A "poor" verdict here is a strong signal that the model has
        drifted away from the local quadratic regime - the user can react by
        switching to a more conservative update method or tightening the TR.
        """
        if (
            len(self._predicted_energy_history) < 2
            or len(self._actual_energy_history) < 2
        ):
            return
        pred = self._predicted_energy_history[-2:]
        actual = self._actual_energy_history[-2:]
        ratios = [
            a / p for a, p in zip(actual, pred) if abs(p) > 1e-12
        ]
        if not ratios:
            return
        avg_ratio = sum(ratios) / len(ratios)
        same_dir = all(a * p > 0.0 for a, p in zip(actual, pred))
        if 0.8 < avg_ratio < 1.2 and same_dir:
            quality = "good"
        elif 0.5 < avg_ratio < 1.5 and same_dir:
            quality = "acceptable"
        else:
            quality = "poor"
        self._log(
            f"  [step quality] last 2 ratios -> avg={avg_ratio:+.3f}, "
            f"same_direction={same_dir} -> {quality}"
        )

    def _apply_level_shift(self, eigvals: np.ndarray) -> np.ndarray:
        """Conditionally apply a level shift to the eigenvalue spectrum.

        Following the multioptpy reference implementation, level-shifting is
        applied uniformly to the diagonal of ``H`` (``H' = H + s I``) and
        then *removed* from the resulting eigenvalues. This is numerically
        equivalent to a non-shifted diagonalisation in exact arithmetic but
        improves conditioning during the eigh call itself.

        Here we work in the eigenbasis, so the equivalent operation is:
        compute ``cond = |lambda_max|/|lambda_min|`` excluding near-zero
        modes; if it exceeds ``condition_number_threshold`` (and
        ``auto_level_shift`` is enabled) or the user forced
        ``use_level_shift``, push any eigenvalue whose absolute value is
        below ``level_shift_value`` away from zero by that amount,
        preserving its sign. Pure zeros default to a positive shift.

        We log whether and why the shift was applied (verbose mode).
        """
        if eigvals.size == 0:
            return eigvals

        abs_min = float(np.min(np.abs(eigvals)))
        abs_max = float(np.max(np.abs(eigvals)))
        cond = abs_max / max(abs_min, 1e-30)

        apply = False
        reason = ""
        if self.use_level_shift:
            apply = True
            reason = "user-enabled"
        elif self.auto_level_shift and cond > self.condition_number_threshold:
            apply = True
            reason = f"auto (kappa={cond:.2e})"

        if not apply:
            if self.verbose:
                self._log(
                    f"  [Level shift] not applied "
                    f"(kappa={cond:.2e} <= threshold)"
                )
            return eigvals

        shifted = eigvals.copy()
        mask = np.abs(shifted) < self.level_shift_value
        shifted[mask] = np.sign(shifted[mask]) * self.level_shift_value
        zero_mask = shifted == 0.0
        shifted[zero_mask] = self.level_shift_value
        n_shifted = int(np.sum(mask) + np.sum(zero_mask))
        self._log(
            f"  [Level shift] {reason}: "
            f"{n_shifted} eigval(s) pushed to +/- {self.level_shift_value:.2e} "
            f"(kappa was {cond:.2e})"
        )
        return shifted

    # -------------------------------------------------------- log helpers
    def _log_eigvals(
        self, eigvals_all: np.ndarray, eigvals_kept: np.ndarray
    ) -> None:
        """Log Hessian eigenvalues. Shows lowest / highest modes plus a
        count of imaginary (negative) eigenvalues, which is the most useful
        diagnostic for saddle-point searches."""
        if self.logfile is None or eigvals_all.size == 0:
            return
        sorted_eig = np.sort(eigvals_all)
        n_neg = int(np.sum(sorted_eig < -self.small_eigval_thresh))
        n_zero = int(np.sum(np.abs(sorted_eig) <= self.small_eigval_thresh))
        n_pos = int(np.sum(sorted_eig > self.small_eigval_thresh))
        n_show = min(self.eigval_log_count, sorted_eig.size)
        n_low = (n_show + 1) // 2
        n_high = n_show - n_low

        low_part = ", ".join(f"{v:+.4e}" for v in sorted_eig[:n_low])
        msg = f"  Hessian eigvals (lowest {n_low}): [{low_part}]"
        if n_high > 0 and sorted_eig.size > n_low:
            high_part = ", ".join(
                f"{v:+.4e}" for v in sorted_eig[-n_high:]
            )
            msg += f"  ... (highest {n_high}): [{high_part}]"
        msg += (
            f"\n  Hessian spectrum: {n_neg} negative, {n_zero} near-zero, "
            f"{n_pos} positive  (target order = {self.order})"
        )
        self._log(msg)

    def _log(self, msg: str) -> None:
        """Write a debug line to the optimiser's log file.

        Note: ASE >= 3.23 flushes log files automatically on every write,
        so we deliberately avoid calling ``flush()`` here.
        """
        if self.logfile is not None:
            try:
                self.logfile.write(msg + "\n")
            except Exception:  # noqa: BLE001
                pass
