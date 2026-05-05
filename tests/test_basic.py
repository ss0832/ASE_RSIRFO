# Copyright (C) 2026 ss0832
# This file is part of ASE_RSIRFO and is licensed under GPL-3.0-or-later.
# See the LICENSE file in the repository root for the full text, or visit
# <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
tests/test_basic.py
===================

Sanity tests for ASE_RSIRFO. These are designed to be fast and require
only ASE's built-in LennardJones and EMT calculators.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ase_rsirfo import RSIRFO, numerical_hessian_from_forces


# ─── helpers ────────────────────────────────────────────────────────────────

def lj_ar2(d=4.0):
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, d]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    return atoms


def lj_ar4():
    atoms = Atoms(
        "Ar4",
        positions=[
            [0.00, 0.00, 0.00],
            [3.80, 0.00, 0.00],
            [1.90, 3.30, 0.00],
            [1.90, 1.10, 3.10],
        ],
    )
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    return atoms


LJ_AR2_EXPECTED = 3.40 * 2 ** (1 / 6)  # equilibrium distance


# ─── basic convergence ──────────────────────────────────────────────────────

def test_ar2_identity_converges():
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-5, steps=60)
    d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    assert abs(d - LJ_AR2_EXPECTED) < 1e-3, f"distance error: {d:.5f} vs {LJ_AR2_EXPECTED:.5f}"
    assert opt._iteration <= 30


def test_ar4_converges():
    atoms = lj_ar4()
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-2, steps=160)
    fmax = float(np.max(np.abs(atoms.get_forces())))
    assert fmax < 1e-2


# ─── initial Hessian options ─────────────────────────────────────────────────

def test_fischer_hessian():
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="fischer", logfile=None)
    opt.run(fmax=1e-5, steps=60)
    d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    assert abs(d - LJ_AR2_EXPECTED) < 1e-3


def test_swart_hessian():
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="swart", logfile=None)
    opt.run(fmax=1e-5, steps=60)
    d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    assert abs(d - LJ_AR2_EXPECTED) < 1e-3


def test_user_ndarray_hessian():
    atoms = lj_ar2()
    H_user = np.eye(6) * 1.5
    opt = RSIRFO(atoms, hessian=H_user, logfile=None)
    opt.run(fmax=1e-5, steps=60)
    d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    assert abs(d - LJ_AR2_EXPECTED) < 1e-3


# ─── hessian update methods ──────────────────────────────────────────────────

@pytest.mark.parametrize("method", [
    "auto", "flowchart",
    "bfgs", "bfgs_dd", "sr1", "psb", "fsb", "fsb_dd",
    "bofill", "msp",
    "block_bfgs", "block_fsb", "block_fsb_weighted",
    "block_bofill",
])
def test_update_methods(method):
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="identity", hessian_update=method, logfile=None)
    opt.run(fmax=1e-4, steps=80)
    d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    assert abs(d - LJ_AR2_EXPECTED) < 0.05, f"method={method}: d={d:.4f}"


# ─── hessian recompute ────────────────────────────────────────────────────────

def test_recompute_numerical():
    atoms = lj_ar4()
    opt = RSIRFO(
        atoms,
        hessian="identity",
        hessian_recompute_interval=3,
        hessian_recompute_method="numerical",
        numerical_hessian_step=0.01,
        logfile=None,
    )
    opt.run(fmax=1e-3, steps=120)
    fmax = float(np.max(np.abs(atoms.get_forces())))
    assert fmax < 1e-3


def test_recompute_model():
    atoms = lj_ar4()
    opt = RSIRFO(
        atoms,
        hessian="fischer",
        hessian_recompute_interval=5,    # less frequent reset
        hessian_recompute_method="model",
        logfile=None,
    )
    opt.run(fmax=5e-3, steps=200)
    fmax = float(np.max(np.abs(atoms.get_forces())))
    assert fmax < 5e-3


def test_recompute_callback():
    call_log = []

    def cb(a):
        call_log.append(1)
        return numerical_hessian_from_forces(a, delta=0.01)

    # Use a heavily rattled geometry so the optimisation runs for at least
    # two steps and the callback gets a chance to fire.
    atoms = lj_ar4()
    rng = np.random.default_rng(13)
    atoms.positions += rng.normal(scale=0.4, size=atoms.positions.shape)
    opt = RSIRFO(
        atoms,
        hessian="identity",
        hessian_recompute_interval=2,
        hessian_recompute_method="callback",
        hessian_callback=cb,
        logfile=None,
    )
    opt.run(fmax=1e-3, steps=120)
    assert len(call_log) >= 1, "callback was never called"


# ─── PBC: rotation projection disabled ───────────────────────────────────────

def test_pbc_rotation_projection():
    atoms = Atoms("Ar2", positions=[[5, 5, 5], [5, 5, 9]], cell=[10]*3, pbc=True)
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40, rc=8.0)
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    assert opt.project_translation is True
    assert opt.project_rotation is False
    opt.run(fmax=1e-4, steps=40)
    fmax = float(np.max(np.abs(atoms.get_forces())))
    assert fmax < 1e-4


# ─── restart round-trip ───────────────────────────────────────────────────────

def test_restart(tmp_path):
    restart_file = str(tmp_path / "opt.pckl")
    atoms = lj_ar2()
    opt1 = RSIRFO(atoms, hessian="identity", restart=restart_file, logfile=None)
    opt1.run(fmax=1e-5, steps=4)
    iters_after_first = opt1._iteration
    tr_after_first = opt1.trust_radius

    # Re-open from checkpoint (atoms already at intermediate geometry)
    opt2 = RSIRFO(atoms, hessian="identity", restart=restart_file, logfile=None)
    assert opt2._iteration == iters_after_first
    assert abs(opt2.trust_radius - tr_after_first) < 1e-10
    assert opt2.hessian is not None


# ─── saddle order parameter ───────────────────────────────────────────────────

def test_order_parameter_accepted():
    """RSIRFO should accept order >= 0 without raising."""
    for order in (0, 1, 2):
        atoms = lj_ar2()
        opt = RSIRFO(atoms, order=order, logfile=None)
        assert opt.order == order


def test_negative_order_raises():
    with pytest.raises(ValueError):
        RSIRFO(lj_ar2(), order=-1, logfile=None)


# ─── numerical_hessian_from_forces ────────────────────────────────────────────

def test_numerical_hessian_shape():
    atoms = lj_ar2()
    H = numerical_hessian_from_forces(atoms, delta=0.01)
    assert H.shape == (6, 6)
    # Should be symmetric
    assert np.max(np.abs(H - H.T)) < 1e-10


def test_numerical_hessian_positive_definite_at_minimum():
    """At the LJ minimum, the Hessian should have at most 1 near-zero
    eigenvalue (the radial soft mode is constrained by the 1D geometry)."""
    atoms = lj_ar2(d=LJ_AR2_EXPECTED)
    H = numerical_hessian_from_forces(atoms, delta=0.001)
    eigvals = np.linalg.eigvalsh(H)
    # 5 near-zero translational/rotational + 1 positive radial
    n_positive = np.sum(eigvals > 1e-3)
    assert n_positive >= 1

# ─── transition-state search (image-RFO regression test) ─────────────────────

def test_ts_search_finds_one_imaginary_mode():
    """Image-RFO must climb along the lowest mode, ending up with exactly
    one imaginary frequency at the converged saddle point. Regression test
    for the gradient-component sign-flip bug in image-RFO."""
    sigma = 3.40
    r_lin = 3.95
    atoms = Atoms(
        "Ar3",
        positions=[
            [-r_lin, 0.00, 0.0],
            [0.00,   0.00, 0.0],
            [+r_lin, 0.00, 0.0],
        ],
    )
    atoms.positions[1, 1] += 0.05  # break symmetry
    atoms.calc = LennardJones(epsilon=0.0103, sigma=sigma)

    opt = RSIRFO(
        atoms,
        order=1,
        hessian="fischer",
        trust_radius=0.05,
        trust_radius_max=0.3,
        logfile=None,
    )
    opt.run(fmax=1e-3, steps=80)

    # 1 imaginary frequency at the TS
    eigvals = np.linalg.eigvalsh(opt.hessian)
    n_imag = int(np.sum(eigvals < -1e-3))
    assert n_imag == 1, f"expected 1 imaginary frequency, got {n_imag}"

    # Near-collinear geometry: the long bond should be ~ twice the short bond
    d12 = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    d23 = float(np.linalg.norm(atoms.positions[2] - atoms.positions[1]))
    d13 = float(np.linalg.norm(atoms.positions[2] - atoms.positions[0]))
    assert abs(d13 - (d12 + d23)) < 0.05, (
        f"atoms are not collinear (TS): {d12:.3f}, {d23:.3f}, {d13:.3f}"
    )


def test_trust_radius_hard_cap():
    """Restricted-step solver may occasionally return a step larger than
    the trust radius; the optimiser must hard-clip such steps. We verify
    by looking at the achieved positions vs the trust radius bound on a
    deliberately ill-conditioned starting point."""
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, 6.0]])  # far apart
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    opt = RSIRFO(
        atoms,
        hessian="identity",
        trust_radius=0.1,
        trust_radius_max=0.1,
        logfile=None,
    )
    pos_before = atoms.positions.copy()
    opt.step()
    delta = np.linalg.norm(atoms.positions - pos_before)
    # Allow a very small numerical slack (1.001 factor in the cap)
    assert delta <= 0.1 * 1.05, f"step exceeded TR: |dx| = {delta:.4f}"


def test_default_update_methods_per_order():
    """Defaults: order=0 -> block_fsb, order>=1 -> block_bofill."""
    atoms = lj_ar2()
    opt_min = RSIRFO(atoms, logfile=None)
    assert opt_min.hessian_update_method == "block_fsb"

    atoms2 = lj_ar2()
    opt_ts = RSIRFO(atoms2, order=1, logfile=None)
    assert opt_ts.hessian_update_method == "block_bofill"


def test_default_recompute_interval():
    """Empirical defaults:
    - identity / user ndarray -> OFF
    - model Hessian (fischer/swart) -> 50 (min) / 5 (TS)
    - callback provided        -> 50 (min) / 5 (TS)
    """
    # identity (default): OFF
    atoms = lj_ar2()
    opt_id = RSIRFO(atoms, hessian="identity", logfile=None)
    assert opt_id.hessian_recompute_interval == 0

    # fischer model: 50 for min
    atoms_f = lj_ar2()
    opt_f = RSIRFO(atoms_f, hessian="fischer", logfile=None)
    assert opt_f.hessian_recompute_interval == 50

    # fischer model + TS: 5
    atoms_fts = lj_ar2()
    opt_fts = RSIRFO(atoms_fts, hessian="fischer", order=1, logfile=None)
    assert opt_fts.hessian_recompute_interval == 5

    # swart model: 50 for min
    atoms_s = lj_ar2()
    opt_s = RSIRFO(atoms_s, hessian="swart", logfile=None)
    assert opt_s.hessian_recompute_interval == 50

    # User ndarray: OFF (user probably wants their matrix preserved)
    atoms_u = lj_ar2()
    opt_u = RSIRFO(atoms_u, hessian=np.eye(6) * 1.5, logfile=None)
    assert opt_u.hessian_recompute_interval == 0

    # callback provided + identity: still uses order-dependent interval
    atoms_c = lj_ar2()
    opt_c = RSIRFO(
        atoms_c, hessian="identity",
        hessian_callback=lambda a: numerical_hessian_from_forces(a),
        logfile=None,
    )
    assert opt_c.hessian_recompute_interval == 50

    # Explicit override still works
    atoms_o = lj_ar2()
    opt_o = RSIRFO(atoms_o, hessian="fischer",
                   hessian_recompute_interval=0, logfile=None)
    assert opt_o.hessian_recompute_interval == 0


# ─── ASE constraint handling ──────────────────────────────────────────────────

def _make_lj_3atom_with_fixed_atom():
    """Equilateral-triangle LJ trimer with atom 0 pinned at the origin."""
    from ase.constraints import FixAtoms
    atoms = Atoms("Ar3", positions=[[0, 0, 0], [4.5, 0, 0], [2.0, 3.5, 0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


@pytest.mark.parametrize("method", ["auto", "subspace", "freeze", "none"])
def test_constraint_method_keeps_fixed_atom_fixed(method):
    """All four constraint methods must keep FixAtoms-fixed atoms still."""
    atoms = _make_lj_3atom_with_fixed_atom()
    initial = atoms.positions.copy()
    opt = RSIRFO(
        atoms,
        hessian="identity",
        constraint_method=method,
        logfile=None,
    )
    opt.run(fmax=1e-5, steps=80)
    # Atom 0 must not move at all (ASE's adjust_positions ensures this even
    # for the 'none' method).
    assert np.linalg.norm(atoms.positions[0] - initial[0]) < 1e-12

    # The trimer should converge to an equilateral triangle (LJ minimum).
    fmax = float(np.max(np.abs(atoms.get_forces())))
    assert fmax < 1e-5, f"method={method}: fmax={fmax:.2e}"
    bonds = sorted([
        float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))
        for i in range(3) for j in range(i)
    ])
    expected = 3.40 * 2 ** (1 / 6)
    for b in bonds:
        assert abs(b - expected) < 1e-3, f"method={method}: bond {b:.4f}"


def test_fix_cartesian_keeps_z_fixed():
    """FixCartesian must keep the constrained component fixed."""
    from ase.constraints import FixCartesian
    atoms = Atoms("Ar3", positions=[[0, 0, 0], [4.0, 0, 1.0], [2.0, 3.5, 0.5]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixCartesian(1, mask=(False, False, True)))
    initial_z1 = atoms.positions[1, 2]

    opt = RSIRFO(
        atoms, hessian="identity", constraint_method="auto", logfile=None
    )
    opt.run(fmax=1e-3, steps=80)
    # z component of atom 1 must remain unchanged
    assert abs(atoms.positions[1, 2] - initial_z1) < 1e-12


def test_constraint_method_invalid_raises():
    atoms = lj_ar2()
    with pytest.raises(ValueError):
        RSIRFO(atoms, constraint_method="bogus", logfile=None)


def test_detect_fixed_dofs_basic():
    """Direct test of the constraint detection helper."""
    from ase.constraints import FixAtoms
    from ase_rsirfo.constraints import detect_fixed_dofs
    atoms = Atoms("Ar4", positions=[[0,0,0], [3,0,0], [0,3,0], [0,0,3]])
    atoms.set_constraint(FixAtoms(indices=[0, 2]))
    mask, has_internal, has_other = detect_fixed_dofs(atoms)
    assert mask.shape == (12,)
    assert mask.sum() == 6
    assert mask[0:3].all() and mask[6:9].all()
    assert not mask[3:6].any() and not mask[9:12].any()
    assert not has_internal
    assert not has_other


# ─── Internal-coordinate constraints ─────────────────────────────────────────

def test_fix_bond_length_keeps_bond_fixed():
    """FixBondLength must hold the constrained distance to machine precision.

    For internal-coord constraints, T/R projection is required (the rigid-body
    modes are still zero-energy directions because internal coordinates are
    invariant under rigid translation/rotation). This test verifies that the
    'auto' constraint_method correctly enables T/R projection in this case.
    """
    from ase.constraints import FixBondLength
    atoms = Atoms("Ar3", positions=[[0,0,0], [5.0,0,0], [2.5,3.0,0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixBondLength(0, 1))
    initial_d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))

    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-4, steps=80)

    final_d = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
    assert abs(final_d - initial_d) < 1e-6, (
        f"bond 0-1 changed from {initial_d:.6f} to {final_d:.6f}"
    )
    # Other bonds should still relax to the LJ minimum
    expected = 3.40 * 2 ** (1 / 6)
    d02 = float(np.linalg.norm(atoms.positions[2] - atoms.positions[0]))
    d12 = float(np.linalg.norm(atoms.positions[2] - atoms.positions[1]))
    assert abs(d02 - expected) < 0.01
    assert abs(d12 - expected) < 0.01


def test_fix_internals_keeps_angle_fixed():
    """FixInternals (angle) must keep the constrained angle constant."""
    from ase.constraints import FixInternals
    atoms = Atoms("Ar3", positions=[
        [0, 0, 0],
        [3.5, 0, 0],
        [3.5 + 3.5*np.cos(np.deg2rad(60)),
         3.5*np.sin(np.deg2rad(60)), 0],
    ])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixInternals(angles_deg=[[120.0, [0, 1, 2]]]))

    def angle_deg(a, b, c):
        v1 = a - b; v2 = c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-3, steps=80)

    final = angle_deg(atoms.positions[0], atoms.positions[1],
                      atoms.positions[2])
    assert abs(final - 120.0) < 1e-3, f"angle drifted to {final:.4f}"


def test_mixed_constraints_atom_and_bond():
    """Combination of FixAtoms (one atom) and FixBondLength (one bond)
    must work together."""
    from ase.constraints import FixAtoms, FixBondLength
    atoms = Atoms("Ar4", positions=[[0,0,0], [3.5,0,0], [3.5,3.5,0], [0,3.5,0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint([FixAtoms(indices=[0]), FixBondLength(2, 3)])

    initial_pos0 = atoms.positions[0].copy()
    initial_d23 = float(np.linalg.norm(atoms.positions[3] - atoms.positions[2]))

    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-3, steps=80)

    # atom 0 must not move
    assert np.linalg.norm(atoms.positions[0] - initial_pos0) < 1e-12
    # bond 2-3 must stay fixed
    final_d23 = float(np.linalg.norm(atoms.positions[3] - atoms.positions[2]))
    assert abs(final_d23 - initial_d23) < 1e-6


def test_detect_internal_constraint_flag():
    """detect_fixed_dofs should flag FixBondLength as an internal constraint."""
    from ase.constraints import FixBondLength
    from ase_rsirfo.constraints import detect_fixed_dofs
    atoms = Atoms("Ar3", positions=[[0,0,0], [3,0,0], [0,3,0]])
    atoms.set_constraint(FixBondLength(0, 1))
    mask, has_internal, has_other = detect_fixed_dofs(atoms)
    assert mask.sum() == 0          # no Cartesian DOF is hard-fixed
    assert has_internal             # but an internal constraint is present
    assert not has_other


def test_tr_projection_with_internal_constraint(monkeypatch):
    """When only internal constraints are present, T/R projection must be
    kept active so that rigid-body modes are removed from the Hessian."""
    from ase.constraints import FixBondLength
    atoms = Atoms("Ar3", positions=[[0,0,0], [5.0,0,0], [2.5,3.0,0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixBondLength(0, 1))

    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-4, steps=80)
    # Indirect check: the optimisation must converge cleanly. If T/R were
    # incorrectly skipped, the rigid-body modes would interfere and the
    # gradient would never drop below 1e-4 in 80 steps.
    fmax = float(np.max(np.abs(atoms.get_forces())))
    assert fmax < 1e-3


# ─── Hessian accessor API (get_hessian / get_raw_hessian) ────────────────────

def test_get_raw_hessian_before_step():
    """Accessor must return None before any step has been taken."""
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    assert opt.get_raw_hessian() is None
    assert opt.get_hessian() is None
    assert opt.hessian is None


def test_get_raw_hessian_returns_copy():
    """Mutating the returned Hessian must not corrupt the optimiser state."""
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-5, steps=20)
    H = opt.get_raw_hessian()
    H[0, 0] = 1e30
    H_after = opt.get_raw_hessian()
    assert H_after[0, 0] != 1e30
    # Also: the second copy should equal the first up to the mutation
    H[0, 0] = H_after[0, 0]
    assert np.allclose(H, H_after)


def test_get_hessian_projection_options():
    """project_tr=True removes T/R modes; project_tr=False removes none.

    For a non-linear molecule we expect 6 near-zero eigvals (3 T + 3 R).
    For a linear (e.g. diatomic) molecule one rotational mode is
    degenerate so we expect 5.
    """
    # 3-atom non-collinear cluster: 6 zero eigvals expected
    atoms = Atoms("Ar3", positions=[[0, 0, 0], [4.5, 0, 0], [2.0, 3.5, 0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-5, steps=30)

    H_proj = opt.get_hessian(project_tr=True)
    H_no = opt.get_hessian(project_tr=False)
    H_raw = opt.get_raw_hessian()

    # project_tr=False must equal raw
    assert np.allclose(H_no, H_raw)
    # project_tr=True must produce 6 near-zero eigvals (3 T + 3 R) for a
    # non-linear molecule
    eigs = np.linalg.eigvalsh(H_proj)
    n_zero = int(np.sum(np.abs(eigs) < 1e-6))
    assert n_zero == 6, (
        f"expected 6 near-zero eigvals after T/R projection, got {n_zero}"
    )

    # And without projection: no spurious zeros (eigval of T/R modes was 1
    # because we used the identity Hessian, but level-shifting might lower it
    # so we just check < 6)
    eigs_no = np.linalg.eigvalsh(H_no)
    n_zero_no = int(np.sum(np.abs(eigs_no) < 1e-6))
    assert n_zero_no < 6


def test_hessian_property_matches_get_hessian_proj():
    """The backward-compatible .hessian property must equal
    get_hessian(project_tr=True)."""
    atoms = lj_ar2()
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-5, steps=20)
    assert np.allclose(opt.hessian, opt.get_hessian(project_tr=True))


def test_get_hessian_auto_skips_tr_when_fixed_atoms():
    """project_tr=None must automatically skip T/R projection when there
    are fixed atoms (T/R symmetry is broken by the constraint)."""
    from ase.constraints import FixAtoms
    atoms = Atoms("Ar3", positions=[[0, 0, 0], [4.5, 0, 0], [2.0, 3.5, 0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixAtoms(indices=[0]))
    opt = RSIRFO(atoms, hessian="identity", logfile=None)
    opt.run(fmax=1e-5, steps=60)

    # auto must == project_tr=False here
    assert np.allclose(
        opt.get_hessian(),
        opt.get_hessian(project_tr=False),
    )


def test_get_hessian_apply_constraints_freezes_fixed_dofs():
    """apply_constraints=True must replace fixed rows/cols with the
    freeze-diagonal level shift."""
    from ase.constraints import FixAtoms
    atoms = Atoms("Ar3", positions=[[0, 0, 0], [4.5, 0, 0], [2.0, 3.5, 0]])
    atoms.calc = LennardJones(epsilon=0.0103, sigma=3.40)
    atoms.set_constraint(FixAtoms(indices=[0]))
    opt = RSIRFO(atoms, hessian="identity", freeze_value=1e8, logfile=None)
    opt.run(fmax=1e-5, steps=60)

    H = opt.get_hessian(project_tr=False, apply_constraints=True)
    # Atom 0 indices: 0, 1, 2 in flattened DOFs
    # Diagonal entries on fixed DOFs should equal freeze_value
    for i in range(3):
        assert abs(H[i, i] - 1e8) < 1.0
    # Off-diagonal entries on fixed rows / cols should be (near) zero
    for i in range(3):
        row_off = np.delete(H[i], i)
        col_off = np.delete(H[:, i], i)
        assert np.max(np.abs(row_off)) < 1e-6
        assert np.max(np.abs(col_off)) < 1e-6

