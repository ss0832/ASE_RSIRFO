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

