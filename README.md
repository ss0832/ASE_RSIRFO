# ASE_RSIRFO

**Restricted-Step Image Rational Function Optimisation (RS-I-RFO)**
implemented as a drop-in [`ase.optimize`](https://wiki.fysik.dtu.dk/ase/) subclass.

| Feature | Details |
|---------|---------|
| Saddle order | 0 = minimum, 1 = TS, *n* = *n*-th order saddle |
| Initial Hessian | `'identity'`, `'fischer'`, `'swart'`, or user `ndarray` |
| Hessian refresh | Model rebuild **or** numerical **or** analytic callback, every *N* steps |
| Hessian updates | 25+ quasi-Newton methods (BFGS, SR1, FSB, Bofill, block multi-secant) |
| Periodic systems | Automatic: translational projection only (rotation skipped for PBC) |
| Restart | ASE-standard pickle restart (`restart=` argument) |
| License | GPL-3.0-or-later |

---

## Installation

```bash
git clone https://github.com/ss0832/ASE_RSIRFO.git
cd ASE_RSIRFO
pip install -e .
# or pip install ase-rsirfo
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10, ASE ≥ 3.23

---

## Quick start

### Energy minimisation

```python
from ase.build import molecule
from ase.calculators.emt import EMT
from ase_rsirfo import RSIRFO

atoms = molecule("H2O")
atoms.calc = EMT()

opt = RSIRFO(atoms, trajectory="opt.traj", logfile="opt.log")
opt.run(fmax=0.05)
```

### Transition-state search (order=1) — recommended setup

For a reliable TS search, **start from an exact numerical Hessian and recompute
it every 5 steps**.  A model Hessian (Fischer/Swart) only captures rough
element-pair curvatures; the exact Hessian guarantees that the optimizer begins
with the correct sign along the reaction coordinate from the very first step.

> **Why pre-compute?**  The internal recompute schedule (`hessian_recompute_interval`)
> always skips step 0 — the initial Hessian is always whatever is passed to
> the `hessian=` argument.  To start from the numerical Hessian you must
> compute it beforehand with `numerical_hessian_from_forces` and pass the
> resulting array as `hessian=`.

```python
from ase_rsirfo import RSIRFO, numerical_hessian_from_forces

# 1. Compute the exact Hessian at the TS guess geometry (costs 6N single points)
H0 = numerical_hessian_from_forces(atoms, delta=0.01)

# 2. Run the TS search, refreshing the Hessian every 5 accepted steps
opt = RSIRFO(
    atoms,
    order=1,                              # image-RFO: climb along the lowest mode
    hessian=H0,                           # exact numerical Hessian as starting point
    hessian_recompute_interval=5,         # recompute every 5 steps (must be set
                                          # explicitly when hessian= is an ndarray)
    hessian_recompute_method="numerical", # full numerical recompute at each refresh
    numerical_hessian_step=0.01,          # finite-difference step in Angstrom
    reset_history_on_recompute=True,      # clear secant history after each refresh
    trajectory="ts.traj",
)
opt.run(fmax=0.05)

# 3. Verify exactly one imaginary frequency at the converged geometry
import numpy as np
eigvals = np.linalg.eigvalsh(opt.hessian)
n_imag = int(np.sum(eigvals < -1e-3))
print(f"imaginary frequencies: {n_imag}  (expected 1 for a TS)")
```

**Cost note.** `numerical_hessian_from_forces` performs 6*N* single-point
calculations (symmetric finite differences).  For large systems where this is
prohibitive, see the [cheaper alternatives](#cheaper-alternatives-for-ts-search)
below.


The auto-default for `hessian_recompute_interval` depends on the initial
Hessian type:

| `hessian=` | `order=0` default | `order>=1` default |
|------------|-------------------|--------------------|
| `"callable"` / `"numerical"` / `"fischer"` / `"swart"` | 50 | **5** |
| `"identity"` / `ndarray` | 0 (off) | **0 (off)** |

When passing a pre-computed `ndarray`, you **must set `hessian_recompute_interval`
explicitly** if you want periodic refreshes — the auto-default is 0 (off) for
ndarray inputs.

---

## Accessing the converged Hessian

`opt.hessian` returns the **translational/rotational (T/R) projected** Hessian
``P^T H P``, where ``P = I - Q^T Q`` and the columns of ``Q`` span the
rigid-body modes of the molecule.  This means spurious near-zero or slightly
negative eigenvalues that arise from numerical noise along T/R modes are
explicitly zeroed out, and `np.linalg.eigvalsh(opt.hessian)` gives a clean
count of imaginary frequencies immediately after convergence.

The raw (unprojected) internal Hessian is available as `opt._hessian` if
needed for custom post-processing.

```python
# After opt.run():
import numpy as np

H = opt.hessian                          # T/R-projected  (recommended)
eigvals = np.linalg.eigvalsh(H)
n_imag = int(np.sum(eigvals < -1e-3))
print(f"imaginary frequencies: {n_imag}")

H_raw = opt._hessian                     # raw internal Hessian (advanced use)
```

---

## Analytic Hessian via callback

If your calculator supports analytic second derivatives, pass them through the
`hessian_callback` interface.  The callback is called at every scheduled
refresh (including step 0 if you combine it with a pre-computed `H0`):

```python
def my_hessian(atoms):
    """Return the (3N × 3N) Hessian in eV/Angstrom²."""
    return atoms.calc.get_property("hessian")  # if supported by the calculator

opt = RSIRFO(
    atoms,
    order=1,
    hessian=my_hessian(atoms),            # analytic Hessian at step 0
    hessian_recompute_interval=5,
    hessian_recompute_method="callback",
    hessian_callback=my_hessian,
    reset_history_on_recompute=True,
)
opt.run(fmax=0.05)
```

---

## Energy minimisation with Hessian refresh

### Model Hessian refresh

```python
opt = RSIRFO(
    atoms,
    hessian="fischer",               # initial model Hessian
    hessian_recompute_interval=10,   # rebuild model every 10 steps (default: 50)
    hessian_recompute_method="model",
)
opt.run(fmax=0.05)
```

### Numerical Hessian refresh

```python
opt = RSIRFO(
    atoms,
    hessian="identity",
    hessian_recompute_interval=5,
    hessian_recompute_method="numerical",
    numerical_hessian_step=0.01,
    reset_history_on_recompute=True,
)
opt.run(fmax=0.05)
```

---

## Restart

```python
# First run (saves state to rsirfo.pckl)
opt = RSIRFO(atoms, restart="rsirfo.pckl", trajectory="opt.traj")
opt.run(fmax=0.05)

# Resume from checkpoint
opt2 = RSIRFO(atoms, restart="rsirfo.pckl", trajectory="opt.traj",
              append_trajectory=True)
opt2.run(fmax=0.01)
```

---

## Key constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `order` | `0` | Saddle order (0 = min, 1 = TS) |
| `hessian` | `'identity'` | Initial Hessian: `'identity'`, `'fischer'`, `'swart'`, or `ndarray`. For TS searches, passing `numerical_hessian_from_forces(atoms)` here gives the exact curvature at step 0. |
| `hessian_update` | auto | Quasi-Newton update method. Default: `'block_fsb'` for `order=0`, `'block_bofill'` for `order>=1`. Pass `'auto'` for the Bakó-Császár flowchart selector. |
| `verbose` | `True` | Log Hessian eigenvalue spectrum at every step |
| `eigval_log_count` | `10` | How many eigenvalues to display in verbose mode |
| `hessian_recompute_interval` | auto | Refresh Hessian every N **accepted** steps. Step 0 is always skipped — the initial Hessian is always the `hessian=` argument. Auto-defaults: `5` (`order>=1`) or `50` (`order=0`) when `hessian=` is a model string; **`0` (off) when `hessian=` is `'identity'` or an `ndarray`** — set this explicitly when passing a pre-computed Hessian. Pass `0` to disable refresh entirely. |
| `hessian_recompute_method` | `None` | `'model'`, `'numerical'`, `'callback'`, or `None` (auto). Auto selects `'model'` for model-string starts, `'numerical'` otherwise. |
| `hessian_callback` | `None` | Callable `(Atoms) -> ndarray` for analytic Hessian |
| `numerical_hessian_step` | `0.01` | Finite-difference step (Angstrom) for numerical Hessian |
| `reset_history_on_recompute` | `True` | Clear quasi-Newton secant history after each Hessian refresh |
| `trust_radius` | `0.3` / `0.2` | Initial trust radius in Angstrom (min / TS default) |
| `trust_radius_max` | `0.5` / `0.2` | Maximum trust radius in Angstrom (min / TS default) |
| `use_adaptive_trust_radius` | `True` | Fletcher ratio-based TR adaptation |
| `project_translation` | `True` | Project out translational modes |
| `project_rotation` | auto | Project out rotational modes (auto: False for PBC) |
| `use_level_shift` | `True` | Level-shift near-singular Hessian eigenvalues |
| `block_size` | `4` | History window for block (multi-secant) updates |
| `max_window` | `8` | Maximum secant-pair history length |

Full parameter reference: see the docstring of `RSIRFO.__init__`.

---

## Available Hessian update methods

| Method string | Description |
|---------------|-------------|
| `auto` / `flowchart` | Automatic selection (Bakó & Császár 2016) |
| `bfgs` | BFGS with optional double damping |
| `sr1` | Symmetric Rank-1 |
| `fsb` | FSB (SR1/BFGS mix by Schlegel) |
| `bofill` | Bofill (SR1/PSB mix) |
| `block_bfgs` | Multi-secant BFGS |
| `block_fsb` | Multi-secant FSB |
| `block_bofill` | Multi-secant Bofill |

---

## Model Hessian note

The `'fischer'` and `'swart'` model Hessians in `model_hessian.py` are
independent implementations based solely on the original publications
(Fischer & Almlöf 1992; Swart & Bickelhaupt 2006). Both generators
provide bonded-term empirical curvatures without any dispersion correction.
See `NOTICE.md` for full references.

---

## References

### ASE

- A. H. Larsen, J. J. Mortensen, J. Blomqvist, I. E. Castelli, R. Christensen, M. Dułak, J. Friis, M. N. Groves, B. Hammer, C. Hargus, E. D. Hermes, P. C. Jennings, P. B. Jensen, J. Kermode, J. R. Kitchin, E. L. Kolsbjerg, J. Kubal, K. Kaasbjerg, S. Lysgaard, J. B. Maronsson, T. Maxson, T. Olsen, L. Pastewka, A. Peterson, C. Rostgaard, J. Schiøtz, O. Schütt, M. Strange, K. S. Thygesen, T. Vegge, L. Vilhelmsen, M. Walter, Z. Zeng, and K. W. Jacobsen,  
  "The atomic simulation environment — a Python library for working with atoms,"  
  *J. Phys.: Condens. Matter* **29**, 273002 (2017).  
  DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

- S. R. Bahn and K. W. Jacobsen,  
  "An object-oriented scripting interface to a legacy electronic structure code,"  
  *Comput. Sci. Eng.* **4**, 56–66 (2002).  
  DOI: [10.1109/5992.998641](https://doi.org/10.1109/5992.998641)

See `NOTICE.md` for the complete bibliography.

---

## ASE constraint support

`RSIRFO` honours the standard ASE constraint mechanism
(`atoms.set_constraint(...)`). Three categories of constraints are
recognised, each handled by a tailored strategy:

| Constraint category | Examples | Cartesian mask | Internal | T/R projection | RFO step processing |
|---|---|---|---|---|---|
| **Atom fix** | `FixAtoms`, `FixCartesian` | yes | no | **OFF** (fixed atoms break rigid-body symmetry) | `'auto'` ⇒ `'none'` (rely on ASE) + secant zeroing |
| **Internal-coord fix** | `FixBondLength`, `FixInternals` (angles, dihedrals), `FixedPlane`, `FixedLine`, `Hookean` | no | yes | **ON** (internal coords are T/R-invariant; rigid-body modes remain zero-energy directions) | `'auto'` ⇒ `'none'` (rely on ASE's iterative adjuster) |
| **Mixed** | atom-fix AND internal-coord | yes | yes | **OFF** (atom-fix dominates) | `'auto'` ⇒ `'none'` |

Two universal protections are applied regardless of the chosen method:

1. **Quasi-Newton update protection** — the secant pair `(s, y)` used by
   the Hessian update is zeroed on fixed Cartesian DOFs every step, preventing
   accumulation of spurious information on the constrained subspace.
2. **T/R projection auto-decision** — chosen at the start of every step
   from the constraint signature, so users do not need to think about it.

The `constraint_method` argument controls the RFO-step-level processing:

| Method | What it does | When to use |
|--------|--------------|-------------|
| `'auto'` (default) | Pick `'none'` for recognised constraints + secant zeroing; `'freeze'` as a safety net for unrecognised constraint types | Recommended in most cases |
| `'subspace'` | Project `H`, `g` to the active Cartesian subspace; solve the RFO equations on a smaller matrix; expand the step back at the end | Exact for FixAtoms / FixCartesian; ignored for purely internal constraints |
| `'freeze'` | Replace fixed rows/cols of `H` with `freeze_value` * I, zero matching gradient components | Required when the constraint cannot be expressed as an analytic Cartesian projection |
| `'none'` | Rely entirely on ASE's `adjust_forces` / `adjust_positions` | Diagnostic / control; same as `'auto'` for recognised constraints |

Examples:

```python
from ase.constraints import FixAtoms, FixBondLength, FixInternals
from ase_rsirfo import RSIRFO

# (a) Pin two atoms in space
atoms.set_constraint(FixAtoms(indices=[0, 5]))
RSIRFO(atoms, hessian="fischer").run(fmax=0.05)

# (b) Hold a bond length fixed
atoms.set_constraint(FixBondLength(3, 4))
RSIRFO(atoms, hessian="identity").run(fmax=0.05)

# (c) Hold an angle fixed at 120 deg
atoms.set_constraint(FixInternals(angles_deg=[[120.0, [0, 1, 2]]]))
RSIRFO(atoms, hessian="identity").run(fmax=0.05)

# (d) Mixed: FixAtoms + FixBondLength
atoms.set_constraint([FixAtoms(indices=[0]), FixBondLength(2, 3)])
RSIRFO(atoms, hessian="identity").run(fmax=0.05)
```

See `examples/05_constraints.py` (atom fixes) and
`examples/06_internal_constraints.py` (bond / angle / dihedral / mixed)
for full working demonstrations.

---

## Reading the Hessian

`RSIRFO` keeps two views of its internal Hessian:

* The **raw** quasi-Newton Hessian — what the optimiser writes into during
  `update()` calls. Contains all curvature including translation/rotation
  modes and any DOFs that ASE constraints will subsequently filter.
* The **projected** Hessian — what the RFO solver actually sees at step
  time, with T/R modes removed and (optionally) constraint freezing
  applied.

After the run you can choose either with the public accessors:

```python
opt = RSIRFO(atoms, hessian="fischer")
opt.run(fmax=0.05)

H_raw = opt.get_raw_hessian()                        # bare matrix, no projection
H_default = opt.get_hessian()                        # auto: T/R off if FixAtoms, on otherwise
H_no_tr = opt.get_hessian(project_tr=False)          # explicit: skip T/R
H_tr = opt.get_hessian(project_tr=True)              # explicit: project T/R
H_full = opt.get_hessian(project_tr=True,
                         apply_constraints=True)     # also freeze fixed DOFs

# Backward-compatible shortcut (same as get_hessian(project_tr=True)):
H_compat = opt.hessian
```

| Method | T/R projection | Constraint freeze | Returns a copy? |
|--------|---------------|-------------------|-----------------|
| `get_raw_hessian()` | no | no | yes |
| `get_hessian(project_tr=None, apply_constraints=False)` (default) | auto (off if FixAtoms, on otherwise) | no | yes |
| `get_hessian(project_tr=True/False, apply_constraints=...)` | explicit | optional | yes |
| `.hessian` (property) | yes | no | yes |

For a clean post-run frequency analysis, `get_hessian(project_tr=True)`
is what you usually want; `get_raw_hessian()` is best when you need to
serialise the matrix or pass it to a subsequent RSIRFO instance via
`hessian=...`.

---

## License

Copyright (C) 2026 ss0832  
Licensed under the GNU General Public License, version 3 or later.  
See `LICENSE` for the full text.

