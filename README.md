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

### Transition-state search (order=1)

```python
from ase_rsirfo import RSIRFO

opt = RSIRFO(
    atoms,
    order=1,                  # image-RFO: climb along the lowest mode
    hessian="fischer",        # chemistry-aware initial Hessian (recommended)
    # hessian_update defaults to "block_bofill" when order >= 1
    trust_radius=0.05,        # small TR for TS search
    trajectory="ts.traj",
)
opt.run(fmax=0.05)

# Inspect the converged saddle
import numpy as np
eigvals = np.linalg.eigvalsh(opt.hessian)
n_imag = int(np.sum(eigvals < -1e-3))
print(f"imaginary frequencies: {n_imag}  (expected 1 for a TS)")
```

The image-RFO algorithm flips the sign of both the eigenvalue **and** the
gradient component along the imaged modes (equivalent to applying
``P = I - 2 v v^T`` to the Hessian and gradient, see Heyden et al. 2005).
When `verbose=True` (default), the optimiser logs the Hessian eigenvalue
spectrum and a diagnostic of how many modes are imaginary at every step,
which is the most informative telltale during a TS hunt.

A hard trust-radius cap is applied as a safety net: if the restricted-step
solver returns a step larger than `trust_radius`, the step is rescaled
uniformly while preserving its direction.

### Fischer model Hessian with periodic refresh

```python
opt = RSIRFO(
    atoms,
    hessian="fischer",               # initial model Hessian
    hessian_recompute_interval=10,   # rebuild model every 10 steps
    hessian_recompute_method="model",# use the same model at current geometry
)
opt.run(fmax=0.05)
```

### Numerical Hessian refresh (works with any calculator)

```python
opt = RSIRFO(
    atoms,
    hessian="identity",
    hessian_recompute_interval=5,      # full Hessian every 5 steps
    hessian_recompute_method="numerical",
    numerical_hessian_step=0.01,       # finite-difference step in Angstrom
)
opt.run(fmax=0.05)
```

### Analytic Hessian via callback

```python
def my_hessian(atoms):
    """Return the (3N x 3N) Hessian in eV/Angstrom^2."""
    # e.g. call an external code, read from file, etc.
    return atoms.calc.get_property("hessian")  # if supported

opt = RSIRFO(
    atoms,
    hessian="identity",
    hessian_recompute_interval=3,
    hessian_recompute_method="callback",
    hessian_callback=my_hessian,
)
opt.run(fmax=0.05)
```

### Restart

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
| `hessian` | `'identity'` | Initial Hessian: `'identity'`, `'fischer'`, `'swart'`, or `ndarray` |
| `hessian_update` | auto | Quasi-Newton update method. Default: `'block_fsb'` for `order=0`, `'block_bofill'` for `order>=1`. Pass `'auto'` for the Bakó-Császár flowchart selector. |
| `verbose` | `True` | Log Hessian eigenvalue spectrum at every step |
| `eigval_log_count` | `10` | How many eigenvalues to display in verbose mode |
| `hessian_recompute_interval` | auto | Refresh Hessian every N steps. Default: `0` (off) for `'identity'` or user `ndarray`; `50` (`order=0`) or `5` (`order>=1`) for `'fischer'`, `'swart'`, or when a callback is provided. Pass `0` explicitly to disable. |
| `hessian_recompute_method` | `None` | `'model'`, `'numerical'`, `'callback'`, or `None` (auto) |
| `hessian_callback` | `None` | Callable `(Atoms) -> ndarray` for analytic Hessian |
| `numerical_hessian_step` | `0.01` | Finite-difference step (Angstrom) |
| `trust_radius` | `0.3` / `0.1` | Initial trust radius in Angstrom (min / TS default) |
| `trust_radius_max` | `0.5` | Maximum trust radius (Angstrom) |
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

The Fischer and Swart model Hessians in `model_hessian.py` are
independent implementations based solely on the original publications
(Fischer & Almlöf 1992; Swart & Bickelhaupt 2006). No code from any
other package has been copied. See `NOTICE.md` for full references.

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

## License

Copyright (C) 2026 ss0832  
Licensed under the GNU General Public License, version 3 or later.  
See `LICENSE` for the full text.
