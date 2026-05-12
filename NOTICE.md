# NOTICE — ASE_RSIRFO

## License

ASE_RSIRFO is Copyright (C) 2026 ss0832 and licensed under the
GNU General Public License, version 3 or later (GPL-3.0-or-later).
See `LICENSE` in the repository root for the full text.

## Algorithm and Theory

The RS-I-RFO algorithm implemented in this package is based on the
following peer-reviewed publications. These references describe the
mathematical methods; no source code from these authors has been
copied or adapted.

| Method | Reference |
|--------|-----------|
| RFO step / augmented Hessian | Banerjee, Adams, Simons, Shepard, *J. Phys. Chem.* **89**, 52 (1985) |
| RFO implementation details | Baker, *J. Comput. Chem.* **7**, 385 (1986) |
| RS-RFO restricted step | Besalú, Bofill, *Theor. Chem. Acc.* **100**, 265 (1998) |
| I-RFO image projection (TS) | Heyden, Bell, Keil, *J. Chem. Phys.* **123**, 224101 (2005) |
| Secular equation solver | Moré, Sorensen, *SIAM J. Sci. Stat. Comput.* **4**, 553 (1983) |
| Quasi-Newton / BFGS | Nocedal, Wright, *Numerical Optimization*, 2nd ed., Springer (2006) |
| PSB update | Powell, *Math. Program.* **14**, 31 (1978) |
| SR1/BFGS/PSB mix (Bofill) | Bofill, *J. Comput. Chem.* **15**, 1 (1994) |
| FSB update | Schlegel, *Theor. Chem. Acc.* **103**, 294 (2000) |
| CFD update variants | Wu, Rahman, Wang, Louderaj, Hase, Zhuang, *J. Chem. Phys.* **133**, 074101 (2010); Zhuang, Siebert, Hase, Kay, Ceotto, *J. Chem. Theory Comput.* **9**, 54 (2013) |
| Update flowchart selector | Bakó, Császár, *Theor. Chem. Acc.* **135**, 84 (2016) |
| Powell damping | Nocedal, Wright (2006), eq. 18.15 |
| Double damping | Bofill, Comajuan, *J. Comput. Chem.* **36**, 1557 (2015) |
| Fischer-Almlöf model Hessian | Fischer, Almlöf, *J. Phys. Chem.* **96**, 9768 (1992) |
| Swart-Bickelhaupt model Hessian | Swart, Bickelhaupt, *Int. J. Quantum Chem.* **106**, 2536 (2006) |
| Wilson B-vectors | Wilson, Decius, Cross, *Molecular Vibrations*, McGraw-Hill (1955) |
| B-vector torsion formula | Bakken, Helgaker, *J. Chem. Phys.* **117**, 9160 (2002) |
| T/R projection | Miller, Handy, Adams, *J. Chem. Phys.* **72**, 99 (1980) |

## Software References and Implementation Cross-Checks

The RFO secular-equation implementation was checked against the public
``pysisyphus`` project for numerical behaviour. ``pysisyphus`` is distributed
under GPL-3.0. No pysisyphus source files are vendored in this repository. The relevant software/paper
reference is:

| Software | Reference |
|----------|-----------|
| pysisyphus | Steinmetzer, Kupfer, Gräfe, *Int. J. Quantum Chem.* (2020), DOI: 10.1002/qua.26390; source license GPL-3.0 |

Level-shift heuristics and some optimiser-structuring choices were also
checked against the author's earlier MultiOptPy implementation. MultiOptPy
is the same author's project; no third-party source files are vendored here.

## Atomic Data Tables

The numerical values in `parameters.py` are taken from the
scientific literature listed below. They are tabulated facts
(not copyrightable expression); the table *implementation* is
original work by the author of this package.

| Data | Reference |
|------|-----------|
| Covalent radii | Pyykkö, Atsumi, *Chem. Eur. J.* **15**, 186 (2009) |
| UFF vdW distances | Rappé, Casewit, Colwell, Goddard, Skiff, *J. Am. Chem. Soc.* **114**, 10024 (1992) |

## Dependencies

This package depends on ASE, NumPy, and SciPy, each distributed
under their own licenses (LGPL-2.1+, BSD-3-Clause, and BSD-3-Clause
respectively). Those licenses are compatible with GPL-3.0-or-later
for the purpose of distribution.
