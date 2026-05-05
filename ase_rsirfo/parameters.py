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
parameters.py
=============

Atomic data tables and physical constants used by the model Hessian
generators (Fischer-Almlöf and Swart-Bickelhaupt) and by the bond detection
logic.

All distance parameters are stored in **Bohr**, energies in **Hartree**.
Conversion to/from ASE units (Angstrom, eV) is performed at the boundary of
the public API in :mod:`ase_rsirfo.optimizer`.

References
----------
* Covalent radii (single bond)
    P. Pyykkö, M. Atsumi, *Chem. Eur. J.* **15**, 186 (2009).
* UFF van der Waals distances
    A. K. Rappé et al., *J. Am. Chem. Soc.* **114**, 10024 (1992).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Unit conversions (CODATA 2018 consistent)
# --------------------------------------------------------------------------- #

#: Bohr -> Angstrom conversion factor.
BOHR_TO_ANGSTROM: float = 0.529177210903
#: Hartree -> eV conversion factor.
HARTREE_TO_EV: float = 27.211386245988
#: (Hartree / Bohr^2) -> (eV / Angstrom^2) conversion factor.
HARTREE_PER_BOHR2_TO_EV_PER_A2: float = (
    HARTREE_TO_EV / (BOHR_TO_ANGSTROM ** 2)
)
#: (Hartree / Bohr) -> (eV / Angstrom) conversion factor.
HARTREE_PER_BOHR_TO_EV_PER_A: float = HARTREE_TO_EV / BOHR_TO_ANGSTROM


# --------------------------------------------------------------------------- #
#  Element <-> atomic number mapping
# --------------------------------------------------------------------------- #

ELEMENT_SYMBOLS: tuple[str, ...] = (
    "X",  # placeholder for Z=0
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
)

_SYMBOL_TO_NUMBER: dict[str, int] = {
    sym: z for z, sym in enumerate(ELEMENT_SYMBOLS)
}


def element_to_number(symbol: str | int) -> int:
    """Return the atomic number for an element symbol or pass an int through."""
    if isinstance(symbol, (int, np.integer)):
        return int(symbol)
    if symbol not in _SYMBOL_TO_NUMBER:
        raise KeyError(f"Unknown element symbol: {symbol!r}")
    return _SYMBOL_TO_NUMBER[symbol]


def number_to_element(z: int | str) -> str:
    """Return the element symbol for an atomic number or pass a symbol through."""
    if isinstance(z, str):
        return z
    if z < 0 or z >= len(ELEMENT_SYMBOLS):
        raise IndexError(f"Atomic number out of range: {z}")
    return ELEMENT_SYMBOLS[z]


# --------------------------------------------------------------------------- #
#  Covalent radii (single-bond, Pyykkö & Atsumi 2009)
# --------------------------------------------------------------------------- #
#
# Values are in **Angstrom** as published; conversion to Bohr is done in
# :func:`covalent_radius_bohr`.
# --------------------------------------------------------------------------- #

_COVALENT_RADII_ANGSTROM: dict[str, float] = {
    "H": 0.32, "He": 0.46,
    "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63,
    "F": 0.64, "Ne": 0.67,
    "Na": 1.55, "Mg": 1.39, "Al": 1.26, "Si": 1.16, "P": 1.11, "S": 1.03,
    "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71,
    "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16,
    "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18,
    "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17,
    "Rb": 2.10, "Sr": 1.85,
    "Y": 1.63, "Zr": 1.54, "Nb": 1.47, "Mo": 1.38, "Tc": 1.28, "Ru": 1.25,
    "Rh": 1.25, "Pd": 1.20, "Ag": 1.28, "Cd": 1.36,
    "In": 1.42, "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.33, "Xe": 1.31,
    "Cs": 2.32, "Ba": 1.96,
    "La": 1.80, "Ce": 1.63, "Pr": 1.76, "Nd": 1.74, "Pm": 1.73, "Sm": 1.72,
    "Eu": 1.68, "Gd": 1.69, "Tb": 1.68, "Dy": 1.67, "Ho": 1.66, "Er": 1.65,
    "Tm": 1.64, "Yb": 1.70, "Lu": 1.62,
    "Hf": 1.52, "Ta": 1.46, "W": 1.37, "Re": 1.31, "Os": 1.29, "Ir": 1.22,
    "Pt": 1.23, "Au": 1.24, "Hg": 1.33,
    "Tl": 1.44, "Pb": 1.44, "Bi": 1.51, "Po": 1.45, "At": 1.47, "Rn": 1.42,
    "X": 1.000,
}


def covalent_radius_bohr(element: str | int) -> float:
    """Single-bond covalent radius in Bohr.

    Source: Pyykkö & Atsumi, *Chem. Eur. J.* **15**, 186 (2009).
    """
    sym = number_to_element(element)
    if sym not in _COVALENT_RADII_ANGSTROM:
        raise KeyError(
            f"Covalent radius not tabulated for element {sym!r}"
        )
    return _COVALENT_RADII_ANGSTROM[sym] / BOHR_TO_ANGSTROM


def covalent_radius_angstrom(element: str | int) -> float:
    """Single-bond covalent radius in Angstrom."""
    sym = number_to_element(element)
    return _COVALENT_RADII_ANGSTROM[sym]



# --------------------------------------------------------------------------- #
#  UFF van-der-Waals distances (Rappé 1992)
# --------------------------------------------------------------------------- #
#
# Tabulated as x-distance (sigma * 2^{1/6}) in Angstrom. See Rappé et al.,
# JACS 114, 10024 (1992), Table 1, column "x_I". Light elements are sufficient
# for most molecular optimisations.
# --------------------------------------------------------------------------- #

_UFF_VDW_ANGSTROM: dict[str, float] = {
    "H":  2.886, "He": 2.362,
    "Li": 2.451, "Be": 2.745,
    "B":  4.083, "C": 3.851, "N": 3.660, "O": 3.500, "F": 3.364, "Ne": 3.243,
    "Na": 2.983, "Mg": 3.021,
    "Al": 4.499, "Si": 4.295, "P": 4.147, "S": 4.035, "Cl": 3.947,
    "Ar": 3.868,
    "K": 3.812, "Ca": 3.399,
    "Sc": 3.295, "Ti": 3.175, "V": 3.144, "Cr": 3.023, "Mn": 2.961,
    "Fe": 2.912, "Co": 2.872, "Ni": 2.834, "Cu": 3.495, "Zn": 2.763,
    "Ga": 4.383, "Ge": 4.280, "As": 4.230, "Se": 4.205, "Br": 4.189,
    "Kr": 4.141,
    "Rb": 4.114, "Sr": 3.641,
    "Y":  3.345, "Zr": 3.124, "Nb": 3.165, "Mo": 3.052, "Tc": 2.998,
    "Ru": 2.963, "Rh": 2.929, "Pd": 2.899, "Ag": 3.148, "Cd": 2.848,
    "In": 4.463, "Sn": 4.392, "Sb": 4.420, "Te": 4.470, "I": 4.500,
    "Xe": 4.404,
    "Cs": 4.517, "Ba": 3.703,
    "La": 3.522,
    "X":  3.000,
}


def uff_vdw_distance_bohr(element: str | int) -> float:
    """UFF van-der-Waals distance (x) in Bohr.

    The full vdW minimum distance is ``x``; relate to LJ sigma by
    ``sigma = x / 2**(1/6)``.

    Source: Rappé et al., *J. Am. Chem. Soc.* **114**, 10024 (1992).
    """
    sym = number_to_element(element)
    if sym not in _UFF_VDW_ANGSTROM:
        raise KeyError(f"UFF vdW distance not tabulated for element {sym!r}")
    return _UFF_VDW_ANGSTROM[sym] / BOHR_TO_ANGSTROM


# --------------------------------------------------------------------------- #
#  Convenience helpers used by model-Hessian / connectivity code
# --------------------------------------------------------------------------- #

def covalent_radii_array_bohr(elements: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return covalent radii (Bohr) for a list of element symbols."""
    return np.array([covalent_radius_bohr(e) for e in elements], dtype=float)
