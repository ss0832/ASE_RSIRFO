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
* Grimme D2 dispersion (C6, VDW radii)
    S. Grimme, *J. Comput. Chem.* **27**, 1787 (2006).
* Grimme D3 dispersion (s6, s8, a1, a2, r4r2 reference values, BJ damping)
    S. Grimme, J. Antony, S. Ehrlich, H. Krieg,
    *J. Chem. Phys.* **132**, 154104 (2010).
    S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011).
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
#  Grimme D2 parameters (C6, VDW radii)
# --------------------------------------------------------------------------- #
#
# Values from the original D2 publication (Grimme 2006). C6 in
# (J nm^6 / mol), but the customary form in atomic units is used here:
# C6 in Hartree * Bohr^6, R in Bohr.
# Tabulated up to Xe (Z = 54). Heavier elements raise an exception.
# --------------------------------------------------------------------------- #

_D2_C6_HARTREE_BOHR6: dict[str, float] = {
    "H": 0.140,
    "He": 0.080,
    "Li": 1.610, "Be": 1.610,
    "B": 3.130, "C": 1.750, "N": 1.230, "O": 0.700, "F": 0.750, "Ne": 0.630,
    "Na": 5.710, "Mg": 5.710,
    "Al": 10.79, "Si": 9.230, "P": 7.840, "S": 5.570, "Cl": 5.070, "Ar": 4.610,
    "K": 10.80, "Ca": 10.80,
    "Sc": 10.80, "Ti": 10.80, "V": 10.80, "Cr": 10.80, "Mn": 10.80,
    "Fe": 10.80, "Co": 10.80, "Ni": 10.80, "Cu": 10.80, "Zn": 10.80,
    "Ga": 16.99, "Ge": 17.10, "As": 16.37, "Se": 12.64, "Br": 12.47,
    "Kr": 12.01,
    "Rb": 24.67, "Sr": 24.67,
    "Y":  24.67, "Zr": 24.67, "Nb": 24.67, "Mo": 24.67, "Tc": 24.67,
    "Ru": 24.67, "Rh": 24.67, "Pd": 24.67, "Ag": 24.67, "Cd": 24.67,
    "In": 37.32, "Sn": 38.71, "Sb": 38.44, "Te": 31.74, "I": 31.50,
    "Xe": 29.99,
    "X": 0.0,
}

# Grimme 2006 D2 vdW radii (in Angstrom in the paper, converted to Bohr here).
_D2_VDW_ANGSTROM: dict[str, float] = {
    "H": 1.001,
    "He": 1.012,
    "Li": 0.825, "Be": 1.408,
    "B": 1.485, "C": 1.452, "N": 1.397, "O": 1.342, "F": 1.287, "Ne": 1.243,
    "Na": 1.144, "Mg": 1.364,
    "Al": 1.639, "Si": 1.716, "P": 1.705, "S": 1.683, "Cl": 1.639, "Ar": 1.595,
    "K": 1.485, "Ca": 1.474,
    "Sc": 1.562, "Ti": 1.562, "V": 1.562, "Cr": 1.562, "Mn": 1.562,
    "Fe": 1.562, "Co": 1.562, "Ni": 1.562, "Cu": 1.562, "Zn": 1.562,
    "Ga": 1.650, "Ge": 1.727, "As": 1.760, "Se": 1.771, "Br": 1.749,
    "Kr": 1.727,
    "Rb": 1.628, "Sr": 1.606,
    "Y":  1.639, "Zr": 1.639, "Nb": 1.639, "Mo": 1.639, "Tc": 1.639,
    "Ru": 1.639, "Rh": 1.639, "Pd": 1.639, "Ag": 1.639, "Cd": 1.639,
    "In": 1.672, "Sn": 1.804, "Sb": 1.881, "Te": 1.892, "I": 1.892,
    "Xe": 1.881,
    "X": 1.500,
}


def d2_c6_coefficient(element: str | int) -> float:
    """Grimme D2 C6 coefficient in Hartree*Bohr^6."""
    sym = number_to_element(element)
    if sym not in _D2_C6_HARTREE_BOHR6:
        raise KeyError(f"D2 C6 not tabulated for element {sym!r}")
    return _D2_C6_HARTREE_BOHR6[sym]


def d2_vdw_radius_bohr(element: str | int) -> float:
    """Grimme D2 van-der-Waals radius in Bohr."""
    sym = number_to_element(element)
    if sym not in _D2_VDW_ANGSTROM:
        raise KeyError(f"D2 vdW radius not tabulated for element {sym!r}")
    return _D2_VDW_ANGSTROM[sym] / BOHR_TO_ANGSTROM


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
        # Fall back to D2 vdW radius if UFF entry is missing.
        return d2_vdw_radius_bohr(sym)
    return _UFF_VDW_ANGSTROM[sym] / BOHR_TO_ANGSTROM


# --------------------------------------------------------------------------- #
#  Grimme D3 parameters
# --------------------------------------------------------------------------- #
#
# Becke-Johnson damping form:
#     E_disp = - sum_n=6,8 s_n * C_n^{ij} / (R_ij^n + (a1 * R0_ij + a2)^n)
# with R0_ij = sqrt(Q_i * Q_j) and Q_i = sqrt(r4r2_i).
#
# The s6, s8, a1, a2 values are functional-dependent. We expose a small set
# of presets and default to the widely used PBE0-D3(BJ) parameters.
# --------------------------------------------------------------------------- #

# r4r2 reference values (sqrt(<r4>/<r2>)) in Bohr, atoms 1..86.
# Values from the dftd3 reference implementation.
_R4R2_BOHR: tuple[float, ...] = (
    0.0,                                                # Z = 0 placeholder
    2.00734898, 1.56637132,                             # H, He
    5.01986934, 3.85379032,                             # Li, Be
    3.64446594, 3.10492822, 2.71175247,                 # B, C, N
    2.59361680, 2.38825250, 2.21522516,                 # O, F, Ne
    6.58585536, 5.46295967,                             # Na, Mg
    5.65216669, 4.88284902, 4.29727576,                 # Al, Si, P
    4.04108902, 3.72932356, 3.44677275,                 # S, Cl, Ar
    7.97762753, 7.07623947,                             # K, Ca
    6.60844053, 6.28791364, 6.07728703, 5.54643096,     # Sc-Cr
    5.80491167, 5.58415602, 5.41374528, 5.28497229,     # Mn-Ni
    5.22592821, 5.09817141,                             # Cu, Zn
    6.12149689, 5.54083734, 5.06696878, 4.87005108,     # Ga-Se
    4.59089647, 4.31176304,                             # Br, Kr
    9.55461698, 8.67396077,                             # Rb, Sr
    7.97210197, 7.43439917, 6.58711862, 6.19536215,     # Y-Mo
    6.01517290, 5.81623410, 5.65710424, 5.52640661,     # Tc-Pd
    5.44263305, 5.58285373,                             # Ag, Cd
    7.02081898, 6.46815523, 5.98089120, 5.81686657,     # In-Te
    5.53321815, 5.25477007,                             # I, Xe
    11.02204549, 10.15679528,                           # Cs, Ba
    9.35167836,                                         # La
    9.06926079, 8.97241155, 8.90092807, 8.85984840,     # Ce-Sm
    8.81736827, 8.79317710, 7.89969626, 8.80588454,     # Eu-Dy
    8.42439218, 8.54289262, 8.47583370, 8.45090888,     # Ho-Yb
    8.47339339,                                         # Lu
    7.83525634, 8.20702843, 7.70559063, 7.32755997,     # Hf-Os
    7.03887381, 6.68978720, 6.05450052, 5.88752022,     # Ir-Hg
    7.89629395, 7.97083874, 7.32614983, 7.07689971,     # Tl-Po
    6.76667235, 6.45331945,                             # At, Rn
)


class D3Parameters:
    """Grimme D3-BJ parameters for selected functionals.

    Becke-Johnson damping form following Grimme, Ehrlich & Goerigk
    (*J. Comput. Chem.* **32**, 1456, 2011).

    Parameters
    ----------
    functional
        Functional preset to pick. Currently supported: ``'pbe0'`` (default),
        ``'pbe'``, ``'b3lyp'``, ``'b97-d3'``, ``'tpss'``.
        Custom values can be set by direct attribute assignment after
        construction.
    """

    PRESETS: dict[str, dict[str, float]] = {
        "pbe0":  {"s6": 1.000, "s8": 1.2177, "a1": 0.4145, "a2": 4.8593},
        "pbe":   {"s6": 1.000, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
        "b3lyp": {"s6": 1.000, "s8": 1.9889, "a1": 0.3981, "a2": 4.4211},
        "b97-d3": {"s6": 1.000, "s8": 0.6939, "a1": 0.5546, "a2": 3.2297},
        "tpss":  {"s6": 1.000, "s8": 1.9435, "a1": 0.4535, "a2": 4.4752},
    }

    def __init__(self, functional: str = "pbe0") -> None:
        if functional not in self.PRESETS:
            raise KeyError(
                f"Unknown D3 functional preset {functional!r}. "
                f"Available: {sorted(self.PRESETS)}"
            )
        params = self.PRESETS[functional]
        self.functional = functional
        self.s6 = params["s6"]
        self.s8 = params["s8"]
        self.a1 = params["a1"]
        self.a2 = params["a2"]

    def get_r4r2(self, element: str | int) -> float:
        """Reference quantity ``Q = sqrt(<r^4>/<r^2>)`` in Bohr."""
        z = element_to_number(element)
        if z < 1 or z >= len(_R4R2_BOHR):
            raise IndexError(
                f"D3 r4r2 not tabulated for atomic number {z}"
            )
        return _R4R2_BOHR[z]


# --------------------------------------------------------------------------- #
#  Convenience helpers used by model-Hessian / connectivity code
# --------------------------------------------------------------------------- #

def covalent_radii_array_bohr(elements: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return covalent radii (Bohr) for a list of element symbols."""
    return np.array([covalent_radius_bohr(e) for e in elements], dtype=float)
