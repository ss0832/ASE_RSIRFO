"""
model_hessian.py
================

Two empirical model Hessian generators that provide a chemistry-aware
starting guess for geometry optimisation:

* :class:`FischerD3ModelHessian` -- Fischer & Almlöf (1992) bonded-term model
  augmented with a Grimme D3-BJ dispersion correction for non-bonded pairs.
* :class:`SwartD2ModelHessian`   -- Swart & Bickelhaupt (2006) all-pairs model
  with a Grimme D2 dispersion correction.

Both generators return the **Cartesian** Hessian in atomic units
(Hartree / Bohr^2) and **without** translation / rotation projection. The
calling optimiser is responsible for converting units and projecting modes.

References
----------
* T. H. Fischer, J. Almlöf, *J. Phys. Chem.* **96**, 9768 (1992)
* M. Swart, F. M. Bickelhaupt, *Int. J. Quantum Chem.* **106**, 2536 (2006)
* S. Grimme, *J. Comput. Chem.* **27**, 1787 (2006) - D2 dispersion
* S. Grimme, J. Antony, S. Ehrlich, H. Krieg,
  *J. Chem. Phys.* **132**, 154104 (2010) - D3 dispersion
* S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011) -
  Becke-Johnson damping for D3
* J. M. Anglada, J. M. Bofill, *Theor. Chem. Acc.* **100**, 336 (1998) -
  out-of-plane / off-diagonal handling for empirical Hessians.
"""

from __future__ import annotations

import numpy as np

from .bond_connectivity import BondConnectivity
from .internal_coords import (
    bend, d3_pair_hessian, out_of_plane, stretch, torsion,
    vdw_anisotropic, vdw_isotropic,
)
from .parameters import (
    D3Parameters,
    covalent_radius_bohr,
    d2_c6_coefficient,
    d2_vdw_radius_bohr,
    uff_vdw_distance_bohr,
)


# --------------------------------------------------------------------------- #
#  Fischer-Almlöf + D3 dispersion
# --------------------------------------------------------------------------- #


class FischerD3ModelHessian:
    """Fischer & Almlöf (1992) model Hessian with D3-BJ dispersion correction.

    Parameters
    ----------
    bond_factor
        Connectivity threshold ``r <= bond_factor * (r_cov_i + r_cov_j)``
        (default 1.3).
    d3_functional
        Functional name passed to :class:`~.parameters.D3Parameters`.
        Default ``'pbe0'``.

    Notes
    -----
    Fischer's empirical force constants (atomic units, distances in Bohr):

    * Stretch:  ``k = 0.3601 * exp(-1.944 * (r - r_cov))``
    * Bend:     ``k = 0.089 + 0.11 * (r_cov_ij * r_cov_jk)^{0.42}
                  * exp(-0.44 * (r_ij + r_jk - r_cov_ij - r_cov_jk))``
    * Torsion:  ``k = 0.0015 + 14.0 * sum_bonds^{0.57} / (r * r_cov)^4
                  * exp(-2.85 * (r - r_cov))``
    """

    def __init__(
        self, bond_factor: float = 1.3, d3_functional: str = "pbe0"
    ) -> None:
        self.bond_factor = float(bond_factor)
        self.d3_params = D3Parameters(d3_functional)
        self._connectivity = BondConnectivity(bond_factor=self.bond_factor)

    # ----- empirical force constants ----------------------------------------
    @staticmethod
    def _k_stretch(r: float, r_cov: float) -> float:
        return 0.3601 * float(np.exp(-1.944 * (r - r_cov)))

    @staticmethod
    def _k_bend(
        r_ab: float, r_bc: float, r_ab_cov: float, r_bc_cov: float
    ) -> float:
        prod = r_ab_cov * r_bc_cov
        if prod < 1e-12:
            return 0.0
        return 0.089 + 0.11 * (prod ** 0.42) * float(
            np.exp(-0.44 * (r_ab + r_bc - r_ab_cov - r_bc_cov))
        )

    @staticmethod
    def _k_torsion(r_jk: float, r_jk_cov: float, n_bonds_central: int) -> float:
        prod = r_jk * r_jk_cov
        if prod < 1e-12:
            return 0.0
        return 0.0015 + 14.0 * (max(n_bonds_central, 0) ** 0.57) / (
            prod ** 4
        ) * float(np.exp(-2.85 * (r_jk - r_jk_cov)))

    # ----- Cartesian assembly ------------------------------------------------
    def _add_bonded(
        self,
        H: np.ndarray,
        coord: np.ndarray,
        elements: list[str],
        bond_mat: np.ndarray,
    ) -> None:
        """Bond + bend + torsion contributions in Wilson B form."""
        # Bonds
        for i, j in self._connectivity.bond_connect_table(bond_mat):
            try:
                r, B = stretch(coord[[i, j]])
            except ArithmeticError:
                continue
            r_cov = covalent_radius_bohr(elements[i]) + covalent_radius_bohr(
                elements[j]
            )
            k = self._k_stretch(r, r_cov)
            for a_idx, a in enumerate((i, j)):
                for b_idx, b in enumerate((i, j)):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += k * np.outer(
                        B[a_idx], B[b_idx]
                    )

        # Bends
        for i, j, k in self._connectivity.angle_connect_table(bond_mat):
            try:
                _, Bvec = bend(coord[[i, j, k]])
            except ArithmeticError:
                continue
            r_ij = float(np.linalg.norm(coord[i] - coord[j]))
            r_jk = float(np.linalg.norm(coord[k] - coord[j]))
            r_ij_cov = covalent_radius_bohr(elements[i]) + covalent_radius_bohr(
                elements[j]
            )
            r_jk_cov = covalent_radius_bohr(elements[j]) + covalent_radius_bohr(
                elements[k]
            )
            kk = self._k_bend(r_ij, r_jk, r_ij_cov, r_jk_cov)
            atoms = (i, j, k)
            for a_idx, a in enumerate(atoms):
                for b_idx, b in enumerate(atoms):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += kk * np.outer(
                        Bvec[a_idx], Bvec[b_idx]
                    )

        # Torsions
        torsions = self._connectivity.dihedral_angle_connect_table(bond_mat)
        for i, j, k, l in torsions:
            try:
                _, Bvec = torsion(coord[[i, j, k, l]])
            except ArithmeticError:
                continue
            r_jk = float(np.linalg.norm(coord[k] - coord[j]))
            r_jk_cov = covalent_radius_bohr(elements[j]) + covalent_radius_bohr(
                elements[k]
            )
            n_bonds = int(np.sum(bond_mat[j])) + int(np.sum(bond_mat[k])) - 2
            k_t = self._k_torsion(r_jk, r_jk_cov, n_bonds)

            # Damping near 180-deg interior angles - same idea as the original
            # implementation, scaling by sin^2(theta_1) * sin^2(theta_2).
            v_ji = coord[i] - coord[j]
            v_jk = coord[k] - coord[j]
            v_kl = coord[l] - coord[k]
            n_ji = float(np.linalg.norm(v_ji))
            n_jk = float(np.linalg.norm(v_jk))
            n_kl = float(np.linalg.norm(v_kl))
            if n_ji < 1e-8 or n_jk < 1e-8 or n_kl < 1e-8:
                continue
            cos1 = float(np.dot(v_ji, v_jk) / (n_ji * n_jk))
            cos2 = float(np.dot(-v_jk, v_kl) / (n_jk * n_kl))
            sin2_1 = max(0.0, 1.0 - cos1 ** 2)
            sin2_2 = max(0.0, 1.0 - cos2 ** 2)
            if sin2_1 < 1e-4 or sin2_2 < 1e-4:
                continue
            k_t *= sin2_1 * sin2_2

            atoms = (i, j, k, l)
            for a_idx, a in enumerate(atoms):
                for b_idx, b in enumerate(atoms):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += k_t * np.outer(
                        Bvec[a_idx], Bvec[b_idx]
                    )

    def _add_d3_dispersion(
        self,
        H: np.ndarray,
        coord: np.ndarray,
        elements: list[str],
        bond_mat: np.ndarray,
    ) -> None:
        """D3-BJ dispersion second-derivative correction (non-bonded pairs)."""
        n = coord.shape[0]
        s6, s8, a1, a2 = (
            self.d3_params.s6, self.d3_params.s8,
            self.d3_params.a1, self.d3_params.a2,
        )
        # Pre-compute element-wise C8 components.
        for i in range(n):
            for j in range(i):
                if bond_mat[i, j]:
                    continue
                r_vec = coord[i] - coord[j]
                r_ij = float(np.linalg.norm(r_vec))
                if r_ij < 0.1:
                    continue
                c6_ij = float(np.sqrt(
                    d2_c6_coefficient(elements[i])
                    * d2_c6_coefficient(elements[j])
                ))
                q_i = self.d3_params.get_r4r2(elements[i])
                q_j = self.d3_params.get_r4r2(elements[j])
                c8_ij = 3.0 * c6_ij * float(np.sqrt(q_i * q_j))
                r0 = float(np.sqrt(q_i * q_j))
                hess_block = d3_pair_hessian(
                    r_vec, c6_ij, c8_ij, r0, s6, s8, a1, a2
                )
                si, sj = 3 * i, 3 * j
                H[si:si + 3, si:si + 3] += hess_block
                H[sj:sj + 3, sj:sj + 3] += hess_block
                H[si:si + 3, sj:sj + 3] -= hess_block
                H[sj:sj + 3, si:si + 3] -= hess_block

    # ----- public API --------------------------------------------------------
    def build(
        self, coord_bohr: np.ndarray, elements: list[str]
    ) -> np.ndarray:
        """Return the model Hessian in Hartree / Bohr^2 (Cartesian)."""
        coord = np.asarray(coord_bohr, dtype=float).reshape(-1, 3)
        n = coord.shape[0]
        H = np.zeros((3 * n, 3 * n), dtype=float)

        bond_mat = self._connectivity.bond_connect_matrix(elements, coord)
        self._add_bonded(H, coord, elements, bond_mat)
        self._add_d3_dispersion(H, coord, elements, bond_mat)
        return 0.5 * (H + H.T)


# --------------------------------------------------------------------------- #
#  Swart-Bickelhaupt + D2 dispersion
# --------------------------------------------------------------------------- #


class SwartD2ModelHessian:
    """Swart-Bickelhaupt (2006) all-pairs model Hessian with D2 dispersion.

    Compared to Fischer's model, Swart's variant treats every pair (bonded or
    not) with an exponential covalent term plus a Gaussian VDW term, then
    adds an explicit D2 second-derivative correction. Bend, torsion and
    out-of-plane contributions follow Wilson's B-matrix formulation.

    Parameters
    ----------
    kr, kf, kt
        Coupling constants for stretch, bend, torsion (atomic units).
        Defaults match the original Swart paper.
    kd
        Weight of the Gaussian VDW spring contribution (default 2.0).
    """

    def __init__(
        self,
        kr: float = 0.35,
        kf: float = 0.15,
        kt: float = 0.005,
        kd: float = 2.00,
        bond_factor: float = 1.3,
    ) -> None:
        self.kr = float(kr)
        self.kf = float(kf)
        self.kt = float(kt)
        self.kd = float(kd)
        self.bond_factor = float(bond_factor)
        self._connectivity = BondConnectivity(bond_factor=self.bond_factor)
        self._eps = 1.0e-12

    # ----- empirical pair springs -------------------------------------------
    @staticmethod
    def _exp_covalent_factor(r: float, r_cov: float) -> float:
        """``exp(-1 * (r / r_cov - 1))``."""
        if r_cov < 1e-12:
            return 0.0
        return float(np.exp(-(r / r_cov - 1.0)))

    @staticmethod
    def _gauss_vdw_factor(r: float, r_vdw: float, alpha: float = 5.0) -> float:
        """``exp(-alpha * (r_vdw - r)^2)``."""
        return float(np.exp(-alpha * (r_vdw - r) ** 2))

    # ----- bond stretch + D2 dispersion (acts on every pair) ----------------
    def _add_bond_and_dispersion(
        self,
        H: np.ndarray,
        coord: np.ndarray,
        elements: list[str],
    ) -> None:
        n = coord.shape[0]
        for i in range(n):
            for j in range(i):
                rij = coord[i] - coord[j]
                r2 = float(np.dot(rij, rij))
                if r2 < self._eps:
                    continue
                r = float(np.sqrt(r2))
                r_cov = covalent_radius_bohr(elements[i]) + covalent_radius_bohr(
                    elements[j]
                )
                r_vdw = uff_vdw_distance_bohr(elements[i]) + uff_vdw_distance_bohr(
                    elements[j]
                )
                # Combined "spring" force constant (Swart-style).
                g = (
                    self.kr * self._exp_covalent_factor(r, r_cov)
                    + self.kd * self._gauss_vdw_factor(r, r_vdw)
                )
                # Spring rank-1 outer product r r^T / r^2.
                outer = np.outer(rij, rij) / r2
                spring_block = g * outer
                # D2 dispersion correction.
                c6 = float(np.sqrt(
                    d2_c6_coefficient(elements[i])
                    * d2_c6_coefficient(elements[j])
                ))
                rvdw_d2 = d2_vdw_radius_bohr(elements[i]) + d2_vdw_radius_bohr(
                    elements[j]
                )
                vdw_block = np.empty((3, 3))
                xs = (rij[0], rij[1], rij[2])
                # diagonal terms: d^2 / dx_alpha^2
                for a in range(3):
                    others = (xs[(a + 1) % 3], xs[(a + 2) % 3])
                    vdw_block[a, a] = vdw_isotropic(
                        xs[a], others[0], others[1], c6, rvdw_d2
                    )
                # off-diagonal terms
                for a in range(3):
                    for b in range(a + 1, 3):
                        c = 3 - a - b
                        vdw_block[a, b] = vdw_anisotropic(
                            xs[a], xs[b], xs[c], c6, rvdw_d2
                        )
                        vdw_block[b, a] = vdw_block[a, b]
                block = spring_block - vdw_block

                si, sj = 3 * i, 3 * j
                H[si:si + 3, si:si + 3] += block
                H[sj:sj + 3, sj:sj + 3] += block
                H[si:si + 3, sj:sj + 3] -= block
                H[sj:sj + 3, si:si + 3] -= block

    # ----- bend (only bonded triples) ---------------------------------------
    def _add_bends(
        self,
        H: np.ndarray,
        coord: np.ndarray,
        elements: list[str],
        bond_mat: np.ndarray,
    ) -> None:
        for i, j, k in self._connectivity.angle_connect_table(bond_mat):
            try:
                _, Bvec = bend(coord[[i, j, k]])
            except ArithmeticError:
                continue
            r_ij = float(np.linalg.norm(coord[i] - coord[j]))
            r_jk = float(np.linalg.norm(coord[k] - coord[j]))
            r_ij_cov = covalent_radius_bohr(elements[i]) + covalent_radius_bohr(
                elements[j]
            )
            r_jk_cov = covalent_radius_bohr(elements[j]) + covalent_radius_bohr(
                elements[k]
            )
            r_ij_vdw = uff_vdw_distance_bohr(elements[i]) + uff_vdw_distance_bohr(
                elements[j]
            )
            r_jk_vdw = uff_vdw_distance_bohr(elements[j]) + uff_vdw_distance_bohr(
                elements[k]
            )
            g_ij = (
                self._exp_covalent_factor(r_ij, r_ij_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(r_ij, r_ij_vdw)
            )
            g_jk = (
                self._exp_covalent_factor(r_jk, r_jk_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(r_jk, r_jk_vdw)
            )
            kk = self.kf * g_ij * g_jk
            atoms = (i, j, k)
            for a_idx, a in enumerate(atoms):
                for b_idx, b in enumerate(atoms):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += kk * np.outer(
                        Bvec[a_idx], Bvec[b_idx]
                    )

    # ----- torsions ---------------------------------------------------------
    def _add_torsions(
        self,
        H: np.ndarray,
        coord: np.ndarray,
        elements: list[str],
        bond_mat: np.ndarray,
    ) -> None:
        cos35 = float(np.cos(np.deg2rad(35.0)))
        for i, j, k, l in self._connectivity.dihedral_angle_connect_table(
            bond_mat
        ):
            r_ij = coord[i] - coord[j]
            r_jk = coord[j] - coord[k]
            r_kl = coord[k] - coord[l]
            n_ij = float(np.linalg.norm(r_ij))
            n_jk = float(np.linalg.norm(r_jk))
            n_kl = float(np.linalg.norm(r_kl))
            if n_ij < self._eps or n_jk < self._eps or n_kl < self._eps:
                continue
            cos1 = float(np.dot(r_ij, r_jk) / (n_ij * n_jk))
            cos2 = float(np.dot(r_kl, r_jk) / (n_kl * n_jk))
            if abs(cos1) > cos35 or abs(cos2) > cos35:
                continue
            try:
                _, Bvec = torsion(coord[[i, j, k, l]])
            except ArithmeticError:
                continue
            ij_cov = covalent_radius_bohr(elements[i]) + covalent_radius_bohr(
                elements[j]
            )
            jk_cov = covalent_radius_bohr(elements[j]) + covalent_radius_bohr(
                elements[k]
            )
            kl_cov = covalent_radius_bohr(elements[k]) + covalent_radius_bohr(
                elements[l]
            )
            ij_vdw = uff_vdw_distance_bohr(elements[i]) + uff_vdw_distance_bohr(
                elements[j]
            )
            jk_vdw = uff_vdw_distance_bohr(elements[j]) + uff_vdw_distance_bohr(
                elements[k]
            )
            kl_vdw = uff_vdw_distance_bohr(elements[k]) + uff_vdw_distance_bohr(
                elements[l]
            )
            g_ij = (
                self._exp_covalent_factor(n_ij, ij_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(n_ij, ij_vdw)
            )
            g_jk = (
                self._exp_covalent_factor(n_jk, jk_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(n_jk, jk_vdw)
            )
            g_kl = (
                self._exp_covalent_factor(n_kl, kl_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(n_kl, kl_vdw)
            )
            kk = self.kt * g_ij * g_jk * g_kl
            atoms = (i, j, k, l)
            for a_idx, a in enumerate(atoms):
                for b_idx, b in enumerate(atoms):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += kk * np.outer(
                        Bvec[a_idx], Bvec[b_idx]
                    )

    # ----- out-of-plane -----------------------------------------------------
    def _add_out_of_plane(
        self,
        H: np.ndarray,
        coord: np.ndarray,
        elements: list[str],
        bond_mat: np.ndarray,
    ) -> None:
        for i, j, k, apex in self._connectivity.out_of_plane_connect_table(
            bond_mat
        ):
            try:
                _, Bvec = out_of_plane(coord[[i, j, k, apex]])
            except ArithmeticError:
                continue
            # Skip if any of the three reference bonds is near-collinear.
            atoms = (i, j, k, apex)
            r_ij = coord[i] - coord[apex]
            r_ik = coord[j] - coord[apex]
            r_il = coord[k] - coord[apex]
            n_ij = float(np.linalg.norm(r_ij))
            n_ik = float(np.linalg.norm(r_ik))
            n_il = float(np.linalg.norm(r_il))
            if n_ij < self._eps or n_ik < self._eps or n_il < self._eps:
                continue
            ij_cov = covalent_radius_bohr(elements[i]) + covalent_radius_bohr(
                elements[apex]
            )
            ik_cov = covalent_radius_bohr(elements[j]) + covalent_radius_bohr(
                elements[apex]
            )
            il_cov = covalent_radius_bohr(elements[k]) + covalent_radius_bohr(
                elements[apex]
            )
            ij_vdw = uff_vdw_distance_bohr(elements[i]) + uff_vdw_distance_bohr(
                elements[apex]
            )
            ik_vdw = uff_vdw_distance_bohr(elements[j]) + uff_vdw_distance_bohr(
                elements[apex]
            )
            il_vdw = uff_vdw_distance_bohr(elements[k]) + uff_vdw_distance_bohr(
                elements[apex]
            )
            g_ij = (
                self._exp_covalent_factor(n_ij, ij_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(n_ij, ij_vdw)
            )
            g_ik = (
                self._exp_covalent_factor(n_ik, ik_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(n_ik, ik_vdw)
            )
            g_il = (
                self._exp_covalent_factor(n_il, il_cov)
                + 0.5 * self.kd * self._gauss_vdw_factor(n_il, il_vdw)
            )
            kk = self.kt * g_ij * g_ik * g_il
            for a_idx, a in enumerate(atoms):
                for b_idx, b in enumerate(atoms):
                    H[3 * a:3 * a + 3, 3 * b:3 * b + 3] += kk * np.outer(
                        Bvec[a_idx], Bvec[b_idx]
                    )

    # ----- public API --------------------------------------------------------
    def build(
        self, coord_bohr: np.ndarray, elements: list[str]
    ) -> np.ndarray:
        """Return the model Hessian in Hartree / Bohr^2 (Cartesian)."""
        coord = np.asarray(coord_bohr, dtype=float).reshape(-1, 3)
        n = coord.shape[0]
        H = np.zeros((3 * n, 3 * n), dtype=float)

        bond_mat = self._connectivity.bond_connect_matrix(elements, coord)
        self._add_bond_and_dispersion(H, coord, elements)
        self._add_bends(H, coord, elements, bond_mat)
        self._add_torsions(H, coord, elements, bond_mat)
        self._add_out_of_plane(H, coord, elements, bond_mat)
        return 0.5 * (H + H.T)
