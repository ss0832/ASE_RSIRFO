"""
bond_connectivity.py
====================

Bond / angle / dihedral / out-of-plane discovery for a molecular skeleton.

The bond graph is built from interatomic distances using the Pyykkö & Atsumi
covalent radii multiplied by a configurable scaling factor (default 1.3) -
a common convention in chemistry-aware optimisers.

References
----------
* Pyykkö, M. Atsumi, *Chem. Eur. J.* **15**, 186 (2009)
  - covalent radii used for the connectivity threshold.
* Wilson, Decius, Cross, *Molecular Vibrations*, McGraw-Hill (1955)
  - definition of the internal coordinate set.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .parameters import covalent_radii_array_bohr


class BondConnectivity:
    """Build the bond / angle / dihedral / OOP graphs of a skeleton.

    Parameters
    ----------
    bond_factor
        Two atoms are considered bonded when their interatomic distance is
        less than ``bond_factor * (r_cov_i + r_cov_j)``. Default 1.3.
    """

    def __init__(self, bond_factor: float = 1.3) -> None:
        self.bond_factor = float(bond_factor)

    # --------------------------------------------------------------- bond
    def bond_connect_matrix(
        self,
        elements: list[str] | tuple[str, ...],
        coord_bohr: np.ndarray,
    ) -> np.ndarray:
        """Boolean ``(N, N)`` connectivity matrix (no self-bonds)."""
        coord = np.asarray(coord_bohr, dtype=float).reshape(-1, 3)
        if len(elements) != coord.shape[0]:
            raise ValueError(
                "Number of elements does not match number of atoms."
            )
        dist = cdist(coord, coord)
        rcov = covalent_radii_array_bohr(elements)
        threshold = self.bond_factor * (rcov[:, None] + rcov[None, :])
        bond_mat = dist <= threshold
        np.fill_diagonal(bond_mat, False)
        return bond_mat

    @staticmethod
    def bond_connect_table(bond_mat: np.ndarray) -> list[tuple[int, int]]:
        """List of unique bonded pairs ``(i, j)`` with ``i < j``."""
        i_arr, j_arr = np.where(np.triu(bond_mat, k=1))
        return list(zip(i_arr.tolist(), j_arr.tolist()))

    # -------------------------------------------------------------- angle
    @staticmethod
    def angle_connect_table(
        bond_mat: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """List of unique bend triples ``(i, j, k)`` where ``j`` is central
        and ``i < k``."""
        n = bond_mat.shape[0]
        triples: list[tuple[int, int, int]] = []
        for j in range(n):
            neighbours = np.where(bond_mat[j])[0]
            for a_idx in range(len(neighbours)):
                for b_idx in range(a_idx + 1, len(neighbours)):
                    i, k = neighbours[a_idx], neighbours[b_idx]
                    if i < k:
                        triples.append((int(i), int(j), int(k)))
                    else:
                        triples.append((int(k), int(j), int(i)))
        return triples

    # ----------------------------------------------------------- dihedral
    @staticmethod
    def dihedral_angle_connect_table(
        bond_mat: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """List of unique torsion quadruples ``(i, j, k, l)``.

        For every bond (j, k) we enumerate neighbours i of j (i != k) and
        l of k (l != j) and emit the canonical ordering with ``i < l`` to
        avoid duplicates of the same physical torsion.
        """
        n = bond_mat.shape[0]
        torsions: list[tuple[int, int, int, int]] = []
        for j in range(n):
            for k in range(j + 1, n):
                if not bond_mat[j, k]:
                    continue
                neigh_j = [a for a in np.where(bond_mat[j])[0] if a != k]
                neigh_k = [a for a in np.where(bond_mat[k])[0] if a != j]
                for i in neigh_j:
                    for l in neigh_k:
                        if i == l:
                            continue
                        if i < l:
                            torsions.append((int(i), int(j), int(k), int(l)))
                        else:
                            torsions.append((int(l), int(k), int(j), int(i)))
        # Deduplicate (i, j, k, l) and (l, k, j, i) collisions.
        return list(dict.fromkeys(torsions))

    # ------------------------------------------------------- out of plane
    @staticmethod
    def out_of_plane_connect_table(
        bond_mat: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """List of OOP centres ``(i, j, k, l)``: atom ``l`` is the apex,
        ``i, j, k`` are three of its bonded neighbours."""
        n = bond_mat.shape[0]
        oops: list[tuple[int, int, int, int]] = []
        for centre in range(n):
            neighbours = np.where(bond_mat[centre])[0]
            if len(neighbours) < 3:
                continue
            for a in range(len(neighbours)):
                for b in range(a + 1, len(neighbours)):
                    for c in range(b + 1, len(neighbours)):
                        i, j, k = (
                            int(neighbours[a]),
                            int(neighbours[b]),
                            int(neighbours[c]),
                        )
                        oops.append((i, j, k, int(centre)))
        return oops
