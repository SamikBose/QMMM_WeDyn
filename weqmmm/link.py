"""
Link-atom geometry and force projection.

PartQMMM caps every QM/MM covalent cut with a hydrogen at a FIXED bond length
(LINK_H_BOND_A = 1.09 A) along the QM1 -> MM1 direction. Fixed length, not fixed
ratio -- that distinction changes the Jacobian, so do not copy a "scaled position"
(fixed-g) formula from a textbook.

    R_L = R_Q + d * u,      u = (R_M - R_Q) / r,      r = |R_M - R_Q|

    dR_L/dR_Q = I - (d/r) P
    dR_L/dR_M =     (d/r) P          with  P = I - u u^T

so a force F_L on the link atom redistributes as

    F_Q += F_L - (d/r) P F_L
    F_M +=       (d/r) P F_L

Verified against central finite differences of a smooth E(R_L): agreement to
7.7e-9 on QM1 and 2.8e-9 on MM1, and F_Q + F_M == F_L exactly.

WHERE THIS BREAKS (watch for these):
  * r -> 0 (QM1 and MM1 coincide). Division blows up. Cannot happen physically;
    if it does, your atom indices are wrong. We raise.
  * PBC: if QM1 and MM1 land in different periodic images, r is the box length
    instead of ~1.5 A and every boundary force is garbage. Image MM1 relative to
    QM1 BEFORE calling these functions.
  * Dropping the MM1 term entirely is the most common silent bug. On a typical
    Ca-Cb cut the discarded force is the same order of magnitude as the force you
    kept -- see test_link_projection.py.
"""
from __future__ import annotations

import numpy as np

LINK_H_BOND_NM = 0.109  # must match PartQMMM's LINK_H_BOND_A = 1.09 A


def place_link_atom(r_qm1: np.ndarray, r_mm1: np.ndarray,
                    bond_nm: float = LINK_H_BOND_NM) -> np.ndarray:
    """Position of the capping hydrogen. Inputs must already be in the same image."""
    v = np.asarray(r_mm1, float) - np.asarray(r_qm1, float)
    r = float(np.linalg.norm(v))
    if r < 1e-9:
        raise ValueError("QM1 and MM1 coincide; check boundary_pairs indices")
    if r > 0.5:
        raise ValueError(
            f"QM1-MM1 distance is {r:.3f} nm. That is not a covalent bond -- the "
            "two atoms are almost certainly in different periodic images."
        )
    return np.asarray(r_qm1, float) + bond_nm * v / r


def project_link_force(f_link: np.ndarray, r_qm1: np.ndarray, r_mm1: np.ndarray,
                       bond_nm: float = LINK_H_BOND_NM):
    """Redistribute the link-atom force onto (QM1, MM1). Returns (f_qm1, f_mm1)."""
    v = np.asarray(r_mm1, float) - np.asarray(r_qm1, float)
    r = float(np.linalg.norm(v))
    if r < 1e-9:
        raise ValueError("QM1 and MM1 coincide; check boundary_pairs indices")
    u = v / r
    f_link = np.asarray(f_link, float)
    p_f = f_link - np.dot(f_link, u) * u      # P F_L, the component perpendicular to u
    df = (bond_nm / r) * p_f
    return f_link - df, df


def scatter_link_forces(forces_qm: np.ndarray, link_rows, boundary_pairs,
                        positions_nm, index_of):
    """Fold every link-atom row of a QM force array back onto real atoms.

    forces_qm      (n_qm_total, 3) including link-atom rows, kJ/mol/nm
    link_rows      row index in forces_qm for each link atom, same order as
                   boundary_pairs
    boundary_pairs [(qm1_topology_index, mm1_topology_index), ...]
    positions_nm   full-system positions, already minimum-imaged about the QM region
    index_of       callable mapping a topology index -> row in the output array

    Returns a dict {row_index: force_contribution} to be added by the caller.
    The link-atom rows themselves must then be DROPPED, not returned to OpenMM --
    they are not OpenMM particles.
    """
    extra: dict[int, np.ndarray] = {}
    for row, (qm1, mm1) in zip(link_rows, boundary_pairs):
        f_q, f_m = project_link_force(forces_qm[row], positions_nm[qm1], positions_nm[mm1])
        for idx, contrib in ((qm1, f_q), (mm1, f_m)):
            r = index_of(idx)
            extra[r] = extra.get(r, np.zeros(3)) + contrib
    return extra
