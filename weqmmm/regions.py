"""Build a PartQMMM partition once and freeze it for MD.

The complete shifted MM charge array is retained for charge accounting. A second,
fixed index list selects whole MM residue fragments for finite molecular embedding.
This omits long-range QM/MM electrostatics; MM PME alone does not restore it.
The selected molecules can drift beyond the initial radius during propagation.
Nearest-image changes can also introduce discontinuities on sufficiently long runs.
This model is an approximate validation platform, not production periodic QM/MM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrozenPartition:
    qm_atom_indices: tuple[int, ...]
    qm_all_topology_indices: tuple[int, ...]
    qm_elements: tuple[str, ...]
    boundary_pairs: tuple[tuple[int, int], ...]
    m2_per_cb: tuple[tuple[int, ...], ...]
    qm_formal_charge: int
    spin: int
    mm_atom_indices: tuple[int, ...]
    embedding_charges: tuple[float, ...]
    embedding_atom_indices: tuple[int, ...]
    selected_water_resids: tuple[int, ...]
    embedding_residue_indices: tuple[int, ...]
    anchor_index: int
    system_net_charge: float
    n_atoms: int
    embedding_radius_nm: float
    source_sha256: dict[str, str]

    def __post_init__(self):
        q = set(self.qm_atom_indices)
        qa = set(self.qm_all_topology_indices)
        m = set(self.mm_atom_indices)
        e = set(self.embedding_atom_indices)
        assert len(q) == len(self.qm_atom_indices)
        assert len(qa) == len(self.qm_all_topology_indices)
        assert len(m) == len(self.mm_atom_indices)
        assert len(e) == len(self.embedding_atom_indices)
        assert q <= qa and not qa & m
        assert qa | m == set(range(self.n_atoms))
        assert e <= m
        assert self.anchor_index in q
        assert len(self.qm_elements) == len(q) + len(self.boundary_pairs)
        if self.boundary_pairs:
            assert self.qm_elements[-len(self.boundary_pairs):] == ('H',) * len(self.boundary_pairs)
        assert len(self.m2_per_cb) == len(self.boundary_pairs)
        assert all(qi in q and mi in e for qi, mi in self.boundary_pairs)
        assert len(self.embedding_charges) == len(self.mm_atom_indices)
        assert np.isfinite(self.embedding_charges).all()
        combined_charge = sum(self.embedding_charges) + self.qm_formal_charge
        charge_error = abs(combined_charge - self.system_net_charge)
        assert charge_error < 1e-6, (
            f'Charge conservation failed: QM+MM={combined_charge:.12g} e, '
            f'original topology={self.system_net_charge:.12g} e, '
            f'rounded topology={round(self.system_net_charge)} e; '
            f'error={charge_error:.6g} e exceeds 1e-6 e. '
            'No charges were changed; see validation/partition-charge-diagnostic.json.'
        )
        assert self.embedding_radius_nm > 0

    @property
    def selected_charges(self):
        by_index = dict(zip(self.mm_atom_indices, self.embedding_charges, strict=True))
        return np.array([by_index[i] for i in self.embedding_atom_indices])

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        for key in ('boundary_pairs', 'm2_per_cb'):
            data[key] = tuple(tuple(row) for row in data[key])
        for key in ('qm_atom_indices', 'qm_all_topology_indices', 'qm_elements',
                    'mm_atom_indices', 'embedding_charges', 'embedding_atom_indices',
                    'selected_water_resids', 'embedding_residue_indices'):
            data[key] = tuple(data[key])
        return cls(**data)


def build_partition(prmtop, coordinates, *, frame=0, region='minimal',
                    resnum_offset=-4, embedding_radius_nm=2.0):
    """Freeze a chemical region, or ``zn_only`` for coupling development only.

    The Zn-only fixture has charge +2, no link atoms, and no QM waters. It
    cannot describe ligand charge transfer or enzyme reaction chemistry.
    """
    import mdtraj as md
    import qmmm_partition as qp

    if embedding_radius_nm <= 0:
        raise ValueError('embedding_radius_nm must be positive')
    topology, charges = qp.load_topology_and_charges(prmtop)
    traj = md.load(str(coordinates), top=str(prmtop))
    xyz = traj.xyz[frame].astype(float) * 10
    box = traj.unitcell_vectors[frame].astype(float) * 10
    region_settings = ({'sidechains': {}, 'ligand_resnames': [],
                        'metal_resnames': ['ZN', 'ZN2', 'Zn2+'], 'formal_charge': 2}
                       if region == 'zn_only' else qp.QM_REGIONS[region])
    spec = qp.build_qm_region_spec(topology, charges, region_settings, resnum_offset)
    if region == 'zn_only':
        assert len(spec.qm_static) == 1, 'Zn-only fixture requires exactly one Zn atom'
        assert topology.atom(spec.qm_static[0]).element.symbol == 'Zn'
        # Disable adaptive QM-water promotion explicitly for this test fixture.
        spec = replace(spec, water_records=[])
    fp = qp.partition_frame(topology, charges, spec, xyz, frame,
                            box_vectors_A=box, boundary_charge_method='shift')
    assert fp.n_boundary_virtual_sites == 0
    assert set(fp.mm_site_types) == {'topology'}
    assert fp.n_link_atoms == len(spec.boundary_pairs)
    assert fp.qm_elements[:len(fp.qm_atom_indices)] == [
        topology.atom(i).element.symbol for i in fp.qm_atom_indices]
    if fp.n_link_atoms:
        assert fp.qm_elements[-fp.n_link_atoms:] == ['H'] * fp.n_link_atoms
    assert abs(qp.LINK_H_BOND_A - 1.09) < 1e-12

    centroid = fp.qm_positions_A[:len(fp.qm_atom_indices)].mean(axis=0)
    delta = qp.minimum_image_displacement(xyz[fp.mm_atom_indices] - centroid, box)
    nearby = np.linalg.norm(delta, axis=1) <= embedding_radius_nm * 10
    residues = {topology.atom(i).residue.index
                for i, inside in zip(fp.mm_atom_indices, nearby, strict=True) if inside}
    # Boundary parents and recipients must exist even if a very small radius is requested.
    mandatory = [mi for _, mi in spec.boundary_pairs]
    mandatory += [i for ids in spec.m2_per_cb for i in ids]
    residues.update(topology.atom(i).residue.index for i in mandatory)
    embedding = tuple(i for i in fp.mm_atom_indices if topology.atom(i).residue.index in residues)
    result = FrozenPartition(
        tuple(fp.qm_atom_indices), tuple(fp.qm_all_topology_indices),
        tuple(fp.qm_elements), tuple(map(tuple, spec.boundary_pairs)),
        tuple(map(tuple, spec.m2_per_cb)), int(fp.qm_formal_charge), 0,
        tuple(fp.mm_atom_indices), tuple(map(float, fp.mm_charges)), embedding,
        tuple(fp.selected_water_resids), tuple(sorted(residues)), int(spec.anchor_index),
        float(charges.sum()), topology.n_atoms, float(embedding_radius_nm),
        {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (prmtop, coordinates)},
    )
    audit = {
        'region': region,
        'n_atoms': result.n_atoms,
        'qm_real': len(result.qm_atom_indices),
        'qm_with_links': len(result.qm_elements),
        'qm_virtual_sites': len(result.qm_all_topology_indices) - len(result.qm_atom_indices),
        'boundary_pairs': result.boundary_pairs,
        'selected_water_resids': result.selected_water_resids,
        'mm_all': len(result.mm_atom_indices),
        'embedding_sites': len(embedding),
        'embedding_residues': len(residues),
        'embedding_charge_sum': float(result.selected_charges.sum()),
        'omitted_mm_charge_sum': float(sum(result.embedding_charges) - result.selected_charges.sum()),
        'complete_partition_charge_error': fp.charge_error,
        'original_topology_charge': result.system_net_charge,
        'original_topology_integer_charge_residual': result.system_net_charge - round(result.system_net_charge),
        'boundary_charge_residual': fp.boundary_charge_residual,
        'box_vectors_nm': (box / 10).tolist(),
    }
    return result, audit
