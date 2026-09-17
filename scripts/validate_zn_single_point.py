"""Check def2-SVP Zn forces at the saved enzyme geometry before propagation.

Checks all three Zn force components and all three components of the nearest
nonzero MM charge, plus total-force balance and rotational energy invariance.
This is a sampled single-point check, not a complete enzyme MD validation.
"""
import json
from pathlib import Path
import time

import numpy as np

from weqmmm.config import QMConfig
from weqmmm.engines.pyscf_cpu import PySCFEngine
from weqmmm.regions import FrozenPartition
from weqmmm.units import BOHR_TO_NM, GRAD_AU_TO_FORCE_OMM


def main():
    region = FrozenPartition.load('runs/hca2_azm/zn-only/partition.json')
    geometry = Path('runs/hca2_azm/zn-only/md-smoke/failures/failure-1789547160153550644/coordinates.npz')
    with np.load(geometry, allow_pickle=False) as data:
        qpos, mpos, box = data['qm_positions_nm'], data['mm_positions_nm'], data['box_vectors_nm']
    charges = region.selected_charges
    cfg = QMConfig(method='rks', xc='b3lyp', basis='def2-svp', charge=2, spin=0,
                   conv_tol=1e-10, conv_tol_grad=1e-7, grid_level=5, max_cycle=50)
    provider = PySCFEngine(('Zn',), cfg)
    started = time.perf_counter()
    reference = provider.compute(qpos, mpos, charges, box_vectors_nm=box)
    nonzero = np.flatnonzero(charges)
    nearest = int(nonzero[np.argmin(np.linalg.norm(mpos[nonzero]-qpos[0], axis=1))])
    h = 2e-4 * BOHR_TO_NM
    errors = {}
    for label, positions, index, analytic in (
            ('qm', qpos, 0, reference.qm_forces[0]),
            ('mm_nearest', mpos, nearest, reference.mm_forces[nearest])):
        numerical = []
        for axis in range(3):
            energies = []
            for sign in (1, -1):
                trial = positions.copy()
                trial[index, axis] += sign*h
                provider.begin_segment()
                result = provider.compute(trial if label == 'qm' else qpos,
                    mpos if label == 'qm' else trial, charges, box_vectors_nm=box)
                energies.append(result.energy_kj)
            numerical.append(-(energies[0]-energies[1])/(2*h))
        errors[label] = float(np.max(abs(np.asarray(numerical)-analytic))/GRAD_AU_TO_FORCE_OMM)
        print(f'{label} max force error: {errors[label]:.8g} Ha/Bohr', flush=True)
        assert errors[label] < 1e-6, (label, errors[label])

    forces = np.vstack((reference.qm_forces, reference.mm_forces))
    ratio = float(np.linalg.norm(forces.sum(axis=0))/np.linalg.norm(forces, axis=1).max())
    assert ratio < 1e-6, ratio
    rotation, _ = np.linalg.qr(np.array([[.7,.2,.9],[-.3,.8,.1],[.6,-.5,.4]]))
    provider.begin_segment()
    rotated = provider.compute(qpos @ rotation, mpos @ rotation, charges,
                               box_vectors_nm=box @ rotation)
    rotation_error = abs(rotated.energy_kj-reference.energy_kj)/abs(reference.energy_kj)
    assert rotation_error < 1e-9, rotation_error
    report = {'passed': True, 'basis': 'def2-svp', 'xc': 'b3lyp',
              'geometry': str(geometry), 'force_error_hartree_per_bohr': errors,
              'mm_tested_topology_index': region.embedding_atom_indices[nearest],
              'finite_difference_step_bohr': h/BOHR_TO_NM,
              'force_sum_ratio': ratio, 'rotation_energy_relative_error': rotation_error,
              'scf_calls': provider.n_calls, 'wall_seconds': time.perf_counter()-started}
    Path('validation/zn-def2svp-force-checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
