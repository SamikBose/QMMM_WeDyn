"""OpenMM PythonForce adapter with link projection and complete MM reaction forces.

Nearest-image molecular embedding is the explicitly chosen prototype model.
It does not implement periodic QM electrostatics. Fixed selections prevent
selection changes, but distant charge image changes can still be discontinuous.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import openmm
from openmm import unit
from qmmm_partition import minimum_image_displacement

from .engines.base import QMRegionEngine
from .errors import SCFConvergenceError
from .link import place_link_atom, project_link_force
from .regions import FrozenPartition


class QMMMForce:
    def __init__(self, partition: FrozenPartition, provider: QMRegionEngine,
                 *, periodic=False, check_force_sum=True, failure_directory='runs/failures'):
        self.partition = partition
        self.provider = provider
        self.periodic = periodic
        self.check_force_sum = check_force_sum
        self.failure_directory = Path(failure_directory)
        self.particles = partition.qm_all_topology_indices + partition.embedding_atom_indices
        self.index_of = {atom: row for row, atom in enumerate(self.particles)}
        self.qm_rows = np.array([self.index_of[i] for i in partition.qm_atom_indices], dtype=int)
        self.n_qm_topology = len(partition.qm_all_topology_indices)
        self.charges = partition.selected_charges
        self.geometry = None
        self.failure_exception = None
        self.failure_path = None
        self.last_result = None
        self.last_force_sum_ratio = None
        self.context_labels = {}

    def begin_segment(self):
        if self.failure_exception is not None:
            raise RuntimeError('This force previously failed; construct a new engine') from self.failure_exception
        self.provider.begin_segment(dm0=None)
        self.last_result = None

    def __call__(self, state):
        if self.failure_exception is not None:
            raise self.failure_exception
        try:
            return self._evaluate(state)
        except Exception as exc:
            # Preserve the original exception before OpenMM erases its type.
            # This handler writes context, re-raises, and never resumes the run.
            self.failure_exception = exc
            self.failure_path = self._dump_failure(exc)
            exc.add_note(f'QM/MM failure diagnostics: {self.failure_path}')
            print(f'QM/MM callback failed: {type(exc).__name__}: {exc}\n'
                  f'Diagnostics: {self.failure_path}', flush=True)
            raise

    def _evaluate(self, state):
        positions = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        assert positions.shape == (len(self.particles), 3)
        assert np.isfinite(positions).all()
        box = np.asarray(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)) if self.periodic else None
        self.geometry = {'raw_positions_nm': positions.copy(), 'box_vectors_nm': box}
        # A fixed real QM anchor works even when OpenMM wraps QM atoms separately.
        anchor = positions[self.index_of[self.partition.anchor_index]]
        qpos = anchor + minimum_image_displacement(positions[self.qm_rows]-anchor, box)
        centroid = qpos.mean(axis=0)
        mpos = centroid + minimum_image_displacement(positions[self.n_qm_topology:]-centroid, box)
        coherent = positions.copy()
        coherent[self.qm_rows] = qpos
        coherent[self.n_qm_topology:] = mpos
        links = []
        for qi, mi in self.partition.boundary_pairs:
            qr, mr = self.index_of[qi], self.index_of[mi]
            coherent[mr] = coherent[qr] + minimum_image_displacement(positions[mr]-positions[qr], box)
            links.append(place_link_atom(coherent[qr], coherent[mr]))
        qm_with_links = np.vstack([qpos, np.asarray(links).reshape(-1,3)])
        mpos = coherent[self.n_qm_topology:]
        self.geometry.update(qm_positions_nm=qm_with_links.copy(), mm_positions_nm=mpos.copy())
        result = self.provider.compute(qm_with_links, mpos, self.charges, box_vectors_nm=box)
        if not result.converged:
            raise SCFConvergenceError('QM provider returned an unconverged result')
        assert result.qm_forces.shape == qm_with_links.shape
        assert result.mm_forces.shape == mpos.shape
        assert np.isfinite(result.energy_kj)
        assert np.isfinite(result.qm_forces).all() and np.isfinite(result.mm_forces).all()
        forces = np.zeros_like(positions)
        nreal = len(self.qm_rows)
        forces[self.qm_rows] = result.qm_forces[:nreal]
        forces[self.n_qm_topology:] = result.mm_forces
        for row, (qi, mi) in enumerate(self.partition.boundary_pairs, start=nreal):
            qr, mr = self.index_of[qi], self.index_of[mi]
            fq, fm = project_link_force(result.qm_forces[row], coherent[qr], coherent[mr])
            forces[qr] += fq
            forces[mr] += fm
        scale = float(np.linalg.norm(forces, axis=1).max())
        total = float(np.linalg.norm(forces.sum(axis=0)))
        self.last_force_sum_ratio = total / scale if scale else 0.
        if self.check_force_sum:
            assert self.last_force_sum_ratio < 1e-6, (
                f'QM/MM force sum failed: |sum(F)|/max|F|={self.last_force_sum_ratio:.6g}; '
                'check MM reaction forces, link projection, units, and DFT grid derivatives')
        self.last_result = result
        return result.energy_kj, forces

    def _dump_failure(self, exc):
        directory = self.failure_directory / f'failure-{time.time_ns()}'
        directory.mkdir(parents=True)
        metadata = {'exception_type': type(exc).__name__, 'message': str(exc),
                    'provider_calls': self.provider.n_calls,
                    'context': self.context_labels,
                    'last_force_sum_ratio': self.last_force_sum_ratio,
                    'last_converged_energy_kj': None if self.last_result is None else self.last_result.energy_kj}
        if self.geometry is not None:
            np.savez(directory/'coordinates.npz', **{k:v for k,v in self.geometry.items() if v is not None})
            if 'qm_positions_nm' in self.geometry:
                q = self.geometry['qm_positions_nm']
                m = self.geometry['mm_positions_nm']
                text = f'{len(q)}\nQM atoms including link H; coordinates in Angstrom\n'
                text += ''.join(f'{el} {x:.12f} {y:.12f} {z:.12f}\n'
                                for el,(x,y,z) in zip(self.partition.qm_elements,q*10,strict=True))
                (directory/'qm.xyz').write_text(text)
                np.savetxt(directory/'embedding.pc', np.column_stack((self.charges,m*10)),
                           header='charge_e x_A y_A z_A; frozen embedding order')
                metadata['qm_centroid_nm'] = q[:len(self.qm_rows)].mean(0).tolist()
                nonzero = m[self.charges != 0]
                metadata['minimum_qm_nonzero_charge_distance_nm'] = (
                    float(np.linalg.norm(q[:,None,:]-nonzero[None,:,:],axis=2).min()) if len(nonzero) else None)
        (directory/'failure.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return str(directory)

    def create_openmm_force(self, group=31):
        force = openmm.PythonForce(self)
        force.setParticles(list(self.particles))
        force.setForceGroup(group)
        force.setUsesPeriodicBoundaryConditions(self.periodic)
        return force
