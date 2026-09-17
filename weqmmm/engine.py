"""Standalone OpenMM QM/MM dynamics; no wepy dependency.

Integrator and Context lifetime belong here. Quantum providers own only their
energy/force calculation and segment-local SCF history.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import openmm
from openmm import unit
from .config import QMMMConfig
from .engines.base import QMRegionEngine
from .force import QMMMForce
from .partition import transform_system
from .regions import FrozenPartition


@dataclass
class MDState:
    positions_nm: np.ndarray
    velocities_nm_ps: np.ndarray
    box_vectors_nm: np.ndarray | None
    time_ps: float = 0.
    step_count: int = 0

    def save(self,path):
        values = {'positions_nm':self.positions_nm,'velocities_nm_ps':self.velocities_nm_ps,
                  'time_ps':self.time_ps,'step_count':self.step_count}
        if self.box_vectors_nm is not None:
            values['box_vectors_nm'] = self.box_vectors_nm
        with Path(path).open('wb') as out:
            np.savez(out,**values)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            return cls(data['positions_nm'],data['velocities_nm_ps'],
                       data['box_vectors_nm'] if 'box_vectors_nm' in data else None,
                       float(data['time_ps']),int(data['step_count']))


class QMMMEngine:
    def __init__(self, original_system, partition: FrozenPartition, config: QMMMConfig,
                 *, provider: QMRegionEngine | None = None):
        self.config = config
        self.partition = partition
        if config.qm.charge != partition.qm_formal_charge or config.qm.spin != partition.spin:
            raise ValueError('QM charge/spin settings disagree with frozen partition')
        self.system, self.audit = transform_system(original_system, partition)
        self.periodic = self.system.usesPeriodicBoundaryConditions()
        if provider is None:
            if config.qm.backend == 'gpu4pyscf_molecular':
                from .engines.gpu4pyscf_molecular import GPU4PySCFEngine
                provider = GPU4PySCFEngine(partition.qm_elements, config.qm)
            else:
                from .engines.pyscf_cpu import PySCFEngine
                provider = PySCFEngine(partition.qm_elements, config.qm)
        self.provider = provider
        if tuple(self.provider.elements) != partition.qm_elements:
            raise ValueError('Provider element ordering differs from frozen partition')
        self.callback = QMMMForce(partition,self.provider,periodic=self.periodic,
                                  check_force_sum=config.check_force_sum,
                                  failure_directory=config.failure_directory)
        self.system.addForce(self.callback.create_openmm_force())
        dt = config.timestep_fs * unit.femtosecond
        if config.integrator == 'Verlet':
            self.integrator = openmm.VerletIntegrator(dt)
        else:
            self.integrator = openmm.LangevinMiddleIntegrator(
                config.temperature_K*unit.kelvin, config.friction_per_ps/unit.picosecond, dt)
            self.integrator.setRandomNumberSeed(config.random_seed)
        self.integrator.setConstraintTolerance(1e-8)
        platform = openmm.Platform.getPlatformByName(config.platform)
        self.context = openmm.Context(self.system,self.integrator,platform,config.platform_properties)
        self.audit.update(backend=type(self.provider).__name__,periodic_mm=self.periodic,
                          embedding_sites=len(partition.embedding_atom_indices),config=config.to_dict())

    def _execute(self, fn):
        if self.callback.failure_exception is not None:
            raise RuntimeError('This engine has failed and cannot propagate again') from self.callback.failure_exception
        try:
            return fn()
        except openmm.OpenMMException as exc:
            if self.callback.failure_exception is not None:
                original = self.callback.failure_exception
                original.add_note(f'OpenMM step={self.context.getStepCount()}, '
                                  f'context={self.callback.context_labels}')
                raise original from exc
            raise

    def set_state(self, state: MDState):
        if self.callback.failure_exception is not None:
            raise RuntimeError('Cannot reset a failed engine') from self.callback.failure_exception
        for array in (state.positions_nm,state.velocities_nm_ps):
            assert np.shape(array) == (self.partition.n_atoms,3)
            assert np.isfinite(array).all()
        if self.periodic:
            if state.box_vectors_nm is None or np.shape(state.box_vectors_nm) != (3,3):
                raise ValueError('Periodic dynamics requires three box vectors')
            self.context.setPeriodicBoxVectors(*[openmm.Vec3(*row)*unit.nanometer for row in state.box_vectors_nm])
        elif state.box_vectors_nm is not None:
            raise ValueError('Nonperiodic engine received a periodic state')
        self.context.setPositions(state.positions_nm*unit.nanometer)
        self.context.computeVirtualSites()
        self.context.setVelocities(state.velocities_nm_ps*unit.nanometer/unit.picosecond)
        self.context.setTime(state.time_ps*unit.picosecond)
        self.context.setStepCount(state.step_count)
        # Reset electronic history after installing the incoming nuclear state.
        self.callback.begin_segment()

    def initialize(self, positions_nm, *, box_vectors_nm=None, velocity_seed=1):
        self.set_state(MDState(np.asarray(positions_nm),np.zeros_like(positions_nm),box_vectors_nm))
        # Initial MM constraint projection only. Segment resets do not alter positions.
        self.context.applyConstraints(1e-8)
        self.context.computeVirtualSites()
        self._execute(lambda: self.context.setVelocitiesToTemperature(
            self.config.temperature_K*unit.kelvin, velocity_seed))
        self.context.applyVelocityConstraints(1e-8)

    def step(self, n_steps):
        if not isinstance(n_steps,(int,np.integer)) or n_steps < 0:
            raise ValueError('n_steps must be a nonnegative integer')
        self._execute(lambda: self.integrator.step(int(n_steps)))

    def get_state(self):
        state = self._execute(lambda: self.context.getState(getPositions=True,getVelocities=True))
        return MDState(
            np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
            np.array(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)),
            np.array(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)) if self.periodic else None,
            state.getTime().value_in_unit(unit.picosecond),state.getStepCount())

    def energies(self):
        full = self._execute(lambda: self.context.getState(getEnergy=True))
        qm = self._execute(lambda: self.context.getState(getEnergy=True,groups={31}))
        mm = self._execute(lambda: self.context.getState(getEnergy=True,groups=set(range(31))))
        conv = lambda e: float(e.value_in_unit(unit.kilojoule_per_mole))
        result = {'potential_kj':conv(full.getPotentialEnergy()),'kinetic_kj':conv(full.getKineticEnergy()),
                  'qm_kj':conv(qm.getPotentialEnergy()),'mm_kj':conv(mm.getPotentialEnergy())}
        result['total_kj'] = result['potential_kj']+result['kinetic_kj']
        return result

    def forces(self):
        state = self._execute(lambda: self.context.getState(getForces=True))
        return np.array(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))

    def close(self):
        # Explicit disposal is useful for tests and worker shutdown.
        self.context = None
        self.integrator = None
