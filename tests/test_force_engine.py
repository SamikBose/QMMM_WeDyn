from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import unit
import pytest

from weqmmm.config import QMConfig,QMMMConfig
from weqmmm.engine import MDState,QMMMEngine
from weqmmm.engines.base import QMResult
from weqmmm.errors import SCFConvergenceError
from weqmmm.examples import two_water_system
from weqmmm.force import QMMMForce
from weqmmm.regions import FrozenPartition
from weqmmm.units import GRAD_AU_TO_FORCE_OMM


class PairProvider:
    """Analytic translation/rotation invariant potential for testing force routing."""
    elements = ('He','H')
    failure = None
    n_calls = 0

    def begin_segment(self,dm0=None):
        pass

    def end_segment(self):
        return None

    def compute(self,q,m,charges,*,box_vectors_nm=None):
        self.n_calls += 1
        delta = q[:,None,:]-m[None,:,:]
        fq = -7*delta.sum(axis=1)
        fm = 7*delta.sum(axis=0)
        return QMResult(float(3.5*np.sum(delta**2)),fq,fm,True,{})


def linked_context(tmp_path, periodic=False, provider=None):
    region = FrozenPartition((2,),(2,),('He','H'),((2,4),),((0,),),0,0,
                             (0,1,3,4),(0.,0.,0.,0.),(0,4),(),(0,),2,0.,5,2.,{})
    callback = QMMMForce(region,provider or PairProvider(),periodic=periodic,failure_directory=tmp_path)
    system = mm.System()
    for _ in range(5):
        system.addParticle(4)
    system.addForce(callback.create_openmm_force())
    box = np.array([[2.,0,0],[-.6,1.9,0],[-.6,-.8,1.7]])
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in box])
    integ = mm.VerletIntegrator(.0005)
    ctx = mm.Context(system,integ,mm.Platform.getPlatformByName('Reference'))
    pos = np.array([[.1,.1,.15],[1.,0,0],[.2,.3,.1],[0,1.,0],[.35,.32,.08]])
    ctx.setPositions(pos*unit.nanometer)
    return ctx,integ,callback,pos,box


@pytest.mark.parametrize('periodic',[False,True])
def test_full_callback_link_scatter_finite_difference(tmp_path,periodic):
    ctx,integ,callback,pos,box = linked_context(tmp_path,periodic)
    get_energy = lambda: ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energy = get_energy()
    forces = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    assert np.all(forces[[1,3]] == 0)
    assert np.linalg.norm(forces[4]) > 0
    numerical = np.zeros_like(pos)
    h = 1e-6
    for i in range(5):
        for j in range(3):
            energies = []
            for sign in (1,-1):
                trial = pos.copy()
                trial[i,j] += sign*h
                ctx.setPositions(trial*unit.nanometer)
                energies.append(get_energy())
            numerical[i,j] = -(energies[0]-energies[1])/(2*h)
    assert np.max(abs(numerical-forces)) < 1e-5
    assert callback.last_force_sum_ratio < 1e-12
    if periodic:
        wrapped = pos.copy()
        wrapped[2] += box[1]
        wrapped[4] -= box[2]
        ctx.setPositions(wrapped*unit.nanometer)
        assert get_energy() == pytest.approx(energy,abs=1e-12)
        new = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        assert np.max(abs(new-forces)) < 1e-12


def config(tmp_path,**kwargs):
    qm = QMConfig(method='rhf',charge=0,basis='sto-3g')
    return QMMMConfig(qm=qm,integrator='Verlet',platform='Reference',
                       platform_properties={},failure_directory=str(tmp_path),**kwargs)


def test_real_provider_energy_groups_cache_and_state(tmp_path):
    system,region,pos = two_water_system()
    engine = QMMMEngine(system,region,config(tmp_path))
    try:
        state = MDState(pos,np.zeros_like(pos),None,1.25,12)
        engine.set_state(state)
        energies = engine.energies()
        assert abs(energies['qm_kj']+energies['mm_kj']-energies['potential_kj'])/abs(energies['potential_kj']) < 1e-9
        assert engine.provider.n_calls == 1
        engine.forces()
        engine.energies()
        assert engine.provider.n_calls == 1
        assert engine.callback.last_force_sum_ratio < 1e-6
        restored = engine.get_state()
        assert restored.time_ps == 1.25 and restored.step_count == 12
        assert np.array_equal(restored.positions_nm,pos)
        engine.step(2)
        final = engine.get_state()
        assert final.time_ps == pytest.approx(1.251)
        assert final.step_count == 14
        assert not np.array_equal(final.positions_nm,pos)
        engine.set_state(state)
        assert engine.provider.end_segment() is None
    finally:
        engine.close()


def test_callback_scf_failure_preserves_type_and_dumps(tmp_path):
    system,region,pos = two_water_system()
    cfg = config(tmp_path)
    cfg = replace(cfg,qm=replace(cfg.qm,max_cycle=1))
    engine = QMMMEngine(system,region,cfg)
    try:
        engine.set_state(MDState(pos,np.zeros_like(pos),None))
        with pytest.raises(SCFConvergenceError,match='SCF failed'):
            engine.energies()
        assert engine.provider.n_calls == 1
        directory = Path(engine.callback.failure_path)
        assert (directory/'qm.xyz').is_file()
        assert np.loadtxt(directory/'embedding.pc').shape == (3,4)
        assert json.loads((directory/'failure.json').read_text())['exception_type'] == 'SCFConvergenceError'
        with pytest.raises(RuntimeError,match='failed'):
            engine.step(1)
        assert engine.provider.n_calls == 1
    finally:
        engine.close()


def test_force_sum_rejects_missing_mm_reaction(tmp_path):
    class BrokenProvider(PairProvider):
        def compute(self,*args,**kwargs):
            result = super().compute(*args,**kwargs)
            return replace(result,mm_forces=np.zeros_like(result.mm_forces))
    ctx,integ,callback,pos,box = linked_context(tmp_path,provider=BrokenProvider())
    with pytest.raises(mm.OpenMMException,match='force sum failed'):
        ctx.getState(getForces=True)
    assert isinstance(callback.failure_exception,AssertionError)


def test_mm_virtual_site_reaction_reaches_real_parents(tmp_path):
    class VirtualSiteProvider(PairProvider):
        elements = ('He',)

        def compute(self,q,m,charges,*,box_vectors_nm=None):
            self.n_calls += 1
            delta = q[0]-m[-1]
            fm = np.zeros_like(m)
            fm[-1] = 7*delta
            return QMResult(float(3.5*np.dot(delta,delta)),np.array([-7*delta]),fm,True,{})
    region = FrozenPartition((0,),(0,),('He',),(),(),0,0,(1,2,3,4),
                             (0.,0.,0.,0.),(1,2,3,4),(),(1,),0,0.,5,2.,{})
    system = mm.System()
    nb = mm.NonbondedForce()
    for i in range(5):
        system.addParticle(0 if i==4 else 4)
        nb.addParticle(0,1,0)
    system.addForce(nb)
    weights = np.array([.5,.25,.25])
    system.setVirtualSite(4,mm.ThreeParticleAverageSite(1,2,3,*weights))
    pos = np.array([[.4,.3,.2],[0,0,0],[.1,0,0],[0,.1,0],[0,0,0]])
    engine = QMMMEngine(system,region,config(tmp_path),provider=VirtualSiteProvider())
    try:
        engine.set_state(MDState(pos,np.zeros_like(pos),None))
        forces = engine.forces()
        expected = 7*(pos[0]-weights@pos[1:4])
        assert np.allclose(forces[0],-expected)
        assert np.allclose(forces[1:4],weights[:,None]*expected)
        assert np.linalg.norm(forces[:4].sum(0)) < 1e-12
        for atom in range(4):
            h = 1e-6
            energies = []
            for sign in (1,-1):
                trial = pos.copy()
                trial[atom,0] += sign*h
                engine.set_state(MDState(trial,np.zeros_like(trial),None))
                energies.append(engine.energies()['potential_kj'])
            assert -(energies[0]-energies[1])/(2*h) == pytest.approx(forces[atom,0],abs=1e-8)
    finally:
        engine.close()


def test_short_dft_dynamics(tmp_path):
    system,region,pos = two_water_system()
    cfg = config(tmp_path)
    cfg = replace(cfg,qm=replace(cfg.qm,method='rks'))
    engine = QMMMEngine(system,region,cfg)
    try:
        engine.set_state(MDState(pos,np.zeros_like(pos),None))
        initial = engine.energies()['total_kj']
        engine.step(10)
        final = engine.energies()['total_kj']
        assert abs(final-initial) < .5
        assert np.isfinite(engine.get_state().positions_nm).all()
        assert engine.callback.last_force_sum_ratio < 1e-6
        assert engine.callback.last_result.extras['grid_response']
    finally:
        engine.close()
