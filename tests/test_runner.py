from dataclasses import replace
import pickle
import numpy as np
import openmm
import pytest
from wepy.walker import Walker
from weqmmm.config import QMConfig,QMMMConfig
from weqmmm.engine import MDState,QMMMEngine
from weqmmm.examples import two_water_system
from weqmmm.runner import QMMMRunner,walker_state,close_cached_engines,allocated_gpu_for_worker


def test_scheduler_gpu_tokens_remain_distinct():
    assert allocated_gpu_for_worker(0, '3,7') == '3'
    assert allocated_gpu_for_worker(1, '3,7') == '7'
    assert allocated_gpu_for_worker(1, 'GPU-aaa,GPU-bbb') == 'GPU-bbb'
    with pytest.raises(ValueError, match='one allocated'):
        allocated_gpu_for_worker(1, '3')
    with pytest.raises(ValueError, match='one allocated'):
        allocated_gpu_for_worker(0, '')


@pytest.fixture
def case(tmp_path):
    system,region,pos = two_water_system()
    cfg = QMMMConfig(qm=QMConfig(method='rhf',basis='sto-3g',charge=0),
                      integrator='Verlet',platform='Reference',platform_properties={},
                      failure_directory=str(tmp_path))
    runner = QMMMRunner(openmm.XmlSerializer.serialize(system),region,cfg)
    state = MDState(pos,np.zeros_like(pos),None)
    yield system,region,cfg,runner,state
    close_cached_engines()


def test_gate9_weight_and_direct_openmm_identity(case):
    check_gate9_weight_and_direct_openmm_identity(case)


def compare_trajectory(actual, expected, atol):
    if atol == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


def check_gate9_weight_and_direct_openmm_identity(case, *, position_atol=0, velocity_atol=0):
    system,region,cfg,runner,state = case
    walker = Walker(walker_state(state),.123456789012345)
    propagated = runner.run_segment(walker,5)
    engine = QMMMEngine(system,region,cfg)
    try:
        engine.set_state(state)
        engine.step(5)
        direct = engine.get_state()
        compare_trajectory(propagated.state['positions'], direct.positions_nm, position_atol)
        compare_trajectory(propagated.state['velocities'], direct.velocities_nm_ps, velocity_atol)
        assert propagated.weight.hex() == walker.weight.hex()
        assert np.array_equal(walker.state['positions'],state.positions_nm)
        assert propagated.state['time'] == direct.time_ps
        assert propagated.state['step_count'] == 5
    finally:
        engine.close()


def test_alternating_walkers_and_pickle_restart(case):
    check_alternating_walkers_and_pickle_restart(case)


def check_alternating_walkers_and_pickle_restart(case, *, position_atol=0, velocity_atol=0):
    system,region,cfg,runner,state = case
    a = Walker(walker_state(state),.5)
    moved = state.positions_nm.copy()
    moved[3:] += np.array([.025,.01,0])
    b = Walker(walker_state(MDState(moved,state.velocities_nm_ps,None)),.5)
    first = runner.run_segment(a,3)
    runner.run_segment(b,3)
    second = runner.run_segment(a,3)
    compare_trajectory(first.state['positions'], second.state['positions'], position_atol)
    compare_trajectory(first.state['velocities'], second.state['velocities'], velocity_atol)
    restored_runner,restored_walker = pickle.loads(pickle.dumps((runner,first)))
    close_cached_engines()
    restarted = restored_runner.run_segment(restored_walker,2)
    # CPU agrees exactly; GPU uses only the user-approved absolute tolerances.
    close_cached_engines()
    expected = runner.run_segment(first,2)
    compare_trajectory(expected.state['positions'], restarted.state['positions'], position_atol)
    compare_trajectory(expected.state['velocities'], restarted.state['velocities'], velocity_atol)
    assert restarted.state['time'] == pytest.approx(.0025)
    assert restarted.weight.hex() == first.weight.hex()


def test_cloned_walkers_have_independent_langevin_noise(case):
    check_cloned_walkers_have_independent_langevin_noise(case)


def check_cloned_walkers_have_independent_langevin_noise(case, *, position_atol=0):
    system,region,cfg,runner,state = case
    cfg = replace(cfg,integrator='LangevinMiddle',random_seed=37)
    runner = QMMMRunner(openmm.XmlSerializer.serialize(system),region,cfg)
    parent = Walker(walker_state(state),1.)
    a,b = parent.clone()
    fa,fb = runner.run_segment(a,4),runner.run_segment(b,4)
    assert fa.weight == fb.weight == .5
    # GPU roundoff by itself is not evidence of independent thermal noise.
    assert np.max(abs(fa.state['positions']-fb.state['positions'])) > position_atol
    assert np.array_equal(parent.state['positions'],state.positions_nm)
