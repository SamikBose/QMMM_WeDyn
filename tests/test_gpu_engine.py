"""Real CUDA gates, explicitly enabled in the allocated validation job.

No mocking or CPU fallback: when enabled, an unavailable GPU is a failure.
"""
from dataclasses import replace
import os

import numpy as np
import pytest

from weqmmm.config import QMConfig, QMMMConfig
from weqmmm.engine import MDState, QMMMEngine
from weqmmm.examples import two_water_system
from weqmmm.errors import SCFConvergenceError
from weqmmm.engines.pyscf_cpu import PySCFEngine
from weqmmm.units import HARTREE_TO_KJ_MOL, GRAD_AU_TO_FORCE_OMM
from test_pyscf_engine import check_provider_gates, QPOS, MPOS, CHARGES

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('WEQMMM_GPU_TESTS') != '1', reason='Requires explicitly allocated GPU')]


@pytest.mark.parametrize('method,spin', [('rhf',0), ('uhf',1), ('rks',0), ('uks',1)])
def test_gpu_physics_gates(method, spin):
    from weqmmm.engines.gpu4pyscf_molecular import GPU4PySCFEngine
    check_provider_gates(GPU4PySCFEngine, method, spin)


def test_gpu_cpu_agreement_cache_reset_and_fatal_failure():
    from weqmmm.engines.gpu4pyscf_molecular import GPU4PySCFEngine
    cfg = QMConfig(basis='def2-svp', charge=0, conv_tol=1e-12)
    cpu = PySCFEngine(('O','H','H'), cfg)
    gpu = GPU4PySCFEngine(('O','H','H'), replace(cfg, backend='gpu4pyscf_molecular'))
    reference = cpu.compute(QPOS, MPOS, CHARGES)
    result = gpu.compute(QPOS, MPOS, CHARGES)
    assert abs(result.energy_kj-reference.energy_kj)/HARTREE_TO_KJ_MOL < 1e-8
    for field in ('qm_forces', 'mm_forces'):
        assert np.max(abs(getattr(result,field)-getattr(reference,field)))/GRAD_AU_TO_FORCE_OMM < 1e-6
    assert gpu.compute(QPOS.copy(), MPOS.copy(), CHARGES.copy()) is result
    assert gpu.n_calls == 1 and gpu.cache_hits == 1
    gpu.begin_segment()
    assert gpu.end_segment() is None
    gpu.compute(QPOS, MPOS, CHARGES)
    assert gpu.n_calls == 2
    failed = GPU4PySCFEngine(('O','H','H'), replace(cfg, max_cycle=1))
    with pytest.raises(SCFConvergenceError, match='SCF failed'):
        failed.compute(QPOS, MPOS, CHARGES)
    with pytest.raises(SCFConvergenceError):
        failed.compute(QPOS, MPOS, CHARGES)
    assert failed.n_calls == 1


def test_gpu_openmm_energy_decomposition_and_short_md(tmp_path):
    system, region, pos = two_water_system()
    cfg = QMMMConfig(qm=QMConfig(basis='sto-3g', charge=0, backend='gpu4pyscf_molecular'),
                     integrator='Verlet', platform='Reference', platform_properties={},
                     failure_directory=str(tmp_path))
    engine = QMMMEngine(system, region, cfg)
    try:
        engine.set_state(MDState(pos, np.zeros_like(pos), None))
        for _ in range(3):
            energies = engine.energies()
            assert abs(energies['potential_kj']-energies['qm_kj']-energies['mm_kj'])/max(1.,abs(energies['potential_kj'])) < 1e-9
            assert engine.callback.last_force_sum_ratio < 1e-6
            engine.step(1)
        assert np.isfinite(engine.get_state().positions_nm).all()
    finally:
        engine.close()
