from dataclasses import replace
import numpy as np
import pytest
from pyscf import gto, scf
from weqmmm.config import QMConfig
from weqmmm.engines.pyscf_cpu import PySCFEngine
from weqmmm.errors import SCFConvergenceError
from weqmmm.units import BOHR_TO_NM, HARTREE_TO_KJ_MOL, GRAD_AU_TO_FORCE_OMM

QPOS = np.array([[0,0,.0117], [0,.0757,-.0468], [0,-.0757,-.0468]])
MPOS = np.array([[.29,.01,.04], [.36,.06,-.02], [-.26,-.05,.09]])
CHARGES = np.array([-.834,.417,.417])


@pytest.mark.parametrize('method,spin', [('rhf',0), ('uhf',1), ('rks',0), ('uks',1)])
def test_gates_1_2_3_5_7(method, spin):
    check_provider_gates(PySCFEngine, method, spin)


def check_provider_gates(provider_class, method, spin):
    elements = ('O','H') if spin else ('O','H','H')
    pos = QPOS[:len(elements)]
    config = QMConfig(method=method, charge=0, spin=spin, basis='sto-3g',
                      conv_tol=1e-12, conv_tol_grad=1e-7, grid_level=5)
    provider = provider_class(elements, config)
    result = provider.compute(pos, MPOS, np.zeros(3))
    mol = gto.M(atom=list(zip(elements, pos/BOHR_TO_NM)),unit='Bohr',
                basis='sto-3g',spin=spin,verbose=0)
    plain = {'rhf': scf.RHF,'uhf': scf.UHF,'rks': scf.RKS,'uks': scf.UKS}[method](mol)
    if method in {'rks','uks'}:
        plain.xc = config.xc
        plain.grids.level = config.grid_level
    plain.conv_tol, plain.conv_tol_grad = config.conv_tol, config.conv_tol_grad
    expected = plain.kernel()
    assert plain.converged
    assert abs(result.energy_kj/HARTREE_TO_KJ_MOL-expected) < 1e-10
    provider.begin_segment()
    result = provider.compute(pos, MPOS, CHARGES)
    h = 2e-4 * BOHR_TO_NM
    for positions, forces, is_qm in ((pos,result.qm_forces,True), (MPOS,result.mm_forces,False)):
        numerical = np.zeros_like(positions)
        for i in range(len(positions)):
            for j in range(3):
                energies = []
                for sign in (1,-1):
                    trial = positions.copy()
                    trial[i,j] += sign*h
                    provider.begin_segment()
                    energies.append(provider.compute(trial if is_qm else pos,
                                                     MPOS if is_qm else trial, CHARGES).energy_kj)
                numerical[i,j] = -(energies[0]-energies[1])/(2*h)
        error_au = np.max(abs(numerical-forces))/GRAD_AU_TO_FORCE_OMM
        assert error_au < 1e-6, (method, is_qm, error_au)
    scale = max(np.linalg.norm(result.qm_forces,axis=1).max(),np.linalg.norm(result.mm_forces,axis=1).max())
    assert np.linalg.norm(result.qm_forces.sum(0)+result.mm_forces.sum(0))/scale < 1e-6
    # Negative control: the same force-sum check must reject missing MM reaction forces.
    assert np.linalg.norm(result.qm_forces.sum(0))/scale > 1e-4
    rot, _ = np.linalg.qr(np.array([[.7,.2,.9],[-.3,.8,.1],[.6,-.5,.4]]))
    provider.begin_segment()
    rotated = provider.compute(pos@rot,MPOS@rot,CHARGES)
    assert abs(rotated.energy_kj-result.energy_kj)/abs(result.energy_kj) < 1e-9


def test_gaussian_embedding_mm_gradient():
    provider = PySCFEngine(('O','H','H'), QMConfig(method='rhf',charge=0,basis='sto-3g',
                                                 conv_tol=1e-12,mm_charge_radii_bohr=2.0))
    result = provider.compute(QPOS,MPOS,CHARGES)
    h = 2e-4*BOHR_TO_NM
    energies = []
    for sign in (1,-1):
        pos = MPOS.copy()
        pos[0,0] += sign*h
        provider.begin_segment()
        energies.append(provider.compute(QPOS,pos,CHARGES).energy_kj)
    fd = -(energies[0]-energies[1])/(2*h)
    assert abs(fd-result.mm_forces[0,0])/GRAD_AU_TO_FORCE_OMM < 1e-6


def test_cache_and_segment_reset():
    provider = PySCFEngine(('O','H','H'), QMConfig(method='rhf',charge=0,basis='sto-3g'))
    a = provider.compute(QPOS,MPOS,CHARGES)
    assert provider.compute(QPOS.copy(),MPOS.copy(),CHARGES.copy()) is a
    assert provider.n_calls == 1 and provider.cache_hits == 1
    altered = QPOS.copy()
    altered[0,0] = 1e-10
    provider.compute(altered,MPOS,CHARGES)
    assert provider.n_calls == 2
    dm = provider.end_segment()
    dm[:] = 0
    assert np.any(provider.end_segment())
    provider.begin_segment()
    assert provider.end_segment() is None
    provider.compute(QPOS,MPOS,CHARGES)
    assert provider.n_calls == 3


def test_scf_failure_is_fatal_without_retry():
    provider = PySCFEngine(('O','H','H'), QMConfig(method='rhf',charge=0,basis='sto-3g',max_cycle=1))
    with pytest.raises(SCFConvergenceError, match='SCF failed'):
        provider.compute(QPOS,MPOS,CHARGES)
    assert provider.n_calls == 1
    with pytest.raises(SCFConvergenceError):
        provider.compute(QPOS,MPOS,CHARGES)
    assert provider.n_calls == 1
