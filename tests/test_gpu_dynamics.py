"""GPU trajectory and WE runner gates after the single-point GPU gates pass."""
from dataclasses import replace
import os

import numpy as np
import openmm
import pytest

from weqmmm.config import QMConfig, QMMMConfig
from weqmmm.engine import MDState
from weqmmm.examples import two_water_system
from weqmmm.runner import QMMMRunner, close_cached_engines
from test_nve import check_nve_timestep_convergence
import test_runner as runner_checks

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get('WEQMMM_GPU_TESTS') != '1', reason='Requires explicitly allocated GPU')]

# User-approved 2026-09-16: independent GPU reductions are not bitwise
# reproducible. Zero relative tolerance; CPU and weight comparisons stay exact.
GPU_TRAJECTORY_TOLERANCES = dict(position_atol=1e-12, velocity_atol=1e-10)


@pytest.mark.dynamics
def test_gpu_nve_timestep_convergence(tmp_path):
    check_nve_timestep_convergence(tmp_path, 'gpu4pyscf_molecular', 'gpu-nve')


@pytest.fixture
def gpu_case(tmp_path):
    system, region, pos = two_water_system()
    cfg = QMMMConfig(qm=QMConfig(method='rhf', basis='sto-3g', charge=0,
                                backend='gpu4pyscf_molecular'),
                      integrator='Verlet', platform='Reference', platform_properties={},
                      failure_directory=str(tmp_path))
    runner = QMMMRunner(openmm.XmlSerializer.serialize(system), region, cfg)
    yield system, region, cfg, runner, MDState(pos, np.zeros_like(pos), None)
    close_cached_engines()


def test_gpu_runner_identity(gpu_case):
    runner_checks.check_gate9_weight_and_direct_openmm_identity(
        gpu_case, **GPU_TRAJECTORY_TOLERANCES)


def test_gpu_walker_isolation_and_restart(gpu_case):
    runner_checks.check_alternating_walkers_and_pickle_restart(
        gpu_case, **GPU_TRAJECTORY_TOLERANCES)


def test_gpu_clone_independence(gpu_case):
    runner_checks.check_cloned_walkers_have_independent_langevin_noise(
        gpu_case, position_atol=GPU_TRAJECTORY_TOLERANCES['position_atol'])
