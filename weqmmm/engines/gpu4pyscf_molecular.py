"""GPU molecular embedding with the same finite-charge Hamiltonian as CPU.

Uses direct SCF (no implicit density fitting) and full MM reaction forces.
Gaussian charges are unsupported by the upstream molecular nuclear terms.
Run in a process with one visible allocated GPU; spawned WE workers must receive
their GPU assignment before importing GPU4PySCF. No GPU-to-CPU fallback is used.
"""
import numpy as np

from .pyscf_cpu import PySCFEngine
from ..units import BOHR_TO_NM


class GPU4PySCFEngine(PySCFEngine):
    backend_name = 'gpu4pyscf_molecular'

    def __init__(self, elements, config):
        if config.mm_charge_radii_bohr is not None:
            raise ValueError('GPU molecular embedding supports point charges only')
        import cupy
        import gpu4pyscf
        from gpu4pyscf.qmmm import itrf

        self._cupy = cupy
        self._qmmm = itrf
        # Fail at setup on an inaccessible CUDA runtime/device.
        if cupy.cuda.runtime.getDeviceCount() != 1:
            raise ValueError('Expose exactly one allocated GPU with CUDA_VISIBLE_DEVICES')
        self.gpu4pyscf_version = gpu4pyscf.__version__
        super().__init__(elements, config)

    def _make_mean_field(self, mol, mpos, charges):
        # Transfer the undecorated object so no CPU embedding method remains in
        # the GPU method resolution order. Keep XC and grid settings identical.
        base = super()._make_mean_field(mol, np.empty((0, 3)), np.empty(0)).to_gpu()
        return self._qmmm.mm_charge(base, mpos / BOHR_TO_NM, charges,
                                    unit='Bohr') if len(charges) else base

    def _to_numpy(self, array):
        return self._cupy.asnumpy(array)

    def _initial_density(self):
        return None if self._dm is None else self._cupy.asarray(self._dm)
