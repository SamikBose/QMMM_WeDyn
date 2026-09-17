"""Finite molecular electrostatic embedding with CPU PySCF; no OpenMM imports.

The box is part of the backend interface/cache key, but does not create periodic
QM electrostatics. The coupling layer supplies explicitly chosen charge images.
There are no SCF retry ladders, changed guesses on failure, or failed-walker skips.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
from pyscf import gto, scf, qmmm

from .base import QMResult
from ..config import QMConfig
from ..errors import SCFConvergenceError
from ..units import BOHR_TO_NM, HARTREE_TO_KJ_MOL, GRAD_AU_TO_FORCE_OMM


class PySCFEngine:
    backend_name = 'pyscf_cpu_molecular'

    def __init__(self, elements, config: QMConfig):
        self.elements = tuple(elements)
        self.config = config
        self.n_calls = 0
        self.cache_hits = 0
        self.last_converged_energy_kj = None
        self.begin_segment()

    def begin_segment(self, dm0=None):
        self._dm = None if dm0 is None else np.array(dm0, copy=True)
        self._cache_key = None
        self._cache_value = None
        self.failure = None

    def end_segment(self):
        return None if self._dm is None else self._dm.copy()

    def _make_mean_field(self, mol, mpos, charges):
        cfg = self.config
        base = {'rhf': scf.RHF, 'uhf': scf.UHF, 'rks': scf.RKS, 'uks': scf.UKS}[cfg.method](mol)
        if cfg.method in {'rks', 'uks'}:
            base.xc = cfg.xc
            base.grids.level = cfg.grid_level
        radii = None if cfg.mm_charge_radii_bohr is None else np.full(len(charges), cfg.mm_charge_radii_bohr)
        return qmmm.mm_charge(base, mpos / BOHR_TO_NM, charges, radii=radii, unit='Bohr') if len(charges) else base

    def _to_numpy(self, array):
        return np.asarray(array)

    def _initial_density(self):
        return self._dm

    def _mm_gradient(self, grad, dm_total):
        """Return the embedding gradient on the supplied MM sites."""
        return grad.grad_hcore_mm(dm_total) + grad.grad_nuc_mm()

    def compute(self, qm_positions_nm, mm_positions_nm, mm_charges, *, box_vectors_nm=None):
        if self.failure is not None:
            raise SCFConvergenceError(self.failure)
        qpos = np.asarray(qm_positions_nm, dtype=float)
        mpos = np.asarray(mm_positions_nm, dtype=float)
        charges = np.asarray(mm_charges, dtype=float)
        assert qpos.shape == (len(self.elements), 3)
        assert charges.ndim == 1 and mpos.shape == (len(charges), 3)
        assert all(np.isfinite(a).all() for a in (qpos, mpos, charges))
        digest = hashlib.sha256()
        for array in (qpos, mpos, charges, np.asarray(box_vectors_nm if box_vectors_nm is not None else [])):
            digest.update(str(array.shape).encode())
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        key = digest.digest()
        if key == self._cache_key:
            self.cache_hits += 1
            return self._cache_value
        started = time.perf_counter()
        cfg = self.config
        mol = gto.M(atom=list(zip(self.elements, qpos / BOHR_TO_NM)),
                    unit='Bohr', basis=cfg.basis, charge=cfg.charge,
                    spin=cfg.spin, verbose=cfg.verbose)
        mf = self._make_mean_field(mol, mpos, charges)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = cfg.conv_tol, cfg.conv_tol_grad, cfg.max_cycle
        self.n_calls += 1
        energy = float(mf.kernel(dm0=self._initial_density()))
        scf_seconds = time.perf_counter()-started
        if not mf.converged or not np.isfinite(energy):
            self.failure = (f'PySCF SCF failed: backend={self.backend_name} method={cfg.method} basis={cfg.basis} '
                            f'charge={cfg.charge} spin={cfg.spin} call={self.n_calls} '
                            f'cycles={mf.cycles} energy_Ha={energy!r}')
            print(self.failure, flush=True)
            raise SCFConvergenceError(self.failure)
        grad = mf.nuc_grad_method()
        if cfg.method in {'rks', 'uks'}:
            # Differentiate the atom-centered integration grid as well as the
            # density/basis, so forces match the finite-grid energy used by MD.
            grad.grid_response = True
        g_qm = self._to_numpy(grad.kernel())
        dm = mf.make_rdm1()
        dm_total = dm.sum(axis=0) if dm.ndim == 3 else dm
        g_mm = self._to_numpy(self._mm_gradient(grad, dm_total)) if len(charges) else np.empty((0,3))
        f_qm = -g_qm * GRAD_AU_TO_FORCE_OMM
        f_mm = -g_mm * GRAD_AU_TO_FORCE_OMM
        if not np.isfinite(f_qm).all() or not np.isfinite(f_mm).all():
            self.failure = f'Nonfinite PySCF gradient at call {self.n_calls}'
            raise SCFConvergenceError(self.failure)
        f_qm.setflags(write=False)
        f_mm.setflags(write=False)
        result = QMResult(float(energy * HARTREE_TO_KJ_MOL), f_qm, f_mm, True,
                          {'energy_hartree': float(energy), 'scf_cycles': mf.cycles,
                           'wall_seconds': time.perf_counter()-started,
                           'scf_seconds': scf_seconds,
                           'gradient_seconds': time.perf_counter()-started-scf_seconds,
                           'n_basis': mol.nao_nr(), 'backend': self.backend_name,
                           'grid_level': cfg.grid_level if cfg.method in {'rks', 'uks'} else None,
                           'grid_response': cfg.method in {'rks', 'uks'}})
        self._dm = self._to_numpy(dm).copy()
        self.last_converged_energy_kj = result.energy_kj
        self._cache_key, self._cache_value = key, result
        return result
