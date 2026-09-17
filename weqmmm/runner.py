"""Thin wepy adapter; every live QM/MM engine is owned by its worker process.

No resampling or weight changes occur in run_segment. Stochastic integrators
advance their worker-local random stream; setting a seed is not used to rewind
an existing Context. Schedule-independent stochastic replay is not promised.
"""
from dataclasses import asdict,replace
import hashlib
import json
import os
import numpy as np
import openmm
from wepy.runners.runner import Runner
from wepy.walker import WalkerState
from wepy.work_mapper.mapper import Worker
from .engine import MDState,QMMMEngine

_ENGINES = {}


def allocated_gpu_for_worker(worker_idx, visible_devices):
    """Select one scheduler-visible device token without inventing device IDs."""
    devices = [token.strip() for token in visible_devices.split(',') if token.strip()]
    if not 0 <= worker_idx < len(devices):
        raise ValueError('GPU WE requires one allocated visible GPU per worker')
    return devices[worker_idx]


class GPUWorker(Worker):
    """Bind each spawned worker before it imports/initializes GPU4PySCF."""
    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = allocated_gpu_for_worker(
            self.worker_idx, os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        super().run()


def close_cached_engines():
    for key in list(_ENGINES):
        if key[0] == os.getpid():
            _ENGINES.pop(key).close()


def walker_state(state: MDState, **extras):
    return WalkerState(positions=np.array(state.positions_nm,copy=True),
                       velocities=np.array(state.velocities_nm_ps,copy=True),
                       box_vectors=None if state.box_vectors_nm is None else np.array(state.box_vectors_nm,copy=True),
                       time=state.time_ps,step_count=state.step_count,**extras)


class QMMMRunner(Runner):
    def __init__(self, system_xml, partition, config):
        self.system_xml = system_xml
        self.partition = partition
        self.config = config
        serialized = system_xml+json.dumps(asdict(partition),sort_keys=True)+config.to_json()
        self.cache_key = hashlib.sha256(serialized.encode()).hexdigest()

    def _engine(self):
        key = (os.getpid(),self.cache_key)
        if key not in _ENGINES:
            cfg = self.config
            if cfg.integrator == 'LangevinMiddle' and cfg.random_seed:
                # Distinct workers must not start identical noise streams. This
                # is scheduling-dependent statistical WE, not replay mode.
                seed = int(np.random.SeedSequence([cfg.random_seed,os.getpid()]).generate_state(1)[0])
                cfg = replace(cfg,random_seed=1+seed%2147483646)
            _ENGINES[key] = QMMMEngine(openmm.XmlSerializer.deserialize(self.system_xml),
                                       self.partition,cfg)
        return _ENGINES[key]

    def run_segment(self, walker, segment_length, **kwargs):
        engine = self._engine()
        fields = walker.state.dict()
        incoming_time = float(fields.get('time',0.))
        incoming_steps = int(fields.get('step_count',0))
        state = MDState(np.asarray(fields['positions']),np.asarray(fields['velocities']),
                        fields['box_vectors'],0.,0)
        engine.callback.context_labels = {'cycle':kwargs.get('cycle_idx'),
                                          'walker':kwargs.get('walker_idx'),'pid':os.getpid()}
        before_calls = engine.provider.n_calls
        engine.set_state(state)
        engine.step(segment_length)
        energies = engine.energies()
        out = engine.get_state()
        # Context time is segment-local, while the portable state records trajectory time.
        out.time_ps += incoming_time
        out.step_count += incoming_steps
        extras = {k:v for k,v in fields.items() if k not in {
            'positions','velocities','box_vectors','time','step_count','qm_diagnostics','energies'}}
        extras.update(energies=energies,qm_diagnostics={
            **engine.callback.last_result.extras,
            'segment_scf_calls':engine.provider.n_calls-before_calls,
            'force_sum_ratio':engine.callback.last_force_sum_ratio})
        return type(walker)(walker_state(out,**extras),walker.weight)
