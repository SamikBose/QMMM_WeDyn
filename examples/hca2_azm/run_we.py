"""Equilibrium enzyme WE smoke test with a frozen, recorded pilot metric.

Requires an original prmtop, a constraint-projected nuclear checkpoint,
and an explicit metric file. No recycling or kinetic rates are inferred.
Each GPU worker uses one of the GPUs exposed by the scheduler.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit
import yaml
from wepy.walker import Walker
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper, Worker
from wepy.resampling.resamplers.revo import REVOResampler

from weqmmm.config import QMMMConfig
from weqmmm.distances import GeometryDistance
from weqmmm.engine import MDState
from weqmmm.regions import FrozenPartition
from weqmmm.runner import QMMMRunner, GPUWorker, allocated_gpu_for_worker, walker_state, close_cached_engines
from weqmmm.reporter import QMMMReporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--metric', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--walkers', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    assert args.walkers >= 2 and 1 <= args.workers <= args.walkers
    assert args.cycles > 0 and args.steps > 0
    values = yaml.safe_load(Path(args.config).read_text())
    region = FrozenPartition.load(values['partition_file'])
    cfg = QMMMConfig.from_dict(values['engine'])
    worker_type = Worker
    if cfg.qm.backend == 'gpu4pyscf_molecular':
        allocated_gpu_for_worker(args.workers-1, os.environ.get('CUDA_VISIBLE_DEVICES', ''))
        worker_type = GPUWorker
    assert cfg.integrator == 'LangevinMiddle', 'Independence smoke test requires thermal noise'
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = replace(cfg, failure_directory=str(directory/'failures'))
    metric_record = json.loads(Path(args.metric).read_text())
    prmtop = values['system']['prmtop']
    actual_hash = hashlib.sha256(Path(prmtop).read_bytes()).hexdigest()
    assert actual_hash == metric_record['prmtop_sha256'] == region.source_sha256[prmtop]
    metric_values = metric_record['metric']
    distance = GeometryDistance(**metric_values)
    system = app.AmberPrmtopFile(prmtop).createSystem(nonbondedMethod=app.PME,
        nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds, rigidWater=True)
    state = walker_state(MDState.load(args.state))
    initial_positions = state['positions'].copy()
    walkers = [Walker(state, 1/args.walkers) for _ in range(args.walkers)]
    runner = QMMMRunner(openmm.XmlSerializer.serialize(system), region, cfg)
    resampler = REVOResampler(distance=distance, init_state=state,
        merge_dist=metric_record['revo']['merge_dist'], char_dist=metric_record['revo']['char_dist'],
        pmin=1e-12, pmax=.75, seed=41)
    manager = Manager(walkers, runner=runner, resampler=resampler,
        work_mapper=WorkerMapper(num_workers=args.workers, worker_type=worker_type, proc_start_method='spawn'),
        reporters=[QMMMReporter(directory)])
    (directory/'config.json').write_text(json.dumps({'engine':cfg.to_dict(), **vars(args),
        'state_sha256':hashlib.sha256(Path(args.state).read_bytes()).hexdigest(),
        'metric':metric_record, 'recycling':False}, indent=2)+'\n')
    manager.init()
    try:
        for cycle in range(args.cycles):
            before = walkers
            walkers, _ = manager.run_cycle(walkers, args.steps, cycle)
            propagated = manager._last_report['new_walkers']
            assert all(a.weight.hex() == b.weight.hex() for a,b in zip(before,propagated,strict=True))
            assert len(walkers) == args.walkers and abs(sum(w.weight for w in walkers)-1.) < 1e-14
            assert all(np.isfinite(w.state['positions']).all() and np.isfinite(w.state['velocities']).all() for w in walkers)
            if cycle == 0:
                assert len({w.state['positions'].tobytes() for w in propagated}) == args.walkers
                assert np.array_equal(state['positions'], initial_positions)
            print(f'cycle={cycle} walkers={len(walkers)} weight_sum={sum(w.weight for w in walkers):.16g}', flush=True)
        (directory/'PASSED.json').write_text(json.dumps({'cycles':args.cycles,
            'walkers':args.walkers,'workers':args.workers,'segment_steps':args.steps,
            'total_weight':sum(w.weight for w in walkers),'recycling':False,
            'scope':'equilibrium WE software smoke test; not a kinetic-rate calculation'}, indent=2)+'\n')
    finally:
        manager.cleanup()
        close_cached_engines()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
