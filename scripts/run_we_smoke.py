"""Small-system WE integration test: 4 walkers, 2 workers, 20 steps, 3 REVO cycles.

This is equilibrium WE with a toy geometry metric, without product recycling or
rate estimation. It validates the software integration, not enzyme kinetics.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import openmm
from wepy.walker import Walker
from wepy.sim_manager import Manager
from wepy.work_mapper.mapper import WorkerMapper,Worker
from wepy.resampling.resamplers.revo import REVOResampler

from weqmmm.config import QMConfig,QMMMConfig
from weqmmm.engine import MDState
from weqmmm.examples import two_water_system
from weqmmm.runner import QMMMRunner,walker_state,close_cached_engines
from weqmmm.distances import GeometryDistance
from weqmmm.reporter import QMMMReporter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    directory = Path(args.output or f'runs/we-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    directory.mkdir(parents=True,exist_ok=False)
    system,region,pos = two_water_system()
    cfg = QMMMConfig(qm=QMConfig(method='rhf',basis='sto-3g',charge=0),
                      integrator='LangevinMiddle',platform='Reference',platform_properties={},
                      failure_directory=str(directory/'failures'),random_seed=0)
    runner = QMMMRunner(openmm.XmlSerializer.serialize(system),region,cfg)
    state = walker_state(MDState(pos,np.zeros_like(pos),None))
    walkers = [Walker(state,.25) for _ in range(4)]
    # Fixed toy scales: a pilot-derived enzyme metric must be configured separately.
    distance = GeometryDistance(((0,3),(0,1),(0,2)),(.3,.1,.1),(.1,.1,.1))
    resampler = REVOResampler(distance=distance,init_state=state,merge_dist=.1,
                              char_dist=1.,pmin=1e-12,pmax=.75,seed=41)
    mapper = WorkerMapper(num_workers=2,worker_type=Worker,proc_start_method='spawn')
    manager = Manager(walkers,runner=runner,work_mapper=mapper,resampler=resampler,
                       reporters=[QMMMReporter(directory)])
    (directory/'config.json').write_text(json.dumps({'engine':cfg.to_dict(),'metric':asdict(distance),
        'walkers':4,'workers':2,'cycles':3,'segment_steps':20,'recycling':False},indent=2)+'\n')
    manager.init()
    try:
        for cycle in range(3):
            before = walkers
            walkers,_ = manager.run_cycle(walkers,20,cycle)
            propagated = manager._last_report['new_walkers']
            assert all(a.weight.hex()==b.weight.hex() for a,b in zip(before,propagated,strict=True))
            assert len(walkers)==4 and abs(sum(w.weight for w in walkers)-1.) < 1e-14
            assert all(np.isfinite(w.state['positions']).all() for w in walkers)
            if cycle==0:
                assert len({w.state['positions'].tobytes() for w in propagated})==4
            print(f'cycle={cycle} walkers={len(walkers)} weight_sum={sum(w.weight for w in walkers):.16g}',flush=True)
        (directory/'PASSED.json').write_text(json.dumps({'cycles':3,'total_weight':sum(w.weight for w in walkers)})+'\n')
    finally:
        manager.cleanup()
        close_cached_engines()


if __name__ == '__main__':
    # The upstream Worker derives from multiprocessing.Process. Select spawn
    # globally as well as for the mapper's queues to avoid forking native engines.
    mp.set_start_method('spawn')
    main()
