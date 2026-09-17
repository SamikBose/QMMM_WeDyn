"""Standalone enzyme run with recorded configuration, audit, energies, and restart.

Run from the project root after sourcing scripts/activate.sh. Each run gets a new
directory. SCF errors terminate immediately and write molecular diagnostics.
"""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import mdtraj as md
import numpy as np
from openmm import app,unit
import yaml

from weqmmm.config import QMMMConfig
from weqmmm.engine import MDState,QMMMEngine
from weqmmm.regions import FrozenPartition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='examples/hca2_azm/config.yaml')
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--basis',help='Explicit basis override, saved in the run configuration')
    parser.add_argument('--output')
    parser.add_argument('--report-interval',type=int,default=1)
    parser.add_argument('--restart',help='Nuclear-state NPZ from a previous standalone run')
    parser.add_argument('--save-qm-result', action='store_true',
                        help='Save imaged QM/MM coordinates and analytic forces at each report')
    args = parser.parse_args()
    if args.steps < 0 or args.report_interval < 1:
        raise ValueError('steps must be nonnegative and report-interval positive')
    values = yaml.safe_load(Path(args.config).read_text())
    cfg = QMMMConfig.from_dict(values.pop('engine'))
    inputs = values.pop('system')
    prmtop,coordinates,frame = inputs.pop('prmtop'),inputs.pop('coordinates'),inputs.pop('frame',0)
    partition_file = values.pop('partition_file')
    if values or inputs:
        raise ValueError(f'Unsupported configuration entries: {values}, {inputs}')
    if args.basis:
        cfg = replace(cfg,qm=replace(cfg.qm,basis=args.basis))
    region = FrozenPartition.load(partition_file)
    for name in (prmtop,coordinates):
        expected = region.source_sha256[name]
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Input checksum differs from frozen partition: {name}')
    directory = Path(args.output or f'runs/hca2_azm/md-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')
    directory.mkdir(parents=True,exist_ok=False)
    cfg = replace(cfg,failure_directory=str(directory/'failures'))
    run_config = {'engine':cfg.to_dict(),'prmtop':prmtop,'coordinates':coordinates,'frame':frame,
                  'partition_file':partition_file,'steps':args.steps,'report_interval':args.report_interval,
                  'restart':args.restart,'save_qm_result':args.save_qm_result}
    (directory/'config.json').write_text(json.dumps(run_config,indent=2)+'\n')
    traj = md.load(coordinates,top=prmtop)
    system = app.AmberPrmtopFile(prmtop).createSystem(nonbondedMethod=app.PME,
        nonbondedCutoff=1*unit.nanometer,constraints=app.HBonds,rigidWater=True)
    print(f'Output: {directory}\nBuilding {cfg.platform} OpenMM Context; '
          f'{len(region.qm_elements)} QM rows, {len(region.embedding_atom_indices)} embedding sites; '
          f'{cfg.qm.method}/{cfg.qm.xc}/{cfg.qm.basis}',flush=True)
    engine = QMMMEngine(system,region,cfg)
    (directory/'system.audit.json').write_text(json.dumps(engine.audit,indent=2)+'\n')
    started = time.perf_counter()
    try:
        engine.callback.context_labels = {'run_directory':str(directory),'stage':'initialization'}
        if args.restart:
            engine.set_state(MDState.load(args.restart))
        else:
            engine.initialize(traj.xyz[frame].astype(float),box_vectors_nm=traj.unitcell_vectors[frame].astype(float),velocity_seed=17)
        samples = [0]+list(range(args.report_interval,args.steps+1,args.report_interval))
        if samples[-1] != args.steps:
            samples.append(args.steps)
        previous = 0
        with (directory/'energies.jsonl').open('w') as log:
            for step in samples:
                engine.callback.context_labels = {'run_directory':str(directory),'requested_step':step}
                engine.step(step-previous)
                energies = engine.energies()
                error = abs(energies['potential_kj']-energies['qm_kj']-energies['mm_kj'])
                assert error/max(1,abs(energies['potential_kj'])) < 1e-9
                state = engine.get_state()
                assert np.isfinite(state.positions_nm).all()
                record = {'step':step,'time_ps':state.time_ps,**energies,
                          'scf_calls':engine.provider.n_calls,'qm':engine.callback.last_result.extras,
                          'force_sum_ratio':engine.callback.last_force_sum_ratio,
                          'elapsed_seconds':time.perf_counter()-started}
                log.write(json.dumps(record)+'\n')
                log.flush()
                print(json.dumps(record),flush=True)
                state.save(directory/f'state-{step:06d}.npz')
                if args.save_qm_result:
                    geometry = engine.callback.geometry
                    result = engine.callback.last_result
                    np.savez(directory/f'qm-result-{step:06d}.npz',
                        qm_positions_nm=geometry['qm_positions_nm'],
                        mm_positions_nm=geometry['mm_positions_nm'],
                        box_vectors_nm=geometry['box_vectors_nm'],
                        mm_charges=engine.callback.charges,
                        energy_kj=result.energy_kj, qm_forces=result.qm_forces,
                        mm_forces=result.mm_forces)
                previous = step
    finally:
        engine.close()


if __name__ == '__main__':
    main()
