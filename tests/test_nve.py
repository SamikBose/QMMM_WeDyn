"""Gate 8. Criteria are preregistered in docs/NVE_ACCEPTANCE.md."""
import json
from pathlib import Path
import time
import numpy as np
import pytest
from openmm import unit
from weqmmm.config import QMConfig,QMMMConfig
from weqmmm.engine import MDState,QMMMEngine
from weqmmm.examples import two_water_system


@pytest.mark.dynamics
def test_gate8_nve_timestep_convergence(tmp_path):
    check_nve_timestep_convergence(tmp_path)


def check_nve_timestep_convergence(tmp_path, backend='pyscf_cpu', output_prefix='nve'):
    system,region,pos = two_water_system()
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(6)])
    rng = np.random.default_rng(17)
    vel = rng.normal(size=(6,3))*np.sqrt(.00831446261815324*100/masses[:,None])
    vel -= np.sum(vel*masses[:,None],axis=0)/masses.sum()
    reports = []
    for dt in (.5,.25):
        cfg = QMMMConfig(qm=QMConfig(method='rhf',basis='sto-3g',charge=0,backend=backend),integrator='Verlet',
                         timestep_fs=dt,temperature_K=100,platform='Reference',platform_properties={},
                         failure_directory=str(tmp_path))
        engine = QMMMEngine(system,region,cfg)
        try:
            engine.set_state(MDState(pos,vel,None))
            engine.context.applyVelocityConstraints(1e-8)
            started = time.perf_counter()
            rows = []
            for sample in range(201):
                if sample:
                    engine.step(round(5/dt))
                energies = engine.energies()
                rows.append([sample*.005,energies['total_kj'],energies['potential_kj'],energies['kinetic_kj']])
                if sample%20==0:
                    print(f'Gate 8: dt={dt} fs, time={sample*.005:.3f} ps, '
                          f'dE={rows[-1][1]-rows[0][1]:.6g} kJ/mol',flush=True)
            array = np.asarray(rows)
            delta = array[:,1]-array[0,1]
            final = engine.get_state()
            np.savetxt(f'validation/{output_prefix}-{dt:g}fs.csv',array,delimiter=',',
                       header='time_ps,total_kj_mol,potential_kj_mol,kinetic_kj_mol',comments='')
            np.savez(f'validation/{output_prefix}-{dt:g}fs-final.npz',positions_nm=final.positions_nm,
                     velocities_nm_ps=final.velocities_nm_ps)
            report = {'timestep_fs':dt,'duration_ps':1.,'peak_to_peak_kj':float(np.ptp(delta)),
                      'max_absolute_error_kj':float(np.max(abs(delta))),
                      'drift_kj_per_ps':float(np.polyfit(array[:,0],delta,1)[0]),
                      'scf_calls':engine.provider.n_calls,'seconds':time.perf_counter()-started,
                      'finite':bool(np.isfinite(array).all() and np.isfinite(final.positions_nm).all()),
                      'config':cfg.to_dict()}
            reports.append(report)
            Path(f'validation/{output_prefix}-report.json').write_text(json.dumps(reports,indent=2)+'\n')
        finally:
            engine.close()
    for report in reports:
        assert report['finite']
        assert report['peak_to_peak_kj'] < .5, report
        assert abs(report['drift_kj_per_ps']) < .2, report
    coarse,fine = [r['peak_to_peak_kj'] for r in reports]
    assert fine <= .75*coarse or max(coarse,fine) < 1e-5, reports
