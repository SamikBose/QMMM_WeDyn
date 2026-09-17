"""Per-cycle diagnostics and nuclear-state checkpoints for a wepy Manager.

JSON is the cycle commit record; an interrupted cycle must not be treated as
complete until that record exists. Density matrices are intentionally omitted.
"""
import json
from pathlib import Path
import numpy as np
from wepy.reporter.reporter import Reporter


def _json_default(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unsupported diagnostic type: {type(value).__name__}')


class QMMMReporter(Reporter):
    def __init__(self,directory):
        super().__init__()
        self.directory = Path(directory)

    def init(self,**kwargs):
        self.directory.mkdir(parents=True,exist_ok=True)

    def report(self,*,cycle_idx,new_walkers,resampled_walkers,resampling_data,resampler_data,**kwargs):
        prefix = self.directory/f'cycle-{cycle_idx:06d}'
        if prefix.with_suffix('.json').exists():
            raise FileExistsError(f'Cycle already committed: {prefix}')
        weights = np.array([w.weight for w in resampled_walkers])
        states = [w.state.dict() for w in resampled_walkers]
        data = {'positions':np.array([s['positions'] for s in states]),
                'velocities':np.array([s['velocities'] for s in states]),
                'time':np.array([s['time'] for s in states]),
                'step_count':np.array([s['step_count'] for s in states]),'weights':weights}
        if states[0]['box_vectors'] is not None:
            data['box_vectors'] = np.array([s['box_vectors'] for s in states])
        tmp = prefix.with_suffix('.partial.npz')
        with tmp.open('wb') as out:
            np.savez(out,**data)
        tmp.replace(prefix.with_suffix('.npz'))
        record = {'cycle_idx':cycle_idx,'weights':weights,'weight_sum':float(weights.sum()),
                  'propagated_energies':[w.state['energies'] for w in new_walkers],
                  'qm_diagnostics':[w.state['qm_diagnostics'] for w in new_walkers],
                  'resampling_data':resampling_data,'resampler_data':resampler_data}
        tmp = prefix.with_suffix('.partial.json')
        tmp.write_text(json.dumps(record,default=_json_default,indent=2)+'\n')
        tmp.replace(prefix.with_suffix('.json'))

    def cleanup(self,**kwargs):
        pass
