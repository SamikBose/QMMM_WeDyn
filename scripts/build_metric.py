"""Freeze development REVO feature scales from saved enzyme pilot states.

The short pilot is a software test, not a production reaction-coordinate
calibration. A preregistered 0.01 floor prevents nearly rigid features from
receiving enormous weights. No scales are updated during WE.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mdtraj as md
import numpy as np

from weqmmm.distances import GeometryDistance
from weqmmm.engine import MDState
from weqmmm.regions import FrozenPartition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    partition = FrozenPartition.load('runs/hca2_azm/partition.json')
    top = md.load_prmtop('system_1264.parm7')
    zn = partition.anchor_index
    assert top.atom(zn).element.symbol == 'Zn'

    def atom(resname, resseq, name):
        hits = [a.index for a in top.atoms if a.residue.name == resname and
                a.residue.resSeq == resseq and a.name == name]
        assert len(hits) == 1, (resname, resseq, name, hits)
        return hits[0]

    # PDB residue numbering has the audited -4 topology offset.
    donor_ids = [atom('HIS',90,'NE2'), atom('HIS',92,'NE2'), atom('HIS',115,'ND1'),
                 atom('AZM',258,'N1'), atom('THR',195,'OG1')]
    for resid in partition.selected_water_resids:
        oxygens = [a.index for a in top.residue(resid).atoms if a.element.symbol == 'O']
        assert len(oxygens) == 1
        donor_ids.extend(oxygens)
    pairs = tuple((zn, i) for i in donor_ids) + ((atom('AZM',258,'N1'), atom('AZM',258,'H1')),)
    torsion = tuple(atom('AZM',258,name) for name in ('N1','S1','C1','N3'))
    radii = (.25,) * len(donor_ids) + (.11,)
    raw = GeometryDistance(pairs, radii, (1.,) * (len(pairs)+2), (torsion,))
    paths = sorted(Path(args.pilot_dir).glob('state-*.npz'))
    assert len(paths) >= 3, 'Need at least three pilot states'
    states = [MDState.load(path) for path in paths]
    images = np.array([raw.image({'positions':s.positions_nm, 'box_vectors':s.box_vectors_nm}) for s in states])
    std = np.std(images, axis=0, ddof=1)
    assert np.isfinite(std).all()
    scales = np.maximum(std, .01)
    metric = GeometryDistance(pairs, radii, tuple(map(float, scales)), (torsion,))
    record = {'metric':asdict(metric), 'scope':'development-only short-pilot calibration',
              'pilot_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'prmtop_sha256':hashlib.sha256(Path('system_1264.parm7').read_bytes()).hexdigest(),
              'feature_standard_deviations':std.tolist(), 'scale_floor':.01,
              'pilot_duration_ps':states[-1].time_ps-states[0].time_ps,
              'labels':[f'Zn-{top.atom(i)}' for i in donor_ids]+['AZM N1-H1','cos(N1-S1-C1-N3)','sin(N1-S1-C1-N3)'],
              'revo':{'merge_dist':.1,'char_dist':1.}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'output':str(output), 'features':len(scales), 'pilot_states':len(paths),
                      'pilot_duration_ps':record['pilot_duration_ps']}, indent=2))


if __name__ == '__main__':
    main()
