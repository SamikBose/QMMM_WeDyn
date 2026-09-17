"""Freeze the enzyme partition before MD; never select new waters in a force call."""
import argparse
import json
from pathlib import Path
from weqmmm.regions import build_partition

parser = argparse.ArgumentParser()
parser.add_argument('--prmtop', default='system_1264.parm7')
parser.add_argument('--coordinates', default='hca2_azm_withwater_run1_combined.pdb')
parser.add_argument('--output', default='runs/hca2_azm/partition.json')
parser.add_argument('--embedding-radius-nm', type=float, default=2.0)
parser.add_argument('--region', choices=['minimal', 'larger', 'zn_only'], default='minimal')
args = parser.parse_args()
partition, audit = build_partition(args.prmtop, args.coordinates,
                                   region=args.region,
                                   embedding_radius_nm=args.embedding_radius_nm)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
partition.save(output)
output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
