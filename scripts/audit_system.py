"""Transform and audit the enzyme MM System without a quantum force call."""
import json
from pathlib import Path
import openmm
from openmm import app, unit
from weqmmm.partition import transform_system
from weqmmm.regions import FrozenPartition

region = FrozenPartition.load('runs/hca2_azm/partition.json')
original = app.AmberPrmtopFile('system_1264.parm7').createSystem(
    nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,
    constraints=app.HBonds, rigidWater=True)
system, audit = transform_system(original, region)
Path('runs/hca2_azm/system.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
Path('runs/hca2_azm/mm-system.xml').write_text(openmm.XmlSerializer.serialize(system))
print(json.dumps(audit, indent=2))
