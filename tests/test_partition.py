from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app, unit
import pytest
from weqmmm.errors import PartitionError
from weqmmm.partition import transform_system
from weqmmm.regions import FrozenPartition, build_partition


def test_zn_only_partition_keeps_waters_and_ligands_in_mm(tmp_path):
    import qmmm_partition as qp

    region, audit = build_partition('system_1264.parm7',
        'hca2_azm_withwater_run1_combined.pdb', region='zn_only')
    topology, charges = qp.load_topology_and_charges('system_1264.parm7')
    assert region.qm_elements == ('Zn',)
    assert region.qm_formal_charge == 2 and region.spin == 0
    assert region.qm_all_topology_indices == region.qm_atom_indices
    assert not region.boundary_pairs and not region.selected_water_resids
    assert audit['complete_partition_charge_error'] == pytest.approx(0, abs=1e-6)
    np.testing.assert_array_equal(region.embedding_charges, charges[list(region.mm_atom_indices)])
    selected = set(region.embedding_atom_indices)
    for residue in topology.residues:
        ids = {atom.index for atom in residue.atoms}
        if ids & selected:
            assert ids <= selected
    path = tmp_path / 'partition.json'
    region.save(path)
    assert FrozenPartition.load(path) == region


@pytest.fixture(scope='module')
def enzyme():
    region = FrozenPartition.load('runs/hca2_azm/partition.json')
    system = app.AmberPrmtopFile('system_1264.parm7').createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,
        constraints=app.HBonds, rigidWater=True)
    return region, system


def test_enzyme_audit_and_caller_unchanged(enzyme):
    region, original = enzyme
    before = mm.XmlSerializer.serialize(original)
    system, audit = transform_system(original, region)
    assert mm.XmlSerializer.serialize(original) == before
    assert audit['qm_constraints_removed'] == 31
    assert len(audit['qm_virtual_sites']) == 2
    counts = {f['type']: f for f in audit['forces']}
    assert counts['HarmonicBondForce']['internal_qm_terms_zeroed'] == 35
    assert counts['HarmonicAngleForce']['internal_qm_terms_zeroed'] == 86
    assert counts['PeriodicTorsionForce']['internal_qm_terms_zeroed'] == 129
    assert counts['CMAPTorsionForce']['qm_touching_terms'] == 0
    assert counts['NonbondedForce']['original_exceptions_touching_real_qm'] == 326
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    custom = next(f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce))
    assert {tuple(sorted(nb.getExceptionParameters(k)[:2])) for k in range(nb.getNumExceptions())} == {
        tuple(sorted(custom.getExclusionParticles(k))) for k in range(custom.getNumExclusions())}
    Path('validation/system-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


def test_charge_conservation_retains_input_residual(enzyme):
    region, _ = enzyme
    assert abs(region.system_net_charge) > 1e-6
    assert abs(sum(region.embedding_charges) + region.qm_formal_charge - region.system_net_charge) < 1e-6
    with pytest.raises(AssertionError, match='Charge conservation failed'):
        replace(region, embedding_charges=(region.embedding_charges[0]+0.001, *region.embedding_charges[1:]))


def test_reject_barostat_and_crossing_constraint(enzyme):
    region, system = enzyme
    clone = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    clone.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin))
    with pytest.raises(PartitionError, match='Unsupported force'):
        transform_system(clone, region)
    clone = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    clone.addConstraint(*region.boundary_pairs[0], 0.15)
    with pytest.raises(PartitionError, match='Constraint crosses'):
        transform_system(clone, region)


def test_1264_mask_preserves_mm_pairs_with_shared_atom_type():
    region = FrozenPartition((0,), (0,), ('He',), (), (), 0, 0,
                             (1,2), (0.,0.), (1,2), (), (0,), 0, 0., 3, 2., {})
    s = mm.System()
    nb = mm.NonbondedForce()
    custom = mm.CustomNonbondedForce('-c/r^4; c=ccoef(type1,type2)')
    custom.addPerParticleParameter('type')
    custom.addTabulatedFunction('ccoef', mm.Discrete2DFunction(1, 1, [0.002]))
    for _ in range(3):
        s.addParticle(4)
        nb.addParticle(0, 0.3, 0)
        custom.addParticle([0])
    s.addForce(nb)
    s.addForce(custom)
    out, _ = transform_system(s, region)
    integ = mm.VerletIntegrator(0.0005)
    ctx = mm.Context(out, integ, mm.Platform.getPlatformByName('Reference'))
    ctx.setPositions(np.array([[0,0,0], [0.3,0,0], [0.8,0,0]])*unit.nanometer)
    state = ctx.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    assert energy == pytest.approx(-0.002/0.5**4, abs=1e-12)
    assert np.max(abs(forces[0])) < 1e-12
    assert np.max(abs(forces.sum(axis=0))) < 1e-12
    assert forces[1,0] == pytest.approx(4*0.002/0.5**5, abs=1e-12)
