"""Audited additive QM/MM transformation of an OpenMM System.

Retain QM/MM LJ and boundary bonded terms; replace classical QM electrostatics
and internal bonded/LJ interactions. The ZIP's chosen model also removes the
12-6-4 r^-4 correction from QM-touching pairs. The physical treatment of that
correction remains an open modelling question, not a generally validated rule.
"""
from __future__ import annotations

import copy
from itertools import combinations

import openmm as mm
from openmm import unit

from .errors import PartitionError
from .regions import FrozenPartition


def transform_system(original: mm.System, region: FrozenPartition, *, force_group=31):
    system = copy.deepcopy(original)
    if system.getNumParticles() != region.n_atoms:
        raise PartitionError('System and frozen partition particle counts differ')
    qm = set(region.qm_all_topology_indices)
    real = set(region.qm_atom_indices)
    audit = {'qm_particles': len(qm), 'qm_real_atoms': len(real),
             'forces': [], 'constraints_before': system.getNumConstraints(),
             'qm_force_group': force_group}
    nb_forces = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if len(nb_forces) != 1:
        raise PartitionError('Exactly one NonbondedForce is required')
    for force in system.getForces():
        if force.getForceGroup() == force_group:
            raise PartitionError(f'QM force group {force_group} already in use')
        if isinstance(force, mm.NonbondedForce) and force.getReciprocalSpaceForceGroup() == force_group:
            raise PartitionError('QM force group already used by reciprocal-space electrostatics')
    ghosts = []
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            site = system.getVirtualSite(i)
            parents = {site.getParticle(j) for j in range(site.getNumParticles())}
            if parents & qm:
                if not parents <= qm or i not in qm:
                    raise PartitionError(f'Virtual site {i} crosses the QM/MM boundary')
                ghosts.append(i)
    if set(ghosts) != qm - real:
        raise PartitionError('Frozen QM virtual-site indices do not match System')
    audit['qm_virtual_sites'] = ghosts
    for i, element in zip(region.qm_atom_indices, region.qm_elements):
        mass = system.getParticleMass(i).value_in_unit(unit.dalton)
        if mass <= 0 or (element == 'H' and not 1.0 <= mass <= 1.2):
            raise PartitionError(f'Invalid or repartitioned QM mass at atom {i}: {mass}')

    for force in system.getForces():
        name = type(force).__name__
        info = {'type': name}
        if isinstance(force, mm.NonbondedForce):
            if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                raise PartitionError('Nonbonded parameter offsets are unsupported')
            for i in qm:
                _, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, 0, sigma, 0 if i in ghosts else epsilon)
            existing = set()
            changed = real_touches = 0
            for k in range(force.getNumExceptions()):
                i, j, charge_product, sigma, epsilon = force.getExceptionParameters(k)
                existing.add(tuple(sorted((i, j))))
                if i in real or j in real:
                    real_touches += 1
                if i in qm or j in qm:
                    force.setExceptionParameters(k, i, j, 0, sigma,
                                                 0 if (i in qm and j in qm) or i in ghosts or j in ghosts else epsilon)
                    changed += 1
            added = 0
            for i, j in combinations(sorted(qm), 2):
                if (i, j) not in existing:
                    force.addException(i, j, 0, 1, 0)
                    added += 1
            for k in range(force.getNumExceptions()):
                i, j, charge_product, _, epsilon = force.getExceptionParameters(k)
                if i in qm or j in qm:
                    assert charge_product.value_in_unit(unit.elementary_charge**2) == 0
                if i in qm and j in qm:
                    assert epsilon.value_in_unit(unit.kilojoule_per_mole) == 0
            info.update(particle_charges_zeroed=len(qm), exceptions_zeroed=changed,
                        original_exceptions_touching_real_qm=real_touches,
                        qm_pair_exceptions_added=added)
        elif isinstance(force, (mm.HarmonicBondForce, mm.HarmonicAngleForce, mm.PeriodicTorsionForce)):
            if isinstance(force, mm.HarmonicBondForce):
                number, get, put, nindices = force.getNumBonds, force.getBondParameters, force.setBondParameters, 2
            elif isinstance(force, mm.HarmonicAngleForce):
                number, get, put, nindices = force.getNumAngles, force.getAngleParameters, force.setAngleParameters, 3
            else:
                number, get, put, nindices = force.getNumTorsions, force.getTorsionParameters, force.setTorsionParameters, 4
            count = 0
            for k in range(number()):
                parameters = list(get(k))
                if set(parameters[:nindices]) <= qm:
                    parameters[-1] *= 0
                    put(k, *parameters)
                    count += 1
            info['internal_qm_terms_zeroed'] = count
        elif isinstance(force, mm.CMAPTorsionForce):
            touching = sum(bool(set(force.getTorsionParameters(k)[1:]) & qm)
                           for k in range(force.getNumTorsions()))
            if touching:
                raise PartitionError(f'{touching} CMAP terms touch QM; treatment is undefined')
            info['qm_touching_terms'] = touching
        elif isinstance(force, mm.CustomNonbondedForce):
            expression = force.getEnergyFunction().replace(' ', '')
            if (expression != '-c/r^4;c=ccoef(type1,type2)' or
                    force.getNumPerParticleParameters() != 1 or
                    force.getPerParticleParameterName(0) != 'type' or
                    force.getNumTabulatedFunctions() != 1 or
                    force.getTabulatedFunctionName(0) != 'ccoef' or
                    force.getNumInteractionGroups() or force.getNumGlobalParameters()):
                raise PartitionError(f'Unsupported CustomNonbondedForce: {expression}')
            # Preserve the type-pair table; changing a shared type would change MM/MM pairs.
            force.addPerParticleParameter('is_mm')
            for i in range(force.getNumParticles()):
                force.setParticleParameters(i, [*force.getParticleParameters(i), int(i not in qm)])
            force.setEnergyFunction('-is_mm1*is_mm2*c/r^4; c=ccoef(type1,type2)')
            excluded = {tuple(sorted(force.getExclusionParticles(k))) for k in range(force.getNumExclusions())}
            added = 0
            for pair in combinations(sorted(qm), 2):
                if pair not in excluded:
                    force.addExclusion(*pair)
                    added += 1
            info.update(model='disable r^-4 for all QM-touching pairs',
                        qm_pair_exclusions_added=added, masked_particles=len(qm))
        elif isinstance(force, mm.CMMotionRemover):
            info['action'] = 'retained'
        else:
            raise PartitionError(f'Unsupported force: {name}')
        audit['forces'].append(info)

    removed = 0
    for k in reversed(range(system.getNumConstraints())):
        i, j, _ = system.getConstraintParameters(k)
        if i in qm and j in qm:
            system.removeConstraint(k)
            removed += 1
        elif i in qm or j in qm:
            raise PartitionError(f'Constraint crosses QM/MM boundary: {i}, {j}')
    audit['qm_constraints_removed'] = removed
    audit['constraints_after'] = system.getNumConstraints()
    assert audit['constraints_before'] - audit['constraints_after'] == removed
    return system, audit
