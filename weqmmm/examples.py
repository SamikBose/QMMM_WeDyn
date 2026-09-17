"""Small, explicit validation fixtures; never substitutes for enzyme validation."""
import numpy as np
import openmm
from .regions import FrozenPartition


def two_water_system():
    """One flexible QM water and one constrained, fixed-charge three-site MM water.

    The short example is nonperiodic. LJ remains between the two oxygen atoms.
    Full MM reaction forces move the second water; it is not a static field.
    """
    water = np.array([[0,0,0], [0,.0757,.0585], [0,-.0757,.0585]])
    positions = np.vstack((water, water+np.array([.30,0,0])))
    system = openmm.System()
    nb = openmm.NonbondedForce()
    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    for i in range(6):
        oxygen = i%3 == 0
        system.addParticle(15.999 if oxygen else 1.008)
        nb.addParticle(-.834 if oxygen else .417, .315075 if oxygen else 1., .636386 if oxygen else 0.)
    for o in (0,3):
        for h in (o+1,o+2):
            distance = float(np.linalg.norm(positions[h]-positions[o]))
            bonds.addBond(o,h,distance,462750.)
            system.addConstraint(o,h,distance)
        hh = float(np.linalg.norm(positions[o+1]-positions[o+2]))
        system.addConstraint(o+1,o+2,hh)
        angle = np.arccos(np.dot(water[1],water[2])/np.linalg.norm(water[1])/np.linalg.norm(water[2]))
        angles.addAngle(o+1,o,o+2,float(angle),836.8)
        for i,j in ((o,o+1),(o,o+2),(o+1,o+2)):
            nb.addException(i,j,0,1,0)
    system.addForce(nb)
    system.addForce(bonds)
    system.addForce(angles)
    region = FrozenPartition((0,1,2),(0,1,2),('O','H','H'),(),(),0,0,
                             (3,4,5),(-.834,.417,.417),(3,4,5),(),(1,),0,0.,6,2.,{})
    return system,region,positions
