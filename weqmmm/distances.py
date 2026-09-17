"""Geometry-only REVO descriptors with frozen feature scales; no wepy imports."""
from dataclasses import dataclass
import numpy as np
from qmmm_partition import minimum_image_displacement


@dataclass(frozen=True)
class GeometryDistance:
    pairs: tuple[tuple[int,int],...]
    radii_nm: tuple[float,...]
    scales: tuple[float,...]
    torsions: tuple[tuple[int,int,int,int],...] = ()
    exponent: int = 6

    def __post_init__(self):
        assert len(self.pairs) == len(self.radii_nm)
        assert len(self.scales) == len(self.pairs)+2*len(self.torsions)
        assert len(self.scales) > 0
        assert all(s>0 for s in self.scales) and all(r>0 for r in self.radii_nm)
        assert self.exponent > 0

    def image(self,state):
        pos = np.asarray(state['positions'])
        box = state['box_vectors']
        values = []
        for (i,j),radius in zip(self.pairs,self.radii_nm,strict=True):
            r = np.linalg.norm(minimum_image_displacement(pos[j]-pos[i],box))
            values.append(1/(1+(r/radius)**self.exponent))
        for i,j,k,l in self.torsions:
            b0,b1,b2 = minimum_image_displacement(np.array([pos[i]-pos[j],pos[k]-pos[j],pos[l]-pos[k]]),box)
            b1 /= np.linalg.norm(b1)
            v,w = b0-np.dot(b0,b1)*b1,b2-np.dot(b2,b1)*b1
            denom = np.linalg.norm(v)*np.linalg.norm(w)
            if denom < 1e-14:
                raise ValueError('Torsion undefined for collinear or coincident atoms')
            values += [np.dot(v,w)/denom,np.dot(np.cross(b1,v),w)/denom]
        return np.asarray(values)/np.asarray(self.scales)

    def image_distance(self,a,b):
        return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))

    def distance(self,a,b):
        return self.image_distance(self.image(a),self.image(b))
