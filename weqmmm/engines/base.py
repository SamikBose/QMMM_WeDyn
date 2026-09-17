"""Backend contract. An ASE adapter must also supply embedding-site reaction forces.

Supporting only the standard ASE nuclear energy/forces contract is insufficient
for this electrostatic-embedding interface. Periodic backends must declare and
validate their electrostatics separately; a box argument alone does not add Ewald.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class QMResult:
    energy_kj: float
    qm_forces: np.ndarray
    mm_forces: np.ndarray
    converged: bool
    extras: dict


class QMRegionEngine(Protocol):
    elements: tuple[str, ...]
    failure: str | None
    n_calls: int

    def begin_segment(self, dm0=None) -> None: ...
    def end_segment(self): ...
    def compute(self, qm_positions_nm, mm_positions_nm, mm_charges,
                *, box_vectors_nm=None) -> QMResult: ...
