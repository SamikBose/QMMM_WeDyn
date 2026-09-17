"""Serializable provider and propagation settings; no live engine objects."""
from dataclasses import asdict, dataclass, field
import json


@dataclass(frozen=True)
class QMConfig:
    method: str = 'rks'
    xc: str = 'b3lyp'
    basis: str = 'def2-svp'
    charge: int = 1
    spin: int = 0
    conv_tol: float = 1e-10
    conv_tol_grad: float = 1e-7
    max_cycle: int = 50
    grid_level: int = 5
    mm_charge_radii_bohr: float | None = None
    carry_density_matrix: bool = False
    verbose: int = 0
    backend: str = 'pyscf_cpu'

    def __post_init__(self):
        if self.backend not in {'pyscf_cpu', 'gpu4pyscf_molecular'}:
            raise ValueError(f'Unsupported QM backend: {self.backend}')
        if self.backend == 'gpu4pyscf_molecular' and self.mm_charge_radii_bohr is not None:
            raise ValueError('GPU molecular embedding currently supports point charges only')
        if self.method not in {'rhf', 'uhf', 'rks', 'uks'}:
            raise ValueError(f'Unsupported QM method: {self.method}')
        if self.method in {'rhf', 'rks'} and self.spin != 0:
            raise ValueError('Restricted closed-shell method requires spin=0')
        if self.spin < 0 or self.max_cycle < 1 or self.conv_tol <= 0 or self.conv_tol_grad <= 0:
            raise ValueError('Invalid SCF convergence settings')
        if self.mm_charge_radii_bohr is not None and self.mm_charge_radii_bohr <= 0:
            raise ValueError('Gaussian charge radius must be positive')
        if self.carry_density_matrix:
            raise ValueError('Cross-segment density-matrix checkpoints are deferred')


@dataclass(frozen=True)
class QMMMConfig:
    qm: QMConfig = field(default_factory=QMConfig)
    integrator: str = 'LangevinMiddle'
    timestep_fs: float = 0.5
    temperature_K: float = 300.
    friction_per_ps: float = 1.
    platform: str = 'CPU'
    platform_properties: dict[str, str] = field(default_factory=lambda: {'Threads': '1'})
    random_seed: int = 0
    check_force_sum: bool = True
    failure_directory: str = 'runs/failures'

    def __post_init__(self):
        if self.integrator not in {'Verlet', 'LangevinMiddle'}:
            raise ValueError('Supported integrators: Verlet, LangevinMiddle')
        if not 0 < self.timestep_fs <= 0.5:
            raise ValueError('Unconstrained QM bonds require 0 < timestep_fs <= 0.5')
        if self.temperature_K <= 0 or self.friction_per_ps < 0:
            raise ValueError('Invalid thermostat settings')

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, values):
        data = dict(values)
        data['qm'] = QMConfig(**data.get('qm', {}))
        return cls(**data)
