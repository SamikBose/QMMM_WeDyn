# weqmmm

Modular QM/MM molecular dynamics: PySCF supplies quantum energies and forces,
OpenMM supplies classical forces and propagation, and wepy supplies weighted
ensemble propagation and resampling.

**Status:** the standalone engine and wepy adapter are implemented and tested on
small systems. The full 70-row B3LYP/def2-SVP SCF converges on CPU and GPU; the
GPU single-point energy/forces pass. Zn-only enzyme MD and equilibrium WE smoke
tests passed. Full-region GPU MD (10 steps) and small-system GPU NVE (1 ps at
each timestep) passed. The user approved GPU-only numerical trajectory
tolerances after independent GPU runs showed last-digit differences. All three
GPU runner checks and the two-GPU enzyme WE test now pass. Exact
weights and all physics gates are unchanged. See [GPU backend details](docs/GPU_BACKEND.md).

The architecture and runnable workflow are described in
[the undergraduate handoff](docs/UNDERGRAD_HANDOFF.md).

**Resume update (2026-09-16):** the old enzyme runs stopped without results.
A separate Zn2+-only fixture is available in `examples/hca2_azm/zn_only.yaml`.
Its original STO-3G tests failed at 50 cycles. With the user-selected def2-SVP
basis, isolated and embedded Zn2+ converge in 6 and 8 cycles, and force checks
pass. A 10-step (5 fs) Zn-only CPU MD smoke test also passed. See the progress
log for results. GPU4PySCF is installed in `.venv-gpu`, and the initial GPU
physics tests passed on an allocated L40S.

For the model, commands, validated scope, limitations, and ordered TODO list,
see [docs/UNDERGRAD_HANDOFF.md](docs/UNDERGRAD_HANDOFF.md).
The complete HCA2/acetazolamide example input bundle is committed under
[`examples/hca2_azm/data/`](examples/hca2_azm/data/).

## Environment and validation

A fresh conda environment named `weqmmm` is installed at
`/mnt/home/bosesami/.conda/envs/weqmmm`, with CPU PySCF 2.14.0, OpenMM 8.6.1
and upstream wepy. From the project root:

```bash
source scripts/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_prereqs.py
python scripts/validate.py
# Include the separately validated 1 ps NVE timestep study:
python scripts/validate.py --nve
```

For a new checkout, `conda env create -f environment.yml` creates the base
environment. Install the development wepy checkout and PartQMMM separately as
described in the handoff guide; they are intentionally not vendored into this
repository. GPU4PySCF is an optional extra and should only be installed on a
CUDA node.

Activation also sets the project and `external/PartQMMM` import paths and disables
user-site imports. Record installed package versions and external repository
commits when creating a new research run.

| Check | Measured result |
| --- | --- |
| Prerequisites | All 15 checks passed |
| Consolidated non-dynamics tests | 30 passed after backend refactor |
| Provider forces | RHF, UHF, RKS and UKS finite-difference and rotation checks passed |
| Standalone integration | Force mapping, energy decomposition, fatal-error handling and short DFT integration passed |
| NVE, small RHF system | 1 ps at both 0.5 and 0.25 fs passed preregistered criteria |
| Runner | Standalone agreement, restart, walker isolation and independent clone noise passed |
| REVO smoke test, small RHF system | 4 walkers, 2 workers, 3 cycles passed; weights sum to 1 |
| Full 70-row enzyme quantum evaluation | B3LYP/def2-SVP converged; GPU energy/forces passed |
| Zn-only enzyme WE | 4 walkers, 2 CPU workers, 3 cycles of 20 steps passed |
| GPU provider | Six numerical tests, 1 ps NVE, three runner checks and full-region enzyme WE passed |

The original four provider-test failures and their user-approved corrections
remain documented. There are no automatic SCF retries or silent solver fallbacks.

## Standalone enzyme setup

```bash
python scripts/build_partition.py
python scripts/audit_system.py
# A new output directory is created for every MD invocation:
python scripts/run_md.py --steps 0
python scripts/run_md.py --steps 10
python scripts/run_md.py --restart PATH_TO_STATE.npz --steps 10
```

These commands are implemented; enzyme quantum runtime and numerical behavior
remain unvalidated. The example in `examples/hca2_azm/config.yaml` explicitly
uses B3LYP/STO-3G for coupling tests. The provider default is def2-SVP. Each run
records its configuration, transformation audit, energies and nuclear-state
checkpoints. A nuclear restart resets electronic history and does not reproduce
the previous Langevin random stream bit for bit.

The frozen region has 66 real QM atoms plus four link hydrogens, two inert QM
virtual sites and 4,646 embedding sites selected by whole residue fragments.
Charge conservation is checked against the original unrounded topology total;
the input's approximately +4.6903e-5 e residual is reported without changing charges.

## Weighted ensemble

```bash
python scripts/run_we_smoke.py
```

This small-system REVO test requires permission for local multiprocessing IPC.
`QMMMRunner` takes an **original, unmodified** OpenMM System XML, frozen partition
and configuration, and constructs a process-local engine. Do not pass the
already transformed `runs/hca2_azm/mm-system.xml` as its original System.

Enzyme reaction descriptors, normalization from a pilot, source/product
boundaries and recycling remain to be specified. The current smoke test does
not estimate kinetic rates.

## Architecture and scope

- `weqmmm/engines/base.py`: replaceable QM provider contract, including MM
  reaction forces and segment lifecycle; independent of OpenMM and wepy.
- `regions.py`, `partition.py`, `link.py`, `force.py`: frozen partition,
  classical System transformation, link projection and OpenMM coupling.
- `engine.py`: standalone MD and portable nuclear states.
- `runner.py`, `reporter.py`, `distances.py`: wepy propagation, cycle artifacts
  and geometry descriptors.

Other QM engines and ASE adapters can implement the provider contract; none is
implemented yet. The molecular GPU provider is implemented and passes initial
physics checks; periodic QM embedding remains future work.

The user-approved prototype combines MM PME with fixed finite molecular QM
embedding and omits long-range QM/MM electrostatics. It is not validated for
production kinetics.

The selected production-development path is finite QM plus converged finite
embedding. The OpenMM triclinic box is still used for MM PME and coordinate
imaging; PySCF itself receives the fixed embedding charges as a molecular
calculation. The next gate is convergence versus fixed whole-residue embedding
radius and selection. Periodic Ewald QM/MM remains a future alternative.

[NVE acceptance criteria](docs/NVE_ACCEPTANCE.md) replace the exact fourfold
linear-drift-ratio requirement with a preregistered timestep-convergence test.
**We will examine this choice again later**, including enzyme-specific trajectories.
