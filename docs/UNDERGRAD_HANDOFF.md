# weqmmm undergraduate handoff

This document is the practical starting point for a student working on this
repository. It explains the model, the code layout, the commands that have
already been tested, and the work that remains. Read this together with
the README and the focused GPU/NVE notes in `docs/`. Those files contain the
operational setup and acceptance criteria for this compact repository.

## What this project does

The project couples three layers:

```text
wepy manager and resampler
        |
QMMMRunner: one walker segment
        |
QMMMEngine: one OpenMM Context and integrator
        |
OpenMM forces + PythonForce callback
        |
PySCF or GPU4PySCF molecular QM provider
```

There is one OpenMM integrator for all real atoms. The QM provider never moves
atoms. At each force evaluation it receives the current QM coordinates and
fixed embedding MM coordinates, returns QM forces and MM reaction forces, and
OpenMM adds those forces to the other force groups before advancing all
particles once. Link hydrogens are temporary QM rows; their forces are
projected back to the boundary atoms. wepy operates above this engine: it runs
segments, then clones/merges walker states at cycle boundaries.

The selected physical model is **finite QM plus fixed, whole-residue finite
embedding**. OpenMM retains periodic MM PME and triclinic minimum-image
handling. PySCF receives a molecular QM cluster and the selected MM point
charges. This is electrostatic embedding during every MD force call, but MM
charges outside the selected list do not contribute to the QM Hamiltonian.
Periodic Ewald QM/MM embedding is a deferred alternative; it is not part of
the current production-development path.

## System and frozen partition

The target is human carbonic anhydrase II with acetazolamide, Zn2+, and
explicit OPC water. The supplied topology has 38,361 OpenMM particles,
including virtual sites. The frozen minimal partition contains:

* 66 real QM atoms, four link hydrogens, and two QM virtual sites;
* QM formal charge +1 and spin 0;
* 4,646 selected MM embedding sites from 525 complete residue fragments;
* four covalent boundary cuts.

The partition is built once and then remains fixed during a trajectory. Never
re-run adaptive PartQMMM selection inside an MD or WE segment. The complete
partition charge check uses the original unrounded topology charge and a
`1e-6 e` tolerance; the small input residual (`+4.690297447962344e-5 e`) is
reported and is not silently corrected.

## Environment and first checks

The validated CPU environment is the conda environment `weqmmm` with Python
3.11, OpenMM 8.6.1, PySCF 2.14.0, MDTraj, NumPy, SciPy, h5py, pytest, wepy,
and the pinned PartQMMM checkout. GPU work uses the separate `.venv-gpu` and
the scripts under `scripts/`.

From the repository root:

```bash
source scripts/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_prereqs.py
python scripts/validate.py
python scripts/validate.py --nve
```

Gate 0 must pass before changing scientific code. The normal CPU regression
has passed 30 tests; the separate small-system NVE gate has also passed. A
test that is skipped because a GPU is unavailable is not a passing GPU result.
On an allocated GPU node, use `source scripts/activate_gpu.sh` and the commands
in `docs/GPU_BACKEND.md`.

## Reproducing the existing workflows

Build the frozen partition and audit the original OpenMM system:

```bash
python scripts/build_partition.py
python scripts/audit_system.py
```

Run bounded standalone enzyme tests. Each invocation must use a new output
directory; the runner records configuration, input/audit data, energies, and
nuclear state checkpoints:

```bash
python scripts/run_md.py --steps 0
python scripts/run_md.py --steps 10
python scripts/run_md.py --restart PATH_TO_STATE.npz --steps 10
```

The example configuration uses B3LYP/STO-3G for a deliberately inexpensive
coupling test. The provider default is def2-SVP. The full 70-row GPU
def2-SVP single point, a 10-step GPU MD pilot, and a short four-walker/two-GPU
WE run completed. The earlier full-region CPU attempts were stopped before a
validated force result; do not describe them as successful enzyme dynamics.

Run the small WE plumbing test with:

```bash
python scripts/run_we_smoke.py
```

The enzyme WE entry point is `examples/hca2_azm/run_we.py`. It requires the
original untransformed OpenMM System XML, the frozen partition, and a frozen
metric file. Do not pass `runs/hca2_azm/mm-system.xml` as the original System.
The existing enzyme WE runs are equilibrium plumbing tests only: source and
product states, recycling, and rates have not been scientifically defined.

The Zn-only development fixture is useful for inexpensive debugging:

```bash
python scripts/validate_zn_single_point.py
```

It uses def2-SVP because the requested 6-31+G basis was unavailable for Zn in
the installed PySCF basis library. It is a development fixture, not a model of
the enzyme reaction.

## Where to make changes

Keep the dependency direction intact:

* `weqmmm/engines/base.py`: provider protocol and `QMResult`; no OpenMM/wepy.
* `weqmmm/engines/pyscf_cpu.py`: molecular CPU PySCF finite embedding.
* `weqmmm/engines/gpu4pyscf_molecular.py`: molecular GPU finite embedding.
* `weqmmm/regions.py`: PartQMMM bridge and frozen partition.
* `weqmmm/partition.py`: clone and transform the OpenMM System.
* `weqmmm/link.py`: link placement and force projection.
* `weqmmm/force.py`: OpenMM `PythonForce` callback and force routing.
* `weqmmm/engine.py`: OpenMM Context, integrator, state and restart handling.
* `weqmmm/runner.py`: wepy segment propagation and process-local engines.
* `weqmmm/reporter.py`: atomic cycle/state artifacts.
* `weqmmm/distances.py`: geometry-only WE distances.
* `examples/` and `scripts/`: runnable workflows and audits.
* `tests/`: numerical and plumbing gates.

Do not modify wepy or PartQMMM source in `external/`. Do not add a retry loop,
silent SCF fallback, frozen walker, or error-and-continue branch. An SCF
failure is evidence about the model and must terminate the run after writing
its geometry/embedding diagnostic. Do not put density matrices in walker state.

## What has been validated

The following results were completed during development; this compact repository
contains runnable code and configurations rather than generated run outputs:

* all 15 Gate 0 prerequisite checks;
* partition charge conservation and OpenMM transformation audits;
* RHF, UHF, RKS, and UKS provider/link force checks after the approved
  `conv_tol_grad=1e-7`, DFT grid level 5, and grid-response settings;
* force mapping, force sum, energy decomposition, fatal-error propagation,
  and restart behavior;
* small-system NVE at 0.5 and 0.25 fs under the documented convergence
  criteria (the exact fourfold drift rule is deferred for later review);
* CPU and GPU runner isolation, restart, clone independence, and weight
  conservation;
* full 70-row GPU B3LYP/def2-SVP single point and 10-step MD;
* short Zn-only CPU MD/WE and full-region GPU WE plumbing.

These results establish implementation progress. They do not establish
production enzyme kinetics, converged finite embedding, or a validated reaction
coordinate.

## Todo list, in order

### A. Make the selected finite-embedding Hamiltonian defensible

1. Define candidate fixed whole-residue embedding radii (for example 1.2,
   1.6, 2.0, 2.5, and 3.0 nm) and record the exact residue-selection policy.
2. Build each partition once and save its JSON, audit, source hashes, and
   selected/omitted charge totals. Never select by individual atoms.
3. Compare QM energies, QM forces, MM reaction forces, force-sum residuals,
   Zn coordination, ligand geometry, and other observables relevant to the
   intended study across radii and nearby alternative residue selections.
4. Define acceptance criteria before running the campaign. If observables do
   not converge, stop and report that finite embedding is inadequate for the
   intended property; do not move the cutoff until a desired result appears.
5. Record cost versus radius and select the smallest demonstrably converged
   embedding for production development.

### B. Validate full-enzyme dynamics

1. Run sampled full-enzyme single points at several independent geometries,
   checking SCF convergence, force magnitudes, QM/MM finite differences,
   rotational/translation behavior, and force balance.
2. Run an enzyme-specific timestep study and document criteria before analysis.
   Keep the approved note that the criterion will be revisited later.
3. Run progressively longer GPU MD with checkpoints. Monitor temperature,
   structure, minimum QM/MM distances, force-sum ratios, SCF cycles, and
   wall-clock cost. Treat any SCF failure as a stopped run requiring diagnosis.
4. Test restart from checkpoints and verify state, configuration, partition,
   and provenance checksums.

### C. Define and validate the weighted-ensemble science

1. Define source and product states from chemically meaningful structural
   criteria. Document whether and how product walkers recycle to the source.
2. Run a longer metric pilot, freeze feature scales with its artifact, and
   justify Zn coordination, AZM sulfonamide, torsion, or other features.
3. Tune REVO characteristic and merge distances in the frozen metric units.
4. Run multiple independent WE replicates, inspect resampling, effective
   sample size, source occupancy, product flux, and weight conservation.
5. Add rate estimation only after source/product and recycling definitions are
   fixed. Use `k_AB = J_B / p_A` and `MFPT = p_A / J_B`; do not assume
   `p_A=1` without measuring it.
6. Report uncertainty and convergence across independent replicates. A short
   WE smoke test is not a rate calculation.

### D. Package and release

1. Add a clean-install test and pin CPU/GPU dependencies and external commits.
2. Make GPU tests explicit and skip-safe while preserving clear failure output.
3. Add documented Slurm examples, resource estimates, checkpoint policy, and
   output-artifact schemas.
4. Add user-facing tutorials for partition, single point, MD, restart, and WE.
5. Add CI for non-GPU tests and a manually triggered GPU validation workflow.
6. Review every public API, type annotation, error message, and docstring for
   undergraduate readability.
7. Confirm the MIT license and input-data redistribution terms with the lab,
   then create a tagged release only after the finite-embedding convergence and
   full-enzyme dynamics gates are complete.

## Deferred alternatives

Periodic Ewald QM/MM embedding remains a possible future project. It would
require a separate provider and new force/energy benchmarks, including general
triclinic support and a no-double-counting audit. ASE is an adapter/framework,
not automatically a periodic QM/MM engine; any ASE backend would need to prove
triclinic support, long-range electrostatics, and MM reaction forces before it
could replace the current provider. These alternatives should not distract
from the finite-embedding convergence campaign unless that campaign fails.

## Reporting standards

Every change should state: what changed, why, exact command/configuration,
test result, artifact path, and remaining limitation. Never call a submitted job
a pass until its exit status and output artifacts have been inspected.
