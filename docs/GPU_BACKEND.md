# Molecular GPU backend

`QMConfig.backend = 'gpu4pyscf_molecular'` selects GPU4PySCF; the default is
`pyscf_cpu`. OpenMM can still use its CPU platform. The QM provider has no
OpenMM or wepy imports. Both providers use the same fixed finite embedding,
direct SCF (no density fitting), XC, basis, unit conversions, grid level, full
grid response, cache, segment lifecycle, and fatal-SCF policy.

This is **not** periodic QM embedding. MM PME remains periodic; missing
long-range QM/MM electrostatics remains a model limitation. Point charges are
supported; Gaussian-smearing requests are rejected because GPU4PySCF 1.8.1's
molecular nuclear energy/gradient methods do not implement them.

## Environment

The CPU conda environment remains intact. `.venv-gpu` uses its interpreter and
system packages, with the GPU-specific versions installed locally. GPU4PySCF
requires a NumPy version compatible with CuPy, so this environment uses NumPy
2.2.6 while the CPU environment keeps 2.4.6. `pip check` and all 15 prerequisite
checks passed in both. Pins are in `requirements-gpu.txt`; the full resolved
Record the installed GPU environment versions and CUDA device details for each
research run.

On an allocated GPU node:

```bash
source scripts/activate_gpu.sh
export WEQMMM_GPU_TESTS=1
python -m pytest tests/test_gpu_engine.py -q -s
python -m pytest tests/test_gpu_dynamics.py -q -s -x
```

The activation script loads CUDA/12.4.0 and preserves Slurm's device assignment.
The engine requires exactly one visible GPU per process. WE uses spawned
`GPUWorker` processes; each selects one token from the scheduler's original
`CUDA_VISIBLE_DEVICES` before importing GPU4PySCF. The worker index must map to
an allocated GPU. The first enzyme WE job requests two GPUs for two workers.
No upstream wepy source was modified.

## Validation and job order

The initial L40S job, 17413872, passed six tests (188.30 seconds):

- RHF/UHF/RKS/UKS isolated limits, finite-difference QM/MM forces, force balance,
  and rotational invariance, using the original numerical thresholds.
- CPU/GPU energy and force agreement, exact cache reuse, fresh segment history,
  and fatal SCF failure without retry.
- OpenMM force-group energy decomposition and short DFT propagation.

The logged one-cycle SCF failure is the deliberate negative-control test.
It is not a failed production segment.

CPU regression after extracting shared provider hooks: 30 passed. GPU worker
assignment plus CPU runner checks: four passed. The full 1 ps GPU NVE runs at
0.5 and 0.25 fs and GPU runner identity/isolation/clone checks run separately.
For this tiny three-atom QM fixture, GPU overhead can dominate CPU execution;
the NVE gate is a correctness test, not a speed benchmark.

Submitted jobs and dependencies for the initial full-region campaign:

| Job | Purpose | Dependency |
| --- | --- | --- |
| 17413872 | Initial GPU numerical gates | None; passed |
| 17413946 | 70-row B3LYP/def2-SVP single point | Initial gates passed; completed |
| 17413947 | 1 ps timestep study and GPU runner checks | NVE passed; bitwise runner identity failed |
| 17414572 | 10-step 70-row MD, analytic snapshots, pilot metric | Completed successfully |
| 17414582 | 4 walkers, 2 GPUs, 3 cycles, 2 steps/segment | Cancelled after failed prerequisite; never executed |
| 17423312 | 4 walkers, 2 GPUs, 3 cycles, 2 steps/segment | Completed successfully after runner prerequisites passed |

Single-point result: -4278.602747997007 Ha, 20 SCF cycles, 708 basis functions,
4,646 embedding sites; 156.16 s through SCF and 45.21 s for gradients on one
L40S. Force-sum ratio 9.63e-11, below 1e-6. This is the entire 70-row region,
including four links, with the existing full solvated OpenMM system.

All jobs use new run directories and preserve failed outputs. An `afterok`
dependency prevented the first WE job from executing after a failed prerequisite;
the replacement job 17423312 completed successfully. A submitted job is not a
passing validation result: inspect its exit status, XML, and committed run
artifacts; retain those artifacts outside this compact source repository.

### Exact identity failure and independent diagnostic

Job 17413947 passed the unchanged NVE criteria: energy peak-to-peak variation
was 0.32373684 kJ/mol at 0.5 fs and 0.08238450 kJ/mol at 0.25 fs, each over
1 ps. The following runner test failed `np.array_equal` on coordinates from
independent GPU calculations. The suite stopped; isolation and clone tests
were not reached. This failure remains recorded, not converted to a pass.

Diagnostic job 17423242 reproduced non-bitwise results **without wepy**.
Three independent five-step direct OpenMM runs differed by up to
5.6899e-16 nm in position and 1.9451e-13 nm/ps in velocity. Independent RHF
single points differed by 1.42e-14 Ha in energy and 1.64e-12 Ha/Bohr in QM
gradient; B3LYP repeats also differed in their last digits. Raw outputs and
the report should be saved with the run artifacts.

The user subsequently **approved and implemented** GPU-specific absolute
trajectory tolerances of 1e-12 nm and 1e-10 nm/ps, with zero relative tolerance.
Walker weights remain bitwise identical, CPU trajectory tests remain exact,
and physics thresholds remain unchanged. See the handoff guide for the
explicit amendment. The original failed artifacts are preserved.

Job 17423311 **passed all three GPU runner checks in 26.75 s**: direct-runner
agreement, alternating-walker isolation/pickle restart, and clone independence.
Clone divergence must exceed the approved position tolerance, so last-digit
differences alone cannot pass that test. Exact CPU runner checks passed again
(four tests, 2.79 s). The already-passed NVE gate is unchanged.
Job 17423312 **completed successfully in 53 min 45 s** using the completed MD
pilot from job 17414572: 4 walkers, 2 GPU workers, 3 cycles, 2 steps/segment
at 0.5 fs. All 36 SCFs converged. Propagation preserved each incoming weight
bitwise; all cycle weight sums were 1.0. The largest saved segment-end
force-sum ratio was 3.94e-10, below 1e-6. Actual clone/merge events occurred in
cycles 1 and 2; cycle 3 recorded only keep decisions. Evidence:
`runs/hca2_azm/def2svp-gpu-we-17423312/PASSED.json`, cycle JSON/checkpoints,
and the scheduler exit status should be recorded with the run artifacts.

## Enzyme WE scope

`examples/hca2_azm/run_we.py` accepts a config, portable nuclear state, and
frozen metric JSON. `scripts/build_metric.py` builds geometry features for
Zn coordination, AZM sulfonamide N-H, and cosine/sine of N1-S1-C1-N3. Feature
scales come from pilot state standard deviations with a fixed 0.01 floor.
Input/checkpoint checksums, feature indices, scales, and REVO parameters are
saved. Short pilot scales are for software testing only; production metric
calibration and REVO parameter selection require a longer scientific pilot.

These WE runs are equilibrium software tests. Product/source boundaries and
recycling are not defined, and no kinetic rates are reported. A later rate
estimator must use k_AB = J_B / p_A and MFPT = p_A / J_B.
