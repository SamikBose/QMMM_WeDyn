# Gate 8: timestep convergence

These acceptance criteria were written before running the trajectories.

## Initial validation system

One RHF/STO-3G QM water and one constrained three-site MM water, with full MM
reaction forces and retained inter-water LJ, in a nonperiodic box-free system.
Use the OpenMM Reference platform, Verlet, SCF energy tolerance 1e-10 Ha and
orbital-gradient tolerance 1e-7, and MM constraint tolerance 1e-8.

Compare **1 ps** at **0.5 fs** and **0.25 fs**, starting from identical position and
velocity arrays (NumPy RNG seed 17, Maxwell velocities at 100 K, initial MM
velocity constraint projection). Measure total energy every 5 fs, including t=0.
OpenMM reports the integrator's kinetic-energy convention; each trajectory's
energy error is measured from its own initial total energy.

## Criteria for this fixture

1. Every energy and coordinate must be finite; every SCF must converge. The
   callback force-sum check must pass at every force evaluation.
2. Total-energy peak-to-peak variation must be less than **0.5 kJ/mol** at either
   timestep. This is an absolute bound for this small system, not a bound scaled
   by its large negative electronic energy.
3. The absolute least-squares linear drift must be below **0.2 kJ/mol/ps** at both
   timesteps.
4. Halving the timestep must reduce peak-to-peak variation: the fine-step value
   must be at most **0.75 times** the coarse-step value, unless both are already
   below a **1e-5 kJ/mol** numerical floor.

We do not demand an exact fourfold reduction in a fitted drift slope. Report the
measured variation ratio, drift slopes, maximum deviations, SCF counts and timing.
Record failed results before asserting, and do not loosen these criteria after
seeing them. This is a deliberately bounded test of the numerical coupling.

**Revisit these criteria later**, as requested by the user. Passing this test does
not establish 1 ps B3LYP/def2-SVP stability of the 38,361-particle enzyme. The
enzyme needs its own single-point, short-trajectory, and longer NVE evaluations
with explicit reporting of the finite-embedding approximation and CPU cost.
