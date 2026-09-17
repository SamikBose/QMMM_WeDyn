#!/usr/bin/env python3
"""
Environment gate for weqmmm. Run this FIRST, in the fresh conda env, before writing
any package code. It fails loudly and returns nonzero if anything is missing or if
an API this project depends on has changed.

    python scripts/check_prereqs.py
    python scripts/check_prereqs.py --gpu      # also require gpu4pyscf + a visible GPU
"""
import argparse
import importlib
import sys
import traceback

FAILURES = []
NOTES = []


def check(name):
    def deco(fn):
        try:
            msg = fn()
            print(f"  PASS  {name}" + (f"  [{msg}]" if msg else ""))
        except Exception as exc:
            FAILURES.append((name, exc))
            print(f"  FAIL  {name}")
            print("        " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
        return fn
    return deco


print("=" * 78)
print("weqmmm prerequisite check")
print("=" * 78)
print(f"python {sys.version.split()[0]}")

# ---------------------------------------------------------------- imports
print("\n[1] imports")


@check("numpy")
def _():
    import numpy
    return numpy.__version__


@check("openmm")
def _():
    import openmm
    v = openmm.version.version
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) < (8, 6):
        raise RuntimeError(f"openmm {v} is too old; PythonForce needs >= 8.6")
    return v


@check("pyscf")
def _():
    import pyscf
    return pyscf.__version__


@check("mdtraj (PartQMMM dependency)")
def _():
    import mdtraj
    return mdtraj.__version__


@check("wepy")
def _():
    import wepy
    return getattr(wepy, "__version__", "unknown")


@check("PartQMMM importable as qmmm_partition")
def _():
    import qmmm_partition
    for attr in ("load_topology_and_charges", "build_qm_region_spec",
                 "partition_frame", "QM_REGIONS", "minimum_image_displacement"):
        if not hasattr(qmmm_partition, attr):
            raise RuntimeError(f"PartQMMM is missing {attr}")
    return f"{len(qmmm_partition.QM_REGIONS)} built-in regions"


# ---------------------------------------------------------------- OpenMM API
print("\n[2] OpenMM API contracts")


@check("openmm.PythonForce exists")
def _():
    import openmm
    if not hasattr(openmm, "PythonForce"):
        raise RuntimeError("no PythonForce: wrong OpenMM build")
    return ""


@check("PythonForce.setParticles subsets State and scatters forces")
def _():
    import numpy as np
    import openmm
    from openmm import unit
    seen = {}

    def prov(state):
        p = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        seen["n"] = p.shape[0]
        return 0.0, -100.0 * p

    s = openmm.System()
    for _i in range(4):
        s.addParticle(1.0)
    f = openmm.PythonForce(prov)
    f.setParticles([1, 3])
    s.addForce(f)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    pos = np.array([[.1, 0, 0], [0, .2, 0], [0, 0, .3], [.1, .1, .1]])
    ctx.setPositions(pos * unit.nanometer)
    F = ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    if seen["n"] != 2:
        raise RuntimeError(f"callback saw {seen['n']} particles, expected 2")
    exp = np.zeros_like(pos)
    exp[[1, 3]] = -100.0 * pos[[1, 3]]
    if not np.allclose(F, exp):
        raise RuntimeError("forces were not scattered to the right global indices")
    return "subset semantics confirmed"


@check("PythonForce callback is the live object (warm starts persist)")
def _():
    import numpy as np
    import openmm
    from openmm import unit

    class P:
        def __init__(self):
            self.n = 0

        def __call__(self, state):
            self.n += 1
            p = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            return 0.0, np.zeros_like(p)

    p = P()
    s = openmm.System()
    s.addParticle(1.0)
    s.addForce(openmm.PythonForce(p))
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0]] * unit.nanometer)
    ctx.getState(getEnergy=True)
    if p.n != 1:
        raise RuntimeError("OpenMM did not invoke our object; state will not persist")
    return ""


@check("callback exception surfaces as OpenMMException (type is erased)")
def _():
    import openmm
    from openmm import unit

    def boom(state):
        raise RuntimeError("SENTINEL")

    s = openmm.System()
    s.addParticle(1.0)
    s.addForce(openmm.PythonForce(boom))
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0]] * unit.nanometer)
    try:
        ctx.getState(getEnergy=True)
    except openmm.OpenMMException as exc:
        if "SENTINEL" not in str(exc):
            raise RuntimeError("message not preserved through the callback boundary")
        return "message preserved, Python type lost"
    raise RuntimeError("exception was swallowed -- SCF failures would go unnoticed")


@check("force groups isolate an energy term")
def _():
    import numpy as np
    import openmm
    from openmm import unit
    s = openmm.System()
    s.addParticle(1.0)
    f = openmm.PythonForce(lambda st: (7.0, np.zeros((1, 3))))
    f.setForceGroup(31)
    s.addForce(f)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0]] * unit.nanometer)
    e = ctx.getState(getEnergy=True, groups={31}).getPotentialEnergy()
    if abs(e.value_in_unit(unit.kilojoule_per_mole) - 7.0) > 1e-9:
        raise RuntimeError("force-group energy decomposition is broken")
    return ""


@check("NonbondedForce exception chargeProd survives particle-charge zeroing")
def _():
    import openmm
    from openmm import unit
    nb = openmm.NonbondedForce()
    nb.addParticle(0.5, 0.3, 0.5)
    nb.addParticle(-0.5, 0.3, 0.5)
    nb.addException(0, 1, -0.2083, 0.3, 0.4)
    nb.setParticleParameters(0, 0.0, 0.3, 0.5)
    nb.setParticleParameters(1, 0.0, 0.3, 0.5)
    qq = nb.getExceptionParameters(0)[2].value_in_unit(unit.elementary_charge ** 2)
    if abs(qq) < 1e-9:
        NOTES.append("OpenMM now auto-updates exception chargeProd; "
                     "partition.py can drop its manual exception fix")
        return "behaviour CHANGED -- see notes"
    return "trap confirmed; partition.py MUST fix exceptions by hand"


# ---------------------------------------------------------------- PySCF API
print("\n[3] PySCF QM/MM API contracts")


@check("qmmm.mm_charge accepts radii= and unit=")
def _():
    import inspect
    from pyscf import qmmm
    params = inspect.signature(qmmm.mm_charge).parameters
    for p in ("radii", "unit"):
        if p not in params:
            raise RuntimeError(f"mm_charge has no '{p}' keyword; API changed")
    return "radii=, unit= present"


@check("grad_hcore_mm + grad_nuc_mm reproduce finite differences")
def _():
    import numpy as np
    from pyscf import gto, scf, qmmm
    BOHR = 0.52917721092
    mmc = np.array([[2.9, .1, .4], [3.6, .6, -.2], [-2.6, -.5, .9]])
    mmq = np.array([-0.834, 0.417, 0.417])
    atoms = [("O", (0, 0, .117)), ("H", (0, .757, -.468)), ("H", (0, -.757, -.468))]

    def build(mm):
        mol = gto.M(atom=atoms, basis="sto-3g", unit="Angstrom", verbose=0)
        mf = qmmm.mm_charge(scf.RHF(mol), mm, mmq, unit="Angstrom")
        mf.conv_tol = 1e-12
        return mf

    mf = build(mmc)
    mf.kernel()
    g = mf.nuc_grad_method()
    dm = np.asarray(mf.make_rdm1())
    dmt = dm[0] + dm[1] if dm.ndim == 3 else dm
    g_mm = g.grad_hcore_mm(dmt) + g.grad_nuc_mm()

    h = 2e-4
    fd = np.zeros_like(mmc)
    for i in range(mmc.shape[0]):
        for j in range(3):
            up = mmc.copy(); up[i, j] += h
            dn = mmc.copy(); dn[i, j] -= h
            fd[i, j] = (build(up).kernel() - build(dn).kernel()) / (2 * h)
    fd *= BOHR
    err = np.abs(g_mm - fd).max()
    if err > 1e-6:
        raise RuntimeError(f"MM gradient disagrees with FD by {err:.2e} Hartree/Bohr")
    return f"max|analytic-FD| = {err:.1e}"


@check("Newton's third law holds for the QM/MM gradient")
def _():
    import numpy as np
    from pyscf import gto, scf, qmmm
    mol = gto.M(atom=[("O", (0, 0, .117)), ("H", (0, .757, -.468)), ("H", (0, -.757, -.468))],
                basis="sto-3g", unit="Angstrom", verbose=0)
    mm = np.array([[2.9, .1, .4], [3.6, .6, -.2]])
    q = np.array([-0.5, 0.5])
    mf = qmmm.mm_charge(scf.RHF(mol), mm, q, unit="Angstrom")
    mf.conv_tol = 1e-12
    mf.kernel()
    g = mf.nuc_grad_method()
    dm = np.asarray(mf.make_rdm1())
    dmt = dm[0] + dm[1] if dm.ndim == 3 else dm
    tot = g.kernel().sum(0) + (g.grad_hcore_mm(dmt) + g.grad_nuc_mm()).sum(0)
    if np.abs(tot).max() > 1e-8:
        raise RuntimeError(f"sum of gradients = {tot}, expected ~0")
    return f"|sum| = {np.abs(tot).max():.1e}"


# ---------------------------------------------------------------- optional GPU
args = argparse.ArgumentParser()
args.add_argument("--gpu", action="store_true")
opts = args.parse_args()

if opts.gpu:
    print("\n[4] GPU backend")

    @check("gpu4pyscf imports")
    def _():
        import gpu4pyscf
        return getattr(gpu4pyscf, "__version__", "unknown")

    @check("cupy sees a device")
    def _():
        import cupy
        n = cupy.cuda.runtime.getDeviceCount()
        if n == 0:
            raise RuntimeError("no CUDA device visible")
        return f"{n} device(s)"

    @check("gpu4pyscf molecular QM/MM available")
    def _():
        m = importlib.import_module("gpu4pyscf.qmmm.itrf")
        if not hasattr(m, "mm_charge"):
            raise RuntimeError("no molecular mm_charge in gpu4pyscf.qmmm.itrf")
        NOTES.append("periodic QM/MM is deferred; general-triclinic support and "
                     "its independent validation gates are not part of this release")
        return "molecular finite-embedding API present"

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 78)
if NOTES:
    print("NOTES:")
    for n in NOTES:
        print("  * " + n)
    print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED. Do not start the build.")
    for name, exc in FAILURES:
        print(f"  - {name}: {exc}")
    sys.exit(1)
print("All prerequisite checks passed. Proceed to STAGE 1.")
sys.exit(0)
