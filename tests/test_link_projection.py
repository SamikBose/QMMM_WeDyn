"""Regression test for link.py. MUST assert, not print."""
import numpy as np, pytest
from weqmmm.link import place_link_atom, project_link_force, LINK_H_BOND_NM

def test_projection_matches_finite_difference():
    rng = np.random.default_rng(3)
    RQ = rng.normal(size=3)*0.1
    RM = RQ + np.array([0.15, 0.03, -0.04])
    A = rng.normal(size=3); B = rng.normal(size=(3,3)); B = B + B.T
    E = lambda L: float(A @ L + 0.5 * L @ B @ L)
    F = lambda L: -(A + B @ L)
    Etot = lambda q, m: E(place_link_atom(q, m))

    fQ, fM = project_link_force(F(place_link_atom(RQ, RM)), RQ, RM)
    h = 1e-7
    numQ = np.array([-(Etot(RQ+h*np.eye(3)[k], RM)-Etot(RQ-h*np.eye(3)[k], RM))/(2*h) for k in range(3)])
    numM = np.array([-(Etot(RQ, RM+h*np.eye(3)[k])-Etot(RQ, RM-h*np.eye(3)[k]))/(2*h) for k in range(3)])
    assert np.abs(fQ-numQ).max() < 1e-5, f"QM1 projection wrong: {np.abs(fQ-numQ).max()}"
    assert np.abs(fM-numM).max() < 1e-5, f"MM1 projection wrong: {np.abs(fM-numM).max()}"

def test_force_is_conserved():
    RQ = np.zeros(3); RM = np.array([0.15, 0., 0.])
    FL = np.array([1., 2., -3.])
    fQ, fM = project_link_force(FL, RQ, RM)
    assert np.allclose(fQ+fM, FL)

def test_mm1_contribution_is_not_negligible():
    """If someone 'simplifies' by dropping f_mm1, this documents the damage."""
    RQ = np.zeros(3); RM = np.array([0.15, 0., 0.])
    FL = np.array([1., 5., -3.])
    _, fM = project_link_force(FL, RQ, RM)
    assert np.linalg.norm(fM) > 0.1*np.linalg.norm(FL)

def test_rejects_wrong_image():
    with pytest.raises(ValueError, match="periodic images"):
        place_link_atom(np.zeros(3), np.array([6.0, 0., 0.]))

def test_rejects_coincident():
    with pytest.raises(ValueError, match="coincide"):
        place_link_atom(np.zeros(3), np.zeros(3))
