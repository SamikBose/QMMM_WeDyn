import json
from pathlib import Path
import numpy as np
import pytest
from weqmmm.distances import GeometryDistance
from weqmmm.engine import MDState


def test_metric_wrap_and_torsion_continuity():
    metric = GeometryDistance(((0,3),),(.3,),(.1,1.,1.),((0,1,2,3),))
    pos = np.array([[.1,.1,0],[0,0,0],[0,0,.1],[.1,0,.2]])
    box = np.array([[2.,0,0],[-.6,1.9,0],[-.6,-.8,1.7]])
    state = {'positions':pos,'box_vectors':box}
    wrapped = pos.copy()
    wrapped[0] += box[1]
    wrapped[3] -= box[2]
    assert np.allclose(metric.image(state),metric.image({'positions':wrapped,'box_vectors':box}),atol=1e-12)
    assert metric.distance(state,state) == 0
    assert abs(np.linalg.norm(metric.image(state)[1:])-1) < 1e-12
    with pytest.raises(AssertionError):
        GeometryDistance(((0,1),),(.1,),(0.,))


@pytest.mark.parametrize('periodic',[False,True])
def test_nuclear_state_portable_restart(tmp_path,periodic):
    state = MDState(np.arange(18.).reshape(6,3),np.ones((6,3)),np.eye(3) if periodic else None,.4,800)
    path = tmp_path/'state.npz'
    state.save(path)
    loaded = MDState.load(path)
    assert np.array_equal(loaded.positions_nm,state.positions_nm)
    assert np.array_equal(loaded.velocities_nm_ps,state.velocities_nm_ps)
    assert np.array_equal(loaded.box_vectors_nm,state.box_vectors_nm)
    assert loaded.time_ps == state.time_ps and loaded.step_count == state.step_count
