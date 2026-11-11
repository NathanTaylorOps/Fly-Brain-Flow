import numpy as np
import pytest
pytest.importorskip('numba')
from flybrainflow.live_data import LiveConnectome
from flybrainflow.spiking.live_engine import csr_from_edges
from flybrainflow.brains.live_connectome import LiveConnectomeBrain


def graph():
    names = ['sugar', 'loom', 'ORN_L', 'ORN_R', 'MN9', 'DNp01', 'DNp09', 'DNa02_L', 'DNa02_R']
    pops = {name: np.array([i], dtype=np.int32) for i, name in enumerate(names)}
    ptr, idx, w = csr_from_edges(np.array([1]), np.array([5]), [400], len(names))
    return LiveConnectome(np.arange(len(names)), ptr, idx, w, np.zeros((len(names), 3)),
                          np.array(names), np.array(names), pops, {})


class Field:
    def sample(self, positions):
        return np.ones(len(positions))


def step(brain, ids=(1,), odor=False):
    return brain.desired_velocities(ids, np.tile([2., 4.], (len(ids), 1)),
        np.full(len(ids), .1), 1., personalities=[None] * len(ids), odor_field=Field(), dt=.1)


def test_real_adapter_requires_spikes_to_move_and_lesion_stops_it():
    brain = LiveConnectomeBrain(graph())
    np.testing.assert_array_equal(step(brain), [[0, 0]])
    brain.set_controls(stimulus_hz=200)
    assert np.linalg.norm(step(brain)) > 0
    assert brain.neural_snapshot()['rates_hz']['DNp01'] > 0
    brain.set_controls(silenced=True)
    np.testing.assert_array_equal(step(brain), [[0, 0]])
    assert brain.neural_snapshot()['rates_hz']['DNp01'] == 0


def test_independent_state_and_odor_interface_and_forget():
    brain = LiveConnectomeBrain(graph())
    brain.set_controls(odor_enabled=True)
    step(brain, (1, 2))
    assert not np.shares_memory(brain.engines[1].v, brain.engines[2].v)
    assert brain.engines[1].weights is brain.engines[2].weights
    assert brain.last_sense[1][0] == 50
    brain.forget([1])
    assert 1 not in brain.engines and 2 in brain.engines


def test_missing_mapping_fails_instead_of_substituting():
    g = graph()
    g.populations['DNp09'] = np.array([], dtype=np.int32)
    with pytest.raises(ValueError, match='DNp09'):
        LiveConnectomeBrain(g)
