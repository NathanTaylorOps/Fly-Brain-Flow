"""CI-tier integration tests for SpikingSimulator -- ties connectivity.py and dynamics.py together
against small, hand-built connectomes. The real ~164,740-neuron MaleCNS connectome is Kaggle-only,
never loaded here -- see scripts/run_full_connectome_check.py for that."""

import torch

from flybrainflow.spiking.connectivity import load_connectivity
from flybrainflow.spiking.dynamics import LIFParams
from flybrainflow.spiking.simulator import SpikingSimulator


def _two_neuron_feedforward_connectivity():
    # body id 1 -> body id 2, weight 150 -- same shape as test_spiking_dynamics.py's own
    # propagation test, but built through load_connectivity this time (real integration path).
    return load_connectivity(body_pre=[1], body_post=[2], weight=[150])


def test_a_strong_drive_propagates_through_one_synapse_with_a_one_tick_delay():
    conn = _two_neuron_feedforward_connectivity()
    params = LIFParams(threshold=100, refractory_ticks=2)
    sim = SpikingSimulator(conn, params, n_flies=1)

    pre, post = conn.index_of(1), conn.index_of(2)
    drive = torch.zeros((1, 2), dtype=torch.int64)
    drive[0, pre] = 200  # well above threshold

    spikes_tick0 = sim.step(drive)
    assert spikes_tick0[0, pre].item() == 1
    assert spikes_tick0[0, post].item() == 0

    no_drive = torch.zeros((1, 2), dtype=torch.int64)
    spikes_tick1 = sim.step(no_drive)
    assert spikes_tick1[0, pre].item() == 0  # refractory now
    assert spikes_tick1[0, post].item() == 1  # fired purely from the propagated synaptic input


def test_step_rejects_external_input_with_the_wrong_shape():
    conn = _two_neuron_feedforward_connectivity()
    sim = SpikingSimulator(conn, n_flies=1)
    wrong_shape = torch.zeros((1, 5), dtype=torch.int64)  # conn has 2 neurons, not 5
    try:
        sim.step(wrong_shape)
    except ValueError as e:
        assert "shape" in str(e)
        return
    raise AssertionError("expected a ValueError for a mismatched external_input shape")


def test_multiple_flies_run_independently_with_no_cross_talk():
    conn = _two_neuron_feedforward_connectivity()
    params = LIFParams(threshold=100, refractory_ticks=2)
    sim = SpikingSimulator(conn, params, n_flies=2)
    pre, post = conn.index_of(1), conn.index_of(2)

    drive = torch.zeros((2, 2), dtype=torch.int64)
    drive[0, pre] = 200  # only fly 0 gets driven; fly 1 gets nothing, ever

    spikes_tick0 = sim.step(drive)
    assert spikes_tick0[0, pre].item() == 1, "fly 0's own driven neuron should fire"
    assert spikes_tick0[1].sum().item() == 0, "fly 1 got no input and must show no activity at all"

    no_drive = torch.zeros((2, 2), dtype=torch.int64)
    spikes_tick1 = sim.step(no_drive)
    assert spikes_tick1[0, post].item() == 1, "fly 0's propagated spike should still arrive on schedule"
    assert spikes_tick1[1].sum().item() == 0, "fly 1 must still show no activity -- no leakage from fly 0"


def test_run_returns_a_stacked_history_of_the_right_shape():
    conn = _two_neuron_feedforward_connectivity()
    sim = SpikingSimulator(conn, n_flies=1)
    history = sim.run(n_steps=5)
    assert tuple(history.shape) == (5, 1, conn.n_neurons)


def test_reset_returns_the_simulator_to_a_clean_zero_state():
    conn = _two_neuron_feedforward_connectivity()
    params = LIFParams(threshold=100, refractory_ticks=2)
    sim = SpikingSimulator(conn, params, n_flies=1)
    pre = conn.index_of(1)
    drive = torch.zeros((1, 2), dtype=torch.int64)
    drive[0, pre] = 200
    sim.step(drive)
    assert sim.t_step == 1
    assert sim.state.spikes.sum().item() > 0

    sim.reset()
    assert sim.t_step == 0
    assert sim.state.spikes.sum().item() == 0
    assert sim.state.membrane.sum().item() == 0
