"""CI-tier tests for flybrainflow/spiking/dynamics.py's LIF update rule -- hand-worked expected
values, not statistical/approximate checks, since every operation here is exact integer arithmetic
(that's the whole point -- see dynamics.py's own module docstring)."""

import torch

from flybrainflow.spiking.dynamics import LIFParams, LIFState, step


def _empty_weight_matrix(n: int) -> torch.Tensor:
    """A valid n x n sparse int64 matrix with no edges at all -- for tests that only care about a
    neuron's own decay/refractory behavior, not synaptic input."""
    indices = torch.zeros((2, 0), dtype=torch.int64)
    values = torch.zeros((0,), dtype=torch.int64)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


def test_membrane_decays_toward_zero_with_no_input_by_the_exact_integer_ratio():
    params = LIFParams()  # leak 9/10, threshold 100 -- defaults, see dynamics.py
    state = LIFState(
        membrane=torch.tensor([[50]], dtype=torch.int64),
        refractory_remaining=torch.tensor([[0]], dtype=torch.int64),
        spikes=torch.tensor([[0]], dtype=torch.int64),
    )
    weight_matrix = _empty_weight_matrix(1)
    zero_input = torch.zeros((1, 1), dtype=torch.int64)

    state = step(state, weight_matrix, zero_input, params)
    assert state.membrane.item() == 45  # floor(50 * 9 / 10) = floor(45.0) = 45, exact
    assert state.spikes.item() == 0

    state = step(state, weight_matrix, zero_input, params)
    assert state.membrane.item() == 40  # floor(45 * 9 / 10) = floor(40.5) = 40


def test_strong_external_input_fires_the_neuron_and_resets_it_into_refractory():
    params = LIFParams(threshold=100, refractory_ticks=2, reset_value=0)
    state = LIFState.zeros(n_flies=1, n_neurons=1, params=params)
    weight_matrix = _empty_weight_matrix(1)
    strong_input = torch.tensor([[150]], dtype=torch.int64)  # above threshold in one step

    state = step(state, weight_matrix, strong_input, params)
    assert state.spikes.item() == 1
    assert state.membrane.item() == params.reset_value
    assert state.refractory_remaining.item() == 2


def test_refractory_neuron_ignores_new_input_entirely_and_cannot_re_fire():
    params = LIFParams(threshold=100, refractory_ticks=2, reset_value=0)
    # Manually place the neuron mid-refractory (as if it fired one tick ago).
    state = LIFState(
        membrane=torch.tensor([[0]], dtype=torch.int64),
        refractory_remaining=torch.tensor([[2]], dtype=torch.int64),
        spikes=torch.tensor([[0]], dtype=torch.int64),
    )
    weight_matrix = _empty_weight_matrix(1)
    huge_input = torch.tensor([[999999]], dtype=torch.int64)  # would fire instantly if not refractory

    state = step(state, weight_matrix, huge_input, params)
    assert state.spikes.item() == 0, "a refractory neuron must not fire no matter how strong the input is"
    assert state.membrane.item() == 0, "a refractory neuron's membrane is frozen, not integrating input"
    assert state.refractory_remaining.item() == 1, "refractory count ticks down by exactly one"

    # One more step should clear refractory (2 -> 1 -> 0) and let it fire normally again.
    state = step(state, weight_matrix, huge_input, params)
    assert state.refractory_remaining.item() == 0
    state = step(state, weight_matrix, huge_input, params)
    assert state.spikes.item() == 1, "once refractory clears, a strong input should fire it again"


def test_activity_never_exceeds_its_guard_rail_under_an_adversarial_drive():
    # Mirrors ToyBrain's own test_hidden_activity_never_exceeds_its_guard_rail_under_an_adversarial_drive
    # (tests/test_toy.py) -- same concern (docs/PLAN.md's M1 risk section: "a few wrong signs
    # across tens of millions of connections could make a brain run away or go dead").
    #
    # Real bug caught here, fixed: the first version of this test used
    # threshold=10**6, activity_max=1000 -- LIFParams' own validator correctly rejects that
    # (threshold above activity_max means the neuron could never fire, which is a
    # misconfiguration, not a guard rail). Fixing the params surfaced a deeper problem with an
    # *positive*-drive version of this test: LIFParams requires activity_max >= threshold, so any
    # positive input strong enough to reach the activity_max clamp is, by that same requirement,
    # also strong enough to cross threshold -- meaning the neuron spikes and resets to 0 the same
    # tick, which masks whatever the clamp did. A massive positive adversarial drive can never
    # actually prove the clamp is holding; it only proves the neuron fires and resets.
    #
    # The clamp is only observable, unmasked, on the *negative* (inhibitory) side: a large enough
    # negative drive can never cross a positive threshold, so it never spikes-and-resets, and the
    # membrane after clamping is exactly what gets returned -- this is also the more realistic
    # real-world failure shape ("a brain runs away or goes dead"; this is the "goes dead" half).
    params = LIFParams()  # defaults: threshold=100, activity_min=-1000, activity_max=1000
    state = LIFState.zeros(n_flies=1, n_neurons=1, params=params)
    weight_matrix = _empty_weight_matrix(1)
    adversarial_input = torch.tensor([[-(10**9)]], dtype=torch.int64)  # absurdly large and negative, on purpose

    for _ in range(50):
        state = step(state, weight_matrix, adversarial_input, params)
        assert state.membrane.item() == params.activity_min, (
            f"expected the clamp to hold the membrane exactly at activity_min "
            f"({params.activity_min}); got {state.membrane.item()} -- if this ever fails, the "
            "clamp in dynamics.step() has been weakened or removed"
        )
        assert state.spikes.item() == 0  # a massive negative drive can never cross a positive threshold


def test_synaptic_input_from_a_spiking_presynaptic_neuron_reaches_the_postsynaptic_neuron():
    # Two neurons, one edge 0 -> 1 with weight 150 (post-major: weight_matrix[1, 0] = 150).
    indices = torch.tensor([[1], [0]], dtype=torch.int64)  # [post], [pre]
    values = torch.tensor([150], dtype=torch.int64)
    weight_matrix = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()
    params = LIFParams(threshold=100, refractory_ticks=2, reset_value=0)

    state = LIFState.zeros(n_flies=1, n_neurons=2, params=params)
    # Tick 0: drive neuron 0 hard enough to fire; neuron 1 gets nothing yet.
    state = step(state, weight_matrix, torch.tensor([[200, 0]], dtype=torch.int64), params)
    assert state.spikes.tolist() == [[1, 0]]

    # Tick 1: no external input at all -- neuron 1 should still fire, purely from neuron 0's spike
    # propagating through the weight-150 synapse (150 >= threshold 100). Neuron 0 is refractory.
    state = step(state, weight_matrix, torch.zeros((1, 2), dtype=torch.int64), params)
    assert state.spikes.tolist() == [[0, 1]]


def test_overflow_guard_trips_at_construction_not_per_tick():
    # Moved 2026-09-27 night (see the 2026-09-27 review pass finding this used to check the wrong,
    # much larger quantity -- whole-matrix |weight| sum -- on every single tick instead of the real
    # per-post-neuron worst case, once, at construction): a single self-loop edge with weight right
    # at dynamics._EXACT_INT_CEILING now must raise OverflowError out of SpikingSimulator's own
    # constructor, before any step ever runs -- not out of step()/_synaptic_input() on first use.
    import numpy as np

    from flybrainflow.spiking.connectivity import load_connectivity
    from flybrainflow.spiking.dynamics import _EXACT_INT_CEILING
    from flybrainflow.spiking.simulator import SpikingSimulator

    connectivity = load_connectivity(
        body_pre=np.array([0]), body_post=np.array([0]), weight=np.array([_EXACT_INT_CEILING])
    )
    assert connectivity.max_row_abs_weight_sum == _EXACT_INT_CEILING, (
        "sanity check: this connectome's one self-loop edge should BE the per-row worst case"
    )

    try:
        SpikingSimulator(connectivity, LIFParams())
    except OverflowError:
        return
    raise AssertionError("expected an OverflowError from SpikingSimulator's constructor, but none was raised")


def test_overflow_guard_does_not_trip_on_an_ordinary_small_connectome():
    # The flip side of the test above -- a connectome nowhere near the ceiling must construct fine.
    # Guards against a future edit accidentally making the check too strict (e.g. comparing the
    # wrong quantity again, or an off-by-one on the boundary).
    import numpy as np

    from flybrainflow.spiking.connectivity import load_connectivity
    from flybrainflow.spiking.simulator import SpikingSimulator

    connectivity = load_connectivity(
        body_pre=np.array([0, 1]), body_post=np.array([1, 0]), weight=np.array([150, 150])
    )
    SpikingSimulator(connectivity, LIFParams())  # must not raise


def test_lifparams_rejects_an_unreachable_threshold():
    try:
        LIFParams(threshold=1000, activity_max=100)  # threshold above the guard rail -- can never fire
    except ValueError as e:
        assert "activity_max" in str(e)
        return
    raise AssertionError("expected a ValueError for a threshold above activity_max")
