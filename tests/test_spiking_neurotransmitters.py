"""CI-tier tests for flybrainflow/spiking/neurotransmitters.py -- pure NumPy, no torch, so these
(unlike test_spiking_connectivity/dynamics/simulator.py) were actually run and confirmed correct
before ever reaching a real pytest run, the same way the rest of this package's NumPy-only id
mapping logic was verified in Step 2's own build (see docs/JOURNAL.md)."""

import numpy as np

from flybrainflow.spiking.neurotransmitters import apply_signs, resolve_signs


def test_resolve_signs_follows_shiu_et_al_convention_case_insensitively():
    body_ids = np.array([1, 2, 3, 4])
    nts = np.array(["GABA", "acetylcholine", "Glutamate", "dopamine"])
    signs = resolve_signs(body_ids, nts)
    assert signs == {1: -1, 2: 1, 3: -1, 4: 1}


def test_resolve_signs_unknown_transmitter_defaults_to_excitatory_by_default():
    signs = resolve_signs(np.array([1]), np.array(["some_future_transmitter"]))
    assert signs == {1: 1}


def test_resolve_signs_unknown_sign_none_raises_instead_of_guessing():
    try:
        resolve_signs(np.array([1]), np.array(["some_future_transmitter"]), unknown_sign=None)
    except ValueError as e:
        assert "some_future_transmitter" in str(e)
        return
    raise AssertionError("expected a ValueError when unknown_sign=None and a transmitter isn't recognized")


def test_resolve_signs_rejects_duplicate_body_ids():
    try:
        resolve_signs(np.array([1, 1]), np.array(["GABA", "acetylcholine"]))
    except ValueError as e:
        assert "duplicate" in str(e)
        return
    raise AssertionError("expected a ValueError for duplicate body_ids -- one row per neuron is required")


def test_apply_signs_applies_the_presynaptic_neurons_sign_to_every_outgoing_edge():
    # Neuron 1 is inhibitory (GABA), neuron 2 and 3 are excitatory.
    signs = {1: -1, 2: 1, 3: 1}
    body_pre = np.array([1, 1, 2, 3])
    weight = np.array([10, 20, 5, 7])
    signed = apply_signs(body_pre, weight, signs)
    assert list(signed) == [-10, -20, 5, 7]


def test_apply_signs_raises_on_a_body_pre_with_no_known_sign():
    try:
        apply_signs(np.array([1, 99]), np.array([10, 20]), {1: -1})
    except ValueError as e:
        assert "99" in str(e)
        return
    raise AssertionError("expected a ValueError for an edge whose body_pre has no resolved sign")


def test_apply_signs_default_sign_fills_gaps_when_explicitly_allowed():
    signed = apply_signs(np.array([1, 99]), np.array([10, 20]), {1: -1}, default_sign=1)
    assert list(signed) == [-10, 20]


def test_apply_signs_matches_a_naive_per_edge_lookup_at_moderate_scale():
    # The real function is vectorized (searchsorted, not a per-edge Python loop) for speed at real
    # MaleCNS scale (25M+ edges) -- this checks the vectorized result against a naive, obviously
    # correct per-edge reference implementation, so a future optimization can't silently change
    # the actual answer.
    rng = np.random.default_rng(0)
    n = 2000
    body_pre = rng.integers(0, 50, n)
    weight = rng.integers(1, 100, n)
    signs = {i: (-1 if i % 3 == 0 else 1) for i in range(50)}
    result = apply_signs(body_pre, weight, signs)
    expected = np.array([weight[k] * signs[int(body_pre[k])] for k in range(n)])
    assert np.array_equal(result, expected)
