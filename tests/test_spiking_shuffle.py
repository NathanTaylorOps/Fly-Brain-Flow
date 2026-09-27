"""CI-tier tests for flybrainflow/spiking/shuffle.py -- pure NumPy, no torch, actually run and
confirmed correct (degree preservation, weight-multiset preservation, no self-loops/duplicates,
error paths) before this reached a real pytest run. Real-scale (25M-edge) performance has NOT been
measured -- see the module's own docstring; these tests only cover correctness at small scale."""

from collections import Counter

import numpy as np

from flybrainflow.spiking.shuffle import shuffle_connectivity


def _small_synthetic_graph(rng, n_nodes=30, n_edges=150):
    """A directed graph with no self-loops and no duplicate (pre, post) pairs -- matches what a
    real, already-coalesced Connectivity's edge list looks like."""
    pre = rng.integers(0, n_nodes, n_edges)
    post = rng.integers(0, n_nodes, n_edges)
    seen = set()
    keep_pre, keep_post = [], []
    for p, q in zip(pre, post):
        if p == q:
            continue
        key = (int(p), int(q))
        if key in seen:
            continue
        seen.add(key)
        keep_pre.append(p)
        keep_post.append(q)
    weight = rng.integers(1, 100, len(keep_pre))
    return np.array(keep_pre), np.array(keep_post), weight


def test_shuffle_preserves_every_nodes_in_and_out_degree_exactly():
    rng = np.random.default_rng(1)
    pre, post, weight = _small_synthetic_graph(rng)
    n = len(pre)

    new_pre, new_post, new_weight = shuffle_connectivity(pre, post, weight, rng=rng, n_swaps=5 * n)

    assert Counter(new_pre.tolist()) == Counter(pre.tolist())
    assert Counter(new_post.tolist()) == Counter(post.tolist())


def test_shuffle_preserves_the_exact_weight_multiset():
    rng = np.random.default_rng(2)
    pre, post, weight = _small_synthetic_graph(rng)
    _, _, new_weight = shuffle_connectivity(pre, post, weight, rng=rng, n_swaps=5 * len(pre))
    assert sorted(weight.tolist()) == sorted(new_weight.tolist())


def test_shuffle_introduces_no_self_loops_or_duplicate_edges():
    rng = np.random.default_rng(3)
    pre, post, weight = _small_synthetic_graph(rng)
    n = len(pre)
    new_pre, new_post, _ = shuffle_connectivity(pre, post, weight, rng=rng, n_swaps=5 * n)

    assert not np.any(new_pre == new_post), "shuffle introduced a self-loop"
    pairs = set(zip(new_pre.tolist(), new_post.tolist()))
    assert len(pairs) == n, "shuffle introduced a duplicate edge"


def test_shuffle_actually_changes_most_edges_not_just_a_few():
    rng = np.random.default_rng(4)
    pre, post, weight = _small_synthetic_graph(rng)
    n = len(pre)
    new_pre, new_post, _ = shuffle_connectivity(pre, post, weight, rng=rng, n_swaps=5 * n)
    unchanged = int(np.sum((new_pre == pre) & (new_post == post)))
    assert unchanged < n * 0.5, f"only {unchanged}/{n} edges changed -- shuffle barely did anything"


def test_shuffle_rejects_fewer_than_two_edges():
    rng = np.random.default_rng(5)
    try:
        shuffle_connectivity(np.array([1]), np.array([2]), np.array([5]), rng=rng)
    except ValueError as e:
        assert "at least 2" in str(e)
        return
    raise AssertionError("expected a ValueError for fewer than 2 edges")


def test_shuffle_rejects_mismatched_array_lengths():
    rng = np.random.default_rng(6)
    try:
        shuffle_connectivity(np.array([1, 2]), np.array([2]), np.array([5, 6]), rng=rng)
    except ValueError as e:
        assert "same length" in str(e)
        return
    raise AssertionError("expected a ValueError for mismatched array lengths")


def test_shuffle_gives_up_loudly_on_a_graph_with_no_valid_swap():
    # Two edges, 1->2 and 2->1: the only possible swap produces 1->1 and 2->2, both self-loops --
    # there is no valid swap for this graph, ever, so this must raise rather than loop forever.
    rng = np.random.default_rng(7)
    try:
        shuffle_connectivity(
            np.array([1, 2]), np.array([2, 1]), np.array([5, 6]),
            rng=rng, n_swaps=1, max_attempts_factor=5,
        )
    except RuntimeError as e:
        assert "attempts" in str(e)
        return
    raise AssertionError("expected a RuntimeError instead of hanging on an unshuffleable graph")
