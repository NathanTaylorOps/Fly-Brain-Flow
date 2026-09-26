"""CI-tier tests for flybrainflow/spiking/connectivity.py -- small, hand-built edge lists with a
known expected matrix, not the real MaleCNS connectome (that's Kaggle-only, see
scripts/run_full_connectome_check.py)."""

import numpy as np
import torch

from flybrainflow.spiking.connectivity import load_connectivity


def test_edges_land_at_the_expected_post_major_matrix_positions():
    # body ids 1, 2, 3 -> sorted indices 0, 1, 2. Edges: 1->2 (w=5), 1->3 (w=-2), 2->3 (w=10).
    conn = load_connectivity(
        body_pre=np.array([1, 1, 2]),
        body_post=np.array([2, 3, 3]),
        weight=np.array([5, -2, 10]),
    )
    assert conn.n_neurons == 3
    assert list(conn.body_ids) == [1, 2, 3]
    assert conn.index_of(1) == 0
    assert conn.index_of(2) == 1
    assert conn.index_of(3) == 2

    dense = conn.weight_matrix.to_dense()
    # post-major: dense[post, pre]
    assert dense[conn.index_of(2), conn.index_of(1)] == 5
    assert dense[conn.index_of(3), conn.index_of(1)] == -2
    assert dense[conn.index_of(3), conn.index_of(2)] == 10
    # every other entry (no edge) is exactly zero
    assert dense.abs().sum().item() == 5 + 2 + 10


def test_index_of_an_unknown_body_id_raises_a_clear_keyerror_not_a_bare_indexerror():
    conn = load_connectivity(body_pre=np.array([1]), body_post=np.array([2]), weight=np.array([1]))
    try:
        conn.index_of(99)
    except KeyError as e:
        assert "99" in str(e)
        return
    raise AssertionError("expected a KeyError naming the missing body id")


def test_duplicate_pre_post_pairs_are_summed_not_overwritten():
    conn = load_connectivity(
        body_pre=np.array([1, 1]),
        body_post=np.array([2, 2]),
        weight=np.array([3, 4]),
    )
    dense = conn.weight_matrix.to_dense()
    assert dense[conn.index_of(2), conn.index_of(1)] == 7  # 3 + 4, not 4 (last-write-wins)


def test_extra_body_ids_get_a_zero_row_and_column_not_left_out_entirely():
    conn = load_connectivity(
        body_pre=np.array([1]),
        body_post=np.array([2]),
        weight=np.array([5]),
        extra_body_ids=np.array([999]),
    )
    assert conn.n_neurons == 3
    idx = conn.index_of(999)
    dense = conn.weight_matrix.to_dense()
    assert dense[idx, :].abs().sum().item() == 0
    assert dense[:, idx].abs().sum().item() == 0


def test_mismatched_array_lengths_raise_a_clear_error_not_a_silent_truncation():
    try:
        load_connectivity(body_pre=np.array([1, 2]), body_post=np.array([2]), weight=np.array([1, 2]))
    except ValueError as e:
        assert "same length" in str(e)
        return
    raise AssertionError("expected a ValueError for mismatched array lengths")


def test_weight_matrix_dtype_is_int64_not_float():
    # Real MaleCNS synapse weights are integer counts (confirmed live 2026-09-26: min=1, max=2591)
    # -- this stays int64 end to end, never silently promoted to float, since dynamics.py's whole
    # determinism argument depends on it (see that module's own docstring).
    conn = load_connectivity(body_pre=np.array([1]), body_post=np.array([2]), weight=np.array([7]))
    assert conn.weight_matrix.dtype == torch.int64
