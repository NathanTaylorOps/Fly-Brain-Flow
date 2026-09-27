"""Turns a raw edge list (pre-synaptic neuron, post-synaptic neuron, weight) into one sparse weight
matrix plus a stable neuron-id <-> matrix-index mapping. This is the only place in the spiking
scaffold that knows about "body IDs" at all -- `dynamics.py` and `simulator.py` only ever see plain
integer indices `0..n-1`, so the same code runs unchanged whether `n` is a few dozen (the CI-tier
synthetic connectome) or 164,740 (the real, Kaggle-only MaleCNS pull).

Deliberately does NOT resolve excitatory/inhibitory sign from a neurotransmitter table, and does
NOT filter/join against `flybrainflow/data_config.py`'s `REQUIRED_NEURON_TYPES`/`KNOWN_GAPS` --
both are Step 4 (wiring specific senses/motors to specific neurons) concerns. This module takes
whatever signed weight it's given (positive = excitatory, negative = inhibitory) and builds a
matrix from it; deciding what sign a real MaleCNS synapse should carry is a separate, later step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Connectivity:
    """One connectome's weight matrix, indexed 0..n-1, plus the mapping back to real body IDs.

    `weight_matrix` is a coalesced sparse COO int64 tensor of shape `(n_neurons, n_neurons)` --
    `weight_matrix[post, pre]` is the total signed synaptic weight from neuron `pre` onto neuron
    `post` (post-major so `weight_matrix @ spike_vector` directly gives each neuron's total
    incoming input in one sparse matrix-vector product, without an extra transpose every tick).
    """

    body_ids: np.ndarray  # int64, shape (n_neurons,) -- body_ids[i] is neuron i's real body ID
    weight_matrix: torch.Tensor  # sparse COO int64, shape (n_neurons, n_neurons), coalesced
    max_row_abs_weight_sum: int  # see load_connectivity's own docstring -- the real per-row overflow-guard quantity

    @property
    def n_neurons(self) -> int:
        return len(self.body_ids)

    def index_of(self, body_id: int) -> int:
        """The matrix index for a real body ID -- raises KeyError with the actual id in the
        message (not a bare numpy IndexError) if it isn't in this connectivity at all, since a
        caller asking "where is neuron X" almost always wants to know *which* id was wrong.
        """
        idx = np.searchsorted(self.body_ids, body_id)
        if idx >= len(self.body_ids) or self.body_ids[idx] != body_id:
            raise KeyError(f"body id {body_id} is not in this Connectivity's {self.n_neurons} neurons")
        return int(idx)

    def to(self, device: str | torch.device) -> "Connectivity":
        """A copy of this Connectivity with its weight matrix moved to `device` (e.g. `"cuda"`) --
        `Connectivity` is frozen (see class docstring reasoning: it's identity/topology, not
        mutable simulation state), so this returns a new instance rather than moving in place.
        `body_ids` stays a plain numpy array regardless of device -- it's only ever used for
        id<->index bookkeeping on the CPU side, never inside a per-step tensor op.
        `max_row_abs_weight_sum` is topology/weight-derived, not device-derived, so it carries over
        unchanged -- moving a matrix to a GPU doesn't change what it contains.
        """
        return Connectivity(
            body_ids=self.body_ids,
            weight_matrix=self.weight_matrix.to(device),
            max_row_abs_weight_sum=self.max_row_abs_weight_sum,
        )


def load_connectivity(
    body_pre: np.ndarray,
    body_post: np.ndarray,
    weight: np.ndarray,
    *,
    extra_body_ids: np.ndarray | None = None,
) -> Connectivity:
    """Build a `Connectivity` from parallel edge arrays -- e.g. the `body_pre`/`body_post`/`weight`
    columns straight off `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather`
    (confirmed live 2026-09-26, see `flybrainflow/data_config.py`), or a hand-built synthetic
    connectome for testing. `weight` is taken as-is, signed -- this function does not infer or
    apply excitatory/inhibitory sign itself (see this module's own docstring).

    `extra_body_ids`: neurons that should get a matrix index even though they have no edges at all
    (e.g. a sensory or motor neuron Step 4 needs to inject into or read from, that happens to have
    no synapses in whatever edge subset was loaded). Without this, a neuron with zero edges simply
    never appears -- fine for Step 2's own scaffold-only scope, but Step 4 will need it.

    Duplicate (pre, post) pairs are summed, not overwritten -- this matches what neuPrint's own
    bulk connectivity export already does (one row per pre/post pair, pre-aggregated), but summing
    on load rather than assuming it means a connectome from a different source that
    *hasn't* pre-aggregated duplicate edges still produces the correct total weight per pair.

    Also computes `max_row_abs_weight_sum` -- the largest, over every post-synaptic neuron, of the
    sum of |weight| over that neuron's own incoming edges. This is the actual worst-case quantity
    `dynamics._synaptic_input`'s float64-exact-integer overflow guard needs to check against (every
    pre-synaptic neuron feeding into one post-synaptic neuron firing on the very same tick), not the
    whole matrix's total |weight| sum (a much larger, unrelated number that happened to stay safely
    under the ceiling at MaleCNS's real scale by luck, not because it verified the actual claim --
    found in the 2026-09-27 review pass, fixed 2026-09-27 night). Computed once here, from the
    pre-torch NumPy arrays, rather than recomputed from the sparse tensor on every simulation tick
    (`SpikingSimulator.__init__` checks it once against the ceiling at construction time instead --
    see `dynamics.py`'s and `simulator.py`'s own docstrings).
    """
    if not (len(body_pre) == len(body_post) == len(weight)):
        raise ValueError(
            f"body_pre ({len(body_pre)}), body_post ({len(body_post)}), and weight "
            f"({len(weight)}) must all be the same length -- one row per edge"
        )

    all_ids = np.concatenate(
        [np.asarray(body_pre), np.asarray(body_post)]
        + ([np.asarray(extra_body_ids)] if extra_body_ids is not None else [])
    )
    body_ids = np.unique(all_ids)  # sorted, deduplicated -- np.unique guarantees both

    # searchsorted against the sorted, unique id array is the id -> index lookup, vectorized over
    # every edge at once rather than a per-edge dict lookup (matters once n_edges is 25 million).
    pre_idx = np.searchsorted(body_ids, body_pre)
    post_idx = np.searchsorted(body_ids, body_post)

    weight_arr = np.asarray(weight)
    # This whole spiking package's design rests on exact integer arithmetic, nothing approximate
    # (see dynamics.py's own docstring) -- a non-integer weight cast straight to int64 would be
    # silently truncated toward zero (4.85 -> 4) rather than rejected, undermining that guarantee
    # with no warning. Found in the 2026-09-27 review pass: the real feather column is documented
    # as already-integer, but nothing enforced it here, so a future data pull with a differently
    # typed (e.g. float64-with-nulls) weight column would corrupt silently instead of failing loudly.
    if not np.array_equal(weight_arr, weight_arr.astype(np.int64)):
        raise ValueError(
            "weight must be all integers -- flybrainflow.spiking is built on exact integer "
            "arithmetic throughout (see dynamics.py's module docstring); a fractional weight "
            "would be silently truncated rather than handled correctly"
        )

    n = len(body_ids)
    indices = torch.tensor(np.stack([post_idx, pre_idx]), dtype=torch.int64)  # post-major, see Connectivity docstring
    values = torch.tensor(weight_arr, dtype=torch.int64)
    weight_matrix = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    max_row_abs_weight_sum = _max_row_abs_weight_sum(post_idx, weight_arr, n)

    return Connectivity(body_ids=body_ids, weight_matrix=weight_matrix, max_row_abs_weight_sum=max_row_abs_weight_sum)


def _max_row_abs_weight_sum(post_idx: np.ndarray, weight_arr: np.ndarray, n: int) -> int:
    """The real quantity `dynamics._synaptic_input`'s overflow guard needs (see
    `load_connectivity`'s own docstring): the largest, over every post-synaptic neuron, of the sum
    of |weight| over that neuron's own incoming edges. Split out as its own plain-NumPy function
    (rather than inlined in `load_connectivity`) so it's independently testable without needing
    torch at all beyond the module-level import -- see `tests/test_spiking_connectivity.py`'s tests
    for this function specifically, added 2026-09-27 night alongside the fix that introduced it.

    Vectorized with `np.bincount` rather than a per-edge Python loop/groupby -- matters at 25M+
    edges, same reasoning as `load_connectivity`'s own `searchsorted` id lookups. `minlength=n`
    covers a neuron with zero incoming edges (bincount would otherwise just omit it from the output
    array rather than error, but explicit `minlength` keeps this correct even in the edge case where
    every edge happens to share one post index).
    """
    if n == 0:
        return 0
    row_abs_sums = np.bincount(post_idx, weights=np.abs(weight_arr).astype(np.float64), minlength=n)
    return int(row_abs_sums.max())
