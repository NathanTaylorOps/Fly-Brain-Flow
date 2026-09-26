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
        """
        return Connectivity(body_ids=self.body_ids, weight_matrix=self.weight_matrix.to(device))


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

    n = len(body_ids)
    indices = torch.tensor(np.stack([post_idx, pre_idx]), dtype=torch.int64)  # post-major, see Connectivity docstring
    values = torch.tensor(np.asarray(weight), dtype=torch.int64)
    weight_matrix = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()

    return Connectivity(body_ids=body_ids, weight_matrix=weight_matrix)
