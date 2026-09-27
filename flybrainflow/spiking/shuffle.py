"""Degree-preserving edge shuffle, for Step 3's calibration-gate control condition: "a
degree-preserving shuffled version of the same connectome (same node degrees, randomised edges)"
(docs/M1_PLAN.md's Step 3). Nothing like this existed anywhere in this repo before the 2026-09-27
review pass found Step 3's own written protocol assumed it.

Decisions made here, written down because Step 3's pre-work list said this needed deciding, not
guessing:
  - Algorithm: the classic double-edge-swap / configuration-model randomization for directed
    graphs. Repeatedly pick two distinct edges (a -> b) and (c -> d) and swap their endpoints to
    (a -> d) and (c -> b), skipping (retrying) a swap that would create a self-loop (a == d or
    c == b) or a duplicate edge that already exists. This preserves each node's in-degree AND
    out-degree EXACTLY (not just their aggregate distribution) -- every edge keeps its original
    tail or head, just recombined with a different partner.
  - Weights move WITH the edge they're attached to, not resampled -- edge (a -> b)'s weight is
    still the weight now carried by whichever new edge (a -> d) it became. This preserves the exact
    multiset of edge weights in the whole graph, on top of preserving degrees, rather than only
    preserving degrees and letting weights land arbitrarily.
  - Duplicate (pre, post) pairs from `load_connectivity`'s own summing-on-coalesce are handled
    upstream, before this function ever runs: this operates on and returns one row per edge in
    whatever edge list it's given (already-summed or not), and does not itself re-aggregate.

What this does NOT do: this is correctness-first, not yet benchmarked at real MaleCNS scale (25.5M
edges) -- see docs/M1_PLAN.md's Step 3 pre-work note. The edge-membership check below is a Python
set of int64-encoded (pre, post) keys; this is correct and reasonably fast at the small/synthetic
scale the tests below exercise, but its real cost (time and memory) at 25M edges has not been
measured. Measure it for real on Kaggle before assuming this is fast enough to run inside the
actual calibration gate at full scale -- if it's too slow, the fix is almost certainly batching the
swap attempts as vectorized NumPy operations (propose many candidate swaps at once, then reject the
invalid ones as a boolean mask) rather than the one-swap-at-a-time loop this function uses now.
"""

from __future__ import annotations

import numpy as np


def shuffle_connectivity(
    body_pre: np.ndarray,
    body_post: np.ndarray,
    weight: np.ndarray,
    *,
    rng: np.random.Generator,
    n_swaps: int | None = None,
    max_attempts_factor: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a new `(body_pre, body_post, weight)` triple, same length and same multiset of
    weights as the input, with edges randomized subject to every node's in-degree and out-degree
    being preserved exactly (see module docstring for the algorithm and what it preserves).

    `n_swaps`: how many successful swaps to perform. Defaults to `10 * n_edges` -- enough that,
    for a graph of this density, the result is not recognizably close to the original ordering
    (this is the standard rule of thumb for edge-swap randomization: a small constant multiple of
    the edge count). Pass a larger multiple for a graph with many high-degree nodes, where valid
    swaps become harder to find and more attempts are needed to actually randomize thoroughly.

    `max_attempts_factor`: gives up (raises `RuntimeError`) if `max_attempts_factor * n_swaps`
    total attempts (successful + rejected) are made without reaching `n_swaps` successes -- a
    graph with too few edges, or an oddly clustered degree sequence, can make valid swaps
    genuinely hard to find; this fails loudly rather than looping effectively forever.

    Raises `ValueError` on fewer than 2 edges (nothing to swap).

    Implementation note: only `body_post` is ever reassigned -- a swap of (a, b) and (c, d) into
    (a, d) and (c, b) always keeps each edge's original `body_pre` value, moving only where it
    points. The returned `body_pre` is therefore always identical to the input array (this is what
    makes out-degree-per-node trivially exact: node `a`'s set of outgoing edges is the same rows it
    always had, just now pointing elsewhere).
    """
    body_pre = np.asarray(body_pre, dtype=np.int64).copy()
    body_post = np.asarray(body_post, dtype=np.int64).copy()
    weight = np.asarray(weight).copy()
    n = len(body_pre)
    if not (n == len(body_post) == len(weight)):
        raise ValueError(
            f"body_pre ({n}), body_post ({len(body_post)}), and weight ({len(weight)}) must all "
            "be the same length"
        )
    if n < 2:
        raise ValueError(f"need at least 2 edges to shuffle, got {n}")

    if n_swaps is None:
        n_swaps = 10 * n

    # Encode each edge as one int64 key for O(1) set membership, rather than a (pre, post) tuple --
    # meaningfully cheaper at real scale. Body ids are >= 0 in every real MaleCNS pull seen so far
    # (see data_config.py) and the multiplier below only needs to exceed the largest id actually
    # present, not some fixed global constant -- decided from the real data at call time, not
    # hardcoded, so this stays correct however large real body ids turn out to be.
    max_id = int(max(body_pre.max(initial=0), body_post.max(initial=0))) + 1
    edge_keys = set((body_pre * max_id + body_post).tolist())

    successes = 0
    attempts = 0
    max_attempts = max_attempts_factor * n_swaps
    while successes < n_swaps:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only completed {successes}/{n_swaps} swaps after {attempts} attempts -- this "
                "graph's degree sequence may make valid swaps too rare to find this way; consider "
                "a smaller n_swaps or investigate whether the graph has an unusual structure "
                "(e.g. very few edges relative to nodes, or a few extremely high-degree nodes)"
            )
        attempts += 1

        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = int(body_pre[i]), int(body_post[i])
        c, d = int(body_pre[j]), int(body_post[j])

        if a == c or b == d:
            # Swapping would either leave both edges unchanged (a==c and b==d, i.e. i==j in
            # effect) or isn't a meaningful degree-preserving recombination for this pair --
            # skip and try a different pair rather than attempt a no-op swap.
            continue
        if a == d or c == b:
            continue  # would create a self-loop

        new_key_1 = a * max_id + d
        new_key_2 = c * max_id + b
        if new_key_1 in edge_keys or new_key_2 in edge_keys:
            continue  # would create a duplicate edge that already exists elsewhere in the graph

        # Perform the swap: edge i's weight now travels with (a -> d); edge j's weight travels
        # with (c -> b). Each node's in/out-degree is unchanged -- a and c keep the same
        # out-degree (each still has exactly one outgoing edge where it had one before), b and d
        # keep the same in-degree, for the same reason.
        edge_keys.discard(a * max_id + b)
        edge_keys.discard(c * max_id + d)
        edge_keys.add(new_key_1)
        edge_keys.add(new_key_2)
        body_post[i] = d
        body_post[j] = b
        successes += 1

    return body_pre, body_post, weight
