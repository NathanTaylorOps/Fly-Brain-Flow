"""Excitatory/inhibitory sign resolution for raw synapse-count weights, from per-neuron predicted
neurotransmitter identity. `connectivity.load_connectivity` deliberately takes weight as-is, signed,
and does not do this itself -- this module is the separate, later step its docstring pointed to.

Convention adopted, and why: this project uses the exact same rule Shiu et al. (2023/2024, "A Drosophila
computational brain model reveals sensorimotor processing", the paper Step 3's calibration gate
reproduces) used to build the model whose published result this project is checking itself against --
see docs/JOURNAL.md's 2026-09-27 research entry for the sourcing. Using a different, perhaps more
"correct" convention here would confound the calibration gate: a failure could then mean either "my
simulator is wrong" or "my sign convention differs from theirs", and the whole point of the gate is
to isolate the first from the second (PLAN.md's own "one variable at a time").

The rule, exactly as they stated it:
  - Sign is a property of the PRE-SYNAPTIC NEURON, not of an individual synapse -- a neuron
    releases one primary small-molecule transmitter, so every one of its outgoing edges gets the
    same sign.
  - GABAergic and glutamatergic neurons are INHIBITORY. Every other predicted transmitter
    (acetylcholine, and the neuromodulators: dopamine, serotonin, octopamine) is treated as
    EXCITATORY in this binary model. They say so explicitly: this "generally models GABAergic and
    cholinergic neurons reasonably well, but other neurons (e.g., dopaminergic or serotonergic)
    will be modeled less well" -- a real, acknowledged simplification, not an oversight; adopted
    here for the same reason, not because it's the most biologically accurate choice available.

What this module does NOT know, on purpose: the real column name(s)/schema of
`body-neurotransmitters-male-cns-v1.0.feather` -- that file has been pulled (Step 1) but never
actually inspected for its real columns in this codebase. `resolve_signs` takes plain parallel
arrays (body_id, predicted_transmitter) rather than a dataframe, so whatever script reads the real
feather file is responsible for picking the right column(s) out of it -- and should print the
columns it found before assuming, the same pattern `pin_dataset.py` already uses for the annotation
table's own columns. Confirming that schema live is scripts/run_calibration_gate.py's job, not this
module's.
"""

from __future__ import annotations

import numpy as np

# Lowercase, exactly matching Shiu et al.'s rule (see module docstring). -1 = inhibitory,
# +1 = excitatory. A transmitter not in this map is *not* silently defaulted -- see
# `resolve_signs`'s `unknown_sign` parameter -- because a real predicted-transmitter value this
# project hasn't seen before deserves an explicit decision, not a guess.
SIGN_CONVENTION: dict[str, int] = {
    "gaba": -1,
    "glutamate": -1,
    "acetylcholine": 1,
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
    # Seen in the broader flyconnectome transmitter panel (not confirmed present in MaleCNS's own
    # predictions -- see funkelab/synister_malecns, which lists 7: acetylcholine, dopamine, gaba,
    # glutamate, histamine, octopamine, serotonin) -- included so a real value isn't rejected just
    # because Shiu et al.'s own paper (FlyWire, an earlier prediction panel) didn't need to name it.
    # Excitatory by the same "everything except GABA/glutamate" rule.
    "histamine": 1,
}


def resolve_signs(
    body_ids: np.ndarray,
    predicted_transmitter: np.ndarray,
    *,
    sign_convention: dict[str, int] = SIGN_CONVENTION,
    unknown_sign: int | None = 1,
) -> dict[int, int]:
    """One sign (+1 or -1) per neuron, keyed by body_id -- Shiu et al.'s rule (see module
    docstring): GABA/glutamate -> inhibitory, everything else -> excitatory.

    `predicted_transmitter` entries are matched case-insensitively (real data has been seen with
    inconsistent casing across sources) after stripping whitespace; a `None`/NaN/empty entry is
    treated as unknown, same as a real string this map doesn't recognize.

    `unknown_sign`: what to assign a neuron whose predicted transmitter isn't in `sign_convention`
    at all (a typo, a transmitter this map hasn't seen, or a missing/NaN prediction). Defaults to
    `+1` (excitatory), matching Shiu et al.'s own "everything except GABA/glutamate is excitatory"
    rule applied consistently -- but this is a real, silent-by-default choice, so it's a keyword
    argument, not baked in unconditionally: pass `unknown_sign=None` to instead raise on any
    unrecognized value, if a caller wants to fail loudly on missing/bad neurotransmitter data
    rather than quietly assuming excitatory.

    Raises `ValueError` if `body_ids` contains a duplicate -- this function assigns one sign per
    *neuron*, so a caller passing per-synapse rows (multiple rows for the same body_id) instead of
    one row per neuron has a real bug, not something to resolve by an arbitrary pick.
    """
    body_ids = np.asarray(body_ids)
    predicted_transmitter = np.asarray(predicted_transmitter, dtype=object)
    if len(body_ids) != len(predicted_transmitter):
        raise ValueError(
            f"body_ids ({len(body_ids)}) and predicted_transmitter ({len(predicted_transmitter)}) "
            "must be the same length -- one row per neuron"
        )
    unique_ids, counts = np.unique(body_ids, return_counts=True)
    dupes = unique_ids[counts > 1]
    if len(dupes):
        raise ValueError(
            f"body_ids has {len(dupes)} duplicate id(s) (e.g. {int(dupes[0])}) -- resolve_signs "
            "expects one row per neuron (a per-neuron majority-vote transmitter call already made), "
            "not one row per synapse"
        )

    signs: dict[int, int] = {}
    unresolved: list[str] = []
    for bid, nt in zip(body_ids, predicted_transmitter):
        key = None if nt is None else str(nt).strip().lower()
        if key in sign_convention:
            signs[int(bid)] = sign_convention[key]
        elif unknown_sign is not None:
            signs[int(bid)] = unknown_sign
        else:
            unresolved.append(f"{bid}: {nt!r}")

    if unresolved:
        preview = "; ".join(unresolved[:5])
        more = f" (+{len(unresolved) - 5} more)" if len(unresolved) > 5 else ""
        raise ValueError(
            f"{len(unresolved)} neuron(s) have a predicted transmitter not in sign_convention and "
            f"unknown_sign=None was passed, so nothing was assumed: {preview}{more} -- either add "
            "the transmitter to sign_convention or pass unknown_sign=1/-1 to pick a default"
        )
    return signs


def apply_signs(
    body_pre: np.ndarray,
    weight: np.ndarray,
    signs: dict[int, int],
    *,
    default_sign: int | None = None,
) -> np.ndarray:
    """Signed weight = `weight[i] * signs[body_pre[i]]`, per Shiu et al.'s rule that sign belongs
    to the pre-synaptic neuron, applied to every one of its outgoing edges.

    `default_sign`: what to use for an edge whose `body_pre` isn't a key in `signs` at all (e.g. a
    neuron with no neurotransmitter prediction row in whatever table `signs` was built from).
    Defaults to `None`, meaning raise `ValueError` naming the first missing id -- an edge with no
    sign decided for it is exactly the kind of silent gap `load_connectivity`'s own integer-weight
    validation was added to stop happening elsewhere in this package (see the 2026-09-27 review
    pass); pass an explicit `default_sign=1` or `-1` only if that gap is expected and its size has
    already been checked.
    """
    body_pre = np.asarray(body_pre)
    weight = np.asarray(weight)
    if len(body_pre) != len(weight):
        raise ValueError(f"body_pre ({len(body_pre)}) and weight ({len(weight)}) must be the same length")

    # Vectorized via searchsorted against a sorted (id, sign) table, same technique
    # `connectivity.load_connectivity` uses for its own id -> index lookup -- matters here for the
    # same reason: this runs over every edge (25M+ at real MaleCNS scale), so a per-edge Python
    # loop/dict-lookup is not an option.
    if signs:
        items = sorted(signs.items())
        sign_ids = np.array([k for k, _ in items], dtype=np.int64)
        sign_vals = np.array([v for _, v in items], dtype=np.int64)
    else:
        sign_ids = np.zeros(0, dtype=np.int64)
        sign_vals = np.zeros(0, dtype=np.int64)

    idx = np.searchsorted(sign_ids, body_pre)
    idx_clipped = np.clip(idx, 0, max(len(sign_ids) - 1, 0))
    found = (len(sign_ids) > 0) & (idx < len(sign_ids)) & (sign_ids[idx_clipped] == body_pre)

    sign_arr = np.empty(len(body_pre), dtype=np.int64)
    sign_arr[found] = sign_vals[idx_clipped[found]]

    if not found.all():
        missing_mask = ~found
        n_missing = int(missing_mask.sum())
        missing_example = int(body_pre[missing_mask][0])
        if default_sign is None:
            raise ValueError(
                f"{n_missing} edge(s) have a body_pre not present in signs (e.g. {missing_example}) "
                "-- pass default_sign=1/-1 if this is expected (e.g. a neuron with no "
                "neurotransmitter prediction at all), otherwise this is a real gap between the "
                "weight table and the neurotransmitter table that should be understood before "
                "it's papered over"
            )
        sign_arr[missing_mask] = default_sign

    return weight * sign_arr
