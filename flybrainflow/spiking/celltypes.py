"""Cell-type name -> body-id lookup, against an already-downloaded annotation table (a pandas
DataFrame from `body-annotations-male-cns-v1.0-minconf-0.5.feather`, per `data_config.py`).

Deliberately separate from `connectivity.py`: `Connectivity` carries only `body_ids`, no type
information at all, by design (so the spiking core stays connectome-source-agnostic -- see
`docs/M1_PLAN.md`'s Step 3 pre-work list for the reasoning). This is glue code that joins a type
name back to real body ids, for the two things that actually need one -- Step 3's "stimulate
sugar-GRNs, read out MN9" calibration protocol, and Step 4's real sensory/motor wiring -- not a
change to `Connectivity` itself.

Mirrors `scripts/pin_dataset.py`'s own `NeuronCriteria(type=..., regex=...)` pattern, applied to a
downloaded dataframe instead of a live neuPrint query (this runs against the bulk-pulled table, not
neuPrint, since that's what Step 3/4's Kaggle scripts actually have on disk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def body_ids_for_type(
    annotations: pd.DataFrame,
    type_pattern: str,
    *,
    type_column: str = "type",
    regex: bool = False,
    body_id_column: str | None = None,
) -> np.ndarray:
    """Body ids whose `type_column` matches `type_pattern` -- an exact string match by default, or
    a regex if `regex=True` (e.g. `body_ids_for_type(ann, "LC10.*", regex=True)` for the LC10
    subtype family, same reasoning as `data_config.LC10_SUBTYPE_PREFIX`).

    Raises `KeyError` if `type_column` isn't a real column in `annotations` (names the actual
    columns present, so a caller with a schema mismatch sees what's really there rather than a bare
    `KeyError` on the column name alone) and `ValueError` if the pattern matches nothing at all --
    a calibration or wiring step silently proceeding with zero neurons for "sugar-GRN" or "MN9"
    would produce a confusing downstream failure (or, worse, a falsely-passing shuffled-vs-real
    comparison with nothing actually stimulated) far from this obvious, checkable cause.

    `body_id_column`: which column holds the body id, if it isn't one of the spellings
    `_find_body_id_column` already tries (`bodyId`, `body_id`, `bodyid`) -- pass this explicitly
    once the real schema is confirmed live, rather than relying on the guess list.
    """
    if type_column not in annotations.columns:
        raise KeyError(
            f"{type_column!r} is not a column in this annotation table -- columns present: "
            f"{list(annotations.columns)}"
        )
    if regex:
        mask = annotations[type_column].astype(str).str.match(type_pattern, na=False)
    else:
        mask = annotations[type_column] == type_pattern

    if not mask.any():
        raise ValueError(
            f"no rows matched type_column={type_column!r} pattern={type_pattern!r} (regex={regex}) "
            f"-- this table has {len(annotations)} rows; double-check the exact type string/prefix "
            "against the real data before assuming this cell type is absent (see "
            "data_config.KNOWN_GAPS for the ones already confirmed genuinely absent)"
        )

    id_col = body_id_column or _find_body_id_column(annotations)
    return annotations.loc[mask, id_col].to_numpy(dtype=np.int64)


def _find_body_id_column(annotations: pd.DataFrame) -> str:
    """The real MaleCNS annotation table's body-id column name has not been confirmed live in this
    codebase as of 2026-09-27 (found in the review pass alongside the neurotransmitter-schema gap --
    see docs/JOURNAL.md) -- tries the two spellings actually seen across this project's other
    pulled tables (`body_pre`/`body_post` on the weights table) and neuPrint's own convention
    (`bodyId`), in that order, and fails loudly with the real column list rather than guessing
    silently if none match.
    """
    for candidate in ("bodyId", "body_id", "bodyid"):
        if candidate in annotations.columns:
            return candidate
    raise KeyError(
        "could not find a body-id column in this annotation table (tried 'bodyId', 'body_id', "
        f"'bodyid') -- columns actually present: {list(annotations.columns)} -- confirm the real "
        "column name against the live file and pass it explicitly if this guess list is wrong"
    )
