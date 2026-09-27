"""CI-tier tests for flybrainflow/spiking/celltypes.py -- pure pandas/NumPy, no torch, actually run
and confirmed correct before this reached a real pytest run (same reasoning as
test_spiking_neurotransmitters.py)."""

import numpy as np
import pandas as pd

from flybrainflow.spiking.celltypes import body_ids_for_type


def _annotations() -> pd.DataFrame:
    return pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5],
        "type": ["Or42b", "LC10a", "LC10b", "MN9", "DNa02"],
    })


def test_exact_type_match_returns_the_right_body_ids():
    ids = body_ids_for_type(_annotations(), "Or42b")
    assert list(ids) == [1]


def test_regex_prefix_match_finds_a_whole_subtype_family():
    ids = body_ids_for_type(_annotations(), "LC10.*", regex=True)
    assert sorted(ids.tolist()) == [2, 3]


def test_missing_type_column_raises_with_the_real_columns_listed():
    try:
        body_ids_for_type(_annotations(), "Or42b", type_column="nonexistent")
    except KeyError as e:
        assert "bodyId" in str(e) and "type" in str(e)
        return
    raise AssertionError("expected a KeyError naming the real columns present")


def test_no_matching_rows_raises_rather_than_returning_empty_silently():
    try:
        body_ids_for_type(_annotations(), "NoSuchType")
    except ValueError as e:
        assert "NoSuchType" in str(e)
        return
    raise AssertionError("expected a ValueError when a type pattern matches nothing")


def test_alternate_body_id_column_spelling_is_auto_detected():
    ann = _annotations().rename(columns={"bodyId": "body_id"})
    ids = body_ids_for_type(ann, "MN9")
    assert list(ids) == [4]


def test_unrecognized_body_id_column_name_raises_with_real_columns_listed():
    ann = _annotations().rename(columns={"bodyId": "weird_id_col"})
    try:
        body_ids_for_type(ann, "MN9")
    except KeyError as e:
        assert "weird_id_col" in str(e)
        return
    raise AssertionError("expected a KeyError when no recognizable body-id column exists")


def test_explicit_body_id_column_override_works_with_any_name():
    ann = _annotations().rename(columns={"bodyId": "weird_id_col"})
    ids = body_ids_for_type(ann, "MN9", body_id_column="weird_id_col")
    assert list(ids) == [4]
