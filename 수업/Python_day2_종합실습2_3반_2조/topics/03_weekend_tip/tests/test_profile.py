from pathlib import Path

import polars as pl
import pytest

from src.data_profile import (
    pandas_profile,
    polars_profile,
    timed_load_pandas,
    timed_load_polars,
    validate_profiles,
)


def test_profiles_agree_and_timed_loaders_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"x": [1, None, 1], "label": ["a", "b", "a"]}).write_parquet(path)

    pandas_frame, pandas_seconds = timed_load_pandas(path)
    polars_frame, polars_seconds = timed_load_polars(path)
    pandas_info = pandas_profile(pandas_frame, pandas_seconds)
    polars_info = polars_profile(polars_frame, polars_seconds)

    validate_profiles(pandas_info, polars_info)
    assert pandas_info["rows"] == 3
    assert pandas_info["null_cells"] == 1
    assert pandas_info["duplicate_rows"] == 1
    assert pandas_seconds >= 0
    assert polars_seconds >= 0


def test_validate_profiles_rejects_mismatch() -> None:
    pandas_info = {
        "rows": 1, "columns": 1, "duplicate_rows": 0,
        "null_cells": 0, "null_by_column": {"x": 0},
    }
    polars_info = {**pandas_info, "rows": 2}

    with pytest.raises(ValueError, match="rows"):
        validate_profiles(pandas_info, polars_info)
