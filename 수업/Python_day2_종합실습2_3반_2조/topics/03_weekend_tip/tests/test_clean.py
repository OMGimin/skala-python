import pandas as pd
import polars as pl
import pytest

from src.clean import clean_nulls, clean_taxi_with_polars


def test_clean_nulls_fills_numeric_median_without_mutating_source() -> None:
    source = pd.DataFrame({"value": [1.0, None, 5.0], "label": ["a", "b", "c"]})

    result = clean_nulls(source, cols=["value"], strategy="median")

    assert result["value"].tolist() == [1.0, 3.0, 5.0]
    assert source["value"].isna().sum() == 1


def test_clean_nulls_drops_selected_null_rows() -> None:
    source = pd.DataFrame({"value": [1, 2], "label": ["a", None]})

    result = clean_nulls(source, cols=["label"], strategy="drop")

    assert result.to_dict("records") == [{"value": 1, "label": "a"}]


def test_clean_nulls_rejects_non_numeric_median() -> None:
    source = pd.DataFrame({"label": ["a", None]})

    with pytest.raises(TypeError, match="수치형"):
        clean_nulls(source, cols=["label"], strategy="median")


def test_clean_nulls_rejects_unknown_strategy() -> None:
    source = pd.DataFrame({"value": [1, None]})

    with pytest.raises(ValueError, match="strategy"):
        clean_nulls(source, strategy="unknown")  # type: ignore[arg-type]


def test_clean_taxi_with_polars_deduplicates_and_fills_expected_nulls() -> None:
    row = {
        "tpep_pickup_datetime": None,
        "tpep_dropoff_datetime": None,
        "PULocationID": 1,
        "DOLocationID": 2,
        "passenger_count": None,
        "RatecodeID": None,
        "payment_type": 0,
        "store_and_fwd_flag": None,
        "congestion_surcharge": None,
        "Airport_fee": None,
        "cbd_congestion_fee": None,
    }
    valid = {
        **row,
        "tpep_pickup_datetime": "2026-05-01 10:00:00",
        "tpep_dropoff_datetime": "2026-05-01 10:10:00",
        "passenger_count": 2,
    }
    frame = pl.DataFrame([valid, valid])

    result = clean_taxi_with_polars(frame)

    assert result.height == 1
    assert sum(result.null_count().row(0)) == 0
    assert result["RatecodeID"].item() == 99
    assert result["store_and_fwd_flag"].item() == "UNKNOWN"

