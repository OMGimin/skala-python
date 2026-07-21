from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from src.features import build_analysis_dataset, create_tip_features


def raw_frame() -> pl.DataFrame:
    pickup = datetime(2026, 5, 2, 10)
    valid = {
        "payment_type": 1,
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": pickup + timedelta(minutes=15),
        "trip_distance": 2.0,
        "fare_amount": 10.0,
        "tip_amount": 2.0,
        "total_amount": 12.0,
        "PULocationID": 1,
        "DOLocationID": 2,
    }
    cash = {**valid, "payment_type": 2, "trip_distance": 7.0}
    return pl.DataFrame([valid, valid, cash])


def test_create_tip_features_filters_and_builds_features() -> None:
    result, audit = create_tip_features(raw_frame())

    assert result.height == 1
    assert result["tip_rate"].item() == 0.2
    assert result["distance_group"].item() == "short_lt_5mi"
    assert result["high_tip"].item() == 1
    assert audit["step"].to_list()[0] == "remove_exact_duplicates"
    assert audit["rows"].to_list()[-1] == 1


def test_build_analysis_dataset_writes_parquet(tmp_path: Path) -> None:
    source = tmp_path / "raw.parquet"
    output = tmp_path / "processed" / "analysis.parquet"
    raw_frame().write_parquet(source)

    result, _ = build_analysis_dataset(source, output)

    assert output.exists()
    assert pl.read_parquet(output).height == result.height == 1

