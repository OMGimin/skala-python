from datetime import datetime
from pathlib import Path

import polars as pl

from src.eda import daily_summary, numeric_summary, payment_summary, write_eda


def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tpep_pickup_datetime": [datetime(2026, 5, 1, 10), datetime(2026, 5, 1, 11)],
            "trip_distance": [1.0, 3.0],
            "total_amount": [10.0, 20.0],
            "tip_amount": [2.0, 4.0],
            "payment_type": [1, 1],
        }
    )


def test_eda_summaries() -> None:
    frame = sample_frame()

    assert "trip_distance" in numeric_summary(frame).columns
    assert daily_summary(frame)["trip_count"].item() == 2
    payment = payment_summary(frame)
    assert payment["trip_count"].item() == 2
    assert payment["ratio"].item() == 1.0


def test_write_eda_creates_expected_files(tmp_path: Path) -> None:
    write_eda(sample_frame(), tmp_path)

    assert (tmp_path / "eda_numeric_summary.csv").exists()
    assert (tmp_path / "eda_daily_summary.csv").exists()
    assert (tmp_path / "eda_payment_type.csv").exists()

