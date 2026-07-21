from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

import src.analysis_pipeline as analysis_pipeline
from src.pipeline import run_pipeline


def raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(8):
        pickup = datetime(2026, 5, 1 if index < 4 else 2, 8 + index)
        distance = float(1 + index) if index < 4 else float(5 + index)
        base = 10.0 + distance
        rate = 0.12 + index * 0.01 if index < 4 else 0.08 + index * 0.005
        tip = base * rate
        rows.append(
            {
                "VendorID": 1,
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + timedelta(minutes=10 + index),
                "passenger_count": 1,
                "trip_distance": distance,
                "RatecodeID": 1,
                "store_and_fwd_flag": "N",
                "PULocationID": index + 1,
                "DOLocationID": index + 2,
                "payment_type": 1,
                "fare_amount": base,
                "extra": 0.0,
                "mta_tax": 0.5,
                "tip_amount": tip,
                "tolls_amount": 0.0,
                "improvement_surcharge": 1.0,
                "total_amount": base + tip,
                "congestion_surcharge": 0.0,
                "Airport_fee": 0.0,
                "cbd_congestion_fee": 0.75,
            }
        )
    return rows


def test_analysis_pipeline_with_injected_output_paths(
    tmp_path: Path, monkeypatch
) -> None:
    raw_path = tmp_path / "raw.parquet"
    pl.DataFrame(raw_rows()).write_parquet(raw_path)

    def fake_train(frame, model_path, metrics_path):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"model")
        metrics_path.write_text("{}")
        return {
            "model": "test-model",
            "sample_rows": len(frame),
            "accuracy": 0.5,
            "f1": 0.5,
            "roc_auc": 0.5,
        }

    monkeypatch.setattr(analysis_pipeline, "train_and_evaluate", fake_train)
    result = analysis_pipeline.run_analysis(
        raw_path,
        processed_path=tmp_path / "processed" / "analysis.parquet",
        table_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        model_dir=tmp_path / "models",
        report_path=tmp_path / "report.md",
        html_report_path=tmp_path / "report.html",
        template_dir=Path("templates"),
    )

    assert result["analysis_rows"] == 8
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "tables" / "distance_ttest.json").exists()
    assert (tmp_path / "figures" / "hourly_tip_interactive.html").exists()


def test_preprocessing_pipeline_with_small_parquet(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    pl.DataFrame(raw_rows()).write_parquet(raw_path)

    result = run_pipeline(raw_path, tmp_path / "processed", tmp_path / "tables")

    assert result["validation"]["shape_equal"] is True
    assert (tmp_path / "processed" / "yellow_tripdata_2026-05_cleaned.csv").exists()
    assert (tmp_path / "tables" / "preprocessing_report.json").exists()
