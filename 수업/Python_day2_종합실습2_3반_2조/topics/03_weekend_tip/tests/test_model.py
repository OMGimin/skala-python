from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.model import build_model_pipeline, train_and_evaluate


def model_frame(rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [float(index % 10 + 1) for index in range(rows)],
            "fare_amount": [float(10 + index % 20) for index in range(rows)],
            "trip_duration_min": [float(5 + index % 30) for index in range(rows)],
            "pickup_hour": [index % 24 for index in range(rows)],
            "day_of_week": [index % 7 + 1 for index in range(rows)],
            "PULocationID": [index % 5 + 1 for index in range(rows)],
            "DOLocationID": [index % 7 + 1 for index in range(rows)],
            "high_tip": [index % 2 for index in range(rows)],
        }
    )


def test_build_model_pipeline_has_expected_steps() -> None:
    pipeline = build_model_pipeline()

    assert list(pipeline.named_steps) == ["preprocessing", "classifier"]


def test_train_evaluate_and_save_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_and_evaluate(model_frame(), model_path, metrics_path, max_rows=100)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert model_path.exists() and metrics_path.exists()
    assert list(joblib.load(model_path).named_steps) == ["preprocessing", "classifier"]


@pytest.mark.parametrize("max_rows", [0, -1, 9, 10.5, True])
def test_train_rejects_invalid_max_rows(tmp_path: Path, max_rows: object) -> None:
    with pytest.raises(ValueError, match="max_rows"):
        train_and_evaluate(
            model_frame(),
            tmp_path / "model.joblib",
            tmp_path / "metrics.json",
            max_rows=max_rows,  # type: ignore[arg-type]
        )


def test_train_rejects_missing_columns_and_single_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="필수 컬럼"):
        train_and_evaluate(
            model_frame().drop(columns=["fare_amount"]),
            tmp_path / "model.joblib",
            tmp_path / "metrics.json",
        )
    single_class = model_frame()
    single_class["high_tip"] = 1
    with pytest.raises(ValueError, match="두 클래스"):
        train_and_evaluate(single_class, tmp_path / "model.joblib", tmp_path / "metrics.json")
