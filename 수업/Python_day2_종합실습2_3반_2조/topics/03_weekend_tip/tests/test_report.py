import json
from pathlib import Path

import pandas as pd
import pytest

from src.report import generate_report, load_json


def test_generate_report_and_load_json(tmp_path: Path) -> None:
    audit = pd.DataFrame({"step": ["card_payment_only"], "rows": [100], "removed": [10]})
    distance = pd.DataFrame(
        {
            "distance_group": ["short_lt_5mi", "long_ge_5mi"],
            "trip_count": [70, 30],
            "mean_tip_amount": [3.0, 7.0],
            "mean_tip_rate": [0.18, 0.14],
            "std_tip_rate": [0.1, 0.1],
            "median_tip_rate": [0.2, 0.15],
            "q25_tip_rate": [0.1, 0.1],
            "q75_tip_rate": [0.2, 0.2],
        }
    )
    day = pd.DataFrame(
        {
            "is_weekend": [False, True],
            "trip_count": [70, 30],
            "mean_tip_amount": [4.0, 4.1],
            "mean_tip_rate": [0.16, 0.17],
            "std_tip_rate": [0.1, 0.1],
            "median_tip_rate": [0.2, 0.2],
            "q25_tip_rate": [0.1, 0.1],
            "q75_tip_rate": [0.2, 0.2],
        }
    )
    ttest = {
        "hypothesis_null": "same",
        "p_value": 0.0001,
        "t_statistic": 2.0,
        "difference_percentage_points": 4.0,
        "cohens_d": 0.4,
    }
    metrics = {
        "model": "SGDClassifier(loss='log_loss')",
        "sample_rows": 100,
        "accuracy": 0.7,
        "f1": 0.7,
        "roc_auc": 0.75,
    }
    report_path = tmp_path / "report.md"

    engines = {
        "pandas": {"rows": 100, "columns": 20, "load_seconds": 1.0, "memory_mb": 10.0, "null_cells": 0, "duplicate_rows": 0},
        "polars": {"rows": 100, "columns": 20, "load_seconds": 0.2, "memory_mb": 5.0, "null_cells": 0, "duplicate_rows": 0},
    }
    generate_report(report_path, engines, audit, distance, day, ttest, metrics)

    assert "거리 그룹" in report_path.read_text()
    assert "통계적으로 유의하다" in report_path.read_text()
    nonsignificant = {**ttest, "p_value": 0.2}
    generate_report(report_path, engines, audit, distance, day, nonsignificant, metrics)
    assert "유의하다고 보기 어렵다" in report_path.read_text()
    json_path = tmp_path / "value.json"
    json_path.write_text(json.dumps({"ok": True}))
    assert load_json(json_path) == {"ok": True}


def test_report_interpretation_follows_actual_values(tmp_path: Path) -> None:
    distance = pd.DataFrame(
        {
            "distance_group": ["short_lt_5mi", "long_ge_5mi"],
            "mean_tip_amount": [8.0, 3.0],
            "mean_tip_rate": [0.14, 0.18],
        }
    )
    day = pd.DataFrame(
        {"is_weekend": [False, True], "mean_tip_rate": [0.18, 0.16]}
    )
    metrics = {"accuracy": 0.81, "f1": 0.79, "roc_auc": 0.88, "sample_rows": 10}
    engines = {
        name: {
            "rows": 10, "columns": 2, "load_seconds": 0.1, "memory_mb": 1.0,
            "null_cells": 0, "duplicate_rows": 0,
        }
        for name in ("pandas", "polars")
    }
    output = tmp_path / "report.md"

    generate_report(
        output, engines, pd.DataFrame(), distance, day,
        {"hypothesis_null": "same", "p_value": 0.2, "t_statistic": 1.0,
         "difference_percentage_points": -4.0, "cohens_d": -0.2},
        metrics,
    )

    text = output.read_text()
    assert "장거리 평균 팁 금액이 단거리보다 $5.00 낮았다" in text
    assert "장거리 평균 팁률이 단거리보다 4.00%p 높았다" in text
    assert "주말 평균 팁률이 주중보다 2.00%p 낮았다" in text
    assert "Accuracy 0.8100, F1 0.7900, ROC-AUC 0.8800" in text


def test_generate_report_rejects_missing_required_group(tmp_path: Path) -> None:
    distance = pd.DataFrame(
        {
            "distance_group": ["short_lt_5mi"],
            "mean_tip_amount": [3.0],
            "mean_tip_rate": [0.18],
        }
    )
    with pytest.raises(ValueError, match="장거리"):
        generate_report(
            tmp_path / "report.md",
            {"pandas": {}, "polars": {}},
            pd.DataFrame(),
            distance,
            pd.DataFrame(),
            {"p_value": 0.1},
            {},
        )
