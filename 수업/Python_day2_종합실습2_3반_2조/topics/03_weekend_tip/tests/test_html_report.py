from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import src.html_report as html_report


def context() -> dict[str, object]:
    return {
        "engine_comparison": {
            "pandas": {"rows": 100, "load_seconds": 1.0, "null_cells": 2},
            "polars": {"rows": 100, "load_seconds": 0.2, "null_cells": 2},
        },
        "audit": pd.DataFrame({"step": ["card"], "rows": [80], "removed": [20]}),
        "distance_summary": pd.DataFrame(
            {
                "distance_group": ["short_lt_5mi", "long_ge_5mi"],
                "trip_count": [60, 20],
                "mean_tip_amount": [3.0, 8.0],
                "mean_tip_rate": [0.17, 0.14],
            }
        ),
        "day_summary": pd.DataFrame(
            {"is_weekend": [False, True], "trip_count": [50, 30], "mean_tip_rate": [0.16, 0.17]}
        ),
        "ttest": {
            "p_value": 0.01,
            "t_statistic": 3.0,
            "difference_percentage_points": 3.0,
            "cohens_d": 0.4,
        },
        "model_metrics": {"model": "test", "accuracy": 0.6, "f1": 0.6, "roc_auc": 0.65},
        "chart_html": "<div id='plotly-test'>chart</div>",
    }


def test_generate_jinja_html_report_with_loop_if_and_chart(tmp_path: Path) -> None:
    output = tmp_path / "report.html"

    html_report.generate_html_report(output, Path("templates"), **context())

    text = output.read_text()
    assert "short_lt_5mi" in text and "long_ge_5mi" in text
    assert "통계적으로 유의합니다" in text
    assert "plotly-test" in text


def test_html_report_can_open_browser(tmp_path: Path, monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(html_report.webbrowser, "open", lambda url: opened.append(url))

    html_report.generate_html_report(
        tmp_path / "report.html",
        Path("templates"),
        open_browser=True,
        **context(),
    )

    assert opened and opened[0].startswith("file:")


def test_html_report_non_significant_branch(tmp_path: Path) -> None:
    values = context()
    values["ttest"] = {**values["ttest"], "p_value": 0.2}  # type: ignore[arg-type]

    html_report.generate_html_report(tmp_path / "report.html", Path("templates"), **values)

    assert "유의하다고 보기 어렵습니다" in (tmp_path / "report.html").read_text()


def test_html_report_generated_time_can_be_fixed(tmp_path: Path) -> None:
    fixed = datetime(2026, 5, 31, tzinfo=timezone.utc)

    html_report.generate_html_report(
        tmp_path / "report.html", Path("templates"), generated_at=fixed, **context()
    )

    assert "2026-05-31T00:00:00+00:00" in (tmp_path / "report.html").read_text()
