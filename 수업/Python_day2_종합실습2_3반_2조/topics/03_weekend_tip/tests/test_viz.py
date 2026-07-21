import matplotlib
import pandas as pd
import pytest
from pathlib import Path

matplotlib.use("Agg")

from src.viz import (
    hourly_interactive_fragment,
    plot_distribution,
    save_distance_comparison,
    save_hourly_interactive,
)


def test_plot_distribution_returns_axes() -> None:
    frame = pd.DataFrame({"tip_amount": [1.0, 2.0, 3.0]})

    axes = plot_distribution(frame, "tip_amount", bins=3, title="Tips")

    assert axes.get_title() == "Tips"
    assert axes.get_xlabel() == "tip_amount"


def test_plot_distribution_rejects_missing_column() -> None:
    with pytest.raises(KeyError, match="존재하지 않는"):
        plot_distribution(pd.DataFrame({"x": [1]}), "missing")


def test_save_static_and_interactive_charts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "distance_group": ["short_lt_5mi", "long_ge_5mi"] * 2,
            "tip_rate": [0.2, 0.1, 0.18, 0.12],
            "tip_amount": [2.0, 5.0, 2.2, 6.0],
            "pickup_hour": [8, 8, 18, 18],
            "is_weekend": [False, False, True, True],
        }
    )
    png = save_distance_comparison(frame, tmp_path / "chart.png")
    html = save_hourly_interactive(frame, tmp_path / "chart.html")

    assert png.exists() and png.stat().st_size > 0
    assert html.exists() and "plotly" in html.read_text().lower()
    assert "plotly" in hourly_interactive_fragment(frame).lower()


def test_charts_reject_missing_groups(tmp_path: Path) -> None:
    only_short = pd.DataFrame(
        {
            "distance_group": ["short_lt_5mi"],
            "tip_rate": [0.2],
            "tip_amount": [2.0],
            "pickup_hour": [8],
            "is_weekend": [False],
        }
    )
    with pytest.raises(ValueError, match="필수 그룹"):
        save_distance_comparison(only_short, tmp_path / "distance.png")
    with pytest.raises(ValueError, match="주중·주말"):
        save_hourly_interactive(only_short, tmp_path / "hourly.html")
