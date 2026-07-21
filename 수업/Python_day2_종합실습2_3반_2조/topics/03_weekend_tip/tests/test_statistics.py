import pandas as pd
import numpy as np
import pytest

from src.statistical_analysis import (
    cohens_d,
    correlation_matrix,
    descriptive_statistics,
    distance_ttest,
    grouped_tip_statistics,
)


def statistics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [1.0, 2.0, 6.0, 8.0],
            "fare_amount": [10.0, 12.0, 30.0, 40.0],
            "base_amount": [10.0, 12.0, 30.0, 40.0],
            "tip_amount": [2.0, 2.4, 3.0, 4.0],
            "tip_rate": [0.18, 0.22, 0.08, 0.12],
            "trip_duration_min": [10.0, 15.0, 30.0, 40.0],
            "pickup_hour": [8, 9, 18, 19],
            "distance_group": ["short", "short", "long", "long"],
        }
    )


def test_descriptive_grouped_and_correlation_outputs() -> None:
    frame = statistics_frame()

    assert "mean" in descriptive_statistics(frame).columns
    assert len(grouped_tip_statistics(frame, "distance_group")) == 2
    assert correlation_matrix(frame).shape == (7, 7)


def test_ttest_and_effect_size() -> None:
    frame = statistics_frame()
    result = distance_ttest(frame)

    assert result["short_n"] == 2
    assert result["long_n"] == 2
    assert result["difference_percentage_points"] == 10.0
    assert cohens_d(frame.tip_rate[:2].to_numpy(), frame.tip_rate[2:].to_numpy()) > 0


def test_ttest_rejects_missing_group_and_zero_variance() -> None:
    only_short = statistics_frame().loc[lambda data: data["trip_distance"] < 5]
    with pytest.raises(ValueError, match="그룹별"):
        distance_ttest(only_short)
    with pytest.raises(ValueError, match="결합 분산"):
        cohens_d(np.array([0.2, 0.2]), np.array([0.1, 0.1]))


def test_ttest_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="필수 컬럼"):
        distance_ttest(pd.DataFrame({"tip_rate": [0.1, 0.2]}))
