from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


NUMERIC_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "base_amount",
    "tip_amount",
    "tip_rate",
    "trip_duration_min",
    "pickup_hour",
]


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """핵심 수치 변수의 평균·표준편차·분위수를 반환한다."""
    return (
        frame[NUMERIC_COLUMNS]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .T.reset_index(names="variable")
    )


def grouped_tip_statistics(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """그룹별 팁 금액과 팁률 기술통계를 계산한다."""
    return (
        frame.groupby(group, observed=True)
        .agg(
            trip_count=("tip_rate", "size"),
            mean_tip_amount=("tip_amount", "mean"),
            mean_tip_rate=("tip_rate", "mean"),
            std_tip_rate=("tip_rate", "std"),
            median_tip_rate=("tip_rate", "median"),
            q25_tip_rate=("tip_rate", lambda values: values.quantile(0.25)),
            q75_tip_rate=("tip_rate", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )


def correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """핵심 수치 변수의 Pearson 상관계수를 계산한다."""
    return frame[NUMERIC_COLUMNS].corr(method="pearson")


def cohens_d(first: np.ndarray, second: np.ndarray) -> float:
    """독립된 두 표본의 pooled-standard-deviation Cohen's d를 계산한다."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        raise ValueError("Cohen's d 계산에는 각 그룹에 유효한 관측치가 2개 이상 필요합니다.")
    pooled_variance = (
        ((n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1))
        / (n1 + n2 - 2)
    )
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        raise ValueError("Cohen's d를 계산할 수 없습니다: 결합 분산이 0이거나 유효하지 않습니다.")
    return float((np.mean(first) - np.mean(second)) / math.sqrt(pooled_variance))


def distance_ttest(frame: pd.DataFrame) -> dict[str, Any]:
    """5마일 미만/이상 그룹의 평균 팁률을 Welch t-test로 비교한다."""
    required = {"trip_distance", "tip_rate"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"t-test 필수 컬럼이 없습니다: {missing}")
    short = frame.loc[frame["trip_distance"] < 5, "tip_rate"].dropna().to_numpy(dtype=float)
    long = frame.loc[frame["trip_distance"] >= 5, "tip_rate"].dropna().to_numpy(dtype=float)
    short = short[np.isfinite(short)]
    long = long[np.isfinite(long)]
    if len(short) < 2 or len(long) < 2:
        raise ValueError("t-test에는 단거리와 장거리 그룹별 유효 관측치가 2개 이상 필요합니다.")
    effect_size = cohens_d(short, long)
    result = ttest_ind(short, long, equal_var=False, nan_policy="omit")
    p_value = float(result.pvalue)
    if not np.isfinite(result.statistic) or not np.isfinite(p_value):
        raise ValueError("t-test 결과가 유효하지 않습니다. 그룹 분산과 입력값을 확인하세요.")
    return {
        "hypothesis_null": "5마일 미만과 5마일 이상 카드 운행의 평균 팁률은 같다.",
        "short_n": int(len(short)),
        "long_n": int(len(long)),
        "short_mean_tip_rate": float(np.mean(short)),
        "long_mean_tip_rate": float(np.mean(long)),
        "difference_percentage_points": float((np.mean(short) - np.mean(long)) * 100),
        "t_statistic": float(result.statistic),
        "p_value": p_value,
        "cohens_d": effect_size,
        "significant_at_0_05": p_value < 0.05,
    }
