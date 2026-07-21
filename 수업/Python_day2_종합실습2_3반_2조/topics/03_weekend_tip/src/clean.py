from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
import polars as pl


NullStrategy = Literal["median", "drop"]


def clean_nulls(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    strategy: NullStrategy = "median",
) -> pd.DataFrame:
    """선택한 열의 결측치를 중앙값으로 대체하거나 해당 행을 제거한다.

    ``cols``를 생략하면 수치형 열만 대상으로 한다. 원본 DataFrame을 변경하지
    않고 복사본을 반환한다.
    """
    result = df.copy()
    target_cols = cols or result.select_dtypes(include="number").columns.tolist()

    if strategy == "median":
        non_numeric = [name for name in target_cols if not pd.api.types.is_numeric_dtype(result[name])]
        if non_numeric:
            raise TypeError(f"median 전략은 수치형 열에만 사용할 수 있습니다: {non_numeric}")
        result[target_cols] = result[target_cols].fillna(result[target_cols].median())
        return result
    if strategy == "drop":
        return result.dropna(subset=target_cols)
    raise ValueError("strategy는 'median' 또는 'drop'이어야 합니다.")


def clean_taxi_with_polars(df: pl.DataFrame) -> pl.DataFrame:
    """전체 데이터용 기본 전처리: 완전 중복 제거 및 주요 결측치 보정."""
    required = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    ]
    result = df.unique(maintain_order=True).drop_nulls(required)
    passenger_median = result.get_column("passenger_count").median()
    return result.with_columns(
        pl.col("passenger_count").fill_null(passenger_median).cast(pl.Int64),
        pl.col("RatecodeID").fill_null(99),
        pl.col("payment_type").fill_null(99),
        pl.col("store_and_fwd_flag").fill_null("UNKNOWN"),
        pl.col("congestion_surcharge").fill_null(0.0),
        pl.col("Airport_fee").fill_null(0.0),
        pl.col("cbd_congestion_fee").fill_null(0.0),
    )

