from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import polars as pl


FilterRule = tuple[str, Callable[[pl.DataFrame], pl.Expr]]


def create_tip_features(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """카드 팁 분석용 행을 정제하고 파생 변수를 생성한다.

    반환값은 ``(분석 데이터, 단계별 행 수 감사표)``이다.
    """
    audit: list[dict[str, int | str]] = []
    current = frame.unique(maintain_order=True)
    audit.append(
        {
            "step": "remove_exact_duplicates",
            "rows": current.height,
            "removed": frame.height - current.height,
        }
    )

    current = current.with_columns(
        (pl.col("total_amount") - pl.col("tip_amount")).alias("base_amount"),
        (
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
            .dt.total_seconds()
            / 60
        ).alias("trip_duration_min"),
    ).with_columns((pl.col("tip_amount") / pl.col("base_amount")).alias("tip_rate"))

    rules: list[FilterRule] = [
        ("card_payment_only", lambda _: pl.col("payment_type") == 1),
        (
            "pickup_in_2026_05",
            lambda _: pl.col("tpep_pickup_datetime").is_between(
                datetime(2026, 5, 1), datetime(2026, 6, 1), closed="left"
            ),
        ),
        ("positive_trip_distance", lambda _: pl.col("trip_distance") > 0),
        ("positive_fare_amount", lambda _: pl.col("fare_amount") > 0),
        ("nonnegative_tip", lambda _: pl.col("tip_amount") >= 0),
        ("positive_base_amount", lambda _: pl.col("base_amount") > 0),
        (
            "tip_rate_0_to_1",
            lambda _: pl.col("tip_rate").is_finite() & pl.col("tip_rate").is_between(0, 1),
        ),
        (
            "duration_0_to_180_min",
            lambda _: pl.col("trip_duration_min").is_between(0, 180, closed="right"),
        ),
    ]
    for name, expression in rules:
        before = current.height
        current = current.filter(expression(current))
        audit.append({"step": name, "rows": current.height, "removed": before - current.height})

    current = current.with_columns(
        pl.col("tpep_pickup_datetime").dt.weekday().alias("day_of_week"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
    ).with_columns(
        (pl.col("day_of_week") >= 6).alias("is_weekend"),
        pl.when(pl.col("trip_distance") < 5)
        .then(pl.lit("short_lt_5mi"))
        .otherwise(pl.lit("long_ge_5mi"))
        .alias("distance_group"),
        pl.when(pl.col("pickup_hour") < 6)
        .then(pl.lit("late_night"))
        .when(pl.col("pickup_hour") < 12)
        .then(pl.lit("morning"))
        .when(pl.col("pickup_hour") < 18)
        .then(pl.lit("afternoon"))
        .otherwise(pl.lit("evening"))
        .alias("time_group"),
        (pl.col("tip_rate") >= 0.20).cast(pl.Int8).alias("high_tip"),
    )

    return current, pl.DataFrame(audit)


def build_analysis_dataset(input_path: Path, output_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """원본 Parquet을 읽어 분석용 Parquet과 필터 감사표를 생성한다."""
    frame = pl.read_parquet(input_path)
    analysis, audit = create_tip_features(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_parquet(output_path, compression="zstd")
    return analysis, audit
