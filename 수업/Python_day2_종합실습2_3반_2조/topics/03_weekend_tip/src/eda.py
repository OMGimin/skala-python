from __future__ import annotations

from pathlib import Path

import polars as pl


def numeric_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """수치형 컬럼의 기술통계를 반환한다."""
    numeric_columns = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    return frame.select(numeric_columns).describe()


def daily_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """승차일별 운행 건수와 거리·금액 평균을 반환한다."""
    return (
        frame.group_by(pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"))
        .agg(
            pl.len().alias("trip_count"),
            pl.col("trip_distance").mean().alias("avg_trip_distance"),
            pl.col("total_amount").mean().alias("avg_total_amount"),
            pl.col("tip_amount").mean().alias("avg_tip_amount"),
        )
        .sort("pickup_date")
    )


def payment_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """결제 유형별 운행 건수, 비율, 평균 결제액을 반환한다."""
    return (
        frame.group_by("payment_type")
        .agg(
            pl.len().alias("trip_count"),
            pl.col("total_amount").mean().alias("avg_total_amount"),
        )
        .with_columns((pl.col("trip_count") / pl.col("trip_count").sum()).alias("ratio"))
        .sort("trip_count", descending=True)
    )


def write_eda(frame: pl.DataFrame, output_dir: Path) -> None:
    """기본 EDA 결과를 재사용 가능한 CSV 표로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_summary(frame).write_csv(output_dir / "eda_numeric_summary.csv")
    daily_summary(frame).write_csv(output_dir / "eda_daily_summary.csv")
    payment_summary(frame).write_csv(output_dir / "eda_payment_type.csv")

