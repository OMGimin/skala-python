from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl


def timed_load_pandas(path: Path) -> tuple[pd.DataFrame, float]:
    """Parquet 파일을 Pandas로 읽고 소요 시간을 반환한다."""
    started = time.perf_counter()
    frame = pd.read_parquet(path)
    return frame, time.perf_counter() - started


def timed_load_polars(path: Path) -> tuple[pl.DataFrame, float]:
    """Parquet 파일을 Polars로 읽고 소요 시간을 반환한다."""
    started = time.perf_counter()
    frame = pl.read_parquet(path)
    return frame, time.perf_counter() - started


def pandas_profile(frame: pd.DataFrame, elapsed: float) -> dict[str, Any]:
    """Pandas DataFrame의 형태, 메모리, 결측치와 중복을 요약한다."""
    return {
        "engine": "pandas",
        "load_seconds": round(elapsed, 4),
        "rows": len(frame),
        "columns": len(frame.columns),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 2),
        "duplicate_rows": int(len(frame) - len(frame.drop_duplicates())),
        "null_cells": int(frame.isna().sum().sum()),
        "null_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
    }


def polars_profile(frame: pl.DataFrame, elapsed: float) -> dict[str, Any]:
    """Polars DataFrame의 형태, 메모리, 결측치와 중복을 요약한다."""
    nulls = frame.null_count().row(0, named=True)
    return {
        "engine": "polars",
        "load_seconds": round(elapsed, 4),
        "rows": frame.height,
        "columns": frame.width,
        "memory_mb": round(frame.estimated_size("mb"), 2),
        "duplicate_rows": int(frame.height - frame.unique().height),
        "null_cells": int(sum(nulls.values())),
        "null_by_column": {key: int(value) for key, value in nulls.items()},
        "dtypes": {key: str(value) for key, value in frame.schema.items()},
    }


def validate_profiles(pandas_info: dict[str, Any], polars_info: dict[str, Any]) -> None:
    """두 엔진의 핵심 로딩 결과가 동일한지 검증한다."""
    for key in ("rows", "columns", "duplicate_rows", "null_cells", "null_by_column"):
        if pandas_info[key] != polars_info[key]:
            raise ValueError(f"Pandas/Polars 결과 불일치: {key}")
