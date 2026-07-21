from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from src.clean import clean_taxi_with_polars
from src.eda import write_eda
from src.data_profile import (
    pandas_profile,
    polars_profile,
    timed_load_pandas,
    timed_load_polars,
    validate_profiles,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "03_weekend_tip"
DEFAULT_INPUT = Path(
    os.getenv(
        "TAXI_RAW_PATH",
        PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet",
    )
)
DEFAULT_PROCESSED_DIR = Path(
    os.getenv("TAXI_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
)
DEFAULT_REPORT_DIR = Path(os.getenv("TAXI_REPORT_DIR", OUTPUT_ROOT / "tables"))


def run_pipeline(input_path: Path, processed_dir: Path, report_dir: Path) -> dict[str, object]:
    """로딩 비교, 전처리, EDA 저장을 한 번에 실행한다."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    pandas_df, pandas_seconds = timed_load_pandas(input_path)
    pandas_info = pandas_profile(pandas_df, pandas_seconds)
    del pandas_df
    gc.collect()

    polars_df, polars_seconds = timed_load_polars(input_path)
    polars_info = polars_profile(polars_df, polars_seconds)
    validate_profiles(pandas_info, polars_info)

    cleaned = clean_taxi_with_polars(polars_df)
    cleaned_nulls = int(sum(cleaned.null_count().row(0)))
    cleaned_duplicates = int(cleaned.is_duplicated().sum())
    if cleaned_nulls or cleaned_duplicates:
        raise ValueError(
            f"전처리 검증 실패: null_cells={cleaned_nulls}, duplicate_rows={cleaned_duplicates}"
        )

    csv_path = processed_dir / "yellow_tripdata_2026-05_cleaned.csv"
    cleaned.write_csv(csv_path, datetime_format="%Y-%m-%d %H:%M:%S")
    write_eda(cleaned, report_dir)

    comparison: dict[str, object] = {
        "source": str(input_path),
        "pandas": pandas_info,
        "polars": polars_info,
        "validation": {
            "shape_equal": True,
            "null_counts_equal": True,
            "duplicate_counts_equal": True,
        },
        "cleaning": {
            "rows_before": polars_df.height,
            "rows_after": cleaned.height,
            "rows_removed": polars_df.height - cleaned.height,
            "null_cells_after": cleaned_nulls,
            "duplicate_rows_after": cleaned_duplicates,
            "output_csv": str(csv_path),
        },
    }
    (report_dir / "preprocessing_report.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="NYC taxi preprocessing with Pandas/Polars comparison")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    result = run_pipeline(args.input, args.processed_dir, args.report_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
