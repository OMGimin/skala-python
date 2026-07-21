from pathlib import Path
from time import perf_counter

import duckdb
import pandas as pd
import polars as pl
from pandas.testing import assert_frame_equal


# --------------------------------------------------
# 1. 데이터 파일 경로 찾기
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

data_candidates = [
    BASE_DIR / "data" / "events_large.csv",
    BASE_DIR.parent
    / "Python_data_0720_Day1"
    / "data"
    / "events_large.csv",
]

DATA_PATH = None

for candidate in data_candidates:
    if candidate.exists():
        DATA_PATH = candidate
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "events_large.csv 파일을 찾을 수 없습니다."
    )

print(f"데이터 경로: {DATA_PATH}")


# --------------------------------------------------
# 2. Pandas 집계
# --------------------------------------------------

def aggregate_with_pandas(
    path: Path,
) -> tuple[pd.DataFrame, float]:
    start = perf_counter()

    df = pd.read_csv(path)

    result = (
        df.groupby(
            "event_type",
            as_index=False,
        )
        .agg(
            event_count=("event_id", "count"),
            unique_users=("user_id", "nunique"),
            total_amount=("amount", "sum"),
        )
        .sort_values("event_type")
        .reset_index(drop=True)
    )

    elapsed = perf_counter() - start

    return result, elapsed


# --------------------------------------------------
# 3. Polars Lazy API 집계
# --------------------------------------------------

def aggregate_with_polars(
    path: Path,
) -> tuple[pd.DataFrame, float]:
    start = perf_counter()

    lazy_df = pl.scan_csv(path)

    result = (
        lazy_df.group_by("event_type")
        .agg(
            pl.len().alias("event_count"),
            pl.col("user_id")
            .n_unique()
            .alias("unique_users"),
            pl.col("amount")
            .sum()
            .alias("total_amount"),
        )
        .sort("event_type")
        .collect()
    )

    elapsed = perf_counter() - start

    return result.to_pandas(), elapsed


# --------------------------------------------------
# 4. DuckDB SQL 집계
# --------------------------------------------------

def aggregate_with_duckdb(
    path: Path,
) -> tuple[pd.DataFrame, float]:
    connection = duckdb.connect()

    start = perf_counter()

    query = """
        SELECT
            event_type,
            COUNT(*) AS event_count,
            COUNT(DISTINCT user_id) AS unique_users,
            SUM(amount) AS total_amount
        FROM read_csv_auto(?)
        GROUP BY event_type
        ORDER BY event_type
    """

    result = connection.execute(
        query,
        [str(path)],
    ).fetchdf()

    elapsed = perf_counter() - start

    connection.close()

    return result, elapsed


# --------------------------------------------------
# 5. 세 엔진 실행
# --------------------------------------------------

print("\n집계를 시작합니다.")

pandas_result, pandas_time = (
    aggregate_with_pandas(DATA_PATH)
)

polars_result, polars_time = (
    aggregate_with_polars(DATA_PATH)
)

duckdb_result, duckdb_time = (
    aggregate_with_duckdb(DATA_PATH)
)


# --------------------------------------------------
# 6. 타입 통일
# --------------------------------------------------

integer_columns = [
    "event_count",
    "unique_users",
    "total_amount",
]

for column in integer_columns:
    pandas_result[column] = (
        pandas_result[column].astype("int64")
    )

    polars_result[column] = (
        polars_result[column].astype("int64")
    )

    duckdb_result[column] = (
        duckdb_result[column].astype("int64")
    )


# --------------------------------------------------
# 7. 결과 출력
# --------------------------------------------------

print("\n=== Pandas 집계 결과 ===")
print(pandas_result.to_string(index=False))

print("\n=== Polars 집계 결과 ===")
print(polars_result.to_string(index=False))

print("\n=== DuckDB 집계 결과 ===")
print(duckdb_result.to_string(index=False))


# --------------------------------------------------
# 8. 결과 일치 여부 검증
# --------------------------------------------------

try:
    assert_frame_equal(
        pandas_result,
        polars_result,
        check_dtype=False,
    )

    assert_frame_equal(
        pandas_result,
        duckdb_result,
        check_dtype=False,
    )

    results_match = True

except AssertionError as error:
    results_match = False

    print("\n집계 결과가 일치하지 않습니다.")
    print(error)


# --------------------------------------------------
# 9. 실행시간 비교
# --------------------------------------------------

performance = pd.DataFrame(
    {
        "engine": [
            "Pandas",
            "Polars",
            "DuckDB",
        ],
        "seconds": [
            pandas_time,
            polars_time,
            duckdb_time,
        ],
    }
)

performance["milliseconds"] = (
    performance["seconds"] * 1000
)

performance = performance.sort_values(
    "seconds"
).reset_index(drop=True)

fastest_time = performance.loc[0, "seconds"]

performance["relative_speed"] = (
    performance["seconds"] / fastest_time
)

print("\n=== 실행시간 비교 ===")

print(
    performance[
        [
            "engine",
            "milliseconds",
            "relative_speed",
        ]
    ].to_string(
        index=False,
        formatters={
            "milliseconds": lambda x: f"{x:,.2f} ms",
            "relative_speed": lambda x: f"{x:.2f}배",
        },
    )
)


# --------------------------------------------------
# 10. 최종 성공 여부
# --------------------------------------------------

print("\n=== 최종 검증 ===")

if results_match:
    print("성공: 세 엔진의 집계 결과가 모두 같습니다.")
else:
    print("실패: 세 엔진의 집계 결과를 확인하세요.")

print(
    f"가장 빠른 엔진: "
    f"{performance.loc[0, 'engine']}"
)