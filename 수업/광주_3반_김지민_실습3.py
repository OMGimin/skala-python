"""
실습명 : [실습 3] Pandas EDA, Polars Lazy, Duck DB SQL 비교
작성자 : 광주 3반 김지민
활용 데이터 : sales_100k.csv

[프로그램 설명]
sales_100k.csv를 Pandas로 읽어 기본적인 탐색적 데이터 분석(EDA)을
수행하고, IQR 기준으로 amount 이상치를 제거한다.

정제된 데이터를 region·category 기준으로 그룹화하여 총매출, 평균,
거래 건수를 계산한다. 같은 집계를 Pandas named aggregation,
Polars Lazy API, DuckDB SQL로 각각 구현하고 실행 시간을 비교한다.

[변경 내역]
- 2026-07-21: 최초 작성
"""

from pathlib import Path
from timeit import repeat

import duckdb
import pandas as pd
import polars as pl

CSV_PATH = Path(__file__).resolve().parent / "sales_100k.csv"
REQUIRED_COLUMNS = {"region", "category", "amount"}

def validate_csv_file(csv_path: Path) -> None:
    """csv 파일의 존재 여부와 필수 컬럼을 확인한다."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다 : {csv_path}")
    
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing_columns = REQUIRED_COLUMNS - columns
    
    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다 : {sorted(missing_columns)}"
            f"/ 실제 컬럼: {sorted(columns)}"
        )
    

def pandas_eda(csv_path:Path) -> tuple[pd.DataFrame, float, float]:
    
    sales = pd.read_csv(csv_path)

    sales["amount"] = pd.to_numeric(sales["amount"],errors="coerce")

    print("\n[1. 데이터 크기]")
    print(sales.shape)

    print("\n[2. 상위 5개 행]")
    print(sales.head())

    print("\n[3. 컬럼별 자료형]")
    print(sales.dtypes)

    print("\n[4. 결측치 개수]")
    print(sales.isna().sum())

    print("\n[5. amount 기술통계]")
    print(sales["amount"].describe())

    q1 = sales["amount"].quantile(0.25)
    q3 = sales["amount"].quantile(0.75)

    iqr = q3 - q1
    
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    cleaned_sales = sales[
        sales["amount"].between(lower_bound,upper_bound)
    ].copy()

    print("\n[6. IQR 이상치 처라]")
    print(f"Q1: {q1:,.2f}")
    print(f"Q3: {q3:,.2f}")
    print(f"IQR: {iqr:,.2f}")
    print(f"정상 범위: {lower_bound:,.2f} ~ {upper_bound:,.2f}")
    print(f"처리 전 행 수: {len(sales):,}")
    print(f"처리 후 행 수: {len(cleaned_sales):,}")
    print(f"제거된 이상치 수: {len(sales) - len(cleaned_sales):,}")

    return cleaned_sales, lower_bound, upper_bound

def aggregate_with_pandas(cleaned_sales: pd.DataFrame) -> pd.DataFrame:

    result = (
        cleaned_sales
        .groupby(["region", "category"], as_index=False, dropna=False)
        .agg(
            total=("amount", "sum"),
            average=("amount", "mean"),
            count=("amount","count"),
        )
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )
    return result

def aggregate_with_polars(
        csv_path:Path,
        lower_bound:float,
        upper_bound:float,
)-> pl.DataFrame:
    
    result = (
        pl.scan_csv(csv_path)
        .filter(
            pl.col("amount").is_between(
                lower_bound,
                upper_bound,
                closed="both",
            )
        )
        .group_by(["region", "category"])
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("amount").mean().alias("average"),
            pl.col("amount").count().alias("count"),
        )
        .sort("total",descending=True)
        .collect()
    )
    return result

def aggregate_with_duckdb(
        csv_path:Path,
        lower_bound: float,
        upper_bound: float,
) -> pd.DataFrame:
    
    safe_path = str(csv_path.resolve()).replace("'","''")

    query = f"""
        SELECT
            region,
            category,
            SUM(amount) AS total,
            AVG(amount) AS average,
            COUNT(amount) AS count
        FROM read_csv_auto('{safe_path}')
        WHERE amount BETWEEN ? AND ?
        GROUP BY region, category
        ORDER BY total DESC
    """

    return duckdb.execute(
        query,
        [lower_bound, upper_bound],
    ).fetchdf()

def measure_execution_time(function, repeat_count: int = 5) -> float:
    execution_times = repeat(function, repeat=repeat_count, number=1)
    return min(execution_times)

def validate_results(
        pandas_result:pd.DataFrame,
        polars_result:pl.DataFrame,
        duckdb_result:pd.DataFrame,
)-> None:
    
    pandas_total = pandas_result["total"].sum()
    polars_total = polars_result["total"].sum()
    duckdb_total = duckdb_result["total"].sum()

    assert abs(pandas_total - polars_total) < 0.001
    assert abs(pandas_total - duckdb_total) < 0.001

    assert len(pandas_result) == len(polars_result)
    assert len(pandas_result) == len(duckdb_result)

    assert pandas_result["total"].is_monotonic_decreasing
    assert polars_result["total"].is_sorted(descending=True)
    assert duckdb_result["total"].is_monotonic_decreasing

def main() -> None :

    validate_csv_file(CSV_PATH)

    cleaned_sales, lower_bound, upper_bound = pandas_eda(CSV_PATH)

    pandas_result = aggregate_with_pandas(cleaned_sales)
    polars_result = aggregate_with_polars(
        CSV_PATH,
        lower_bound,
        upper_bound,
    )
    duckdb_result = aggregate_with_duckdb(
        CSV_PATH,
        lower_bound,
        upper_bound,
    )

    print("\n[Pandas 집계 결과]")
    print(pandas_result)

    print("\n[Polars Lazy 집계 결과]")
    print(polars_result)

    print("\n[DuckDB SQL 집계 결과]")
    print(duckdb_result)
    
    validate_results(
        pandas_result,
        polars_result,
        duckdb_result,
    )

    pandas_time = measure_execution_time(
        lambda : aggregate_with_pandas(cleaned_sales)
    )

    polars_time = measure_execution_time(
        lambda : aggregate_with_polars(
            CSV_PATH,
            lower_bound,
            upper_bound,
        )
    )
    duckdb_time = measure_execution_time(
        lambda: aggregate_with_duckdb(
            CSV_PATH,
            lower_bound,
            upper_bound,
        )
    )

    perfomance = pd.DataFrame(
        {
            "tool" : ["Pandas", "Polars Lazy", "Duck DB"],
            "seconds" : [
                pandas_time,
                polars_time,
                duckdb_time,
            ],
        }
    ).sort_values("seconds")

    print("\n[세 도구 실행 시간 비교]")
    print(perfomance.to_string(index=False))

    print("\n모든 집계 결과 검증을 통과했습니다.")

if __name__ == "__main__":
    main()



    
