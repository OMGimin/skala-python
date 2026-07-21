from pathlib import Path

import pandas as pd

#pandas 2.x Copy-on-Write 활성화
pd.options.mode.copy_on_write = True

#현재 파일이 practice4 폴더 안에 있다고 가정
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sales_raw.csv"

# 1. 데이터 불러오기
df = pd.read_csv(DATA_PATH)

print("===정제 전 데이터 정보===")
print(f"행 개수: {len(df):,}")
print("\n결측치 개수")
print(df.isna().sum())

print("\n수치형 데이터 요약")
print(df[["quantity","unit_price","discount"]].describe())

# 2. 타입 정규화
df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors = "coerce"
)

numeric_columns = [
    "quantity",
    "unit_price",
    "discount"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df["region"] = df["region"].astype("string")
df["category"] = df["category"].astype("string")

# 3. 잘못된 가격을 결측치로 변경
# 판매 가격은 0보다 커야한다고 가정

df.loc[df["unit_price"] <= 0, "unit_price"] = pd.NA

# 4. 결측치 처리
# 지역이 없는 행은 UnKnown으로 변경
df["region"] = df["region"].fillna("Unknown")

# 상품 가격 결측치는 같은 카테고리의 중앙값으로 대체
category_median = df.groupby("category")[
    "unit_price"
].transform("median")

df["unit_price"] = df["unit_price"].fillna(
    category_median
)

# 그래도 남은 결측치는 전체 중앙값으로 대체
df["unit_price"] = df["unit_price"].fillna(
    df["unit_price"].median()
)

# 날짜 결측치는 제거
df = df.dropna(subset=["order_date"]).copy()

# 5. IQR 방식으로 이상치 원저라이징
def winsorize_iqr(
        data : pd.DataFrame,
        column : str,
) -> pd.DataFrame:
    result = data.copy()

    q1 = result[column].quantile(0.25)
    q3 = result[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    print(f"\n[{column} 이상치 기준]")
    print(f"하한값 : {lower_bound:,.2f}")
    print(f"상한값 : {upper_bound:,.2f}")

    result[column] = result[column].clip(
        lower = lower_bound,
        upper = upper_bound,
    )

    return result

df = winsorize_iqr(df,"quantity")
df = winsorize_iqr(df, "unit_price")

# 6. 실제 판매금액 계산
df["sales"] = (
    df["quantity"]
    * df["unit_price"]
    * (1-df["discount"])
)

# 7. 지역 및 카테고리별 집계
summary = (
    df.groupby(
        ["region","category"],
        as_index=False,
    )
    .agg(
        order_count=("order_id","count"),
        total_quantity=("quantity","sum"),
        average_price=("unit_price","mean"),
        total_sales=("sales","sum"),
    )
    .sort_values(
        "total_sales",
        ascending=False,
    )
)

print("\n===지역&카테고리별 집계===")
print(summary.head(20).to_string(index=False))

# 8. 지역별&카테고리별 매출 피벗 테이블
sales_pivot = df.pivot_table(
    index="region",
    columns="category",
    values="sales",
    aggfunc="sum",
    fill_value=0,
)

print("\n===지역별 카테고리 매출 피벗===")
print(sales_pivot.round(0))

# 9. 지역 정보 테이블과 merge
region_info = pd.DataFrame(
    {
        "region":[
            "Seoul",
            "Busan",
            "Incheon",
            "Daegu",
            "Gwangju",
            "Unknown",
        ],
        "region_group" : [
            "Capital",
            "Southeast",
            "Capital",
            "Southeast",
            "Southwest",
            "Missing",
        ],
    }
)

merged_summary = summary.merge(
    region_info,
    on="region",
    how="left",
)

print("\n===지역 정보 병합 결과===")
print(merged_summary.head(20).to_string(index=False))

# 10. 정제 결과 확인
print("\n===정제 후 데이터 정보===")
print(f"행 개수 : {len(df):,}")

print("\n결측치 개수")
print(df.isna().sum())

print("\n정제된 데이터 일부")
print(df.head(10).to_string(index=False))

