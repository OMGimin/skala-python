"""
실습명: [실습 4] 시각화 4종 · 통계 검정 · sklearn Pipeline
작성자: 광주 3반 김지민
활용 데이터: sales_100k.csv

[프로그램 설명]
실습 3에서 사용한 sales_100k.csv를 읽고 IQR 기준으로 amount
이상치를 제거한다.

정제된 데이터를 이용하여 2×2 서브플롯으로 EDA 시각화 4종을 만들고,
서울·부산 매출의 평균 차이를 독립표본 t-test로 검정한다.
category와 payment_method의 독립성을 카이제곱 검정으로 확인한다.

ColumnTransformer와 sklearn Pipeline을 이용해 amount 예측 모델을
훈련·평가·저장·재로딩하고, 지역·카테고리별 총매출을 Plotly
인터랙티브 차트로 저장한다.

[변경 내역]
- 2026-07-21: 최초 작성
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import joblib
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CSV_PATH = Path(__file__).resolve().parent / "sales_100k.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

FIGURE_PATH = OUTPUT_DIR / "practice4_eda.png"
MODEL_PATH = OUTPUT_DIR / "sales_pipeline.joblib"
HTML_PATH = OUTPUT_DIR / "sales_interactive.html"

def load_and_clean_data(csv_path:Path)-> pd.DataFrame:
    """CSV를 읽고 amount의 IQR 이상치를 제거한다."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    sales = pd.read_csv(csv_path)

    required_columns = {
        "order_date",
        "region",
        "category",
        "quantity",
        "unit_price",
        "payment_method",
        "customer_age",
        "customer_gender",
        "amount",
    }

    missing_columns = required_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {sorted(missing_columns)}"
        )

    sales["amount"] = pd.to_numeric(
        sales["amount"],
        errors="coerce",
    )

    sales["order_date"] = pd.to_datetime(
        sales["order_date"],
        errors="coerce",
    )

    q1 = sales["amount"].quantile(0.25)
    q3 = sales["amount"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    cleaned_sales = sales[
        sales["amount"].between(
            lower_bound,
            upper_bound,
        )
    ].copy()

    print("\n[IQR 이상치 처리]")
    print(f"정상 범위: {lower_bound:,.2f} ~ {upper_bound:,.2f}")
    print(f"처리 전: {len(sales):,}건")
    print(f"처리 후: {len(cleaned_sales):,}건")
    print(f"제거 건수: {len(sales) - len(cleaned_sales):,}건")

    return cleaned_sales

def create_eda_visualization(sales: pd.DataFrame) -> None:
    """2×2 서브플롯에 EDA 차트 4종을 작성하고 저장한다."""

    sample_size = min(50_000, len(sales))
    plot_data = sales.sample(
        n=sample_size,
        random_state=42,
    )

    monthly_sales = (
        sales
        .dropna(subset=["order_date"])
        .assign(
            month=lambda data: (
                data["order_date"]
                .dt.to_period("M")
                .astype(str)
            )
        )
        .groupby("month", as_index=False)
        .agg(total_amount=("amount", "sum"))
        .sort_values("month")
    )

    numeric_columns = [
        "quantity",
        "unit_price",
        "customer_age",
        "amount",
    ]

    correlation = plot_data[numeric_columns].corr()

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
    )

    sns.histplot(
        data=plot_data,
        x="amount",
        kde=True,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Amount Histogram and KDE")

    sns.boxplot(
        data=plot_data,
        x="category",
        y="amount",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Amount by Category")
    axes[0, 1].tick_params(axis="x", rotation=30)

    sns.lineplot(
        data=monthly_sales,
        x="month",
        y="total_amount",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Monthly Total Sales")
    axes[1, 0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Correlation Heatmap")

    fig.suptitle(
        "Sales EDA Dashboard",
        fontsize=18,
    )
    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"\nEDA 차트 저장 완료: {FIGURE_PATH}")

def perform_t_test(sales: pd.DataFrame) -> None:
    """서울과 부산의 평균 매출 차이를 t-test로 검정한다."""

    seoul_amount = sales.loc[
        sales["region"] == "서울",
        "amount",
    ].dropna()

    busan_amount = sales.loc[
        sales["region"] == "부산",
        "amount",
    ].dropna()

    t_statistic, p_value = ttest_ind(
        seoul_amount,
        busan_amount,
        equal_var=False,
        nan_policy="omit",
    )

    print("\n[서울 vs 부산 평균 매출 t-test]")
    print(f"서울 평균 매출: {seoul_amount.mean():,.2f}")
    print(f"부산 평균 매출: {busan_amount.mean():,.2f}")
    print(f"t 통계량: {t_statistic:.4f}")
    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("해석: 두 지역의 평균 매출에는 통계적으로 유의한 차이가 있습니다.")
    else:
        print("해석: 두 지역의 평균 매출에는 통계적으로 유의한 차이가 없습니다.")

def perform_chi_square_test(sales: pd.DataFrame) -> None:
    """category와 payment_method의 독립성을 검정한다."""

    contingency_table = pd.crosstab(
        sales["category"],
        sales["payment_method"],
    )

    chi2, p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    print("\n[category × payment_method 카이제곱 검정]")
    print("\n분할표:")
    print(contingency_table)

    print(f"\n카이제곱 통계량: {chi2:.4f}")
    print(f"자유도: {degrees_of_freedom}")
    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("해석: category와 payment_method는 서로 독립적이지 않습니다.")
    else:
        print("해석: category와 payment_method가 관련 있다고 보기 어렵습니다.")

def train_and_save_pipeline(sales: pd.DataFrame) -> None:
    """전처리와 회귀 모델을 Pipeline으로 묶어 학습하고 저장한다."""

    numeric_features = [
        "quantity",
        "unit_price",
        "customer_age",
    ]

    categorical_features = [
        "region",
        "category",
        "payment_method",
        "customer_gender",
    ]

    feature_columns = (
        numeric_features
        + categorical_features
    )

    model_data = sales.dropna(subset=["amount"]).copy()

    x = model_data[feature_columns]
    y = model_data["amount"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    sales_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    sales_pipeline.fit(x_train, y_train)

    predictions = sales_pipeline.predict(x_test)

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\n[sklearn Pipeline 평가]")
    print(f"학습 데이터: {len(x_train):,}건")
    print(f"평가 데이터: {len(x_test):,}건")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² 점수: {r2:.4f}")

    joblib.dump(
        sales_pipeline,
        MODEL_PATH,
    )

    loaded_pipeline = joblib.load(MODEL_PATH)
    reloaded_predictions = loaded_pipeline.predict(
        x_test.head(5)
    )

    assert np.allclose(
        predictions[:5],
        reloaded_predictions,
    )

    print(f"모델 저장 완료: {MODEL_PATH}")
    print("저장 모델 재로딩 및 예측 검증 통과")

def create_plotly_chart(sales: pd.DataFrame) -> None:
    """지역·카테고리별 총매출 차트를 HTML로 저장한다."""

    grouped_sales = (
        sales
        .groupby(
            ["region", "category"],
            as_index=False,
            dropna=False,
        )
        .agg(total_amount=("amount", "sum"))
        .sort_values(
            "total_amount",
            ascending=False,
        )
    )

    grouped_sales["region"] = (
        grouped_sales["region"]
        .fillna("미분류")
    )

    grouped_sales["category"] = (
        grouped_sales["category"]
        .fillna("미분류")
    )

    figure = px.bar(
        grouped_sales,
        x="region",
        y="total_amount",
        color="category",
        barmode="group",
        title="지역·카테고리별 총매출",
        labels={
            "region": "지역",
            "category": "카테고리",
            "total_amount": "총매출",
        },
    )

    figure.write_html(
        HTML_PATH,
        include_plotlyjs="cdn",
    )

    print(f"\nPlotly HTML 저장 완료: {HTML_PATH}")

def main() -> None:
    """실습 4의 전체 분석 과정을 실행한다."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    cleaned_sales = load_and_clean_data(CSV_PATH)

    create_eda_visualization(cleaned_sales)
    perform_t_test(cleaned_sales)
    perform_chi_square_test(cleaned_sales)
    train_and_save_pipeline(cleaned_sales)
    create_plotly_chart(cleaned_sales)

    assert FIGURE_PATH.exists()
    assert MODEL_PATH.exists()
    assert HTML_PATH.exists()

    print("\n[실습 4 완료]")
    print("시각화, 통계 검정, Pipeline, 파일 저장을 완료했습니다.")


if __name__ == "__main__":
    main()