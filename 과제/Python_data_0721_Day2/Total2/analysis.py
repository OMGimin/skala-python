"""종합실습 2: Polars EDA, 통계검정, 이탈 예측 모델을 한 번에 실행한다.

실제 ``telco_churn.csv``의 snake_case 필드를 사용한다. 분석 결과는 Plotly HTML,
joblib 모델, JSON 지표로 저장되며 모든 상대 경로는 이 파일 위치를 기준으로 한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "output"

REQUIRED_COLUMNS = {
    "customer_id",
    "gender",
    "senior",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract",
    "payment_method",
    "num_services",
    "churn",
}
NUMERIC_FEATURES = [
    "senior",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_services",
]
CATEGORICAL_FEATURES = ["gender", "contract", "payment_method"]


def load_data(path: Path = DATA_PATH) -> pl.DataFrame:
    """CSV를 Polars DataFrame으로 읽고 필수 필드와 타깃 값을 검증한다."""
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    frame = pl.read_csv(path, null_values=["", "NA", "null"], try_parse_dates=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"필수 필드가 누락되었습니다: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError("분석할 데이터가 없습니다.")

    invalid_targets = set(frame["churn"].drop_nulls().unique().to_list()) - {0, 1}
    if frame["churn"].null_count() or invalid_targets:
        raise ValueError("churn 필드는 결측값 없이 0 또는 1이어야 합니다.")
    return frame


def run_polars_eda(frame: pl.DataFrame) -> dict[str, Any]:
    """Polars 표현식으로 결측치, 이탈률, 계약별 요약 통계를 계산한다."""
    missing = frame.select(pl.all().null_count()).to_dicts()[0]
    churn_summary = (
        frame.group_by("churn")
        .agg(
            pl.len().alias("customers"),
            pl.col("monthly_charges").mean().round(2).alias("avg_monthly_charges"),
            pl.col("tenure_months").mean().round(2).alias("avg_tenure_months"),
        )
        .sort("churn")
    )
    contract_summary = (
        frame.group_by("contract")
        .agg(
            pl.len().alias("customers"),
            pl.col("churn").mean().round(4).alias("churn_rate"),
        )
        .sort("churn_rate", descending=True)
    )
    return {
        "rows": frame.height,
        "columns": frame.width,
        "missing": missing,
        "churn_summary": churn_summary,
        "contract_summary": contract_summary,
    }


def run_statistical_tests(data: pd.DataFrame) -> dict[str, float]:
    """이탈 여부별 월요금 Welch t-test와 계약유형·이탈 카이제곱 검정을 수행한다."""
    churned = data.loc[data["churn"] == 1, "monthly_charges"].dropna()
    retained = data.loc[data["churn"] == 0, "monthly_charges"].dropna()
    t_statistic, t_pvalue = ttest_ind(churned, retained, equal_var=False)

    contingency = pd.crosstab(data["contract"], data["churn"])
    chi2_statistic, chi2_pvalue, _, _ = chi2_contingency(contingency)
    return {
        "t_statistic": float(t_statistic),
        "t_pvalue": float(t_pvalue),
        "chi2_statistic": float(chi2_statistic),
        "chi2_pvalue": float(chi2_pvalue),
    }


def create_eda_report(data: pd.DataFrame, output_path: Path) -> None:
    """월요금 분포와 계약별 이탈률을 한 Plotly HTML 보고서로 저장한다."""
    contract_rates = data.groupby("contract", observed=True)["churn"].mean().sort_values()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("이탈 여부별 월 요금", "계약 유형별 이탈률"),
    )
    for churn_value, label, color in ((0, "유지", "#3B82F6"), (1, "이탈", "#EF4444")):
        figure.add_trace(
            go.Box(
                y=data.loc[data["churn"] == churn_value, "monthly_charges"],
                name=label,
                marker_color=color,
                boxmean=True,
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=contract_rates.index,
            y=contract_rates.values,
            name="이탈률",
            marker_color="#10B981",
            text=[f"{value:.1%}" for value in contract_rates.values],
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="월 요금", row=1, col=1)
    figure.update_yaxes(title_text="이탈률", tickformat=".0%", row=1, col=2)
    figure.update_layout(
        title="통신 고객 이탈 EDA 리포트",
        template="plotly_white",
        height=560,
        showlegend=True,
    )
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def build_model(data: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    """학습 데이터에만 전처리를 적합한 뒤 RandomForest 모델과 평가 지표를 반환한다."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, x_test, y_train, y_test = train_test_split(
        data[features],
        data["churn"],
        test_size=0.2,
        random_state=42,
        stratify=data["churn"],
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    return model, metrics


def main() -> None:
    """전체 분석을 실행하고 콘솔 및 output 폴더에 결과를 남긴다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    eda = run_polars_eda(frame)
    data = frame.to_pandas()
    statistics = run_statistical_tests(data)

    report_path = OUTPUT_DIR / "churn_eda_report.html"
    model_path = OUTPUT_DIR / "churn_model.joblib"
    metrics_path = OUTPUT_DIR / "metrics.json"
    create_eda_report(data, report_path)
    model, model_metrics = build_model(data)
    joblib.dump(model, model_path)

    metrics = {**statistics, **model_metrics}
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[Polars EDA]")
    print(f"크기: {eda['rows']:,}행 × {eda['columns']}열")
    print(f"결측치: {eda['missing']}")
    print("\n이탈 여부별 요약")
    print(eda["churn_summary"])
    print("\n계약 유형별 요약")
    print(eda["contract_summary"])
    print("\n[통계 검정]")
    print(f"Welch t-test p-value: {statistics['t_pvalue']:.6e}")
    print(f"카이제곱 검정 p-value: {statistics['chi2_pvalue']:.6e}")
    print("\n[머신러닝]")
    print(f"ROC-AUC: {model_metrics['roc_auc']:.4f}")
    print(f"EDA 리포트: {report_path}")
    print(f"학습 모델: {model_path}")
    print(f"평가 지표: {metrics_path}")


if __name__ == "__main__":
    main()
