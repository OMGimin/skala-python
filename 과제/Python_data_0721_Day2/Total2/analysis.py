"""종합실습 2: EDA + 통계 검정 + 머신러닝 파이프라인.

분석 순서는 접근 가이드의 권장 흐름을 따른다.
Polars EDA -> Plotly 시각화 -> 통계 검정 -> 누수 없는 ML Pipeline -> 저장.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DAY2_DIR = BASE_DIR.parent
DATA_PATH = DAY2_DIR / "data" / "telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "output"

TARGET = "churn"
ID_COLUMN = "customer_id"
NUMERIC_COLUMNS = [
    "senior",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_services",
]
CATEGORICAL_COLUMNS = ["gender", "contract", "payment_method"]
RANDOM_STATE = 42


def load_and_explore(path: Path = DATA_PATH) -> tuple[pl.DataFrame, dict[str, Any]]:
    """CSV를 Polars로 읽고 재현 가능한 핵심 EDA 지표를 계산한다."""
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    frame = pl.read_csv(path, try_parse_dates=True)
    required = {ID_COLUMN, TARGET, *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    churn_counts = {
        str(row[TARGET]): row["len"]
        for row in frame.group_by(TARGET).len().sort(TARGET).to_dicts()
    }
    contract_summary = (
        frame.group_by("contract")
        .agg(
            pl.len().alias("customers"),
            pl.col(TARGET).sum().alias("churned"),
            (pl.col(TARGET).mean() * 100).round(2).alias("churn_rate_percent"),
        )
        .sort("churn_rate_percent", descending=True)
    )
    numeric_summary = frame.select(NUMERIC_COLUMNS).describe()
    summary = {
        "rows": frame.height,
        "columns": frame.width,
        "duplicate_customer_ids": frame.select(pl.col(ID_COLUMN).is_duplicated().sum()).item(),
        "null_counts": {column: frame[column].null_count() for column in frame.columns},
        "churn_counts": churn_counts,
        "churn_rate_percent": round(frame[TARGET].mean() * 100, 2),
        "contract_summary": contract_summary.to_dicts(),
        "numeric_summary": numeric_summary.to_dicts(),
    }
    return frame, summary


def run_statistical_tests(frame: pl.DataFrame) -> dict[str, Any]:
    """요금 차이는 Welch t-test, 계약과 이탈은 카이제곱으로 검정한다."""
    pandas_frame = frame.to_pandas()
    stayed = pandas_frame.loc[pandas_frame[TARGET] == 0, "monthly_charges"].dropna()
    churned = pandas_frame.loc[pandas_frame[TARGET] == 1, "monthly_charges"].dropna()
    t_statistic, t_pvalue = ttest_ind(churned, stayed, equal_var=False)

    contingency = pd.crosstab(pandas_frame["contract"], pandas_frame[TARGET])
    chi2_statistic, chi2_pvalue, degrees_of_freedom, _ = chi2_contingency(contingency)

    return {
        "welch_t_test": {
            "variable": "monthly_charges",
            "churn_mean": round(float(churned.mean()), 4),
            "stay_mean": round(float(stayed.mean()), 4),
            "statistic": float(t_statistic),
            "p_value": float(t_pvalue),
            "significant_at_0_05": bool(t_pvalue < 0.05),
            "interpretation": "월 요금과 이탈 여부 사이에 통계적으로 유의한 차이가 있다.",
        },
        "chi_square_test": {
            "variables": ["contract", TARGET],
            "statistic": float(chi2_statistic),
            "p_value": float(chi2_pvalue),
            "degrees_of_freedom": int(degrees_of_freedom),
            "significant_at_0_05": bool(chi2_pvalue < 0.05),
            "interpretation": "계약 유형과 이탈 여부 사이에 통계적으로 유의한 연관성이 있다.",
        },
        "caution": "통계적 유의성과 연관성은 인과관계를 의미하지 않는다.",
    }


def build_model_pipeline() -> Pipeline:
    """결측 처리와 인코딩을 모델 내부에 묶어 데이터 누수를 방지한다."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def train_and_evaluate(
    frame: pl.DataFrame,
) -> tuple[Pipeline, dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    """계층 분할 후 Pipeline을 학습하고 확률 기반 ROC-AUC를 평가한다."""
    pandas_frame = frame.to_pandas()
    features = pandas_frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    target = pandas_frame[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    pipeline = build_model_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    metrics = {
        "random_state": RANDOM_STATE,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "train_churn_rate": round(float(y_train.mean()), 6),
        "test_churn_rate": round(float(y_test.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions)), 6),
        "recall": round(float(recall_score(y_test, predictions)), 6),
        "f1": round(float(f1_score(y_test, predictions)), 6),
        "confusion_matrix": matrix.tolist(),
        "evaluation_note": "불균형 데이터이므로 정확도보다 확률 기반 ROC-AUC를 우선 해석한다.",
    }
    return pipeline, metrics, (false_positive_rate, true_positive_rate)


def create_html_report(
    frame: pl.DataFrame,
    eda: dict[str, Any],
    statistics: dict[str, Any],
    metrics: dict[str, Any],
    roc_points: tuple[np.ndarray, np.ndarray],
    output_path: Path,
) -> None:
    """EDA·통계·모델 결과를 하나의 오프라인 Plotly HTML로 저장한다."""
    pandas_frame = frame.to_pandas()
    contract = pd.DataFrame(eda["contract_summary"])
    false_positive_rate, true_positive_rate = roc_points
    matrix = np.array(metrics["confusion_matrix"])

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "이탈 여부별 월 요금 분포",
            "계약 유형별 이탈률",
            "ROC 곡선",
            "혼동행렬",
        ),
        specs=[[{"type": "histogram"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "heatmap"}]],
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )
    for churn_value, label, color in [(0, "잔류", "#2E86AB"), (1, "이탈", "#E45756")]:
        figure.add_trace(
            go.Histogram(
                x=pandas_frame.loc[pandas_frame[TARGET] == churn_value, "monthly_charges"],
                name=label,
                opacity=0.65,
                nbinsx=35,
                marker_color=color,
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=contract["contract"],
            y=contract["churn_rate_percent"],
            name="이탈률(%)",
            marker_color="#F3A712",
            text=contract["churn_rate_percent"].map(lambda value: f"{value:.1f}%"),
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            name=f"RandomForest (AUC={metrics['roc_auc']:.3f})",
            line={"color": "#2CA58D", "width": 3},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="무작위 기준", line={"dash": "dash", "color": "#999"}
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=matrix,
            x=["예측 잔류", "예측 이탈"],
            y=["실제 잔류", "실제 이탈"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
            name="혼동행렬",
        ),
        row=2,
        col=2,
    )
    figure.update_layout(
        title={
            "text": (
                "통신 고객 이탈 분석 리포트"
                f"<br><sup>고객 {eda['rows']:,}명 · 이탈률 {eda['churn_rate_percent']:.2f}% · "
                f"ROC-AUC {metrics['roc_auc']:.3f}</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        barmode="overlay",
        height=850,
        margin={"l": 70, "r": 50, "t": 115, "b": 60},
        legend={"orientation": "h", "y": -0.1},
    )
    figure.update_xaxes(title_text="월 요금", row=1, col=1)
    figure.update_yaxes(title_text="고객 수", row=1, col=1)
    figure.update_yaxes(title_text="이탈률(%)", range=[0, max(contract["churn_rate_percent"]) * 1.2], row=1, col=2)
    figure.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=2, col=1)
    figure.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=2, col=1)

    t_test = statistics["welch_t_test"]
    chi_square = statistics["chi_square_test"]
    interpretation = f"""
    <section style="font-family:Malgun Gothic,Arial,sans-serif;max-width:1100px;margin:24px auto;
                    padding:22px;border:1px solid #d9e2ec;border-radius:14px;background:#f7fafc">
      <h2 style="color:#173f5f">통계 검정과 해석</h2>
      <ul style="line-height:1.8">
        <li>Welch t-test: 이탈 평균 월 요금 {t_test['churn_mean']:.2f}, 잔류 평균 {t_test['stay_mean']:.2f},
            p={t_test['p_value']:.3e} (유의)</li>
        <li>카이제곱 검정: 계약 유형과 이탈 여부, p={chi_square['p_value']:.3e} (유의)</li>
        <li>모델: ROC-AUC {metrics['roc_auc']:.3f}, Recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f}</li>
      </ul>
      <p><strong>주의:</strong> 통계적 연관성은 인과관계를 의미하지 않습니다.</p>
    </section>
    """
    output_path.write_text(
        interpretation + figure.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False}),
        encoding="utf-8",
    )


def main() -> None:
    """전체 분석을 순서대로 실행하고 공유 가능한 결과물을 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, eda = load_and_explore()
    statistics = run_statistical_tests(frame)
    model, metrics, roc_points = train_and_evaluate(frame)

    combined_metrics = {"eda": eda, "statistics": statistics, "model": metrics}
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(combined_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(eda["contract_summary"]).to_csv(
        OUTPUT_DIR / "contract_churn_summary.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(model, OUTPUT_DIR / "churn_model.joblib")
    create_html_report(
        frame,
        eda,
        statistics,
        metrics,
        roc_points,
        OUTPUT_DIR / "churn_eda_report.html",
    )

    t_test = statistics["welch_t_test"]
    chi_square = statistics["chi_square_test"]
    print("=== 종합실습 2: EDA + 통계 + ML 파이프라인 ===")
    print(f"데이터: {eda['rows']:,}행, {eda['columns']}열")
    print(f"전체 이탈률: {eda['churn_rate_percent']:.2f}%")
    print(f"total_charges 결측치: {eda['null_counts']['total_charges']}건")
    print("\n[통계 검정]")
    print(f"Welch t-test p-value: {t_test['p_value']:.3e} -> 유의")
    print(f"카이제곱 p-value: {chi_square['p_value']:.3e} -> 유의")
    print("주의: 연관성은 인과관계를 의미하지 않습니다.")
    print("\n[머신러닝 평가]")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print("\n[생성 결과]")
    for filename in [
        "churn_eda_report.html",
        "churn_model.joblib",
        "metrics.json",
        "contract_churn_summary.csv",
    ]:
        print(OUTPUT_DIR / filename)


if __name__ == "__main__":
    main()
