"""NYC Yellow Taxi의 시간·지역별 현금 결제 패턴 분석 및 예측.

Pandas·Polars 데이터 비교, 정제, EDA, 통계검정, 시각화, sklearn Pipeline,
모델 평가·저장 및 Markdown 보고서 생성을 하나의 실행 흐름으로 수행한다.

변경 내역:
- 2026-07-21: 카드·현금 결제 방식 예측 프로젝트 최초 작성
- 2026-07-21: 전체 비율 대신 JFK·새벽 시간대의 현금 집중 조건으로 주제 구체화
- 2026-07-21: 조별 코드를 통합하고 프로젝트 기준 경로로 수정
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cash-payment")
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_DIR / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "01_cash_payment"
README_PATH = PROJECT_DIR / "README.md"
REPORT_START = "<!-- AUTO_REPORT_START -->"
REPORT_END = "<!-- AUTO_REPORT_END -->"
RANDOM_STATE = 42
MODEL_SAMPLE_SIZE = 400_000

NUMERIC_FEATURES = [
    "passenger_count",
    "trip_distance",
    "duration_min",
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "cbd_congestion_fee",
    "is_jfk_pickup",
    "is_early_morning",
    "is_jfk_early",
]
CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
]
LEAKAGE_COLUMNS = ["payment_type", "tip_amount", "total_amount", "is_cash"]


def json_default(value: Any) -> Any:
    """Serialize numpy values in JSON outputs."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def benchmark_loaders(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load with Pandas and Polars and verify shape, columns, and missing counts."""
    start = time.perf_counter()
    pdf = pd.read_parquet(path)
    pandas_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pldf = pl.read_parquet(path)
    polars_seconds = time.perf_counter() - start

    pandas_nulls = pdf.isna().sum().astype(int).to_dict()
    polars_nulls = {
        column: int(pldf[column].null_count()) for column in pldf.columns
    }
    result = {
        "pandas_seconds": pandas_seconds,
        "polars_seconds": polars_seconds,
        "pandas_shape": list(pdf.shape),
        "polars_shape": list(pldf.shape),
        "same_shape": pdf.shape == pldf.shape,
        "same_columns": list(pdf.columns) == pldf.columns,
        "same_missing_counts": pandas_nulls == polars_nulls,
        "missing_counts": pandas_nulls,
    }
    if not all(
        result[key]
        for key in ("same_shape", "same_columns", "same_missing_counts")
    ):
        raise ValueError("Pandas and Polars loading results do not match.")
    del pldf
    return pdf, result


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create behavioral features and filter invalid card/cash trips."""
    df = raw.drop_duplicates().copy()
    duplicate_rows = len(raw) - len(df)
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("int8")
    df["pickup_dayofweek"] = (
        df["tpep_pickup_datetime"].dt.dayofweek.astype("int8")
    )
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype("int8")
    df["is_jfk_pickup"] = (df["PULocationID"] == 132).astype("int8")
    df["is_early_morning"] = df["pickup_hour"].isin([4, 5, 6]).astype("int8")
    df["is_jfk_early"] = (
        df["is_jfk_pickup"].eq(1) & df["is_early_morning"].eq(1)
    ).astype("int8")

    masks = {
        "card_or_cash": df["payment_type"].isin([1, 2]),
        "pickup_in_may": df["tpep_pickup_datetime"].between(
            "2026-05-01", "2026-06-01", inclusive="left"
        ),
        "duration_1_to_180": df["duration_min"].between(1, 180),
        "distance_0_1_to_100": df["trip_distance"].between(0.1, 100),
        "fare_0_01_to_500": df["fare_amount"].between(0.01, 500),
        "positive_total": df["total_amount"] > 0,
    }
    valid = pd.Series(True, index=df.index)
    sequential = {"raw": len(raw), "after_duplicates": len(df)}
    for name, mask in masks.items():
        valid &= mask.fillna(False)
        sequential[f"after_{name}"] = int(valid.sum())

    clean = df.loc[valid].copy()
    clean["is_cash"] = (clean["payment_type"] == 2).astype("int8")
    clean["payment_label"] = clean["is_cash"].map({0: "Card", 1: "Cash"})

    # No class resampling: retain the observed card/cash proportions.
    counts = clean["payment_label"].value_counts()
    quality = {
        "duplicates_removed": duplicate_rows,
        "sequential_counts": sequential,
        "final_rows": len(clean),
        "class_counts": counts.to_dict(),
        "class_proportions": (counts / len(clean)).to_dict(),
        "rows_with_any_missing": int(clean.isna().any(axis=1).sum()),
    }
    return clean, quality


def create_eda_tables(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Create EDA aggregations and save reusable CSV tables."""
    hourly = (
        df.groupby("pickup_hour", as_index=False)
        .agg(cash_rate=("is_cash", "mean"), trips=("is_cash", "size"))
    )
    pickup_zone = (
        df.groupby("PULocationID", as_index=False)
        .agg(cash_rate=("is_cash", "mean"), trips=("is_cash", "size"))
        .sort_values("cash_rate", ascending=False)
    )
    distance_bins = [0.1, 0.5, 1, 2, 3, 5, 10, 20, 100]
    distance_labels = [
        "0.1–0.5",
        "0.5–1",
        "1–2",
        "2–3",
        "3–5",
        "5–10",
        "10–20",
        "20–100",
    ]
    df["distance_band"] = pd.cut(
        df["trip_distance"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
        right=False,
    )
    by_distance = (
        df.groupby("distance_band", observed=True, as_index=False)
        .agg(cash_rate=("is_cash", "mean"), trips=("is_cash", "size"))
    )
    segment = (
        df.groupby(["is_jfk_pickup", "is_early_morning"], as_index=False)
        .agg(cash_rate=("is_cash", "mean"), trips=("is_cash", "size"))
    )
    segment["segment"] = segment.apply(
        lambda row: (
            ("JFK" if row["is_jfk_pickup"] else "Non-JFK")
            + " · "
            + ("04–06" if row["is_early_morning"] else "Other hours")
        ),
        axis=1,
    )
    top_zones = df["PULocationID"].value_counts().head(25).index
    zone_hour = (
        df.loc[df["PULocationID"].isin(top_zones)]
        .groupby(["PULocationID", "pickup_hour"], as_index=False)
        .agg(cash_rate=("is_cash", "mean"), trips=("is_cash", "size"))
    )

    hourly.to_csv(output_dir / "cash_rate_by_hour.csv", index=False)
    pickup_zone.to_csv(output_dir / "cash_rate_by_pickup_zone.csv", index=False)
    by_distance.to_csv(output_dir / "cash_rate_by_distance.csv", index=False)
    segment.to_csv(output_dir / "cash_rate_jfk_early_segments.csv", index=False)
    zone_hour.to_csv(output_dir / "cash_rate_zone_hour.csv", index=False)

    numeric_columns = [
        "trip_distance",
        "duration_min",
        "fare_amount",
        "passenger_count",
    ]
    return {
        "overall_cash_rate": float(df["is_cash"].mean()),
        "descriptive_statistics": df[numeric_columns].describe(
            percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
        ).to_dict(),
        "correlation": df[numeric_columns].corr().to_dict(),
        "segment_records": segment.to_dict("records"),
    }


def statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Run Welch t-tests plus the primary JFK-early chi-square test."""
    t_tests: dict[str, Any] = {}
    for metric in ("trip_distance", "duration_min", "fare_amount"):
        cash = df.loc[df["is_cash"] == 1, metric].dropna().astype(float)
        card = df.loc[df["is_cash"] == 0, metric].dropna().astype(float)
        test = ttest_ind(cash, card, equal_var=False, nan_policy="omit")
        pooled = np.sqrt(
            (
                (len(cash) - 1) * cash.var(ddof=1)
                + (len(card) - 1) * card.var(ddof=1)
            )
            / (len(cash) + len(card) - 2)
        )
        effect = float((cash.mean() - card.mean()) / pooled)
        t_tests[metric] = {
            "cash_mean": float(cash.mean()),
            "card_mean": float(card.mean()),
            "mean_difference_cash_minus_card": float(cash.mean() - card.mean()),
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "cohens_d": effect,
            "interpretation": (
                "Statistically significant, but practical magnitude is "
                + ("negligible." if abs(effect) < 0.2 else "non-negligible.")
                if test.pvalue < 0.05
                else "Not statistically significant at alpha=0.05."
            ),
        }

    table = pd.crosstab(df["is_jfk_early"], df["is_cash"])
    chi2, p_value, dof, expected = chi2_contingency(table)
    cramer_v = float(np.sqrt(chi2 / len(df)))
    exposed = df.loc[df["is_jfk_early"] == 1, "is_cash"]
    reference = df.loc[df["is_jfk_early"] == 0, "is_cash"]
    exposed_rate = float(exposed.mean())
    reference_rate = float(reference.mean())
    odds_ratio = float(
        (exposed_rate / (1 - exposed_rate))
        / (reference_rate / (1 - reference_rate))
    )
    chi_square = {
        "contingency_table": table.to_dict(),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": cramer_v,
        "jfk_early_cash_rate": exposed_rate,
        "jfk_early_trip_count": len(exposed),
        "jfk_early_cash_count": int(exposed.sum()),
        "other_cash_rate": reference_rate,
        "percentage_point_difference": (exposed_rate - reference_rate) * 100,
        "odds_ratio": odds_ratio,
        "interpretation": (
            "Cash payment is associated with the JFK 04–06 segment; this is "
            "an observational association, not a causal effect."
        ),
    }
    return {"welch_t_tests": t_tests, "jfk_early_chi_square": chi_square}


def create_charts(output_dir: Path) -> None:
    """Save charts from compact pre-aggregated EDA tables."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    segments = pd.read_csv(output_dir / "cash_rate_jfk_early_segments.csv")
    segments["segment"] = segments["segment"].str.replace(" · ", "\n", regex=False)
    sns.barplot(data=segments, x="segment", y="cash_rate", ax=axes[0], color="#d97706")
    axes[0].set(title="Cash rate by JFK and time", xlabel="Segment", ylabel="Cash rate")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for patch, value in zip(axes[0].patches, segments["cash_rate"]):
        axes[0].text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.012,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    hourly = pd.read_csv(output_dir / "cash_rate_by_hour.csv").rename(
        columns={"cash_rate": "is_cash"}
    )
    sns.lineplot(
        data=hourly,
        x="pickup_hour",
        y="is_cash",
        marker="o",
        ax=axes[1],
        color="#2563eb",
    )
    axes[1].set(title="Cash rate by pickup hour", xlabel="Pickup hour", ylabel="Cash rate")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")

    distance = pd.read_csv(output_dir / "cash_rate_by_distance.csv").rename(
        columns={"cash_rate": "is_cash"}
    )
    sns.lineplot(
        data=distance,
        x="distance_band",
        y="is_cash",
        marker="o",
        ax=axes[2],
        color="#059669",
    )
    axes[2].set(title="Cash rate by trip distance", xlabel="Miles", ylabel="Cash rate")
    axes[2].tick_params(axis="x", rotation=40)
    axes[2].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle("When do NYC yellow-taxi riders pay cash?", fontsize=20)
    fig.tight_layout()
    fig.savefig(output_dir / "cash_payment_eda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    heat = pd.read_csv(output_dir / "cash_rate_zone_hour.csv")
    pivot = heat.pivot(index="PULocationID", columns="pickup_hour", values="cash_rate")
    hover = heat.pivot(index="PULocationID", columns="pickup_hour", values="trips")
    figure = px.imshow(
        pivot,
        labels={"x": "Pickup hour", "y": "Pickup zone ID", "color": "Cash rate"},
        title="Cash-payment rate by pickup zone and hour (top 25 zones)",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=max(0.35, float(np.nanpercentile(pivot.values, 99))),
    )
    figure.update_traces(
        customdata=hover.values,
        hovertemplate="Zone %{y}<br>Hour %{x}:00<br>Cash rate %{z:.1%}<br>Trips %{customdata:,.0f}<extra></extra>",
    )
    figure.update_layout(template="plotly_white", height=720)
    figure.write_html(
        output_dir / "cash_rate_zone_hour_heatmap.html",
        include_plotlyjs="cdn",
        full_html=True,
    )


def _evaluate(y_true: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Return cash-focused classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "cash_precision": precision_score(y_true, prediction, zero_division=0),
        "cash_recall": recall_score(y_true, prediction, zero_division=0),
        "cash_f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
        "classification_report": classification_report(
            y_true,
            prediction,
            target_names=["Card", "Cash"],
            output_dict=True,
            zero_division=0,
        ),
    }


def train_model(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Train a leakage-safe, class-weighted Logistic Regression Pipeline."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    leakage = set(features) & set(LEAKAGE_COLUMNS)
    if leakage:
        raise ValueError(f"Leakage found in features: {sorted(leakage)}")

    sample_size = min(MODEL_SAMPLE_SIZE, len(df))
    cash_rows = int(round(sample_size * df["is_cash"].mean()))
    card_rows = sample_size - cash_rows
    sample = pd.concat(
        [
            df.loc[df["is_cash"] == 1].sample(
                n=cash_rows, random_state=RANDOM_STATE
            ),
            df.loc[df["is_cash"] == 0].sample(
                n=card_rows, random_state=RANDOM_STATE
            ),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE)
    X = sample[features]
    y = sample["is_cash"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
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
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=100,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = pipeline.predict(X_test)
    model_metrics = _evaluate(y_test, prediction, probability)

    majority_prediction = np.zeros(len(y_test), dtype=int)
    majority_probability = np.full(len(y_test), y_train.mean(), dtype=float)
    baseline_metrics = _evaluate(y_test, majority_prediction, majority_probability)

    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    coefficient_table = pd.DataFrame(
        {"feature": transformed_names, "coefficient": coefficients}
    )
    coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
    coefficient_table.sort_values("absolute_coefficient", ascending=False).to_csv(
        output_dir / "model_coefficients.csv", index=False
    )
    joblib.dump(pipeline, output_dir / "cash_payment_model.joblib")

    return {
        "sample_rows": sample_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_cash_rate": float(y_test.mean()),
        "features": features,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "model": model_metrics,
        "majority_baseline": baseline_metrics,
    }


def build_report(
    loader: dict[str, Any],
    quality: dict[str, Any],
    eda: dict[str, Any],
    tests: dict[str, Any],
    model: dict[str, Any],
) -> str:
    """Generate the final Korean Markdown report from computed results."""
    chi = tests["jfk_early_chi_square"]
    metrics = model["model"]
    baseline = model["majority_baseline"]
    t_duration = tests["welch_t_tests"]["duration_min"]
    chi_p_text = "< 1e-300" if chi["p_value"] == 0 else f"{chi['p_value']:.3g}"
    duration_p_text = (
        "< 1e-300"
        if t_duration["p_value"] == 0
        else f"{t_duration['p_value']:.3g}"
    )
    return f"""# 언제 현금을 사용하는가?

## 프로젝트 개요

2026년 5월 NYC Yellow Taxi 운행에서 현금 결제가 집중되는 시간·지역 조건을
분석하고, 운행 정보로 현금 결제 여부를 예측했다. 카드·현금의 자연 비율은
유지했으며 샘플링으로 클래스 비율을 바꾸지 않았다.

## 데이터 준비

- 원본: {loader['pandas_shape'][0]:,}행 × {loader['pandas_shape'][1]}열
- Pandas 로딩: {loader['pandas_seconds']:.3f}초
- Polars 로딩: {loader['polars_seconds']:.3f}초
- 행·열·결측값 결과 일치: {loader['same_shape'] and loader['same_columns'] and loader['same_missing_counts']}
- 정제 후: {quality['final_rows']:,}행
- 카드: {quality['class_counts']['Card']:,}건
- 현금: {quality['class_counts']['Cash']:,}건 ({quality['class_proportions']['Cash']:.2%})

## 핵심 EDA 결과

- 전체 현금 결제율: {eda['overall_cash_rate']:.2%}
- JFK 새벽 4~6시 현금 결제율: {chi['jfk_early_cash_rate']:.2%}
- JFK 새벽 표본: {chi['jfk_early_trip_count']:,}건
- 그 외 조건 현금 결제율: {chi['other_cash_rate']:.2%}
- 차이: {chi['percentage_point_difference']:.2f}%p
- 현금 결제 오즈비: {chi['odds_ratio']:.2f}배

JFK 새벽 운행은 전체적으로 드문 현금 결제가 집중되는 핵심 세그먼트다.
다만 관측 데이터의 연관성이므로 공항이나 시간대가 현금 결제를 유발한다고
해석할 수는 없다.

## 통계검정

JFK 새벽 여부와 결제 방식의 카이제곱 검정 p-value는
{chi_p_text}, Cramér's V는 {chi['cramers_v']:.3f}이다.
두 변수 사이에는 통계적으로 유의한 연관성이 확인됐다.

필수 Welch t-test에서 평균 운행시간은 현금 {t_duration['cash_mean']:.2f}분,
카드 {t_duration['card_mean']:.2f}분이었다. p-value는
{duration_p_text}이지만 Cohen's d는
{t_duration['cohens_d']:.3f}으로 효과크기는 작다. 큰 표본에서는 작은 차이도
통계적으로 유의할 수 있으므로 p-value만으로 큰 차이라고 결론 내리지 않는다.

## ML Pipeline

- 학습 모델: class-weighted Logistic Regression
- 모델링 표본: {model['sample_rows']:,}행
- 전처리: 결측값 대치 + 수치형 표준화 + 범주형 One-Hot Encoding
- 누수 제외: `payment_type`, `tip_amount`, `total_amount`, `is_cash`
- 저장 모델: `cash_payment_model.joblib`

| 평가 지표 | 다수 클래스 기준 | 학습 모델 |
|---|---:|---:|
| Accuracy | {baseline['accuracy']:.3f} | {metrics['accuracy']:.3f} |
| Balanced Accuracy | {baseline['balanced_accuracy']:.3f} | {metrics['balanced_accuracy']:.3f} |
| 현금 Precision | {baseline['cash_precision']:.3f} | {metrics['cash_precision']:.3f} |
| 현금 Recall | {baseline['cash_recall']:.3f} | {metrics['cash_recall']:.3f} |
| 현금 F1 | {baseline['cash_f1']:.3f} | {metrics['cash_f1']:.3f} |
| ROC-AUC | {baseline['roc_auc']:.3f} | {metrics['roc_auc']:.3f} |
| PR-AUC | {baseline['pr_auc']:.3f} | {metrics['pr_auc']:.3f} |

Accuracy는 카드가 많은 불균형 데이터에서 과대평가될 수 있다. 현금 탐지가
목적이므로 현금 Recall·F1, Balanced Accuracy와 PR-AUC를 함께 평가한다.

## 결론

현금 사용은 전체 평균만 보면 11% 수준이지만 지역과 시간대의 조합에서는
큰 차이가 나타난다. 특히 JFK 새벽 운행이 가장 선명한 세그먼트였다. 모델은
이러한 조합 패턴을 이용해 다수 클래스 기준보다 현금 탐지 능력을 개선하는지
평가한다.
"""


def update_readme_report(report: str) -> None:
    """README의 자동 분석 결과 구간만 실제 실행값으로 갱신한다."""

    readme = README_PATH.read_text(encoding="utf-8")
    if REPORT_START not in readme or REPORT_END not in readme:
        raise ValueError("README에 자동 보고서 구간 표시가 없습니다.")
    generated = report.replace("\n## ", "\n### ").replace(
        "# 언제 현금을 사용하는가?",
        "## 자동 생성 분석 결과",
        1,
    )
    before, remainder = readme.split(REPORT_START, maxsplit=1)
    _, after = remainder.split(REPORT_END, maxsplit=1)
    README_PATH.write_text(
        f"{before}{REPORT_START}\n\n{generated.strip()}\n\n{REPORT_END}{after}",
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete analysis, visualization, statistics, and ML workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, loader = benchmark_loaders(args.input)
    clean, quality = prepare_data(raw)
    del raw
    gc.collect()
    eda = create_eda_tables(clean, output_dir)
    tests = statistical_tests(clean)
    create_charts(output_dir)
    model = train_model(clean, output_dir)

    results = {
        "loader_comparison": loader,
        "data_quality": quality,
        "eda": eda,
        "statistical_tests": tests,
        "model_evaluation": model,
    }
    with (output_dir / "cash_project_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, default=json_default)
    update_readme_report(build_report(loader, quality, eda, tests, model))

    print(f"정제 데이터: {len(clean):,}행")
    print(f"전체 현금 결제율: {clean['is_cash'].mean():.2%}")
    print(
        "JFK 새벽 현금 결제율:",
        f"{tests['jfk_early_chi_square']['jfk_early_cash_rate']:.2%}",
        f"({tests['jfk_early_chi_square']['jfk_early_trip_count']:,}건)",
    )
    print(f"현금 F1: {model['model']['cash_f1']:.3f}")
    print(f"결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
