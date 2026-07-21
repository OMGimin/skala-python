from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["trip_distance", "fare_amount", "trip_duration_min", "pickup_hour"]
CATEGORICAL_FEATURES = ["day_of_week", "PULocationID", "DOLocationID"]
TARGET = "high_tip"


def build_model_pipeline() -> Pipeline:
    """전처리와 확률적 로지스틱 분류를 결합한 sklearn Pipeline을 만든다."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=1_000,
                    tol=1e-3,
                    class_weight="balanced",
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(
    frame: pd.DataFrame,
    model_path: Path,
    metrics_path: Path,
    *,
    max_rows: int = 250_000,
) -> dict[str, Any]:
    """결정적 표본으로 모델을 학습·평가하고 Pipeline 전체를 저장한다."""
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"모델 학습 필수 컬럼이 없습니다: {missing_columns}")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 10:
        raise ValueError("max_rows는 10 이상의 정수여야 합니다.")
    data = frame[columns].copy()
    if data[TARGET].isna().any():
        raise ValueError("목표변수 high_tip에 결측치가 있습니다.")
    class_counts = data[TARGET].value_counts()
    if len(class_counts) != 2:
        raise ValueError("모델 학습에는 high_tip의 두 클래스가 모두 필요합니다.")
    if int(class_counts.min()) < 5:
        raise ValueError("안정적인 계층 분할을 위해 각 클래스에 최소 5개 관측치가 필요합니다.")
    if len(data) > max_rows:
        data, _ = train_test_split(
            data,
            train_size=max_rows,
            random_state=42,
            stratify=data[TARGET],
        )

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(columns=[TARGET]),
        data[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET],
    )
    pipeline = build_model_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "model": "SGDClassifier(loss='log_loss')",
        "target": "high_tip (tip_rate >= 20%)",
        "sample_rows": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "positive_rate": float(data[TARGET].mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "excluded_leakage_features": ["tip_amount", "tip_rate", "total_amount", "base_amount"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
