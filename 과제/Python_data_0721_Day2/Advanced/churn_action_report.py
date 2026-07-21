"""Advanced: 예측 확률을 실제 고객 관리 행동으로 연결한다.

Total2에서 저장한 Pipeline을 재사용하고, 검증 데이터에서 여러 분류
임계값을 비교한 뒤 전체 고객의 위험도와 연락 우선순위를 생성한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DAY2_DIR = BASE_DIR.parent
TOTAL2_DIR = DAY2_DIR / "Total2"
DATA_PATH = DAY2_DIR / "data" / "telco_churn.csv"
MODEL_PATH = TOTAL2_DIR / "output" / "churn_model.joblib"
OUTPUT_DIR = BASE_DIR / "output"
TEMPLATE_PATH = BASE_DIR / "templates" / "churn_action_report.html"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "senior",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_services",
    "gender",
    "contract",
    "payment_method",
]


def load_inputs() -> tuple[pd.DataFrame, Any]:
    """고객 데이터와 Total2에서 저장한 전체 Pipeline을 불러온다."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"고객 데이터가 없습니다: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "학습 모델이 없습니다. 먼저 Total2/analysis.py를 실행하세요: "
            f"{MODEL_PATH}"
        )
    return pd.read_csv(DATA_PATH), joblib.load(MODEL_PATH)


def compare_thresholds(
    frame: pd.DataFrame,
    model: Any,
) -> tuple[pd.DataFrame, float]:
    """Total2와 동일한 검증 분할에서 임계값별 성능을 비교한다."""
    _, validation = train_test_split(
        frame,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=frame["churn"],
    )
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    actual = validation["churn"].astype(int)
    rows: list[dict[str, float | int]] = []

    for threshold in np.arange(0.20, 0.71, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision_score(actual, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(actual, predictions, zero_division=0)), 4),
                "f1": round(float(f1_score(actual, predictions, zero_division=0)), 4),
                "customers_flagged": int(predictions.sum()),
            }
        )

    comparison = pd.DataFrame(rows)
    # 이 과제에서는 놓치는 고객과 불필요한 연락을 균형 있게 줄이는 F1을 기준으로 선택한다.
    best_row = comparison.sort_values(
        ["f1", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return comparison, float(best_row["threshold"])


def assign_risk_level(probability: float, threshold: float) -> str:
    """확률과 선택 임계값을 바탕으로 설명 가능한 위험 등급을 부여한다."""
    if probability >= max(0.7, threshold + 0.15):
        return "매우 높음"
    if probability >= threshold:
        return "높음"
    if probability >= max(0.3, threshold - 0.15):
        return "관찰"
    return "낮음"


def score_customers(
    frame: pd.DataFrame,
    model: Any,
    threshold: float,
) -> pd.DataFrame:
    """모든 고객을 점수화하고 연락 우선순위를 계산한다."""
    scored = frame.copy()
    scored["churn_probability"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scored["risk_level"] = scored["churn_probability"].map(
        lambda probability: assign_risk_level(float(probability), threshold)
    )
    scored["contact_recommended"] = scored["churn_probability"] >= threshold

    # 확률만 높은 고객보다, 위험 확률과 월 매출 영향이 함께 큰 고객을 먼저 본다.
    scored["priority_score"] = (
        scored["churn_probability"] * scored["monthly_charges"]
    )
    return scored.sort_values("priority_score", ascending=False).reset_index(drop=True)


def build_summary(
    scored: pd.DataFrame,
    comparison: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """CSV와 HTML에서 공통으로 사용할 행동 중심 지표를 만든다."""
    selected = scored[scored["contact_recommended"]]
    best = comparison.loc[comparison["threshold"] == threshold].iloc[0]
    risk_counts = scored["risk_level"].value_counts().to_dict()
    contract_risk = (
        scored.groupby("contract", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            average_risk=("churn_probability", "mean"),
            contacts=("contact_recommended", "sum"),
        )
        .sort_values("average_risk", ascending=False)
    )
    return {
        "threshold": threshold,
        "validation_precision": float(best["precision"]),
        "validation_recall": float(best["recall"]),
        "validation_f1": float(best["f1"]),
        "total_customers": int(len(scored)),
        "contact_candidates": int(len(selected)),
        "candidate_ratio_percent": round(len(selected) / len(scored) * 100, 2),
        "candidate_monthly_charges": round(float(selected["monthly_charges"].sum()), 2),
        "risk_counts": {str(key): int(value) for key, value in risk_counts.items()},
        "contract_risk": contract_risk.to_dict(orient="records"),
        "caution": (
            "전체 고객 점수는 행동 우선순위 예시이며, 실제 운영 전에는 새로운 기간의 "
            "데이터로 성능과 공정성을 다시 검증해야 한다."
        ),
    }


def render_html(
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    scored: pd.DataFrame,
) -> Path:
    """Jinja2 템플릿으로 실행 가능한 고객 관리 리포트를 만든다."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_PATH.parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template(TEMPLATE_PATH.name)
    output_path = OUTPUT_DIR / "churn_action_report.html"
    output_path.write_text(
        template.render(
            summary=summary,
            thresholds=comparison.to_dict(orient="records"),
            top_customers=scored.head(20).to_dict(orient="records"),
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, model = load_inputs()
    comparison, threshold = compare_thresholds(frame, model)
    scored = score_customers(frame, model, threshold)
    summary = build_summary(scored, comparison, threshold)

    comparison.to_csv(
        OUTPUT_DIR / "threshold_comparison.csv", index=False, encoding="utf-8-sig"
    )
    scored.to_csv(
        OUTPUT_DIR / "customer_churn_priorities.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "action_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = render_html(summary, comparison, scored)

    print("=== Advanced: 고객 이탈 대응 우선순위 생성 완료 ===")
    print(f"선택 임계값: {threshold:.2f} (검증 F1 최대 기준)")
    print(f"검증 Precision: {summary['validation_precision']:.4f}")
    print(f"검증 Recall: {summary['validation_recall']:.4f}")
    print(f"검증 F1: {summary['validation_f1']:.4f}")
    print(
        f"연락 추천 고객: {summary['contact_candidates']:,} / "
        f"{summary['total_customers']:,}명 ({summary['candidate_ratio_percent']:.2f}%)"
    )
    print(f"HTML 리포트: {report_path}")
    print(f"고객 우선순위 CSV: {OUTPUT_DIR / 'customer_churn_priorities.csv'}")
    print(f"임계값 비교 CSV: {OUTPUT_DIR / 'threshold_comparison.csv'}")


if __name__ == "__main__":
    main()
