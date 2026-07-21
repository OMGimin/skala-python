from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_table(frame: pd.DataFrame) -> str:
    """추가 패키지 없이 작은 DataFrame을 Markdown 표로 변환한다."""
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _required_row(frame: pd.DataFrame, column: str, value: object, label: str) -> pd.Series:
    """보고서 필수 그룹 한 행을 반환하고 없으면 이해하기 쉬운 오류를 낸다."""
    if column not in frame.columns:
        raise ValueError(f"{label} 요약에 필수 컬럼이 없습니다: {column}")
    matches = frame.loc[frame[column] == value]
    if matches.empty:
        raise ValueError(f"보고서 생성에 필요한 그룹이 없습니다: {label}")
    return matches.iloc[0]


def _comparison(subject_a: str, value_a: float, subject_b: str, value_b: float, unit: str) -> str:
    """두 실제 값을 비교하는 보고서 문장을 만든다."""
    difference = value_a - value_b
    if abs(difference) < 1e-12:
        return f"{subject_a}과 {subject_b}이 동일했다"
    direction = "높았다" if difference > 0 else "낮았다"
    formatted = f"${abs(difference):.2f}" if unit == "$" else f"{abs(difference):.2f}{unit}"
    return f"{subject_a}이 {subject_b}보다 {formatted} {direction}"


def build_interpretations(
    distance_summary: pd.DataFrame,
    day_summary: pd.DataFrame,
    model_metrics: dict[str, Any],
) -> dict[str, str]:
    """요약표와 평가 지표의 실제 수치에 맞는 해석을 반환한다."""
    short = _required_row(distance_summary, "distance_group", "short_lt_5mi", "단거리")
    long = _required_row(distance_summary, "distance_group", "long_ge_5mi", "장거리")
    weekday = _required_row(day_summary, "is_weekend", False, "주중")
    weekend = _required_row(day_summary, "is_weekend", True, "주말")
    amount_text = _comparison(
        "장거리 평균 팁 금액", float(long["mean_tip_amount"]),
        "단거리", float(short["mean_tip_amount"]), "$",
    )
    rate_text = _comparison(
        "장거리 평균 팁률", float(long["mean_tip_rate"]) * 100,
        "단거리", float(short["mean_tip_rate"]) * 100, "%p",
    )
    day_text = _comparison(
        "주말 평균 팁률", float(weekend["mean_tip_rate"]) * 100,
        "주중", float(weekday["mean_tip_rate"]) * 100, "%p",
    )
    model_text = (
        f"Accuracy {model_metrics['accuracy']:.4f}, F1 {model_metrics['f1']:.4f}, "
        f"ROC-AUC {model_metrics['roc_auc']:.4f}로 측정됐다. 이 값은 현재 입력 특성과 "
        "평가 표본에서의 예측 성능이며 인과관계를 의미하지 않는다."
    )
    return {"distance": f"{amount_text}. {rate_text}.", "day_type": f"{day_text}.", "model": model_text}


def generate_report(
    output_path: Path,
    engine_comparison: dict[str, Any],
    audit: pd.DataFrame,
    distance_summary: pd.DataFrame,
    day_summary: pd.DataFrame,
    ttest: dict[str, Any],
    model_metrics: dict[str, Any],
) -> Path:
    """분석 결과를 하나의 Markdown 보고서로 자동 생성한다."""
    short = _required_row(distance_summary, "distance_group", "short_lt_5mi", "단거리")
    long = _required_row(distance_summary, "distance_group", "long_ge_5mi", "장거리")
    weekday = _required_row(day_summary, "is_weekend", False, "주중")
    weekend = _required_row(day_summary, "is_weekend", True, "주말")
    interpretations = build_interpretations(distance_summary, day_summary, model_metrics)
    p_text = "< 0.001" if ttest["p_value"] < 0.001 else f"= {ttest['p_value']:.4f}"
    significance_text = (
        "p-value가 0.05보다 작으므로 두 거리 그룹의 평균 팁률 차이는 통계적으로 유의하다."
        if ttest["p_value"] < 0.05
        else "p-value가 0.05 이상이므로 두 거리 그룹의 평균 팁률 차이가 통계적으로 유의하다고 보기 어렵다."
    )
    pandas_info = engine_comparison["pandas"]
    polars_info = engine_comparison["polars"]

    text = f"""# NYC Yellow Taxi 카드 팁 분석 보고서

## 1. 분석 목적

2026년 5월 NYC Yellow Taxi 카드 결제 운행에서 거리와 시간적 특성이 팁률 및
고팁 여부에 어떤 관계를 보이는지 분석했다. 현금 팁은 완전하게 기록되지 않을 수
있으므로 팁 분석은 `payment_type=1`로 제한했다.

## 2. 데이터 준비

- 원본: NYC TLC Yellow Taxi Trip Records, 2026-05
- 팁률: `tip_amount / (total_amount - tip_amount)`
- 단거리: 5마일 미만, 장거리: 5마일 이상
- 고팁: 팁률 20% 이상
- 원본 파일은 Git에 포함하지 않고 공식 다운로드 스크립트로 재현한다.

### Pandas·Polars 로딩 비교

| 엔진 | 행 | 열 | 로딩 시간(초) | 추정 메모리(MB) | 결측 셀 | 제거 대상 중복 행 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pandas | {pandas_info['rows']:,} | {pandas_info['columns']} | {pandas_info['load_seconds']:.4f} | {pandas_info['memory_mb']:.2f} | {pandas_info['null_cells']:,} | {pandas_info['duplicate_rows']:,} |
| Polars | {polars_info['rows']:,} | {polars_info['columns']} | {polars_info['load_seconds']:.4f} | {polars_info['memory_mb']:.2f} | {polars_info['null_cells']:,} | {polars_info['duplicate_rows']:,} |

두 엔진의 행·열, 결측치, 중복 결과가 일치하는지 자동 검증했다.

### 필터 단계별 행 수

{_markdown_table(audit)}

## 3. 핵심 기술통계

### 거리 그룹

{_markdown_table(distance_summary)}

단거리 평균 팁 금액은 ${short['mean_tip_amount']:.2f}, 평균 팁률은
{_percent(short['mean_tip_rate'])}였다. 장거리 평균 팁 금액은
${long['mean_tip_amount']:.2f}, 평균 팁률은 {_percent(long['mean_tip_rate'])}였다.
{interpretations['distance']}

### 주중·주말

{_markdown_table(day_summary)}

주중 평균 팁률은 {_percent(weekday['mean_tip_rate'])}, 주말은
{_percent(weekend['mean_tip_rate'])}였다. {interpretations['day_type']}

## 4. 시각화

![거리 그룹별 팁 비교](figures/distance_tip_comparison.png)

- 인터랙티브 차트: [시간대별 주중·주말 팁률](figures/hourly_tip_interactive.html)

## 5. 통계 검정

- 귀무가설: {ttest['hypothesis_null']}
- Welch t 통계량: {ttest['t_statistic']:.4f}
- p-value: {p_text}
- 평균 팁률 차이: {ttest['difference_percentage_points']:.2f}%p
- Cohen's d: {ttest['cohens_d']:.3f}

{significance_text} Cohen's d는 차이의 실질적 크기를 보완해서 보여준다. 표본이 매우
크므로 p-value만으로 중요성을 판단하지 않는다.

## 6. ML Pipeline

### 6.1 목표변수와 입력 변수

- 목표변수: `high_tip` (팁률 20% 이상이면 1, 아니면 0)
- 수치형 입력: 운행 거리, 기본요금, 운행시간, 승차 시간
- 범주형 입력: 요일, 승차 지역 ID, 하차 지역 ID
- 학습 표본: {model_metrics['sample_rows']:,}건

`tip_amount`, `tip_rate`, `total_amount`, `base_amount`는 목표를 직접 또는
간접적으로 포함하므로 데이터 누수를 막기 위해 입력에서 제외했다.

### 6.2 전처리와 모델 구조

```text
sklearn.pipeline.Pipeline
├── ColumnTransformer
│   ├── 수치형: SimpleImputer(median) → StandardScaler
│   └── 범주형: SimpleImputer(most_frequent) → OneHotEncoder
└── SGDClassifier(loss="log_loss")
```

전처리와 분류기를 하나의 Pipeline으로 묶어 학습과 새로운 데이터 예측에 항상
같은 변환이 적용되도록 했다.

### 6.3 SGDClassifier 선택 이유

카드 결제 분석 데이터가 265만 건 이상이고 승하차 지역을 원-핫 인코딩하면
희소 특성이 많아진다. `SGDClassifier(loss="log_loss")`는 대규모 희소 행렬을
빠르고 비교적 적은 메모리로 학습할 수 있으며 확률 출력도 지원해 ROC-AUC를
계산할 수 있다. 반면 선형 모델이므로 복잡한 비선형 관계와 특성 간 상호작용을
충분히 학습하지 못할 수 있다.

### 6.4 평가 결과

| 지표 | 값 | 의미 |
| --- | ---: | --- |
| Accuracy | {model_metrics['accuracy']:.4f} | 전체 테스트 표본 중 올바르게 분류한 비율 |
| F1 | {model_metrics['f1']:.4f} | 고팁 클래스의 Precision과 Recall 조화평균 |
| ROC-AUC | {model_metrics['roc_auc']:.4f} | 임계값 전체에서 고팁과 비고팁을 구분하는 능력 |

{interpretations['model']} 실제 서비스 적용 여부는 별도의 기준 모델, 교차검증,
운영 목적에 맞는 임계값과 함께 판단해야 한다.

### 6.5 대안 모델 비교

| 모델 | 장점 | 단점 |
| --- | --- | --- |
| Logistic Regression | 해석이 쉽고 확률 출력 가능 | 대규모 희소 데이터에서 수렴이 느릴 수 있음 |
| SGDClassifier | 빠르고 대규모·희소 데이터에 적합 | 복잡한 비선형 관계 학습에 한계 |
| Random Forest | 비선형 관계와 상호작용 학습 | 메모리와 학습 시간이 많이 필요 |
| Gradient Boosting | 높은 예측 성능 가능 | 범주형 처리와 튜닝이 더 복잡함 |
| CatBoost | 지역 ID 같은 범주형 변수 처리에 강함 | 추가 라이브러리와 학습 비용 필요 |

이번 실습은 sklearn Pipeline 구성과 재현 가능한 기본 분류 모델이 목적이므로
SGDClassifier를 사용했다. 성능 개선이 우선이라면 CatBoost나 Gradient
Boosting 계열 모델을 비교할 수 있다.

### 6.6 모델 저장과 재사용

- 저장 모델: `models/high_tip_pipeline.joblib`

```python
import joblib

pipeline = joblib.load("models/high_tip_pipeline.joblib")
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]
```

joblib 파일에는 전처리와 분류기가 함께 저장되어 있다. 로딩 환경에서는 학습 때와
동일한 패키지 버전을 사용하고, 신뢰할 수 없는 joblib 파일은 역직렬화하지 않는다.

### 6.7 개선 방향

- 공항 운행 여부, 날씨, 휴일, 지역 특성 추가
- 시간 순서를 반영한 학습·평가 분할
- 임계값 조정과 Precision·Recall 비교
- 비선형 모델과 교차검증 비교
- 거리 기준과 고팁 기준에 대한 민감도 분석

모델 성능은 관측 특성의 예측력을 의미하며, 각 변수가 팁 행동의 원인이라는 것을
입증하지 않는다.

## 7. 결론과 한계

{interpretations['distance']} {interpretations['day_type']} 분석은 카드 결제로 제한되므로 전체 승객의 현금 팁
행동으로 일반화할 수 없고, 관측 데이터이므로 거리와 팁률 사이의 인과관계를
입증하지 않는다.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
