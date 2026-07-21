"""세 가지 NYC Yellow Taxi 분석 결과를 단일 대시보드 데이터로 통합한다.

기존 분석 산출물을 다시 계산하지 않고 검증된 JSON·CSV·Parquet 결과를 읽어
휴대 가능한 HTML 대시보드의 원본인 ``artifact.json``을 생성한다.

변경 내역:
- 2026-07-21: 현금 결제, 초단시간 트립, 카드 팁 결과 통합 기능 최초 작성
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def read_json(path: Path) -> dict[str, Any]:
    """UTF-8 JSON 파일을 딕셔너리로 읽는다."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def csv_records(path: Path) -> list[dict[str, Any]]:
    """CSV를 대시보드 스냅샷에 사용할 레코드 목록으로 변환한다."""

    return pd.read_csv(path).to_dict(orient="records")


def build_short_trip_summary(path: Path) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """정제 Parquet에서 초단시간 트립 핵심지표와 시간대별 비율을 계산한다."""

    columns = ["is_short_trip", "pickup_hour", "is_night_hour", "trip_distance"]
    trips = pd.read_parquet(path, columns=columns)
    short_mask = trips["is_short_trip"].astype(bool)
    night_mask = trips["is_night_hour"].astype(bool)

    hourly = (
        trips.groupby("pickup_hour", as_index=False)
        .agg(short_trip_rate=("is_short_trip", "mean"), trips=("is_short_trip", "size"))
        .sort_values("pickup_hour")
    )
    hourly["period"] = "시간대별 초단시간 트립"

    short_distance = trips.loc[short_mask, "trip_distance"].mean()
    normal_distance = trips.loc[~short_mask, "trip_distance"].mean()
    night_rate = trips.loc[night_mask, "is_short_trip"].mean()
    other_rate = trips.loc[~night_mask, "is_short_trip"].mean()
    summary = {
        "clean_rows": int(len(trips)),
        "short_trips": int(short_mask.sum()),
        "short_trip_rate": float(short_mask.mean()),
        "night_rate": float(night_rate),
        "other_rate": float(other_rate),
        "night_lift": float(night_rate / other_rate),
        "short_mean_distance": float(short_distance),
        "normal_mean_distance": float(normal_distance),
    }
    return summary, hourly.to_dict(orient="records")


def source_specs(generated_at: str) -> list[dict[str, Any]]:
    """대시보드 수치의 파일 및 계산 출처를 정의한다."""

    common_filters = [
        "2026년 5월 승차 데이터",
        "유효 운행 조건 적용",
        "분석별 누수 변수 제외",
    ]
    return [
        {
            "id": "cash_source",
            "label": "현금 결제 분석 산출물",
            "path": "outputs/01_cash_payment/cash_project_results.json",
            "query": {
                "language": "Python",
                "engine": "DuckDB-compatible SQL",
                "sql": """SELECT pickup_hour, PULocationID, COUNT(*) AS trips, AVG(CASE WHEN payment_type = 2 THEN 1.0 ELSE 0.0 END) AS cash_rate FROM read_parquet('data/raw/yellow_tripdata_2026-05.parquet') WHERE payment_type IN (1, 2) GROUP BY pickup_hour, PULocationID""",
                "description": "카드·현금 결제 운행을 정제한 뒤 시간·지역별 현금 비율과 예측 성능을 계산",
                "executed_at": generated_at,
                "filters": common_filters + ["payment_type 1 또는 2"],
                "metric_definitions": [
                    "현금 결제율 = 현금 결제 운행 수 / 카드·현금 전체 운행 수",
                    "JFK 새벽 = PULocationID 132이며 승차시각 04:00–06:59",
                ],
            },
        },
        {
            "id": "short_source",
            "label": "초단시간 트립 분석 산출물",
            "path": "data/processed/cleaned_trips.parquet",
            "query": {
                "language": "Python",
                "engine": "DuckDB-compatible SQL",
                "sql": """SELECT pickup_hour, COUNT(*) AS trips, AVG(CASE WHEN is_short_trip THEN 1.0 ELSE 0.0 END) AS short_trip_rate FROM read_parquet('data/processed/cleaned_trips.parquet') GROUP BY pickup_hour ORDER BY pickup_hour""",
                "description": "정제 운행에서 초단시간 트립 여부를 시간대별로 집계",
                "executed_at": generated_at,
                "filters": common_filters,
                "metric_definitions": [
                    "초단시간 트립률 = 초단시간 조건을 만족한 운행 수 / 정제 운행 수",
                    "심야 시간 = 기존 분석의 is_night_hour 조건",
                ],
            },
        },
        {
            "id": "tip_source",
            "label": "카드 팁 분석 산출물",
            "path": "outputs/03_weekend_tip/tables",
            "query": {
                "language": "Python",
                "engine": "DuckDB-compatible SQL",
                "sql": """SELECT is_weekend, COUNT(*) AS trip_count, AVG(tip_rate) AS mean_tip_rate FROM read_parquet('data/processed/yellow_taxi_card_tip.parquet') GROUP BY is_weekend""",
                "description": "카드 결제 유효 운행의 요일·거리별 팁률과 고팁 예측 성능을 계산",
                "executed_at": generated_at,
                "filters": common_filters + ["카드 결제 운행만 포함"],
                "metric_definitions": [
                    "평균 팁률 = tip_amount / 팁 제외 기본 결제금액의 운행별 비율 평균",
                    "고팁 = 팁률 20% 이상",
                ],
            },
        },
        {
            "id": "model_source",
            "label": "현금·고팁 예측 모델 평가 산출물",
            "path": "outputs/01_cash_payment/cash_project_results.json; outputs/03_weekend_tip/tables/model_metrics.json",
            "query": {
                "language": "SQL",
                "engine": "DuckDB-compatible SQL",
                "sql": """SELECT '현금 결제 예측' AS analysis, model_evaluation.model.cash_f1 AS f1, model_evaluation.model.roc_auc AS roc_auc FROM read_json_auto('outputs/01_cash_payment/cash_project_results.json') UNION ALL SELECT '고팁 예측', f1, roc_auc FROM read_json_auto('outputs/03_weekend_tip/tables/model_metrics.json')""",
                "description": "서로 다른 두 분류 과제의 F1과 ROC AUC를 동일 척도로 정리",
                "executed_at": generated_at,
                "filters": ["각 분석의 고정 평가용 테스트 표본"],
                "metric_definitions": [
                    "F1 = precision과 recall의 조화평균",
                    "ROC AUC = 임계값 전반의 양성·음성 판별력",
                ],
            },
        },
    ]


def build_artifact() -> dict[str, Any]:
    """현재 분석 산출물에서 검증 가능한 대시보드 명세를 만든다."""

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cash_dir = OUTPUT_DIR / "01_cash_payment"
    tip_dir = OUTPUT_DIR / "03_weekend_tip" / "tables"

    cash = read_json(cash_dir / "cash_project_results.json")
    tip_model = read_json(tip_dir / "model_metrics.json")
    tip_test = read_json(tip_dir / "distance_ttest.json")
    short, short_hourly = build_short_trip_summary(
        PROCESSED_DIR / "cleaned_trips.parquet"
    )

    cash_quality = cash["data_quality"]
    cash_test = cash["statistical_tests"]["jfk_early_chi_square"]
    cash_model = cash["model_evaluation"]["model"]
    day_rows = csv_records(tip_dir / "tip_by_day_type.csv")
    for row in day_rows:
        row["day_type"] = "주말" if row["is_weekend"] else "주중"
    distance_rows = csv_records(tip_dir / "tip_by_distance.csv")
    distance_names = {"short_lt_5mi": "5마일 미만", "long_ge_5mi": "5마일 이상"}
    for row in distance_rows:
        row["distance_label"] = distance_names[row["distance_group"]]

    snapshot = {
        "cash_kpi": [{
            "overall_cash_rate": cash["eda"]["overall_cash_rate"],
            "jfk_early_cash_rate": cash_test["jfk_early_cash_rate"],
            "jfk_early_trips": cash_test["jfk_early_trip_count"],
            "odds_ratio": cash_test["odds_ratio"],
            "cash_f1": cash_model["cash_f1"],
            "clean_rows": cash_quality["final_rows"],
        }],
        "short_kpi": [short],
        "tip_kpi": [{
            "card_trips": int(sum(row["trip_count"] for row in day_rows)),
            "weekend_tip_rate": next(row["mean_tip_rate"] for row in day_rows if row["day_type"] == "주말"),
            "weekday_tip_rate": next(row["mean_tip_rate"] for row in day_rows if row["day_type"] == "주중"),
            "distance_difference_pp": tip_test["difference_percentage_points"],
            "tip_model_f1": tip_model["f1"],
            "tip_model_auc": tip_model["roc_auc"],
        }],
        "cash_segments": cash["eda"]["segment_records"],
        "cash_hourly": csv_records(cash_dir / "cash_rate_by_hour.csv"),
        "short_hourly": short_hourly,
        "tip_day": day_rows,
        "tip_distance": distance_rows,
        "model_comparison": [
            {"analysis": "현금 결제 예측", "metric": "F1", "score": cash_model["cash_f1"]},
            {"analysis": "현금 결제 예측", "metric": "ROC AUC", "score": cash_model["roc_auc"]},
            {"analysis": "고팁 예측", "metric": "F1", "score": tip_model["f1"]},
            {"analysis": "고팁 예측", "metric": "ROC AUC", "score": tip_model["roc_auc"]},
        ],
    }
    sources = source_specs(generated_at)

    cards = [
        {
            "id": "cash_card",
            "dataset": "cash_kpi",
            "description": "카드·현금 유효 운행 중 현금 결제 비율",
            "sourceId": "cash_source",
            "metrics": [{"label": "전체 현금 결제율", "field": "overall_cash_rate", "format": "percent"}],
        },
        {
            "id": "jfk_card",
            "dataset": "cash_kpi",
            "description": "JFK 승차, 새벽 4–6시 표본의 현금 결제 비율",
            "sourceId": "cash_source",
            "metrics": [{"label": "JFK 새벽 현금 결제율", "field": "jfk_early_cash_rate", "format": "percent"}],
        },
        {
            "id": "short_card",
            "dataset": "short_kpi",
            "description": "정제 운행 중 초단시간 조건을 만족한 비율",
            "sourceId": "short_source",
            "metrics": [{"label": "초단시간 트립률", "field": "short_trip_rate", "format": "percent"}],
        },
        {
            "id": "night_card",
            "dataset": "short_kpi",
            "description": "일반 시간 대비 심야 초단시간 트립률 배수",
            "sourceId": "short_source",
            "metrics": [{"label": "심야 집중 배수", "field": "night_lift", "format": "number"}],
        },
        {
            "id": "weekend_tip_card",
            "dataset": "tip_kpi",
            "description": "카드 결제 주말 운행의 평균 팁률",
            "sourceId": "tip_source",
            "metrics": [{"label": "주말 평균 팁률", "field": "weekend_tip_rate", "format": "percent"}],
        },
        {
            "id": "distance_tip_card",
            "dataset": "tip_kpi",
            "description": "5마일 미만과 이상 운행의 평균 팁률 차이",
            "sourceId": "tip_source",
            "metrics": [{"label": "단거리 팁률 우위", "field": "distance_difference_pp", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "cash_segment_chart",
            "title": "JFK 여부와 시간대별 현금 결제율",
            "subtitle": "JFK 새벽 4–6시 구간이 48.02%로 가장 높음",
            "type": "bar",
            "intent": "comparison",
            "dataset": "cash_segments",
            "sourceId": "cash_source",
            "encodings": {
                "x": {"field": "segment", "type": "nominal", "label": "구간"},
                "y": {"field": "cash_rate", "type": "quantitative", "format": "percent", "label": "현금 결제율"},
                "tooltip": [{"field": "trips", "format": "compact", "label": "운행 수"}],
            },
            "valueFormat": "percent",
            "layout": "half",
        },
        {
            "id": "cash_hour_chart",
            "title": "시간대별 현금 결제율",
            "subtitle": "24시간 현금 결제 패턴과 표본 수를 함께 확인",
            "type": "line",
            "intent": "trend",
            "dataset": "cash_hourly",
            "sourceId": "cash_source",
            "encodings": {
                "x": {"field": "pickup_hour", "type": "ordinal", "label": "승차 시각"},
                "y": {"field": "cash_rate", "type": "quantitative", "format": "percent", "label": "현금 결제율"},
                "tooltip": [{"field": "trips", "format": "compact", "label": "운행 수"}],
            },
            "valueFormat": "percent",
            "layout": "half",
        },
        {
            "id": "short_hour_chart",
            "title": "시간대별 초단시간 트립률",
            "subtitle": "심야 구간에서 초단시간 트립이 더 자주 관측됨",
            "type": "area",
            "intent": "trend",
            "dataset": "short_hourly",
            "sourceId": "short_source",
            "encodings": {
                "x": {"field": "pickup_hour", "type": "ordinal", "label": "승차 시각"},
                "y": {"field": "short_trip_rate", "type": "quantitative", "format": "percent", "label": "초단시간 트립률"},
                "tooltip": [{"field": "trips", "format": "compact", "label": "운행 수"}],
            },
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "tip_day_chart",
            "title": "주중·주말 카드 결제 평균 팁률",
            "subtitle": "주말이 주중보다 0.25%p 높지만 차이는 작음",
            "type": "bar",
            "intent": "comparison",
            "dataset": "tip_day",
            "sourceId": "tip_source",
            "encodings": {
                "x": {"field": "day_type", "type": "nominal", "label": "요일 구분"},
                "y": {"field": "mean_tip_rate", "type": "quantitative", "format": "percent", "label": "평균 팁률"},
                "tooltip": [{"field": "trip_count", "format": "compact", "label": "운행 수"}],
            },
            "valueFormat": "percent",
            "layout": "half",
        },
        {
            "id": "tip_distance_amount_chart",
            "title": "거리 구간별 카드 결제 평균 팁 금액",
            "subtitle": "5마일 이상 운행은 평균 팁 금액이 더 큼",
            "type": "bar",
            "intent": "comparison",
            "dataset": "tip_distance",
            "sourceId": "tip_source",
            "encodings": {
                "x": {"field": "distance_label", "type": "nominal", "label": "거리 구간"},
                "y": {"field": "mean_tip_amount", "type": "quantitative", "format": "currency", "label": "평균 팁 금액"},
                "tooltip": [{"field": "trip_count", "format": "compact", "label": "운행 수"}],
            },
            "valueFormat": "currency",
            "layout": "half",
        },
        {
            "id": "tip_distance_chart",
            "title": "거리 구간별 카드 결제 평균 팁률",
            "subtitle": "장거리의 팁 금액은 크지만 5마일 미만 운행의 팁률이 3.50%p 높음",
            "type": "bar",
            "intent": "comparison",
            "dataset": "tip_distance",
            "sourceId": "tip_source",
            "encodings": {
                "x": {"field": "distance_label", "type": "nominal", "label": "거리 구간"},
                "y": {"field": "mean_tip_rate", "type": "quantitative", "format": "percent", "label": "평균 팁률"},
                "tooltip": [{"field": "trip_count", "format": "compact", "label": "운행 수"}],
            },
            "valueFormat": "percent",
            "layout": "half",
        },
        {
            "id": "model_chart",
            "title": "예측 모델 평가 지표",
            "subtitle": "불균형 문제를 고려해 정확도보다 F1과 ROC AUC를 비교",
            "type": "bar",
            "intent": "comparison",
            "dataset": "model_comparison",
            "sourceId": "model_source",
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "평가 지표"},
                "y": {"field": "score", "type": "quantitative", "format": "number", "label": "점수"},
                "color": {"field": "analysis", "type": "nominal", "label": "분석"},
            },
            "layout": "full",
        },
    ]

    tables = [{
        "id": "cash_segment_table",
        "title": "현금 결제 세부 구간",
        "subtitle": "현금 결제율과 표본 수를 함께 확인",
        "dataset": "cash_segments",
        "sourceId": "cash_source",
        "defaultSort": {"field": "cash_rate", "direction": "desc"},
        "columns": [
            {"field": "segment", "label": "구간", "type": "text"},
            {"field": "cash_rate", "label": "현금 결제율", "format": "percent"},
            {"field": "trips", "label": "운행 수", "format": "compact"},
        ],
    }]

    blocks = [
        {"id": "intro", "type": "markdown", "layout": "full", "body": "# NYC Yellow Taxi 통합 분석 대시보드\n\n2026년 5월 옐로택시 운행을 세 관점에서 분석한 결과입니다. 현금 결제의 시간·지역 집중, 초단시간 트립의 시간대 패턴, 카드 팁의 요일·거리 차이를 한 화면에서 비교합니다."},
        {"id": "hero", "type": "metric-strip", "layout": "full", "cardIds": [card["id"] for card in cards]},
        {"id": "cash_heading", "type": "markdown", "layout": "full", "body": "## 1. 현금 결제는 언제, 어디에서 증가하는가?"},
        {"id": "cash_relationship", "type": "markdown", "layout": "full", "sourceId": "cash_source", "body": "**관계 요약:** 전체 현금 결제율은 11.59%였지만 JFK 새벽 4–6시에는 48.02%로 상승해, 시간과 승차지역의 조합이 현금 결제 집중과 뚜렷한 관련을 보였습니다."},
        {"id": "cash_segment", "type": "chart", "layout": "half", "chartId": "cash_segment_chart"},
        {"id": "cash_hour", "type": "chart", "layout": "half", "chartId": "cash_hour_chart"},
        {"id": "cash_table", "type": "table", "layout": "full", "tableId": "cash_segment_table"},
        {"id": "short_heading", "type": "markdown", "layout": "full", "body": "## 2. 초단시간 취소성 트립은 언제 집중되는가?"},
        {"id": "short_relationship", "type": "markdown", "layout": "full", "sourceId": "short_source", "body": "**관계 요약:** 심야 2–5시의 초단시간 트립률은 2.19%로 그 외 시간의 0.98%보다 약 2.23배 높아, 심야 시간대와 초단시간 트립 발생 사이의 집중 관계가 확인됐습니다."},
        {"id": "short_hour", "type": "chart", "layout": "full", "chartId": "short_hour_chart"},
        {"id": "tip_heading", "type": "markdown", "layout": "full", "body": "## 3. 주중·주말과 이동거리에 따라 카드 팁은 달라지는가?"},
        {"id": "tip_day", "type": "chart", "layout": "full", "chartId": "tip_day_chart"},
        {"id": "tip_distance_interpretation", "type": "markdown", "layout": "full", "sourceId": "tip_source", "body": "### 이동거리와 팁의 관계\n\n**관계 요약:** 이동거리가 늘면 실제 팁 금액은 증가하지만, 전체 결제금액에서 팁이 차지하는 비율은 오히려 감소했습니다.\n\n5마일 미만 운행의 평균 팁 금액은 **$3.30**, 5마일 이상은 **$8.62**로 장거리 운행에서 실제 팁 금액이 더 컸습니다. 반면 평균 팁률은 단거리 **17.41%**, 장거리 **13.91%**로 장거리 운행이 **3.50%p 낮았습니다**."},
        {"id": "tip_distance_amount", "type": "chart", "layout": "half", "chartId": "tip_distance_amount_chart"},
        {"id": "tip_distance", "type": "chart", "layout": "half", "chartId": "tip_distance_chart"},
        {"id": "model_heading", "type": "markdown", "layout": "full", "body": "## 모델 평가와 해석 주의사항\n\n두 예측 과제는 목적변수와 표본이 다르므로 절대적인 우열 비교가 아니라 각 문제에서의 판별력을 확인하는 용도입니다. 모든 결과는 관찰 자료의 연관성이며 인과관계를 뜻하지 않습니다."},
        {"id": "model", "type": "chart", "layout": "full", "chartId": "model_chart"},
        {"id": "method", "type": "markdown", "layout": "full", "body": "## 데이터와 방법\n\n원본은 NYC TLC Yellow Taxi 2026년 5월 Parquet입니다. Pandas·Polars 로딩 비교, 결측·이상치 처리, 기술통계와 통계 검정, sklearn Pipeline 학습을 거쳐 생성된 산출물만 대시보드에 사용했습니다. 각 카드와 차트의 메뉴에서 계산 정의와 파일 출처를 확인할 수 있습니다."},
    ]

    manifest = {
        "version": 1,
        "surface": "dashboard",
        "title": "NYC Yellow Taxi 통합 분석 대시보드",
        "description": "현금 결제, 초단시간 트립, 카드 팁의 세 가지 분석 결과",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
        "sources": sources,
    }
    return {
        "surface": "dashboard",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshot,
        },
        "sources": sources,
    }


def main() -> None:
    """대시보드 원본 JSON을 생성한다."""

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = DASHBOARD_DIR / "artifact.json"
    artifact_path.write_text(
        json.dumps(build_artifact(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"대시보드 원본 생성: {artifact_path}")


if __name__ == "__main__":
    main()
