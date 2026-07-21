"""판매 데이터를 정제·집계하고 Jinja2 HTML 리포트를 생성한다."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import CONFIG, ReportConfig


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """원본 품질 상태를 기록한 뒤 분석 가능한 형태로 정제한다."""
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    frame = pd.read_csv(path)
    missing_before = {column: int(count) for column, count in frame.isna().sum().items()}

    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    for column in ["quantity", "unit_price", "discount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["region"] = frame["region"].fillna("Unknown")
    frame.loc[frame["unit_price"] <= 0, "unit_price"] = pd.NA
    category_median = frame.groupby("category")["unit_price"].transform("median")
    frame["unit_price"] = frame["unit_price"].fillna(category_median)
    frame["unit_price"] = frame["unit_price"].fillna(frame["unit_price"].median())
    frame = frame.dropna(subset=["order_date", "quantity", "discount"]).copy()

    for column in ["quantity", "unit_price"]:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        frame[column] = frame[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    frame["sales"] = frame["quantity"] * frame["unit_price"] * (1 - frame["discount"])
    return frame, missing_before


def aggregate(frame: pd.DataFrame, top_n: int) -> dict[str, Any]:
    """템플릿에 넘길 KPI와 반복 가능한 표 데이터를 만든다."""
    category = (
        frame.groupby("category", as_index=False)
        .agg(orders=("order_id", "count"), quantity=("quantity", "sum"), sales=("sales", "sum"))
        .sort_values("sales", ascending=False)
    )
    region = (
        frame.groupby("region", as_index=False)
        .agg(orders=("order_id", "count"), sales=("sales", "sum"))
        .sort_values("sales", ascending=False)
    )
    daily = (
        frame.groupby("order_date", as_index=False)["sales"]
        .sum()
        .sort_values("order_date")
    )
    total_sales = float(frame["sales"].sum())
    max_daily_sales = float(daily["sales"].max()) if not daily.empty else 1.0
    daily_rows = [
        {
            "date": row.order_date.strftime("%Y-%m-%d"),
            "sales": float(row.sales),
            "bar_width": round(float(row.sales) / max_daily_sales * 100, 2),
        }
        for row in daily.itertuples()
    ]
    return {
        "kpis": {
            "orders": int(len(frame)),
            "total_sales": total_sales,
            "average_order_value": float(total_sales / len(frame)),
            "customers_regions": int(frame["region"].nunique()),
            "date_start": frame["order_date"].min().strftime("%Y-%m-%d"),
            "date_end": frame["order_date"].max().strftime("%Y-%m-%d"),
        },
        "category_rows": category.to_dict(orient="records"),
        "region_rows": region.to_dict(orient="records"),
        "top_orders": (
            frame.nlargest(top_n, "sales")[
                ["order_id", "order_date", "region", "category", "sales"]
            ]
            .assign(order_date=lambda data: data["order_date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        ),
        "daily_rows": daily_rows,
    }


def read_previous_sales(history_path: Path) -> float | None:
    """직전 실행의 총매출을 읽어 이번 실행과 비교한다."""
    if not history_path.exists():
        return None
    history = pd.read_csv(history_path)
    if history.empty or "total_sales" not in history:
        return None
    return float(history.iloc[-1]["total_sales"])


def append_history(history_path: Path, generated_at: str, data: dict[str, Any]) -> None:
    """리포트 실행 이력을 CSV에 한 줄씩 누적한다."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["generated_at", "orders", "total_sales", "average_order_value"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "generated_at": generated_at,
                "orders": data["kpis"]["orders"],
                "total_sales": round(data["kpis"]["total_sales"], 2),
                "average_order_value": round(data["kpis"]["average_order_value"], 2),
            }
        )


def render_report(
    data: dict[str, Any],
    missing_before: dict[str, int],
    config: ReportConfig = CONFIG,
) -> Path:
    """Jinja2 템플릿을 사용해 타임스탬프 HTML을 생성한다."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone()
    generated_at = generated.isoformat(timespec="seconds")
    # 같은 초에 수동 실행과 스케줄 실행이 겹쳐도 파일을 덮어쓰지 않는다.
    timestamp = generated.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    previous_sales = read_previous_sales(config.history_path)
    current_sales = data["kpis"]["total_sales"]
    change_percent = None
    if previous_sales not in (None, 0):
        change_percent = (current_sales - previous_sales) / previous_sales * 100

    warning = None
    if previous_sales and current_sales < previous_sales * config.low_sales_warning_ratio:
        warning = "직전 실행보다 총매출이 20% 이상 감소했습니다. 원본 데이터를 확인하세요."

    environment = Environment(
        loader=FileSystemLoader(config.template_path.parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template(config.template_path.name)
    output_path = config.output_dir / f"sales_report_{timestamp}.html"
    output_path.write_text(
        template.render(
            title=config.title,
            generated_at=generated_at,
            missing_before=missing_before,
            change_percent=change_percent,
            warning=warning,
            **data,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(output_path, config.output_dir / "latest_report.html")
    append_history(config.history_path, generated_at, data)
    return output_path


def run_once(config: ReportConfig = CONFIG) -> Path:
    """수동·루프·schedule 실행이 공통으로 호출하는 단일 진입점."""
    frame, missing_before = load_and_clean(config.data_path)
    data = aggregate(frame, config.top_n)
    output_path = render_report(data, missing_before, config)
    print("=== 종합실습 3: 자동 분석 리포트 생성 완료 ===")
    print(f"정제 행 수: {data['kpis']['orders']:,}")
    print(f"총매출: {data['kpis']['total_sales']:,.0f}원")
    print(f"평균 주문금액: {data['kpis']['average_order_value']:,.0f}원")
    print(f"원본 결측치: {sum(missing_before.values()):,}건")
    print(f"생성 리포트: {output_path}")
    print(f"최신 리포트: {config.output_dir / 'latest_report.html'}")
    print(f"실행 이력: {config.history_path}")
    return output_path


if __name__ == "__main__":
    run_once()
