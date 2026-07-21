"""종합실습 3: 원시 매출 CSV를 집계하여 타임스탬프 HTML 리포트를 만든다.

실제 데이터에는 amount가 없으므로 quantity * unit_price * (1 - discount)로
매출을 계산한다. ``run_once``는 직접 실행, 반복 루프, schedule, cron이 함께
사용하는 단일 실행 진입점이다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import SETTINGS, Settings

REQUIRED_COLUMNS = {
    "order_id",
    "order_date",
    "region",
    "category",
    "quantity",
    "unit_price",
    "discount",
}


def load_and_prepare_sales(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """CSV를 검증·정제하고 실제 필드로 amount를 계산한다.

    누락된 단가는 같은 카테고리 중앙값으로 보완하고, 그래도 남는 결측값은 전체
    중앙값으로 보완한다. 수량이 음수이거나 단가가 음수인 행은 매출 왜곡을 막기
    위해 제외한다.
    """
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    sales = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(sales.columns)
    if missing:
        raise ValueError(f"필수 필드가 누락되었습니다: {sorted(missing)}")
    if sales.empty:
        raise ValueError("리포트를 생성할 매출 데이터가 없습니다.")

    original_rows = len(sales)
    sales["order_date"] = pd.to_datetime(sales["order_date"], errors="coerce")
    for column in ("quantity", "unit_price", "discount"):
        sales[column] = pd.to_numeric(sales[column], errors="coerce")

    invalid_key_mask = sales[["order_id", "region", "category", "order_date"]].isna().any(axis=1)
    sales = sales.loc[~invalid_key_mask].copy()
    missing_prices = int(sales["unit_price"].isna().sum())
    category_medians = sales.groupby("category")["unit_price"].transform("median")
    sales["unit_price"] = sales["unit_price"].fillna(category_medians)
    sales["unit_price"] = sales["unit_price"].fillna(sales["unit_price"].median())

    invalid_numeric_mask = (
        sales[["quantity", "unit_price", "discount"]].isna().any(axis=1)
        | (sales["quantity"] < 0)
        | (sales["unit_price"] < 0)
        | ~sales["discount"].between(0, 1)
    )
    invalid_rows = int(invalid_key_mask.sum() + invalid_numeric_mask.sum())
    sales = sales.loc[~invalid_numeric_mask].copy()
    if sales.empty:
        raise ValueError("정제 후 유효한 매출 데이터가 없습니다.")

    sales["amount"] = sales["quantity"] * sales["unit_price"] * (1 - sales["discount"])
    quality = {
        "input_rows": original_rows,
        "valid_rows": len(sales),
        "excluded_rows": invalid_rows,
        "imputed_unit_prices": missing_prices,
    }
    return sales, quality


def aggregate_sales(sales: pd.DataFrame, top_n: int) -> dict[str, Any]:
    """전체 KPI와 지역별·카테고리별·일별 매출 집계를 반환한다."""
    order_totals = sales.groupby("order_id", as_index=False)["amount"].sum()
    category = (
        sales.groupby("category", as_index=False)
        .agg(total_sales=("amount", "sum"), orders=("order_id", "nunique"))
        .sort_values("total_sales", ascending=False)
        .head(top_n)
    )
    region = (
        sales.groupby("region", as_index=False)
        .agg(total_sales=("amount", "sum"), orders=("order_id", "nunique"))
        .sort_values("total_sales", ascending=False)
    )
    daily = (
        sales.groupby("order_date", as_index=False)["amount"]
        .sum()
        .sort_values("order_date")
    )
    return {
        "kpi": {
            "total_sales": float(sales["amount"].sum()),
            "transactions": int(sales["order_id"].nunique()),
            "average_order_value": float(order_totals["amount"].mean()),
        },
        "category_rows": category.to_dict(orient="records"),
        "region_rows": region.to_dict(orient="records"),
        "daily_rows": [
            {"date": row.order_date.strftime("%Y-%m-%d"), "total_sales": row.amount}
            for row in daily.itertuples(index=False)
        ],
    }


def render_report(
    analysis: dict[str, Any],
    quality: dict[str, int],
    generated_at: datetime,
    settings: Settings,
) -> Path:
    """Jinja2 템플릿을 렌더링해 덮어쓰지 않는 타임스탬프 HTML로 저장한다."""
    environment = Environment(
        loader=FileSystemLoader(settings.template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template(settings.template_name)
    html = template.render(
        title=settings.report_title,
        generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        quality=quality,
        **analysis,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sales_report_{generated_at.strftime('%Y%m%d_%H%M%S_%f')}.html"
    output_path = settings.output_dir / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_once(settings: Settings = SETTINGS) -> Path:
    """CSV 로드부터 HTML 저장까지 한 번 실행하고 생성 경로를 반환한다."""
    sales, quality = load_and_prepare_sales(settings.data_path)
    analysis = aggregate_sales(sales, settings.top_n)
    output_path = render_report(analysis, quality, datetime.now(), settings)
    print(f"리포트 생성 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    run_once()
