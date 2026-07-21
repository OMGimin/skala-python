from __future__ import annotations

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from src.report import build_interpretations


def generate_html_report(
    output_path: Path,
    template_dir: Path,
    *,
    engine_comparison: dict[str, Any],
    audit: pd.DataFrame,
    distance_summary: pd.DataFrame,
    day_summary: pd.DataFrame,
    ttest: dict[str, Any],
    model_metrics: dict[str, Any],
    chart_html: str,
    open_browser: bool = False,
    generated_at: datetime | None = None,
) -> Path:
    """Jinja2 템플릿에 분석 결과와 Plotly 차트를 주입해 HTML을 생성한다."""
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("report.html")
    generated_at = generated_at or datetime.now().astimezone()
    html = template.render(
        title="NYC Yellow Taxi 카드 팁 분석",
        generated=generated_at.isoformat(timespec="seconds"),
        pandas_info=engine_comparison["pandas"],
        polars_info=engine_comparison["polars"],
        audit=audit.to_dict("records"),
        distance_summary=distance_summary.to_dict("records"),
        day_summary=day_summary.to_dict("records"),
        ttest=ttest,
        significant=bool(ttest["p_value"] < 0.05),
        model=model_metrics,
        interpretations=build_interpretations(distance_summary, day_summary, model_metrics),
        chart_html=chart_html,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path
