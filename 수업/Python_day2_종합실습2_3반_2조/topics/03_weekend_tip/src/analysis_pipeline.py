from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from pathlib import Path

from src.data_profile import (
    pandas_profile,
    polars_profile,
    timed_load_pandas,
    timed_load_polars,
    validate_profiles,
)
from src.features import create_tip_features
from src.html_report import generate_html_report
from src.model import train_and_evaluate
from src.report import generate_report
from src.statistical_analysis import (
    correlation_matrix,
    descriptive_statistics,
    distance_ttest,
    grouped_tip_statistics,
)
from src.viz import hourly_interactive_fragment, save_distance_comparison, save_hourly_interactive


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "03_weekend_tip"
RAW_PATH = Path(
    os.getenv(
        "TAXI_RAW_PATH",
        PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet",
    )
)
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "yellow_taxi_card_tip.parquet"
TABLE_DIR = Path(os.getenv("TAXI_REPORT_DIR", OUTPUT_ROOT / "tables"))
FIGURE_DIR = OUTPUT_ROOT / "figures"
MODEL_DIR = OUTPUT_ROOT / "models"
REPORT_PATH = OUTPUT_ROOT / "weekend_tip_report.txt"
HTML_REPORT_PATH = OUTPUT_ROOT / "report.html"
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"


def run_analysis(
    raw_path: Path = RAW_PATH,
    *,
    processed_path: Path = PROCESSED_PATH,
    table_dir: Path = TABLE_DIR,
    figure_dir: Path = FIGURE_DIR,
    model_dir: Path = MODEL_DIR,
    report_path: Path = REPORT_PATH,
    html_report_path: Path = HTML_REPORT_PATH,
    template_dir: Path = TEMPLATE_DIR,
    open_html_report: bool = False,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """특성 생성부터 통계·시각화·ML·보고서까지 자동 실행한다."""
    pandas_frame, pandas_seconds = timed_load_pandas(raw_path)
    pandas_info = pandas_profile(pandas_frame, pandas_seconds)
    del pandas_frame
    gc.collect()
    polars_frame, polars_seconds = timed_load_polars(raw_path)
    polars_info = polars_profile(polars_frame, polars_seconds)
    validate_profiles(pandas_info, polars_info)
    engine_comparison = {"pandas": pandas_info, "polars": polars_info}
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "data_engine_comparison.json").write_text(
        json.dumps(engine_comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    analysis_pl, audit_pl = create_tip_features(polars_frame)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_pl.write_parquet(processed_path, compression="zstd")
    analysis = analysis_pl.to_pandas()
    audit = audit_pl.to_pandas()

    audit.to_csv(table_dir / "filter_audit.csv", index=False)
    descriptive_statistics(analysis).to_csv(table_dir / "tip_descriptive_statistics.csv", index=False)
    correlation_matrix(analysis).to_csv(table_dir / "tip_correlation_matrix.csv")
    distance_summary = grouped_tip_statistics(analysis, "distance_group")
    day_summary = grouped_tip_statistics(analysis, "is_weekend")
    time_summary = grouped_tip_statistics(analysis, "time_group")
    distance_summary.to_csv(table_dir / "tip_by_distance.csv", index=False)
    day_summary.to_csv(table_dir / "tip_by_day_type.csv", index=False)
    time_summary.to_csv(table_dir / "tip_by_time_group.csv", index=False)

    ttest = distance_ttest(analysis)
    (table_dir / "distance_ttest.json").write_text(
        json.dumps(ttest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_distance_comparison(analysis, figure_dir / "distance_tip_comparison.png")
    save_hourly_interactive(analysis, figure_dir / "hourly_tip_interactive.html")

    metrics = train_and_evaluate(
        analysis,
        model_dir / "high_tip_pipeline.joblib",
        table_dir / "model_metrics.json",
    )
    generate_report(report_path, engine_comparison, audit, distance_summary, day_summary, ttest, metrics)
    generate_html_report(
        html_report_path,
        template_dir,
        engine_comparison=engine_comparison,
        audit=audit,
        distance_summary=distance_summary,
        day_summary=day_summary,
        ttest=ttest,
        model_metrics=metrics,
        chart_html=hourly_interactive_fragment(analysis),
        open_browser=open_html_report,
        generated_at=generated_at,
    )
    return {
        "analysis_rows": len(analysis),
        "processed_data": str(processed_path),
        "report": str(report_path),
        "html_report": str(html_report_path),
        "model": str(model_dir / "high_tip_pipeline.joblib"),
        "ttest": ttest,
        "model_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NYC Yellow Taxi tip analysis")
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="생성된 Jinja2 HTML 보고서를 기본 브라우저에서 엽니다.",
    )
    parser.add_argument(
        "--generated-at",
        type=datetime.fromisoformat,
        help="HTML 생성 시각을 ISO 8601 형식으로 고정합니다(예: 2026-05-31T00:00:00+09:00).",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(
                args.input,
                open_html_report=args.open_report,
                generated_at=args.generated_at,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
