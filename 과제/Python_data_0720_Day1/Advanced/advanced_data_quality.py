"""Advanced project: ETL data-quality monitoring report.

This program reuses the Day1 Total1 pipeline instead of duplicating its ETL
logic. It records validation failures, calculates quality KPIs, and creates
JSON, CSV, and browser-friendly HTML reports.
"""

from __future__ import annotations

import asyncio
import csv
import html
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DAY1_DIR = BASE_DIR.parent
TOTAL1_DIR = DAY1_DIR / "Total1"
OUTPUT_DIR = BASE_DIR / "output"

# Total1 is an assignment folder rather than an installed package.
sys.path.insert(0, str(TOTAL1_DIR))
from pipeline import extract, load, transform  # noqa: E402


def calculate_quality_score(
    requested: int,
    valid: int,
    extract_failed: int,
) -> float:
    """Return a transparent 0-100 quality score.

    Validation failures reduce the score naturally through the valid ratio.
    Collection failures receive an additional penalty because their contents
    could not be inspected at all.
    """
    if requested == 0:
        return 100.0
    valid_ratio = valid / requested
    collection_penalty = (extract_failed / requested) * 20
    return round(max(0.0, valid_ratio * 100 - collection_penalty), 2)


def summarize_errors(invalid: list[dict[str, Any]]) -> Counter[str]:
    """Count validation failures by field and Pydantic error type."""
    counts: Counter[str] = Counter()
    for rejected in invalid:
        for error in rejected["errors"]:
            field = ".".join(str(part) for part in error["loc"])
            counts[f"{field} ({error['type']})"] += 1
    return counts


def quality_grade(score: float) -> str:
    """Convert the numerical quality score into an easy-to-read grade."""
    if score >= 95:
        return "A"
    if score >= 90:
        return "B"
    if score >= 80:
        return "C"
    return "D"


def build_html_report(report: dict[str, Any]) -> str:
    """Create a standalone HTML dashboard without an extra dependency."""
    metrics = report["metrics"]
    categories = report["category_summary"]
    errors = report["validation_errors"]
    generated_at = html.escape(report["generated_at"])

    category_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['category'])}</td>"
        f"<td>{row['count']}</td>"
        f"<td>{row['average_price']:,.0f}</td>"
        f"<td>{row['total_price']:,.0f}</td>"
        "</tr>"
        for row in categories
    )
    error_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in errors.items()
    ) or '<tr><td colspan="2">검증 오류 없음</td></tr>'

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETL 데이터 품질 리포트</title>
  <style>
    :root {{ --navy:#102a43; --blue:#1677ff; --mint:#16a085; --bg:#f4f7fb; --line:#d9e2ec; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:#243b53; font-family:"Malgun Gothic",Arial,sans-serif; }}
    main {{ max-width:1050px; margin:36px auto; padding:0 20px 40px; }}
    header {{ color:white; padding:30px; border-radius:18px; background:linear-gradient(120deg,var(--navy),#245b8a); }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    header p {{ margin:0; color:#d9eaf7; }}
    .grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:18px 0; }}
    .card, section {{ background:white; border:1px solid var(--line); border-radius:14px; box-shadow:0 5px 18px #102a4310; }}
    .card {{ padding:18px; }}
    .label {{ color:#627d98; font-size:13px; }}
    .value {{ margin-top:7px; font-size:27px; font-weight:700; color:var(--navy); }}
    .score {{ color:var(--mint); }}
    section {{ padding:22px; margin-top:16px; }}
    h2 {{ margin:0 0 14px; font-size:19px; color:var(--navy); }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ padding:11px 12px; border-bottom:1px solid #e8eef4; text-align:left; }}
    th {{ background:#edf4fb; color:#334e68; }}
    .note {{ border-left:5px solid var(--blue); }}
    footer {{ margin-top:16px; color:#829ab1; font-size:12px; text-align:right; }}
    @media(max-width:760px) {{ .grid {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body><main>
  <header><h1>ETL 데이터 품질 모니터링</h1><p>비동기 수집부터 검증·적재까지 한눈에 확인하는 Advanced 리포트</p></header>
  <div class="grid">
    <div class="card"><div class="label">요청</div><div class="value">{metrics['requested']}</div></div>
    <div class="card"><div class="label">수집 성공</div><div class="value">{metrics['extracted']}</div></div>
    <div class="card"><div class="label">유효</div><div class="value">{metrics['valid']}</div></div>
    <div class="card"><div class="label">검증 실패</div><div class="value">{metrics['invalid']}</div></div>
    <div class="card"><div class="label">품질 점수</div><div class="value score">{metrics['quality_score']} / {metrics['grade']}</div></div>
  </div>
  <section><h2>카테고리별 유효 데이터</h2><table><thead><tr><th>카테고리</th><th>건수</th><th>평균 가격</th><th>총 가격</th></tr></thead><tbody>{category_rows}</tbody></table></section>
  <section><h2>검증 오류 유형</h2><table><thead><tr><th>필드와 오류 유형</th><th>횟수</th></tr></thead><tbody>{error_rows}</tbody></table></section>
  <section class="note"><h2>자동 판정</h2><p>{html.escape(report['recommendation'])}</p></section>
  <footer>생성 시각: {generated_at} · seed가 고정된 모의 데이터 사용</footer>
</main></body></html>"""


async def run_quality_monitor(ids: range = range(1, 61)) -> dict[str, Any]:
    """Run ETL once and persist quality evidence in multiple formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    requested_ids = list(ids)
    raw, extract_failures = await extract(requested_ids)
    valid, invalid = transform(raw)
    dataframe = load(valid, OUTPUT_DIR / "etl_output")

    category_frame = (
        dataframe.groupby("category", as_index=False)
        .agg(count=("id", "count"), average_price=("price", "mean"), total_price=("price", "sum"))
        .sort_values("category")
    )
    category_frame[["average_price", "total_price"]] = category_frame[
        ["average_price", "total_price"]
    ].round(2)
    category_summary = category_frame.to_dict(orient="records")
    error_counts = summarize_errors(invalid)
    score = calculate_quality_score(len(requested_ids), len(valid), len(extract_failures))

    report: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "metrics": {
            "requested": len(requested_ids),
            "extracted": len(raw),
            "extract_failed": len(extract_failures),
            "valid": len(valid),
            "invalid": len(invalid),
            "valid_ratio_percent": round(len(valid) / len(requested_ids) * 100, 2),
            "quality_score": score,
            "grade": quality_grade(score),
        },
        "validation_errors": dict(error_counts),
        "category_summary": category_summary,
        "recommendation": (
            "품질 기준을 충족했습니다. 검증 실패 데이터는 별도 보관하고 원천 시스템의 가격 규칙을 점검하세요."
            if score >= 90
            else "품질 기준 미달입니다. 수집 실패와 검증 오류를 우선 점검하세요."
        ),
    }

    (OUTPUT_DIR / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "quality_report.html").write_text(
        build_html_report(report), encoding="utf-8"
    )
    with (OUTPUT_DIR / "validation_failures.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["product_id", "field", "error_type", "message"])
        for rejected in invalid:
            for error in rejected["errors"]:
                writer.writerow(
                    [
                        rejected["data"].get("id"),
                        ".".join(str(part) for part in error["loc"]),
                        error["type"],
                        error["msg"],
                    ]
                )
    return report


def main() -> None:
    report = asyncio.run(run_quality_monitor())
    metrics = report["metrics"]
    print("[Advanced] ETL 데이터 품질 모니터링 완료")
    print(f"요청/수집: {metrics['requested']} / {metrics['extracted']}")
    print(f"유효/검증 실패: {metrics['valid']} / {metrics['invalid']}")
    print(f"유효 데이터 비율: {metrics['valid_ratio_percent']}%")
    print(f"품질 점수: {metrics['quality_score']}점 ({metrics['grade']}등급)")
    print(f"오류 유형: {report['validation_errors']}")
    print(f"HTML 리포트: {OUTPUT_DIR / 'quality_report.html'}")
    print(f"JSON 리포트: {OUTPUT_DIR / 'quality_report.json'}")
    print(f"오류 CSV: {OUTPUT_DIR / 'validation_failures.csv'}")


if __name__ == "__main__":
    main()
