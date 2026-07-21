"""광주 3반 2조 NYC Yellow Taxi 세 주제 통합 실행기.

각 주제의 분석 코드는 독립 모듈로 유지하되 사용자는 이 파일 하나만 실행한다.
``--topic``으로 한 주제를 선택하거나 ``all``로 세 주제를 순차 실행할 수 있다.

변경 내역:
- 2026-07-21: 세 주제의 실행 진입점을 하나로 통합
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
TOPIC_COMMANDS = {
    "cash": {
        "title": "현금 결제 시간·지역 분석",
        "script": PROJECT_DIR
        / "topics"
        / "01_cash_payment"
        / "cash_payment_analysis.py",
        "cwd": PROJECT_DIR,
    },
    "short-trip": {
        "title": "초단시간 취소성 트립 분석",
        "script": PROJECT_DIR
        / "topics"
        / "02_short_trip_cancellation"
        / "short_trip_analysis.py",
        "cwd": PROJECT_DIR,
    },
    "weekend-tip": {
        "title": "주중·주말 카드 팁 분석",
        "script": PROJECT_DIR
        / "topics"
        / "03_weekend_tip"
        / "src"
        / "run_analysis.py",
        "cwd": PROJECT_DIR / "topics" / "03_weekend_tip",
    },
}


def parse_args() -> argparse.Namespace:
    """실행할 분석 주제를 명령행에서 선택한다."""

    parser = argparse.ArgumentParser(
        description="NYC Yellow Taxi 세 주제 통합 실행기"
    )
    parser.add_argument(
        "--topic",
        choices=[*TOPIC_COMMANDS, "all"],
        default="all",
        help="실행 주제(기본값: all)",
    )
    return parser.parse_args()


def run_topic(topic: str) -> None:
    """선택한 주제를 현재 가상환경 Python으로 실행한다."""

    config = TOPIC_COMMANDS[topic]
    script = Path(config["script"])
    working_directory = Path(config["cwd"])
    if not script.exists():
        raise FileNotFoundError(f"분석 파일이 없습니다: {script}")

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", "/tmp/kjm-matplotlib")
    if topic == "weekend-tip":
        environment["PYTHONPATH"] = str(working_directory)

    print("\n" + "=" * 72, flush=True)
    print(f"[{topic}] {config['title']} 시작", flush=True)
    print("=" * 72, flush=True)
    subprocess.run(
        [sys.executable, str(script)],
        cwd=working_directory,
        env=environment,
        check=True,
    )
    print(f"[{topic}] 완료", flush=True)


def main() -> None:
    """선택한 한 주제 또는 세 주제 전체를 순차 실행한다."""

    args = parse_args()
    selected = list(TOPIC_COMMANDS) if args.topic == "all" else [args.topic]
    for topic in selected:
        run_topic(topic)
    print("\n선택한 분석을 모두 완료했습니다.")


if __name__ == "__main__":
    main()
