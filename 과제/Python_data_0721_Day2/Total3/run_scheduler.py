"""리포트를 간단한 루프 또는 schedule 라이브러리로 반복 실행한다."""

from __future__ import annotations

import argparse
import time

import schedule

from report import run_once


def run_interval(seconds: int) -> None:
    """외부 라이브러리 없이 지정한 초 간격으로 반복한다."""
    print(f"{seconds}초 간격으로 실행합니다. 종료하려면 Ctrl+C를 누르세요.")
    while True:
        run_once()
        time.sleep(seconds)


def run_schedule(minutes: int) -> None:
    """schedule의 선언적 문법으로 지정한 분 간격으로 반복한다."""
    print(f"{minutes}분 간격으로 실행합니다. 종료하려면 Ctrl+C를 누르세요.")
    run_once()
    schedule.every(minutes).minutes.do(run_once)
    while True:
        schedule.run_pending()
        time.sleep(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="판매 분석 리포트 자동 실행")
    parser.add_argument("--interval", type=int, default=0, help="반복 실행 간격(초)")
    parser.add_argument(
        "--mode", choices=["loop", "schedule"], default="loop", help="스케줄링 방식"
    )
    parser.add_argument("--every-minutes", type=int, default=1, help="schedule 실행 간격(분)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "schedule":
        run_schedule(max(1, args.every_minutes))
    elif args.interval > 0:
        run_interval(args.interval)
    else:
        run_once()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n자동 실행을 종료했습니다.")
