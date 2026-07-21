"""동일한 run_once 함수를 단일·반복·schedule 방식으로 실행한다."""

from __future__ import annotations

import argparse
import time

from report import run_once


def run_loop(interval: float) -> None:
    """지정한 초 간격으로 리포트를 반복 생성한다."""
    if interval <= 0:
        raise ValueError("반복 간격은 0보다 커야 합니다.")
    print(f"{interval:g}초 간격 반복 실행을 시작합니다. 종료하려면 Ctrl+C를 누르세요.")
    while True:
        started_at = time.monotonic()
        run_once()
        time.sleep(max(0, interval - (time.monotonic() - started_at)))


def run_schedule(every_minutes: int) -> None:
    """schedule 라이브러리에 분 단위 작업을 등록하고 계속 실행한다."""
    import schedule

    if every_minutes <= 0:
        raise ValueError("스케줄 간격은 0보다 커야 합니다.")
    schedule.every(every_minutes).minutes.do(run_once)
    run_once()
    print(f"schedule: {every_minutes}분 간격으로 다음 실행을 기다립니다.")
    while True:
        schedule.run_pending()
        time.sleep(1)


def parse_args() -> argparse.Namespace:
    """스케줄러 실행 모드와 간격 명령행 인수를 파싱한다."""
    parser = argparse.ArgumentParser(description="매출 HTML 리포트 자동 생성")
    parser.add_argument(
        "--mode",
        choices=("once", "loop", "schedule"),
        help="once=1회, loop=초 단위 반복, schedule=분 단위 반복",
    )
    parser.add_argument(
        "--interval",
        type=float,
        help="경량 루프 반복 초(이 옵션만 지정해도 loop 모드로 실행)",
    )
    parser.add_argument(
        "--every-minutes", type=int, default=1, help="schedule 모드 반복 분"
    )
    return parser.parse_args()


def main() -> None:
    """선택된 실행 방식으로 공통 run_once 함수를 호출한다."""
    args = parse_args()
    mode = args.mode or ("loop" if args.interval is not None else "once")
    if mode == "once":
        run_once()
    elif mode == "loop":
        run_loop(args.interval if args.interval is not None else 60)
    else:
        run_schedule(args.every_minutes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n스케줄러를 종료했습니다.")
