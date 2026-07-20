"""
실습 1: 대용량 웹 로그 스트리밍 집계
작성자: 광주 3반 김지민
작성일: 2026-07-20

프로그램 설명:
    web_logs.csv 파일을 한 줄씩 읽으면서 전체 요청 수, 인기 페이지,
    5xx 오류율, 시간대별 요청 수, 상위 IP를 집계한다.
    대용량 파일을 한 번에 리스트로 올리지 않고 generator와 Counter,
    defaultdict, reduce를 활용해 메모리 사용량을 줄인다.

변경 내역:
    2026-07-20: Day1 practice1 제출용 파일 최초 작성
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from functools import reduce
from pathlib import Path
from typing import Iterator


DATA_DIR = Path(__file__).resolve().parents[2] / "0720_0721_광주_파이썬_데이터분석_데이터자료"
DEFAULT_CSV_PATH = DATA_DIR / "web_logs.csv"


def stream_log_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    """CSV 파일을 한 행씩 읽어 dict 형태로 반환한다.

    Args:
        csv_path: 웹 로그 CSV 파일 경로

    Yields:
        CSV의 각 행을 표현하는 딕셔너리

    Raises:
        FileNotFoundError: CSV 파일이 없을 때
        ValueError: 필수 컬럼이 누락되었을 때
    """
    required_columns = {"ip", "timestamp", "method", "path", "status", "bytes", "user_agent"}

    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames or [])
            raise ValueError(f"필수 컬럼이 누락되었습니다: {sorted(missing)}")

        for row in reader:
            yield row


def aggregate_web_logs(csv_path: Path) -> dict[str, object]:
    """웹 로그를 스트리밍 방식으로 집계한다.

    Args:
        csv_path: 웹 로그 CSV 파일 경로

    Returns:
        주요 집계 결과를 담은 딕셔너리
    """
    path_counter: Counter[str] = Counter()
    status_counter: Counter[int] = Counter()
    hour_counter: Counter[str] = Counter()
    ip_counter: Counter[str] = Counter()
    bytes_by_path: defaultdict[str, int] = defaultdict(int)
    count_by_path: defaultdict[str, int] = defaultdict(int)
    total_rows = 0

    for row in stream_log_rows(csv_path):
        status = int(row["status"])
        byte_size = int(row["bytes"])
        hour = row["timestamp"][11:13]
        path = row["path"]

        total_rows += 1
        path_counter[path] += 1
        status_counter[status] += 1
        hour_counter[hour] += 1
        ip_counter[row["ip"]] += 1
        bytes_by_path[path] += byte_size
        count_by_path[path] += 1

    total_from_reduce = reduce(lambda acc, count: acc + count, status_counter.values(), 0)
    server_error_count = sum(count for status, count in status_counter.items() if 500 <= status < 600)
    server_error_rate = server_error_count / total_rows if total_rows else 0
    average_bytes_by_path = {
        path: round(bytes_by_path[path] / count_by_path[path], 2)
        for path in path_counter
    }

    return {
        "total_rows": total_rows,
        "total_from_reduce": total_from_reduce,
        "top_paths": path_counter.most_common(5),
        "status_counts": status_counter,
        "server_error_rate": server_error_rate,
        "busy_hours": hour_counter.most_common(5),
        "top_ips": ip_counter.most_common(5),
        "average_bytes_by_path": average_bytes_by_path,
    }


def validate_results(result: dict[str, object]) -> None:
    """집계 결과의 기본 정합성을 assert로 검증한다.

    Args:
        result: aggregate_web_logs()가 반환한 집계 결과
    """
    total_rows = result["total_rows"]
    top_paths = result["top_paths"]
    busy_hours = result["busy_hours"]
    top_ips = result["top_ips"]

    assert isinstance(total_rows, int) and total_rows > 0
    assert result["total_from_reduce"] == total_rows
    assert all(top_paths[i][1] >= top_paths[i + 1][1] for i in range(len(top_paths) - 1))
    assert all(busy_hours[i][1] >= busy_hours[i + 1][1] for i in range(len(busy_hours) - 1))
    assert all(top_ips[i][1] >= top_ips[i + 1][1] for i in range(len(top_ips) - 1))
    assert 0 <= result["server_error_rate"] <= 1


def print_results(result: dict[str, object]) -> None:
    """집계 결과를 읽기 좋게 출력한다.

    Args:
        result: aggregate_web_logs()가 반환한 집계 결과
    """
    print("\n[실습1] 웹 로그 스트리밍 집계 결과")
    print(f"전체 요청 수: {result['total_rows']:,}")
    print(f"reduce 검증 요청 수: {result['total_from_reduce']:,}")
    print(f"5xx 오류율: {result['server_error_rate']:.2%}")
    print(f"인기 페이지 TOP 5: {result['top_paths']}")
    print(f"혼잡 시간대 TOP 5: {result['busy_hours']}")
    print(f"요청 IP TOP 5: {result['top_ips']}")


def main() -> None:
    """실습1 전체 흐름을 실행한다."""
    result = aggregate_web_logs(DEFAULT_CSV_PATH)
    validate_results(result)
    print_results(result)
    print("assert 검증 통과")


if __name__ == "__main__":
    main()
