"""
실습 3: asyncio + httpx 비동기 수집 파이프라인
작성자: 광주 3반 김지민
작성일: 2026-07-20

프로그램 설명:
    asyncio와 httpx 구조를 사용해 여러 API 요청을 동시에 수집하는
    파이프라인을 구현한다. 기본 설정은 USE_REAL_HTTP=False로 두어
    외부 네트워크 없이 mock 데이터 60건을 안정적으로 수집하고 검증한다.

변경 내역:
    2026-07-20: Day1 practice3 제출용 파일 최초 작성
    2026-07-20: 지수 백오프·예외 격리 보완 및 동기/비동기 시간 비교 추가
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


USE_REAL_HTTP = False
MAX_CONCURRENCY = 10
REQUEST_COUNT = 60
MAX_RETRIES = 2
BACKOFF_BASE_SECONDS = 0.1
MOCK_DELAY_SECONDS = 0.01
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class CollectedItem(BaseModel):
    """수집한 API 응답 1건을 검증하는 모델."""

    model_config = ConfigDict(extra="forbid")

    item_id: int = Field(gt=0)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    value: float = Field(ge=0)


async def fetch_mock_item(item_id: int) -> dict[str, Any]:
    """네트워크 없이 비동기 수집을 흉내 내는 mock 응답을 만든다.

    Args:
        item_id: 수집할 데이터 ID

    Returns:
        API 응답과 같은 형태의 딕셔너리
    """
    await asyncio.sleep(MOCK_DELAY_SECONDS)
    return {
        "item_id": item_id,
        "source": "mock-api",
        "title": f"mock item {item_id:02d}",
        "value": round(item_id * 1.5, 2),
    }


def fetch_mock_item_sync(item_id: int) -> dict[str, Any]:
    """동일한 mock 응답을 동기 방식으로 생성한다.

    Args:
        item_id: 수집할 데이터 ID

    Returns:
        API 응답과 같은 형태의 딕셔너리
    """
    time.sleep(MOCK_DELAY_SECONDS)
    return {
        "item_id": item_id,
        "source": "mock-api",
        "title": f"mock item {item_id:02d}",
        "value": round(item_id * 1.5, 2),
    }


def collect_items_sync() -> tuple[list[CollectedItem], float]:
    """동기 방식으로 mock 데이터 60건을 순차 수집하고 시간을 측정한다.

    Returns:
        검증된 동기 수집 데이터와 실행 시간
    """
    start_time = time.perf_counter()
    raw_items = [fetch_mock_item_sync(item_id) for item_id in range(1, REQUEST_COUNT + 1)]
    valid_items, errors = validate_items(raw_items)
    if errors:
        raise ValueError(f"동기 mock 데이터 검증 실패: {errors}")
    return valid_items, time.perf_counter() - start_time


async def fetch_real_item(client: httpx.AsyncClient, item_id: int) -> dict[str, Any]:
    """httpx AsyncClient로 실제 HTTP 요청을 수행한다.

    Args:
        client: httpx 비동기 클라이언트
        item_id: 요청할 게시글 ID

    Returns:
        실습 모델에 맞게 변환한 응답 딕셔너리
    """
    response = await client.get(f"https://jsonplaceholder.typicode.com/posts/{item_id}")
    response.raise_for_status()
    payload = response.json()
    return {
        "item_id": payload["id"],
        "source": "jsonplaceholder",
        "title": payload["title"],
        "value": float(payload["userId"]),
    }


async def fetch_with_retry(
    item_id: int,
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient | None,
    use_real_http: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """세마포어와 재시도로 단일 데이터를 수집한다.

    Args:
        item_id: 수집할 데이터 ID
        semaphore: 동시 실행 수를 제한하는 세마포어
        client: 실제 HTTP 요청에 사용할 클라이언트
        use_real_http: 실제 HTTP 사용 여부

    Returns:
        성공 시 응답 딕셔너리와 None, 실패 시 None과 오류 메시지
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            # 재시도 대기 중에는 세마포어를 점유하지 않고,
            # 실제 요청을 수행하는 동안에만 동시 실행 수를 제한한다.
            async with semaphore:
                if use_real_http:
                    if client is None:
                        raise ValueError("실제 HTTP 모드에는 client가 필요합니다.")
                    return await fetch_real_item(client, item_id), None
                return await fetch_mock_item(item_id), None

        # 개별 요청에서 발생한 예외를 오류 결과로 바꿔 다른 요청과 격리한다.
        except Exception as error:
            if attempt == MAX_RETRIES:
                return None, f"item_id={item_id}, error={type(error).__name__}: {error}"

            delay = BACKOFF_BASE_SECONDS * (2**attempt)
            await asyncio.sleep(delay)

    return None, f"item_id={item_id}, error=unknown"


def validate_items(raw_items: list[dict[str, Any]]) -> tuple[list[CollectedItem], list[dict[str, Any]]]:
    """수집 결과를 Pydantic 모델로 검증한다.

    Args:
        raw_items: 수집된 원본 응답 목록

    Returns:
        검증 통과 목록과 오류 목록
    """
    valid_items: list[CollectedItem] = []
    errors: list[dict[str, Any]] = []

    for row in raw_items:
        try:
            valid_items.append(CollectedItem.model_validate(row))
        except ValidationError as error:
            errors.append({"row": row, "errors": error.errors()})

    return valid_items, errors


async def collect_items(use_real_http: bool = USE_REAL_HTTP) -> tuple[list[CollectedItem], list[dict[str, Any]], float]:
    """여러 API 요청을 asyncio.gather()로 동시에 수집한다.

    Args:
        use_real_http: True면 실제 HTTP, False면 mock 수집 사용

    Returns:
        검증 통과 데이터, 오류 데이터, 실행 시간
    """
    start_time = time.perf_counter()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    raw_items: list[dict[str, Any]] = []
    collected_errors: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [
            fetch_with_retry(item_id, semaphore, client, use_real_http)
            for item_id in range(1, REQUEST_COUNT + 1)
        ]
        # fetch_with_retry() 밖의 예상하지 못한 예외도 전체 수집을 중단시키지 않는다.
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, BaseException):
            collected_errors.append(
                {"error": f"unexpected {type(result).__name__}: {result}"}
            )
            continue

        payload, error = result
        if payload is not None:
            raw_items.append(payload)
        if error is not None:
            collected_errors.append({"error": error})

    valid_items, validation_errors = validate_items(raw_items)
    elapsed = time.perf_counter() - start_time
    return valid_items, collected_errors + validation_errors, elapsed


def save_results(valid_items: list[CollectedItem], errors: list[dict[str, Any]], elapsed: float) -> None:
    """수집 결과를 output 폴더에 저장한다.

    Args:
        valid_items: 검증 통과 수집 데이터
        errors: 수집 또는 검증 오류 목록
        elapsed: 실행 시간
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = {
        "valid_count": len(valid_items),
        "error_count": len(errors),
        "elapsed_seconds": round(elapsed, 4),
        "items": [item.model_dump() for item in valid_items],
        "errors": errors,
    }
    (OUTPUT_DIR / "collected_items.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_results(valid_items: list[CollectedItem], errors: list[dict[str, Any]]) -> None:
    """실습3 결과를 assert로 검증한다.

    Args:
        valid_items: 검증 통과 수집 데이터
        errors: 수집 또는 검증 오류 목록
    """
    assert len(valid_items) == REQUEST_COUNT
    assert not errors
    assert all(item.item_id > 0 for item in valid_items)
    assert len({item.item_id for item in valid_items}) == REQUEST_COUNT


def print_results(
    valid_items: list[CollectedItem],
    errors: list[dict[str, Any]],
    elapsed: float,
    sync_elapsed: float,
) -> None:
    """수집 결과를 읽기 좋게 출력한다.

    Args:
        valid_items: 검증 통과 수집 데이터
        errors: 수집 또는 검증 오류 목록
        elapsed: 실행 시간
        sync_elapsed: 동기 방식 기준 실행 시간
    """
    print("\n[실습3] 비동기 수집 파이프라인 결과")
    print(f"USE_REAL_HTTP: {USE_REAL_HTTP}")
    print(f"요청 수: {REQUEST_COUNT}")
    print(f"동시 실행 제한: {MAX_CONCURRENCY}")
    print(f"정상 수집 수: {len(valid_items)}")
    print(f"오류 수: {len(errors)}")
    print(f"동기 실행 시간: {sync_elapsed:.4f}초")
    print(f"비동기 실행 시간: {elapsed:.4f}초")
    print(f"속도 향상: {sync_elapsed / elapsed:.2f}배")
    print(f"결과 저장 위치: {OUTPUT_DIR}")


async def async_main(sync_elapsed: float) -> None:
    """실습3 비동기 실행 진입점."""
    valid_items, errors, elapsed = await collect_items()
    save_results(valid_items, errors, elapsed)
    validate_results(valid_items, errors)
    print_results(valid_items, errors, elapsed, sync_elapsed)
    print("assert 검증 통과")


def main() -> None:
    """실습3 전체 흐름을 실행한다."""
    sync_items, sync_elapsed = collect_items_sync()
    assert len(sync_items) == REQUEST_COUNT
    asyncio.run(async_main(sync_elapsed))


if __name__ == "__main__":
    main()
