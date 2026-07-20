"""교재 Day 1 종합실습: 세 API를 동시에 수집·검증·저장한다.

Open-Meteo, Countries.dev, ip-api를 ``asyncio.gather()``로 동시에 호출하고,
Pydantic v2 모델로 타입과 범위를 검증한 뒤 CSV와 Parquet으로 저장한다.
두 파일 형식의 읽기·쓰기 시간 및 크기도 측정하여 비교한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError

from models import CountryRecord, IpLocationRecord, WeatherRecord

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PERFORMANCE_FILE = OUTPUT_DIR / "performance.json"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
COUNTRY_URL = "https://countries.dev/alpha/KOR"
IP_URL = "http://ip-api.com/json/8.8.8.8"

REQUESTS: dict[str, tuple[str, dict[str, str | int | float]]] = {
    "weather": (
        WEATHER_URL,
        {
            "latitude": 37.5665,
            "longitude": 126.9780,
            "hourly": "temperature_2m,precipitation_probability",
            "forecast_days": 3,
            "timezone": "Asia/Seoul",
        },
    ),
    "country": (
        COUNTRY_URL,
        {"fields": "name,capital,region,population,alpha3Code"},
    ),
    "ip_location": (
        IP_URL,
        {
            "fields": (
                "status,message,query,country,countryCode,regionName,city,"
                "lat,lon,timezone"
            )
        },
    ),
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RawJson = dict[str, Any]
ValidatedData = dict[str, list[BaseModel]]


async def fetch_json(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    params: Mapping[str, str | int | float],
) -> tuple[str, RawJson]:
    """API 하나를 호출하고 정상적인 JSON 객체 응답을 반환한다."""
    response = await client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise TypeError(f"{name} API 응답은 JSON 객체여야 합니다.")
    logger.info("API 응답 정상: %s (%s)", name, response.status_code)
    return name, data


async def collect_all() -> dict[str, RawJson]:
    """세 API 요청을 asyncio.gather()로 동시에 실행한다."""
    timeout = httpx.Timeout(15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        tasks = [
            fetch_json(client, name, url, params)
            for name, (url, params) in REQUESTS.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    collected: dict[str, RawJson] = {}
    failures: list[str] = []
    for name, result in zip(REQUESTS, results, strict=True):
        if isinstance(result, BaseException):
            failures.append(f"{name}: {result}")
        else:
            result_name, data = result
            collected[result_name] = data

    if failures:
        raise RuntimeError("API 수집 실패 - " + " | ".join(failures))
    return collected


def transform_weather(raw: RawJson) -> list[WeatherRecord]:
    """Open-Meteo의 병렬 배열을 시간대별 WeatherRecord로 변환한다."""
    hourly = raw.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo 응답에 hourly 객체가 없습니다.")

    times = hourly.get("time")
    temperatures = hourly.get("temperature_2m")
    probabilities = hourly.get("precipitation_probability")
    hourly_values = (times, temperatures, probabilities)
    if not all(isinstance(values, list) for values in hourly_values):
        raise TypeError("Open-Meteo 시간대별 필드는 모두 리스트여야 합니다.")
    if not (len(times) == len(temperatures) == len(probabilities)):
        raise ValueError("Open-Meteo 시간대별 배열 길이가 서로 다릅니다.")

    return [
        WeatherRecord.model_validate(
            {
                "time": observed_at,
                "temperature_c": temperature,
                "precipitation_probability": probability,
            }
        )
        for observed_at, temperature, probability in zip(
            times, temperatures, probabilities, strict=True
        )
    ]


def transform_country(raw: RawJson) -> list[CountryRecord]:
    """Countries.dev 응답에서 대한민국 필드를 추출하고 검증한다."""
    return [
        CountryRecord.model_validate(
            {
                "alpha3_code": raw.get("alpha3Code"),
                "name": raw.get("name"),
                "capital": raw.get("capital"),
                "region": raw.get("region"),
                "population": raw.get("population"),
            }
        )
    ]


def transform_ip_location(raw: RawJson) -> list[IpLocationRecord]:
    """ip-api 응답 상태를 확인하고 IP 위치 필드를 검증한다."""
    if raw.get("status") != "success":
        raise ValueError(f"ip-api 오류: {raw.get('message', '알 수 없는 오류')}")
    return [
        IpLocationRecord.model_validate(
            {
                "query": raw.get("query"),
                "country": raw.get("country"),
                "city": raw.get("city"),
                "latitude": raw.get("lat"),
                "longitude": raw.get("lon"),
                "timezone": raw.get("timezone"),
            }
        )
    ]


def validate_all(raw_data: Mapping[str, RawJson]) -> ValidatedData:
    """세 API 응답을 각 Pydantic 모델로 검증하여 데이터셋으로 반환한다."""
    transformers = {
        "weather": transform_weather,
        "country": transform_country,
        "ip_location": transform_ip_location,
    }
    validated: ValidatedData = {}
    for name, transformer in transformers.items():
        try:
            validated[name] = transformer(raw_data[name])
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            logger.error("스키마 검증 실패: %s - %s", name, error)
            raise
        logger.info("스키마 검증 성공: %s (%d건)", name, len(validated[name]))
    return validated


def models_to_frame(records: Sequence[BaseModel]) -> pd.DataFrame:
    """Pydantic 모델 목록을 JSON 직렬화 가능한 DataFrame으로 변환한다."""
    return pd.DataFrame([record.model_dump(mode="json") for record in records])


def benchmark_storage(
    datasets: Mapping[str, Sequence[BaseModel]],
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, dict[str, float | int]]:
    """모든 데이터셋을 CSV·Parquet으로 저장·재로딩하고 성능을 측정한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: models_to_frame(records) for name, records in datasets.items()}
    metrics: dict[str, dict[str, float | int]] = {}

    for file_format in ("csv", "parquet"):
        write_start = time.perf_counter()
        for name, frame in frames.items():
            path = output_dir / f"{name}.{file_format}"
            if file_format == "csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_parquet(path, index=False)
        write_seconds = time.perf_counter() - write_start

        read_start = time.perf_counter()
        reloaded = []
        for name in frames:
            path = output_dir / f"{name}.{file_format}"
            if file_format == "csv":
                reloaded.append(pd.read_csv(path))
            else:
                reloaded.append(pd.read_parquet(path))
        read_seconds = time.perf_counter() - read_start

        expected_rows = sum(len(frame) for frame in frames.values())
        reloaded_rows = sum(len(frame) for frame in reloaded)
        assert reloaded_rows == expected_rows
        metrics[file_format] = {
            "write_seconds": write_seconds,
            "read_seconds": read_seconds,
            "total_bytes": sum(
                (output_dir / f"{name}.{file_format}").stat().st_size for name in frames
            ),
            "rows": reloaded_rows,
        }

    PERFORMANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_FILE.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def print_summary(datasets: ValidatedData, metrics: Mapping[str, Mapping]) -> None:
    """수집 건수와 CSV·Parquet 성능 비교 결과를 읽기 좋게 출력한다."""
    print("\n=== 수집·검증 결과 ===")
    for name, records in datasets.items():
        print(f"{name}: {len(records)}건")

    print("\n=== CSV·Parquet 성능 비교 ===")
    for file_format in ("csv", "parquet"):
        result = metrics[file_format]
        print(
            f"{file_format.upper():7} | "
            f"쓰기 {result['write_seconds']:.6f}초 | "
            f"읽기 {result['read_seconds']:.6f}초 | "
            f"크기 {result['total_bytes']:,} bytes"
        )
    print(f"\n결과 폴더: {OUTPUT_DIR}")


async def main_async() -> int:
    """비동기 수집부터 검증·저장·출력까지 전체 파이프라인을 실행한다."""
    try:
        raw_data = await collect_all()
        datasets = validate_all(raw_data)
        metrics = benchmark_storage(datasets)
        print_summary(datasets, metrics)
        return 0
    except (
        httpx.HTTPError,
        RuntimeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as error:
        logger.error("파이프라인 실행 실패: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
