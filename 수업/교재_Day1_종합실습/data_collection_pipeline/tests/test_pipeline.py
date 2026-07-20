"""교재 Day 1 종합실습의 스키마 및 변환 테스트."""

import pytest
from pydantic import ValidationError

from models import CountryRecord, IpLocationRecord, WeatherRecord
from pipeline import transform_ip_location, transform_weather


def test_weather_record_accepts_valid_values():
    """정상적인 기온과 강수확률은 검증을 통과해야 한다."""
    record = WeatherRecord(
        time="2026-07-20T12:00",
        temperature_c=28.5,
        precipitation_probability=40,
    )

    assert record.temperature_c == 28.5
    assert record.precipitation_probability == 40


@pytest.mark.parametrize("probability", [-1, 101])
def test_weather_probability_rejects_out_of_range(probability):
    """강수확률은 0~100 범위를 벗어나면 안 된다."""
    with pytest.raises(ValidationError):
        WeatherRecord(
            time="2026-07-20T12:00",
            temperature_c=28.5,
            precipitation_probability=probability,
        )


def test_country_population_must_be_positive():
    """국가 인구가 0 이하이면 Pydantic ValidationError가 발생해야 한다."""
    with pytest.raises(ValidationError):
        CountryRecord(
            alpha3_code="KOR",
            name="Korea",
            capital="Seoul",
            region="Asia",
            population=0,
        )


def test_ip_coordinates_must_be_in_range():
    """위도와 경도가 허용 범위를 벗어나면 안 된다."""
    with pytest.raises(ValidationError):
        IpLocationRecord(
            query="8.8.8.8",
            country="United States",
            city="Mountain View",
            latitude=91,
            longitude=-122.08,
            timezone="America/Los_Angeles",
        )


def test_weather_arrays_must_have_same_length():
    """Open-Meteo 배열 길이가 다르면 변환을 중단해야 한다."""
    raw = {
        "hourly": {
            "time": ["2026-07-20T12:00"],
            "temperature_2m": [28.5, 29.0],
            "precipitation_probability": [40],
        }
    }

    with pytest.raises(ValueError, match="배열 길이"):
        transform_weather(raw)


def test_ip_api_failure_is_reported():
    """ip-api가 실패 상태를 반환하면 메시지를 포함한 오류가 발생해야 한다."""
    with pytest.raises(ValueError, match="invalid query"):
        transform_ip_location({"status": "fail", "message": "invalid query"})
