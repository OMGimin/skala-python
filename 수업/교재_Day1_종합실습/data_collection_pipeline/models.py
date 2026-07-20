"""교재 Day 1 종합실습에서 사용하는 Pydantic v2 데이터 모델."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress


class StrictModel(BaseModel):
    """API 응답에서 정의하지 않은 필드를 모델에 저장하지 않는 공통 모델."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class WeatherRecord(StrictModel):
    """서울의 한 시간대별 기온과 강수확률."""

    source: Literal["open_meteo"] = "open_meteo"
    time: datetime
    temperature_c: float = Field(ge=-100, le=70)
    precipitation_probability: float = Field(ge=0, le=100)


class CountryRecord(StrictModel):
    """Countries.dev에서 수집한 대한민국 기본 정보."""

    source: Literal["countries_dev"] = "countries_dev"
    alpha3_code: str = Field(min_length=3, max_length=3)
    name: str = Field(min_length=1)
    capital: str = Field(min_length=1)
    region: str = Field(min_length=1)
    population: int = Field(gt=0)


class IpLocationRecord(StrictModel):
    """ip-api에서 수집한 IP 기반 지역 정보."""

    source: Literal["ip_api"] = "ip_api"
    query: IPvAnyAddress
    country: str = Field(min_length=1)
    city: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = Field(min_length=1)
