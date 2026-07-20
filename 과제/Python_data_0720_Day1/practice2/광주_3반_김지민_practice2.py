"""
실습 2: Pydantic v2 중첩 스키마 검증
작성자: 광주 3반 김지민
작성일: 2026-07-20

프로그램 설명:
    api_response.json 파일의 API 응답을 읽고 Pydantic v2 모델로 검증한다.
    정상 레코드는 valid 목록으로, 실패 레코드는 원본 row와 오류 메시지를
    errors 목록으로 분리한 뒤 JSON 파일로 저장한다.

변경 내역:
    2026-07-20: Day1 practice2 제출용 파일 최초 작성
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_JSON_PATH = DATA_DIR / "api_response.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class UserProfile(BaseModel):
    """사용자 프로필 중첩 객체를 검증하는 모델."""

    model_config = ConfigDict(extra="forbid")

    country: str = Field(min_length=2, max_length=2)
    tier: Literal["free", "pro", "enterprise"]
    score: float = Field(ge=0, le=100)


class UserRecord(BaseModel):
    """API results 배열의 사용자 1명을 검증하는 모델."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    username: str = Field(min_length=1)
    email: str
    age: int = Field(ge=0, le=120)
    is_active: bool
    signup_date: date
    profile: UserProfile
    tags: list[str] = Field(default_factory=list)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        """간단한 이메일 형식 검증을 수행한다."""
        pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        if not re.match(pattern, value):
            raise ValueError("이메일 형식이 올바르지 않습니다.")
        return value


class ApiResponse(BaseModel):
    """전체 API 응답의 최상위 구조를 검증하는 모델."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    count: int = Field(ge=0)
    results: list[dict[str, object]]

    @model_validator(mode="after")
    def validate_count(self) -> "ApiResponse":
        """count 값과 results 길이가 일치하는지 확인한다."""
        if self.count != len(self.results):
            raise ValueError("count 값과 results 개수가 일치하지 않습니다.")
        return self


def load_api_response(json_path: Path) -> ApiResponse:
    """JSON 파일을 읽어 최상위 API 응답을 검증한다.

    Args:
        json_path: api_response.json 파일 경로

    Returns:
        검증된 ApiResponse 객체
    """
    if not json_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_path}")

    try:
        raw_payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 형식이 잘못되었습니다: {error}") from error

    return ApiResponse.model_validate(raw_payload)


def split_valid_and_errors(response: ApiResponse) -> tuple[list[UserRecord], list[dict[str, object]]]:
    """사용자 레코드를 valid와 errors로 분리한다.

    Args:
        response: 검증된 최상위 API 응답

    Returns:
        정상 UserRecord 목록과 오류 정보 목록
    """
    valid_records: list[UserRecord] = []
    error_records: list[dict[str, object]] = []

    for index, row in enumerate(response.results, start=1):
        try:
            valid_records.append(UserRecord.model_validate(row))
        except ValidationError as error:
            error_records.append(
                {
                    "row_number": index,
                    "row": row,
                    "errors": error.errors(include_context=False),
                }
            )

    return valid_records, error_records


def save_results(valid_records: list[UserRecord], error_records: list[dict[str, object]]) -> None:
    """검증 결과를 output 폴더에 저장한다.

    Args:
        valid_records: 검증 통과 레코드
        error_records: 검증 실패 레코드와 오류 정보
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    valid_payload = [record.model_dump(mode="json") for record in valid_records]

    (OUTPUT_DIR / "valid_users.json").write_text(
        json.dumps(valid_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "validation_errors.json").write_text(
        json.dumps(error_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_results(valid_records: list[UserRecord], error_records: list[dict[str, object]]) -> None:
    """실습 데이터 기준 검증 결과를 assert로 확인한다.

    Args:
        valid_records: 검증 통과 레코드
        error_records: 검증 실패 레코드와 오류 정보
    """
    assert len(valid_records) == 36
    assert len(error_records) == 4
    assert {error["row_number"] for error in error_records} == {7, 13, 21, 29}
    assert all(0 <= record.profile.score <= 100 for record in valid_records)


def print_results(valid_records: list[UserRecord], error_records: list[dict[str, object]]) -> None:
    """검증 결과를 읽기 좋게 출력한다.

    Args:
        valid_records: 검증 통과 레코드
        error_records: 검증 실패 레코드와 오류 정보
    """
    print("\n[실습2] Pydantic v2 스키마 검증 결과")
    print(f"정상 레코드 수: {len(valid_records)}")
    print(f"오류 레코드 수: {len(error_records)}")
    print("오류 row 번호:", [error["row_number"] for error in error_records])
    print(f"결과 저장 위치: {OUTPUT_DIR}")


def main() -> None:
    """실습2 전체 흐름을 실행한다."""
    response = load_api_response(DEFAULT_JSON_PATH)
    valid_records, error_records = split_valid_and_errors(response)
    save_results(valid_records, error_records)
    validate_results(valid_records, error_records)
    print_results(valid_records, error_records)
    print("assert 검증 통과")


if __name__ == "__main__":
    main()
