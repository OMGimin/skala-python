"""
실습명: [실습 2] 파일 I/O · 예외 처리 · Pydantic 검증 파이프라인
작성자: 광주 3반 김지민
활용 데이터: Python_Practice1_Data.json

[프로그램 설명]
제공된 Sales 데이터를 안전하게 읽고 Pydantic v2 모델로 검증한다.
정상 데이터와 오류 데이터를 분리하여 각각 CSV와 JSON으로 저장한 뒤,
결과 파일을 다시 읽어 저장 건수가 일치하는지 확인한다.

[변경 내역]
- 2026-07-20: 최초 작성
- 2026-07-20: 실제 자료의 ``sales = [...]`` 형식과 실제 필드명 반영
"""

import ast
import csv
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Python_Practice1_Data.json"
OUTPUT_DIR = BASE_DIR / "output" / "실습2"
VALID_FILE = OUTPUT_DIR / "valid_sales.csv"
ERROR_FILE = OUTPUT_DIR / "validation_errors.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class SalesRecord(BaseModel):
    """월·지역·금액·카테고리로 구성된 정상 매출 한 건을 표현한다.

    month와 region은 공백일 수 없고 amount는 0보다 커야 한다.
    category는 입력 레코드에 없어도 된다.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    month: str
    region: str
    amount: float = Field(gt=0)
    category: str | None = None

    @field_validator("month", "region")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        """필수 문자열에서 빈 값과 공백만 있는 값을 거부한다."""
        if not value:
            raise ValueError("빈 문자열일 수 없습니다.")
        return value


def safe_load_json(file_path: Path) -> list[dict] | None:
    """Sales 원본 파일을 읽어 딕셔너리 리스트를 반환한다.

    표준 JSON과 제공 자료의 ``sales = [...]`` 형식을 모두 안전하게 처리한다.
    파일이 없거나 형식이 잘못되면 오류를 기록하고 None을 반환하며, 성공 여부와
    관계없이 finally에서 로딩 종료를 기록한다.
    """
    try:
        text = file_path.read_text(encoding="utf-8-sig")
        try:
            loaded_data = json.loads(text)
        except json.JSONDecodeError:
            variable_name, separator, value = text.partition("=")
            if not separator or variable_name.strip() != "sales":
                raise ValueError("표준 JSON 또는 sales = [...] 형식이 아닙니다.")
            loaded_data = ast.literal_eval(value.strip())

        if isinstance(loaded_data, dict) and "sales" in loaded_data:
            loaded_data = loaded_data["sales"]
        if not isinstance(loaded_data, list):
            raise TypeError("sales 데이터는 리스트여야 합니다.")
        if not all(isinstance(row, dict) for row in loaded_data):
            raise TypeError("각 sales 레코드는 딕셔너리여야 합니다.")

        logger.info("원본 로딩 성공: %s (%d건)", file_path.name, len(loaded_data))
        return loaded_data
    except (FileNotFoundError, OSError, SyntaxError, ValueError, TypeError) as error:
        logger.error("원본 로딩 실패: %s", error)
        return None
    finally:
        logger.info("원본 로딩 종료")


def safe_load_csv(file_path: Path) -> list[dict[str, str]] | None:
    """CSV를 안전하게 읽어 딕셔너리 리스트를 반환한다.

    파일이 없거나 읽기에 실패하면 logger.error를 남기고 None을 반환한다.
    성공하면 logger.info를 남기며 finally에서는 항상 '로딩 종료'를 기록한다.
    """
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        logger.info("CSV 로딩 성공: %s (%d건)", file_path.name, len(rows))
        return rows
    except (FileNotFoundError, OSError, UnicodeError, csv.Error) as error:
        logger.error("CSV 로딩 실패: %s", error)
        return None
    finally:
        logger.info("CSV 로딩 종료")


def validate_records(
    raw_data: list[dict],
) -> tuple[list[SalesRecord], list[dict]]:
    """원본 레코드를 순회하며 정상 모델과 오류 정보 리스트로 분리한다."""
    valid: list[SalesRecord] = []
    errors: list[dict] = []

    for row_number, row in enumerate(raw_data, start=1):
        try:
            valid.append(SalesRecord.model_validate(row))
        except ValidationError as error:
            errors.append(
                {
                    "row": row_number,
                    "data": row,
                    "error": error.errors(
                        include_url=False,
                        include_context=False,
                    ),
                }
            )

    return valid, errors


def save_valid_csv(records: list[SalesRecord], file_path: Path) -> None:
    """검증에 성공한 Pydantic 모델을 CSV 파일로 저장한다."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["month", "region", "amount", "category"]
    with file_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.model_dump() for record in records)


def save_errors_json(errors: list[dict], file_path: Path) -> None:
    """검증에 실패한 행 번호·원본·오류 내용을 JSON 파일로 저장한다."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reload_errors(file_path: Path) -> list[dict]:
    """저장된 오류 JSON을 다시 읽어 리스트인지 확인한 뒤 반환한다."""
    loaded_data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_data, list):
        raise TypeError("오류 결과 JSON은 리스트여야 합니다.")
    return loaded_data


def main() -> int:
    """원본 로드, 검증, 저장, 재로딩 및 건수 검증을 순서대로 실행한다."""
    raw_data = safe_load_json(INPUT_FILE)
    if raw_data is None:
        return 1

    valid, errors = validate_records(raw_data)

    print("\n=== 1. Pydantic 검증 결과 ===")
    print(f"전체: {len(raw_data)}건")
    print(f"정상: {len(valid)}건")
    print(f"오류: {len(errors)}건")

    save_valid_csv(valid, VALID_FILE)
    save_errors_json(errors, ERROR_FILE)

    reloaded_valid = safe_load_csv(VALID_FILE)
    if reloaded_valid is None:
        return 1
    reloaded_errors = reload_errors(ERROR_FILE)

    assert len(raw_data) == len(valid) + len(errors)
    assert len(reloaded_valid) == len(valid)
    assert len(reloaded_errors) == len(errors)

    print("\n=== 2. 결과 파일 저장 ===")
    print(f"정상 CSV: {VALID_FILE}")
    print(f"오류 JSON: {ERROR_FILE}")
    print("\n=== 3. 재로딩 검증 ===")
    print(f"정상 CSV: {len(reloaded_valid)}건 일치")
    print(f"오류 JSON: {len(reloaded_errors)}건 일치")
    print("모든 저장·재로딩 assert를 통과했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
