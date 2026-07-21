"""종합실습 3에서 공통으로 사용하는 변경 불가능한 설정을 정의한다."""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Settings:
    """입력·출력·템플릿 경로와 리포트 표시 옵션을 보관한다."""

    data_path: Path = BASE_DIR.parent / "data" / "sales_raw.csv"
    output_dir: Path = BASE_DIR / "output"
    template_dir: Path = BASE_DIR / "templates"
    template_name: str = "report.html"
    report_title: str = "지역·카테고리별 매출 자동화 리포트"
    top_n: int = 10


SETTINGS = Settings()
