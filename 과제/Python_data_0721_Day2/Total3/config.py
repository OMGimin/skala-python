"""종합실습 3의 변경 가능한 설정을 한곳에서 관리한다."""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReportConfig:
    """실행 중 실수로 변경되지 않는 리포트 설정."""

    data_path: Path = BASE_DIR.parent / "data" / "sales_raw.csv"
    template_path: Path = BASE_DIR / "templates" / "sales_report.html"
    output_dir: Path = BASE_DIR / "output"
    history_path: Path = BASE_DIR / "output" / "report_history.csv"
    title: str = "판매 데이터 자동 분석 리포트"
    top_n: int = 5
    low_sales_warning_ratio: float = 0.8


CONFIG = ReportConfig()
