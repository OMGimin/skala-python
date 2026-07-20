# 종합실습 1 - 비동기 ETL 파이프라인

60개의 모의 상품 API 응답을 비동기로 수집하고, Pydantic v2로
검증한 뒤 유효한 데이터를 CSV와 Parquet으로 저장합니다.

## 구조

- `models.py`: Product 스키마·정규화 규칙
- `pipeline.py`: Extract, Transform, Load, `run()`
- `test_pipeline.py`: pytest 단위 테스트 6개
- `output/`: 실행 시 CSV·Parquet 생성

## 실행

저장소 최상위 폴더에서 실행합니다.

```bash
.venv/bin/python 과제/Python_data_0720_Day1/Total1/pipeline.py
.venv/bin/pytest 과제/Python_data_0720_Day1/Total1/test_pipeline.py -v
.venv/bin/ruff check 과제/Python_data_0720_Day1/Total1
```
