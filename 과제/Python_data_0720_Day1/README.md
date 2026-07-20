# Python_data_0720_Day1

광주 3반 김지민 Day1 데이터 분석 종합실습 제출 폴더입니다.

## 실행 준비

압축을 풀고 `Python_data_0720_Day1` 폴더로 이동합니다.

```bash
cd Python_data_0720_Day1
python -m pip install -r requirements.txt
```

대용량 데이터 파일이 없다면 원본 데이터 폴더에서 다시 생성합니다.

```bash
python data/generate_data.py
```

## 실행 명령

```bash
python practice1/광주_3반_김지민_practice1.py
python practice2/광주_3반_김지민_practice2.py
python practice3/광주_3반_김지민_practice3.py
python Total1/pipeline.py
python Advanced/advanced_data_quality.py
```

## 구성

- `practice1`: web_logs.csv 스트리밍 집계
- `practice2`: api_response.json Pydantic v2 검증
- `practice3`: asyncio + httpx 비동기 수집
- `Total1`: 비동기 ETL 파이프라인 종합실습
- `Advanced`: ETL 데이터 품질 점수·오류 집계·HTML 모니터링 리포트
