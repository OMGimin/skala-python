# 교재 Day 1 종합실습 - 데이터 수집 미니 파이프라인

교재 `2) 데이터분석 및 AIOps.pdf`의 Day 1 종합실습입니다. 기존 `capstone01_async_etl`과는 별개의 과제이며, 다음 세 API를 `asyncio.gather()`로 동시에 호출합니다.

- Open-Meteo: 서울 3일 시간대별 기온·강수확률
- Countries.dev: 대한민국 국가 정보
- ip-api: `8.8.8.8`의 IP 기반 지역 정보

## 환경 준비

저장소 최상위 폴더에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r 과제/교재_Day1_종합실습/data_collection_pipeline/requirements.txt
```

## 실행

```bash
.venv/bin/python 과제/교재_Day1_종합실습/data_collection_pipeline/pipeline.py
```

실행하면 `output/`에 API별 CSV·Parquet 파일과 `performance.json`이 생성됩니다.

검증 실행 결과:

```text
weather: 72건
country: 1건
ip_location: 1건

CSV     | 쓰기 0.005322초 | 읽기 0.004915초 | 크기 3,238 bytes
PARQUET | 쓰기 2.089401초 | 읽기 3.521477초 | 크기 12,144 bytes
```

API 데이터는 실행 시점에 달라지므로 시간과 값은 다시 실행할 때 달라질 수 있습니다.

## 테스트와 코드 검사

```bash
cd 과제/교재_Day1_종합실습/data_collection_pipeline
../../../.venv/bin/pytest -v
../../../.venv/bin/ruff check .
```

## 코드 분석 의견

- API 요청은 네트워크 응답을 기다리는 I/O 작업이므로 순차 호출보다 `asyncio.gather()`를 이용한 동시 호출이 적합합니다.
- Pydantic 모델을 API별로 분리하면 서로 다른 응답 구조와 범위 오류를 명확하게 확인할 수 있습니다.
- CSV는 사람이 직접 확인하기 쉽고 작은 데이터에서는 준비 비용이 낮습니다. Parquet은 컬럼 타입을 보존하고 반복 분석 및 큰 데이터에서 유리하지만, 이 실습처럼 데이터가 작으면 초기 처리 비용 때문에 항상 더 빠르지는 않습니다.
- 외부 API 장애가 전체 결과를 조용히 왜곡하지 않도록 HTTP 오류와 스키마 오류를 명시적으로 출력하고 비정상 종료 코드 `1`을 반환합니다.

## 개선 아이디어

- 일시적인 네트워크 오류에 대한 지수 백오프 재시도
- 이전 성공 응답을 이용하는 로컬 캐시
- API별 처리 시간을 별도로 기록하는 로깅
- GitHub Actions에서 pytest와 Ruff 자동 실행

## 제출 자료

실행 결과와 본인 의견은 상위 폴더의
`광주_3반_김지민_day1종합실습_실행결과.pdf`에 정리했습니다.
