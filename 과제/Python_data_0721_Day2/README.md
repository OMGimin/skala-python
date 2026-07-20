# Python 데이터 분석 종합실습 Day 2

실제 제공 CSV의 필드명을 기준으로 구현한 종합실습 2·3입니다. 모든 데이터 경로는 소스 파일 위치를 기준으로 계산합니다.

## 설치

압축을 풀고 `Python_data_0721_Day2` 폴더로 이동합니다.

```bash
cd Python_data_0721_Day2
python -m pip install -r requirements.txt
```

## 종합실습 2: EDA + 통계검정 + 머신러닝

```bash
python Total2/analysis.py
```

Polars로 EDA를 수행하고, Welch t-test와 카이제곱 검정을 실행합니다. 전처리와 RandomForest를 하나의 scikit-learn Pipeline으로 학습하며 `output/churn_eda_report.html`, `output/churn_model.joblib`, `output/metrics.json`을 생성합니다.

## 종합실습 3: 자동화 리포트

한 번 실행:

```bash
python Total3/report.py
```

60초 간격 반복 실행(종료: `Ctrl+C`):

```bash
python Total3/run_scheduler.py --interval 60
```

schedule 라이브러리로 1분 간격 실행:

```bash
python Total3/run_scheduler.py --mode schedule --every-minutes 1
```

cron도 동일한 `report.py`를 호출합니다. 예를 들어 매일 오전 9시에 실행하려면 `crontab -e`에 아래 형식으로 등록합니다(경로는 본인의 절대경로로 변경).

```cron
0 9 * * * /절대경로/.venv/bin/python /절대경로/Python_data_0721_Day2/Total3/report.py
```
