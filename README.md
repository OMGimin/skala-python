# skala-python

SKALA Python 데이터 분석 수업 실습과 과제를 관리하는 비공개 저장소입니다.

## 폴더

- `수업/`: 수업 시간에 작성한 실습
- `과제/`: Day1, Day2 종합 실습

## 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -r 과제/Python_data_0720_Day1/requirements.txt
.venv/bin/python -m pip install -r 과제/Python_data_0721_Day2/requirements.txt
```

## 실행

```bash
# Day1 실습 1~3
.venv/bin/python 과제/Python_data_0720_Day1/practice1/광주_3반_김지민_practice1.py
.venv/bin/python 과제/Python_data_0720_Day1/practice2/광주_3반_김지민_practice2.py
.venv/bin/python 과제/Python_data_0720_Day1/practice3/광주_3반_김지민_practice3.py

# 종합실습 1
.venv/bin/python 과제/Python_data_0720_Day1/Total1/pipeline.py

# 종합실습 2
.venv/bin/python 과제/Python_data_0721_Day2/Total2/analysis.py

# 종합실습 3
.venv/bin/python 과제/Python_data_0721_Day2/Total3/report.py
```

각 과제의 자세한 실행 방식은 해당 제출 폴더의 `README.md`에 정리되어 있습니다.
대용량 Day1 샘플 데이터는 `과제/Python_data_0720_Day1/data/generate_data.py`로 다시 생성할 수 있습니다.
