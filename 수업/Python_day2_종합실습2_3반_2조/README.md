# NYC Yellow Taxi 3개 주제 통합 분석

광주 3반 2조 Day 2 End-to-End 데이터 분석 프로젝트입니다. 같은 NYC Yellow
Taxi 2026년 5월 데이터를 사용해 팀원이 각각 제안한 세 가지 질문을 동등한
주제로 분석했습니다.

1. 현금 결제는 언제, 어디에서 증가하는가?
2. 초단시간 취소성 트립은 언제 증가하는가?
3. 카드 팁은 거리와 주중·주말에 따라 어떻게 달라지는가?

세 주제 모두 데이터 정제, Pandas·Polars 비교, EDA, 정적·인터랙티브 시각화,
통계검정, sklearn Pipeline, 모델 평가·저장 과정을 포함합니다.

## 통합 폴더 구조

```text
Python_day2_종합실습2_3반_2조/
├── README.md
├── requirements.txt
├── main.py                         # 세 주제 공통 실행 진입점
├── dashboard/
│   └── build_dashboard.py          # 세 주제 결과를 대시보드 데이터로 통합
├── data/
│   ├── raw/
│   │   └── yellow_tripdata_2026-05.parquet
│   └── processed/
│       ├── cleaned_trips.parquet
│       └── yellow_taxi_card_tip.parquet
├── topics/
│   ├── 01_cash_payment/
│   │   └── cash_payment_analysis.py
│   ├── 02_short_trip_cancellation/
│   │   └── short_trip_analysis.py
│   └── 03_weekend_tip/
│       ├── src/
│       ├── tests/
│       ├── notebooks/
│       └── templates/
└── outputs/
    ├── 01_cash_payment/
    ├── 02_short_trip_cancellation/
    ├── 03_weekend_tip/
    └── dashboard/
        ├── artifact.json
        └── integrated_dashboard.html
```

- `data/raw`: 세 주제가 공유하며 직접 수정하지 않는 원본 데이터
- `data/processed`: 분석 코드로 생성한 재사용 가능한 정제 데이터
- `topics`: 세 주제의 코드. 실행 순서 구분용
- `outputs`: 주제별 표·차트·모델·보고서
- `README.md`: 세 주제 설명과 결과를 합친 유일한 Markdown 문서
- `main.py`: 한 주제 또는 세 주제 전체를 실행하는 유일한 진입점
- `outputs/dashboard/integrated_dashboard.html`: 세 주제의 핵심지표와 차트를
  한 화면에 배치한 자체 포함 통합 대시보드

## 통합 대시보드

아래 파일을 브라우저로 열면 별도 서버나 데이터 파일 로딩 없이 세 주제의
핵심지표, 시간대·구간별 차트, 모델 평가와 출처를 확인할 수 있습니다.

```text
outputs/dashboard/integrated_dashboard.html
```

대시보드는 현금 결제율, JFK 새벽 현금 집중, 시간대별 초단시간 트립률,
주중·주말 및 거리별 카드 팁률과 두 예측 모델의 F1·ROC-AUC를 통합합니다.
`artifact.json`은 표시 수치와 출처를 재검증하기 위한 대시보드 원본입니다.

## 통합 실행 방법

```bash
# 세 주제 전체 순차 실행
.venv/bin/python \
'수업/Python_day2_종합실습2_3반_2조/main.py' --topic all

# 현금 결제 시간·지역 분석
.venv/bin/python \
'수업/Python_day2_종합실습2_3반_2조/main.py' --topic cash

# 초단시간 취소성 트립 분석
.venv/bin/python \
'수업/Python_day2_종합실습2_3반_2조/main.py' --topic short-trip

# 주중·주말 카드 팁 분석
.venv/bin/python \
'수업/Python_day2_종합실습2_3반_2조/main.py' --topic weekend-tip
```

## 환경 설치

저장소 최상위 `data-project` 폴더에서 실행합니다.

```bash
.venv/bin/python -m pip install -r \
'수업/Python_day2_종합실습2_3반_2조/requirements.txt'
```

원본 출처는 [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)이며,
분석 파일은 `data/raw/yellow_tripdata_2026-05.parquet`입니다.

데이터 파일은 용량이 크고 재다운로드할 수 있으므로 Git에 포함하지 않습니다.
저장소를 처음 받은 경우 프로젝트 루트에서 다음 명령으로 원본을 준비합니다.

```bash
mkdir -p '수업/Python_day2_종합실습2_3반_2조/data/raw'
curl -L \
  'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet' \
  -o '수업/Python_day2_종합실습2_3반_2조/data/raw/yellow_tripdata_2026-05.parquet'
```

`data/processed`의 Parquet 파일은 각 분석 코드를 실행하면 자동으로 다시
생성됩니다. 가상환경과 패키지는 위 `requirements.txt` 설치 명령으로
재현할 수 있습니다.

---

## 주제 1. 현금 결제는 언제, 어디에서 증가하는가?

전체 카드·현금 비율만 비교하면 카드가 다수라는 사실 외에 새로운 결론이
부족하므로, 현금 사용이 집중되는 시간과 승차지역의 조합을 탐색했습니다.

<!-- AUTO_REPORT_START -->

## 자동 생성 분석 결과

### 프로젝트 개요

2026년 5월 NYC Yellow Taxi 운행에서 현금 결제가 집중되는 시간·지역 조건을
분석하고, 운행 정보로 현금 결제 여부를 예측했다. 카드·현금의 자연 비율은
유지했으며 샘플링으로 클래스 비율을 바꾸지 않았다.

### 데이터 준비

- 원본: 4,090,836행 × 20열
- Pandas 로딩: 0.237초
- Polars 로딩: 0.104초
- 행·열·결측값 결과 일치: True
- 정제 후: 3,000,809행
- 카드: 2,652,933건
- 현금: 347,876건 (11.59%)

### 핵심 EDA 결과

- 전체 현금 결제율: 11.59%
- JFK 새벽 4~6시 현금 결제율: 48.02%
- JFK 새벽 표본: 6,147건
- 그 외 조건 현금 결제율: 11.52%
- 차이: 36.51%p
- 현금 결제 오즈비: 7.10배

JFK 새벽 운행은 전체적으로 드문 현금 결제가 집중되는 핵심 세그먼트다.
다만 관측 데이터의 연관성이므로 공항이나 시간대가 현금 결제를 유발한다고
해석할 수는 없다.

### 통계검정

JFK 새벽 여부와 결제 방식의 카이제곱 검정 p-value는
< 1e-300, Cramér's V는 0.052이다.
두 변수 사이에는 통계적으로 유의한 연관성이 확인됐다.

필수 Welch t-test에서 평균 운행시간은 현금 17.36분,
카드 18.94분이었다. p-value는
< 1e-300이지만 Cohen's d는
-0.096으로 효과크기는 작다. 큰 표본에서는 작은 차이도
통계적으로 유의할 수 있으므로 p-value만으로 큰 차이라고 결론 내리지 않는다.

### ML Pipeline

- 학습 모델: class-weighted Logistic Regression
- 모델링 표본: 400,000행
- 전처리: 결측값 대치 + 수치형 표준화 + 범주형 One-Hot Encoding
- 누수 제외: `payment_type`, `tip_amount`, `total_amount`, `is_cash`
- 저장 모델: `cash_payment_model.joblib`

| 평가 지표 | 다수 클래스 기준 | 학습 모델 |
|---|---:|---:|
| Accuracy | 0.884 | 0.645 |
| Balanced Accuracy | 0.500 | 0.603 |
| 현금 Precision | 0.000 | 0.174 |
| 현금 Recall | 0.000 | 0.548 |
| 현금 F1 | 0.000 | 0.264 |
| ROC-AUC | 0.500 | 0.658 |
| PR-AUC | 0.116 | 0.251 |

Accuracy는 카드가 많은 불균형 데이터에서 과대평가될 수 있다. 현금 탐지가
목적이므로 현금 Recall·F1, Balanced Accuracy와 PR-AUC를 함께 평가한다.

### 결론

현금 사용은 전체 평균만 보면 11% 수준이지만 지역과 시간대의 조합에서는
큰 차이가 나타난다. 특히 JFK 새벽 운행이 가장 선명한 세그먼트였다. 모델은
이러한 조합 패턴을 이용해 다수 클래스 기준보다 현금 탐지 능력을 개선하는지
평가한다.

<!-- AUTO_REPORT_END -->

### 분석 방법

- `payment_type=1`을 카드, `payment_type=2`를 현금으로 정의
- 시간, 요일, 주말, JFK 승차, 새벽 4~6시 파생변수 생성
- 시간·지역·거리 구간별 현금 결제율 집계
- JFK 새벽 여부×현금 결제의 카이제곱 검정, Cramér's V와 오즈비 계산
- 카드·현금 운행시간 Welch t-test와 Cohen's d 계산
- class-weighted Logistic Regression Pipeline으로 현금 결제 예측
- `payment_type`, `tip_amount`, `total_amount`, `is_cash`는 누수 방지 제외

### 실행

통합 실행기의 `--topic cash` 옵션을 사용합니다.

### 주요 산출물

- `outputs/01_cash_payment/cash_payment_eda.png`
- `outputs/01_cash_payment/cash_rate_zone_hour_heatmap.html`
- `outputs/01_cash_payment/cash_payment_model.joblib`
- `outputs/01_cash_payment/cash_project_results.json`

---

## 주제 2. 초단시간 취소성 트립은 언제 증가하는가?

소요시간 1분 이하인 운행을 미터기 시작 직후 취소된 것으로 추정하고, 해당
운행이 특정 시간대에 집중되는지 분석했습니다. 이 주제에서는 일반 정제에서
노이즈로 제거할 수 있는 `duration<=1분`, `trip_distance=0` 기록 자체가 분석
대상이므로 의도적으로 보존합니다.

### 핵심 결과

- 정제 후 4,038,591건 중 초단시간 트립 41,548건(1.03%)
- 심야 2~5시 초단시간 비율 2.19%, 그 외 시간 0.98%
- 심야 비율이 그 외 시간보다 약 2.23배 높음
- 초단시간 평균 이동 거리 0.058마일, 일반 트립 3.458마일
- 거리 Welch t-test Cohen's d 약 1.117로 큰 효과크기
- 심야 여부×초단시간 여부 카이제곱 검정으로 편중 확인

### 분석 방법

- 0분 초과·24시간 이하 운행을 유지하고 거리 0마일 포함
- 시간대·요일·주말·심야·초단시간 파생변수 생성
- 시간대별 초단시간 비율 정적·인터랙티브 시각화
- 거리 Welch t-test, 심야×초단시간 카이제곱 검정
- 불균형을 고려한 HistGradientBoostingClassifier Pipeline
- 목표 정의에 직접 쓰이는 거리·소요시간은 모델 입력에서 제외

### 실행

통합 실행기의 `--topic short-trip` 옵션을 사용합니다.

### 주요 산출물

- `outputs/02_short_trip_cancellation/short_trip_rate_by_hour.png`
- `outputs/02_short_trip_cancellation/short_trip_rate_by_hour.html`
- `outputs/02_short_trip_cancellation/short_trip_classifier.joblib`
- `data/processed/cleaned_trips.parquet`

---

## 주제 3. 카드 팁은 거리와 주중·주말에 따라 어떻게 달라지는가?

현금 팁은 데이터에 완전하게 기록되지 않을 수 있으므로 카드 결제 운행만
사용했습니다. 거리, 시간대, 주중·주말과 팁률의 관계를 분석하고 팁률 20%
이상의 고팁 여부를 예측했습니다.

### 핵심 결과

- 정제된 카드 운행: 2,658,940건
- 5마일 미만 평균 팁률 17.41%, 5마일 이상 13.91%
- 평균 팁 금액은 5마일 미만 $3.30, 5마일 이상 $8.62로 장거리가 더 큼
- 따라서 거리가 늘면 팁 절대 금액은 증가하지만 결제금액 대비 팁률은 낮아짐
- 거리 그룹 차이 3.50%p, Welch t-test p<0.001, Cohen's d=0.443
- 주중 평균 팁률 16.71%, 주말 16.96%로 차이 0.25%p
- 고팁 분류 모델: Accuracy 0.5613, F1 0.6046, ROC-AUC 0.6394

### 분석 방법

- Pandas·Polars 로딩 속도·메모리·결측치 결과 비교
- 팁률=`tip_amount/(total_amount-tip_amount)` 계산
- 거리·시간대·주중·주말 기술통계와 상관관계
- 5마일 기준 Welch t-test와 Cohen's d
- 수치형 대치·표준화, 범주형 대치·One-Hot Encoding
- 대규모 희소 데이터에 적합한 SGDClassifier Pipeline
- 팁 금액·팁률·총액·팁 제외 청구액은 누수 방지 제외
- pytest 단위 테스트와 Jinja2 HTML 통합 보고서

### 실행

통합 실행기의 `--topic weekend-tip` 옵션을 사용합니다.

테스트:

```bash
PYTHONPATH='수업/Python_day2_종합실습2_3반_2조/topics/03_weekend_tip' \
.venv/bin/pytest \
'수업/Python_day2_종합실습2_3반_2조/topics/03_weekend_tip/tests'
```

### 주요 산출물

- `data/processed/yellow_taxi_card_tip.parquet`
- `outputs/03_weekend_tip/tables/`
- `outputs/03_weekend_tip/figures/distance_tip_comparison.png`
- `outputs/03_weekend_tip/figures/hourly_tip_interactive.html`
- `outputs/03_weekend_tip/models/high_tip_pipeline.joblib`
- `outputs/03_weekend_tip/report.html`

---

## 세 주제 공통 채점 기준 반영

| 평가 항목 | 반영 내용 |
|---|---|
| 데이터 준비 | Pandas·Polars 비교, 결측치·중복·이상값 처리와 필터 감사 기록 |
| EDA | 기술통계, 분위수, 그룹 집계, 상관관계 출력 |
| 시각화 | Seaborn 정적 차트와 Plotly 인터랙티브 차트 |
| 통계 분석 | Welch t-test, p-value 해석, 효과크기 및 카이제곱 검정 |
| ML Pipeline | ColumnTransformer+Pipeline, 불균형 대응, 다중 평가 지표 |
| 모델 저장 | 전처리와 모델을 joblib으로 함께 저장 |
| 자동화 | 주제별 결과표·차트·JSON·HTML 보고서 자동 생성 |
| 재현성 | 공유 requirements, 고정 random state, 주제 3 pytest 테스트 |

## 해석상 주의

- 세 분석은 2026년 5월 관찰 데이터에 한정됩니다.
- 통계적으로 유의한 연관성이 인과관계를 의미하지 않습니다.
- 표본이 매우 크므로 p-value뿐 아니라 Cohen's d, Cramér's V, 비율 차이와
  모델 기준선을 함께 해석했습니다.
- 팁 분석은 기록 특성 때문에 카드 결제로 제한되며 현금 팁 행동으로 일반화할
  수 없습니다.
