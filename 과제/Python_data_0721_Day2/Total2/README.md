# 종합실습 2 - EDA + 통계 + ML 파이프라인

통신 고객 7,000명의 데이터를 이용해 이탈 패턴을 탐색하고, 통계적으로
검증한 뒤 이탈 확률을 예측하는 RandomForest 모델을 학습합니다.

처음부터 모델을 바로 학습하기보다, 데이터를 보면서 생긴 질문을 통계 검정과
모델 비교로 하나씩 확인하는 방식으로 진행했습니다. 구체적인 판단 과정과
Advanced 아이디어로 이어진 흐름은 `../Advanced/아이디어_도출과_고민한점.txt`에
통합했습니다.

## 분석 흐름

1. Polars로 데이터 구조·결측치·이탈률·계약별 이탈률 탐색
2. Plotly로 월 요금 분포·계약별 이탈률 시각화
3. Welch t-test로 이탈/잔류 고객의 월 요금 평균 차이 검정
4. 카이제곱 검정으로 계약 유형과 이탈 여부의 연관성 검정
5. train/test 계층 분할 후 ColumnTransformer로 전처리
6. 전처리와 RandomForest를 하나의 Pipeline으로 학습
7. 확률값으로 ROC-AUC 평가 후 Pipeline 전체 저장

전처리는 train/test 분리 이후 Pipeline 내부에서 학습되므로 테스트 데이터의
정보가 훈련 과정에 들어가는 데이터 누수를 방지합니다. 통계적 연관성은
인과관계를 의미하지 않는다는 점도 결과에 명시합니다.

## 실행

저장소 최상위 폴더에서 실행합니다.

```powershell
python "과제\Python_data_0721_Day2\Total2\analysis.py"
```

## 결과물

- `output/churn_eda_report.html`: Plotly 분석 리포트
- `output/churn_model.joblib`: 전처리와 모델이 결합된 Pipeline
- `output/metrics.json`: EDA·통계·모델 평가 지표
- `output/contract_churn_summary.csv`: 계약 유형별 이탈 요약
- `../Advanced/아이디어_도출과_고민한점.txt`: 구현 중 생긴 질문과 Advanced 도출 과정

## 성공 판정 및 캡처

- t-test와 카이제곱 p값이 모두 0.05 미만
- ROC-AUC가 약 0.66
- 터미널의 통계·모델 지표 및 생성 파일 경로 캡처
- 브라우저에서 `churn_eda_report.html`을 열어 전체 리포트 캡처
