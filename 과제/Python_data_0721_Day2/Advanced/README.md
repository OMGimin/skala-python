# Advanced - 고객 이탈 대응 우선순위

## 시작한 질문

> 모델이 이탈 확률을 계산해도 실제로 누구에게 먼저 연락해야 하는지는 어떻게
> 정할까? 그리고 분류 임계값 0.5는 항상 최선일까?

Total2의 저장 모델을 재사용해 여러 임계값의 Precision·Recall·F1을 비교하고,
F1이 가장 높은 임계값을 선택합니다. 이후 전체 고객을 점수화해 이탈 확률과
월 요금 영향이 모두 큰 고객부터 연락할 수 있는 우선순위 목록을 생성합니다.

## 실행 순서

먼저 Total2 모델을 생성합니다.

```powershell
python "과제\Python_data_0721_Day2\Total2\analysis.py"
python "과제\Python_data_0721_Day2\Advanced\churn_action_report.py"
```

## 결과물

- `output/churn_action_report.html`: 행동 중심 요약 리포트
- `output/customer_churn_priorities.csv`: 전체 고객 위험도·우선순위
- `output/threshold_comparison.csv`: 임계값별 성능 비교
- `output/action_summary.json`: 자동 처리 가능한 요약 지표

## 고민한 점

- 0.5를 무조건 사용하지 않고 왜 임계값을 선택했는지 수치로 남김
- 정확도 대신 Precision·Recall·F1의 trade-off를 확인
- 확률만 높은 고객보다 월 요금 영향까지 고려한 우선순위 사용
- 동일 데이터의 일부로 모델을 평가했으므로 실제 운영 전 새 기간 데이터 검증 필요
- 고객에게 불이익을 주는 용도가 아니라 이탈 방지 혜택 제안 대상으로만 활용

실습 4·5와 종합실습 2·3을 진행하면서 어떤 질문이 생겼고, 그 질문이 어떻게
이 Advanced 주제로 이어졌는지는 `아이디어_도출과_고민한점.txt`에 자세히
정리했습니다.

## 캡처

1. 터미널의 선택 임계값, Precision·Recall·F1, 추천 고객 수
2. 브라우저에서 `churn_action_report.html`을 연 화면
