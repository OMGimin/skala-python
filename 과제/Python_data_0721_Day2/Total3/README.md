# 종합실습 3 - 분석 자동화와 HTML 리포트

매번 CSV를 열어 같은 집계를 반복하지 않도록, 데이터 정제부터 HTML 생성과
실행 이력 기록까지 한 번에 처리합니다. 수동·반복·schedule 실행이 모두 같은
`run_once()`를 호출하므로 실행 방식이 바뀌어도 분석 결과는 같습니다.

## 한 번 실행

```powershell
python "과제\Python_data_0721_Day2\Total3\report.py"
```

## 반복 실행

```powershell
python "과제\Python_data_0721_Day2\Total3\run_scheduler.py" --interval 60
python "과제\Python_data_0721_Day2\Total3\run_scheduler.py" --mode schedule --every-minutes 1
```

반복 실행은 `Ctrl+C`로 종료합니다.

## 결과

- `output/sales_report_날짜_시간.html`: 실행별 리포트
- `output/latest_report.html`: 가장 최근 리포트의 고정된 이름
- `output/report_history.csv`: 실행 시각과 KPI 이력

구현 중 생긴 질문과 Advanced 아이디어로 이어진 과정은
`../Advanced/아이디어_도출과_고민한점.txt`에 통합했습니다.

## 추가한 생각

- 파일명이 고정되면 과거 결과가 사라져서 타임스탬프를 붙임
- 최신 파일을 찾기 번거로워 `latest_report.html`도 함께 갱신
- 실행만 반복하지 않고 직전 매출과 비교할 수 있도록 이력 CSV를 누적
- 직전보다 매출이 20% 이상 감소하면 리포트에 자동 경고 표시
- 원본 결측치 개수를 리포트에 남겨 데이터 품질 상태도 함께 확인
