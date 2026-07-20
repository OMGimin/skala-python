"""
실습명: [실습 1] 자료구조 집계 · 컴프리헨션 · 제너레이터
작성자: 광주 3반 김지민
활용 데이터: Python_Practice1_Data.json

[프로그램 설명]
Sales 데이터를 컴프리헨션, Counter, defaultdict,
제너레이터로 집계하고 체크포인트를 assert로 검증한다.

[변경 내역]
- 2026-07-20: 최초 작성
- 2026-07-20: 슬라이드의 실습 내용과 평가 기준을 기준으로 구조 간소화
"""

import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


DATA_FILE = Path(__file__).with_name("Python_Practice1_Data.json")


def load_sales(file_path):
    """파일을 읽어 Sales 리스트를 반환한다.

    표준 JSON과 실제 자료의 ``sales = [...]`` 형식을 모두 처리한다.
    """
    text = file_path.read_text(encoding="utf-8-sig")

    try:
        loaded_data = json.loads(text)
    except json.JSONDecodeError:
        variable, value = text.split("=", 1)
        if variable.strip() != "sales":
            raise ValueError("sales = [...] 형식이 아닙니다.")
        loaded_data = ast.literal_eval(value.strip())

    if isinstance(loaded_data, dict) and "sales" in loaded_data:
        loaded_data = loaded_data["sales"]

    if not isinstance(loaded_data, list):
        raise TypeError("Sales 데이터는 리스트여야 합니다.")

    return loaded_data


def high_amount_generator(sales_data):
    """amount가 1000보다 큰 거래만 하나씩 yield한다."""
    for sale in sales_data:
        if sale["amount"] > 1000:
            yield sale


def main():
    """데이터 로드, 집계, 메모리 비교, 검증을 순서대로 수행한다."""
    try:
        sales = load_sales(DATA_FILE)

        # 1) 리스트/딕셔너리 컴프리헨션
        filtered_sales = [sale for sale in sales if sale["amount"] >= 1000]

        regions = {sale["region"] for sale in sales}
        region_total = {
            region: sum(
                sale["amount"]
                for sale in sales
                if sale["region"] == region
            )
            for region in regions
        }

        print("\n=== 1. 컴프리헨션 ===")
        print(f"amount >= 1000 거래: {len(filtered_sales)}건")
        print("지역별 총매출:", region_total)

        # 체크포인: region_total 값 정확
        expected_region_total = defaultdict(int)
        for sale in sales:
            expected_region_total[sale["region"]] += sale["amount"]
        assert region_total == dict(expected_region_total)

        # 2) Counter + defaultdict
        region_counts = Counter(sale["region"] for sale in sales)
        region_ranking = region_counts.most_common()

        category_amounts = defaultdict(list)
        for sale in sales:
            category_amounts[sale["category"]].append(sale["amount"])

        print("\n=== 2. Counter + defaultdict ===")
        print("지역별 거래 건수:", region_ranking)
        print("카테고리별 amount:", dict(category_amounts))

        # 체크포인트: most_common() 건수가 내림차순인지 확인
        ranking_counts = [count for _, count in region_ranking]
        assert ranking_counts == sorted(ranking_counts, reverse=True)

        # 3) 제너레이터 - 메모리 비교
        generator_result = high_amount_generator(sales)
        list_result = [sale for sale in sales if sale["amount"] > 1000]

        generator_size = sys.getsizeof(generator_result)
        list_size = sys.getsizeof(list_result)

        print("\n=== 3. 제너레이터 메모리 비교 ===")
        print(f"제너레이터: {generator_size} bytes")
        print(f"리스트: {list_size} bytes")

        # 제너레이터를 list()로 변환하지 않고 크기 비교
        assert generator_size < list_size

        # 체크포인: amount 상위 3건 내림차순 정렬
        top3 = sorted(
            sales,
            key=lambda sale: sale["amount"],
            reverse=True,
        )[:3]
        top3_amounts = [sale["amount"] for sale in top3]
        assert top3_amounts == sorted(top3_amounts, reverse=True)

        print("\n=== amount 상위 3건 ===")
        for rank, sale in enumerate(top3, start=1):
            print(f"{rank}위: {sale}")

        # 4) month·category 기준 월별 카테고리 매출 집계
        monthly_amounts = defaultdict(list)
        for sale in sales:
            key = (sale["month"], sale["category"])
            monthly_amounts[key].append(sale["amount"])

        monthly_category_sales = {
            key: sum(amounts)
            for key, amounts in monthly_amounts.items()
        }

        print("\n=== 4. 월별·카테고리별 총매출 ===")
        for (month, category), total in sorted(monthly_category_sales.items()):
            print(f"{month} / {category}: {total:,}")

        print("\n모든 체크포인트 assert를 통과했습니다.")
        return 0

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {DATA_FILE}", file=sys.stderr)
    except (json.JSONDecodeError, SyntaxError, ValueError, TypeError) as error:
        print(f"오류: 데이터를 처리할 수 없습니다: {error}", file=sys.stderr)
    except (KeyError, OSError) as error:
        print(f"오류: 필드 또는 파일 처리 오류: {error}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
