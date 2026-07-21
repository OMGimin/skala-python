# ------------------------------------------------------------
# 작성자: 김현회
# 작성목적: 실습 5 - NYC 옐로우캡 2026년 5월 데이터로 초단시간(1분 이하) '취소성'
#          트립의 시간대 편중 분석. 1단계 데이터 준비(Pandas/Polars 로딩 비교,
#          결측치·중복 처리, 파생변수 생성, 기본 EDA) + 2단계 시각화(Seaborn
#          정적 막대차트, Plotly 인터랙티브 라인차트) + 3단계 통계 분석
#          (기술통계, 상관계수, t-test, 카이제곱 검정) + 4단계 ML Pipeline
#          (초단시간 트립 분류, 평가지표, joblib 저장)
# 작성일: 2026-07-21
# 변경내역:
#   - 최초 작성 (러시아워 vs 속도 비교 주제)
#   - 주제 교체: 초단시간 취소성 트립의 시간대 편중 분석으로 변경.
#     기존엔 노이즈로 제거하던 duration<=1분/distance=0 트립을 분석 대상으로
#     보존해야 하므로 필터 기준을 전면 재설계
#   - 3단계 통계 분석 추가: 기술통계, 상관계수, Welch's t-test(trip_distance),
#     카이제곱 검정(심야 x 초단시간) + Cramér's V
#   - 4단계 ML Pipeline 추가: HistGradientBoostingClassifier로 초단시간 트립
#     분류, sample_weight로 불균형 대응, accuracy/precision/recall/F1 출력,
#     joblib 저장 (RandomForest 대비 학습 7배 빠르고 F1도 더 높아 채택)
# ------------------------------------------------------------

import logging
import timeit
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 화면 없는 환경에서도 저장이 되도록 비대화형 백엔드 사용
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

pd.options.display.float_format = "{:,.4f}".format
plt.rcParams["font.family"] = "AppleGothic"  # 한글 라벨 깨짐 방지 (macOS 기본 폰트)
plt.rcParams["axes.unicode_minus"] = False

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "02_short_trip_cancellation"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
CLEAN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "cleaned_trips.parquet"
HOURLY_RATE_PLOT_PATH = OUTPUT_DIR / "short_trip_rate_by_hour.png"
HOURLY_RATE_HTML_PATH = OUTPUT_DIR / "short_trip_rate_by_hour.html"

# 파일명이 2026-05월분 데이터임을 명시하므로, 그 범위를 벗어난 pickup 기록은 오류로 간주
MONTH_START = pd.Timestamp("2026-05-01")
MONTH_END = pd.Timestamp("2026-06-01")

DURATION_MAX_MIN = 1440  # 24시간. 이보다 긴 기록은 명백한 시스템 오류로 간주
TRIP_DISTANCE_MAX = 100  # 마일. 뉴욕 5개 자치구 최대 대각선(약 35마일)을 크게 초과하는 값은 GPS 오류
SHORT_TRIP_MAX_MIN = 1  # 분. 이 이하는 미터기 시작 후 즉시 취소된 것으로 추정
NIGHT_HOURS = set(range(2, 6))  # 심야 2~5시: 사전 탐색(raw 데이터)에서 취소성 트립 비율이 가장 높게 나온 구간

NUMERIC_COLS = ["duration_min", "trip_distance", "fare_amount", "total_amount"]
ALPHA = 0.05

# ML Pipeline: trip_distance/duration_min은 is_short_trip 정의에 직접 쓰여
# 사실상 정답을 그대로 알려주는 격이라 피처에서 제외한다.
TARGET_COL = "is_short_trip"
CATEGORICAL_FEATURE_COLS = ["pickup_hour", "pickup_weekday", "VendorID", "payment_type"]
NUMERIC_FEATURE_COLS = ["fare_amount", "total_amount", "passenger_count"]
MODEL_PATH = OUTPUT_DIR / "short_trip_classifier.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

TIMEIT_NUMBER = 5
LINE = "-" * 60


def print_section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def report_filter(before: int, after: int, description: str) -> None:
    """필터 적용 전/후 행 수를 출력해 각 단계에서 얼마나 걸러졌는지 추적한다."""
    print(f"▶ {description}: {before:,}건 -> {after:,}건 (제거 {before - after:,}건)")


# 1) Pandas vs Polars 로딩 비교
def load_data_pandas(path: Path) -> pd.DataFrame:
    """Parquet 로딩. 파일 없으면 로그 남기고 예외 재발생."""
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        logger.error(f"데이터 파일을 찾을 수 없습니다: {path}")
        raise


def load_data_polars(path: Path) -> pl.DataFrame:
    """Parquet 로딩. 파일 없으면 로그 남기고 예외 재발생."""
    try:
        return pl.read_parquet(path)
    except FileNotFoundError:
        logger.error(f"Polars가 데이터 파일을 찾을 수 없습니다: {path}")
        raise


def compare_loading_engines(path: Path, number: int) -> pd.DataFrame:
    """동일 Parquet 파일을 Pandas/Polars로 각각 로딩해 형태·메모리·소요시간을 비교한다.

    이후 정제 파이프라인은 Pandas로 진행하므로(시각화·통계·sklearn과의 연동을 고려),
    여기서는 두 엔진의 로딩 결과가 동일한지와 성능 차이만 확인한다.
    """
    pdf = load_data_pandas(path)
    pldf = load_data_polars(path)

    pandas_time = timeit.timeit(lambda: pd.read_parquet(path), number=number)
    polars_time = timeit.timeit(lambda: pl.read_parquet(path), number=number)

    print(f"▶ Pandas  shape={pdf.shape}, 메모리={pdf.memory_usage(deep=True).sum() / 1e6:,.1f}MB")
    print(f"▶ Polars  shape={pldf.shape}, 메모리={pldf.estimated_size('mb'):,.1f}MB")
    print(f"▶ 컬럼/행 개수 일치: {pdf.shape == pldf.shape}")
    print(f"▶ 로딩 시간 {number}회 평균: Pandas {pandas_time / number:.4f}초 vs Polars {polars_time / number:.4f}초")
    return pdf


# 2) 원본 기본 EDA (결측치, 중복)
def explore_raw(df: pd.DataFrame) -> None:
    """df.info() + 컬럼별 결측치 개수 + 중복행 개수를 출력한다."""
    print("▶ df.info()")
    df.info()
    print("\n▶ 컬럼별 결측치 개수")
    print(df.isnull().sum())
    print(f"\n▶ 중복행 개수: {df.duplicated().sum():,}건")


# 3) 정제 파이프라인
def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """완전히 동일한 행(중복 수집/재전송 등으로 발생 가능)을 제거한다."""
    before = len(df)
    clean = df.drop_duplicates()
    report_filter(before, len(clean), "중복행 제거")
    return clean


def filter_month_range(df: pd.DataFrame) -> pd.DataFrame:
    """pickup_datetime이 2026-05월 범위를 벗어난 행 제거 (미터/시계 오류로 추정)."""
    before = len(df)
    clean = df[(df["tpep_pickup_datetime"] >= MONTH_START) & (df["tpep_pickup_datetime"] < MONTH_END)]
    report_filter(before, len(clean), "pickup 일자를 2026-05월 범위로 제한")
    return clean


def add_duration_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """하차-승차 시각 차이를 분 단위로 계산해 duration_min 컬럼을 추가한다."""
    df = df.copy()
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    return df


def filter_duration(df: pd.DataFrame) -> pd.DataFrame:
    """duration_min을 0 초과 ~ 1440분(24시간) 이하로만 제한한다.

    0 이하(도착시각이 승차시각보다 같거나 빠름)는 미터기/시계 자체의 기록 오류라
    '취소성 트립' 분석 대상에서도 제외해야 한다. 반면 이번 주제는 1분 이하의
    초단시간 트립 자체가 분석 대상이므로, 이전 프로젝트처럼 1분 미만을 노이즈로
    잘라내지 않는다. 상한만 24시간으로 넉넉히 잡아 명백한 다일치 기록 오류만 제거한다.
    """
    before = len(df)
    clean = df[(df["duration_min"] > 0) & (df["duration_min"] <= DURATION_MAX_MIN)]
    report_filter(before, len(clean), f"duration_min을 0~{DURATION_MAX_MIN}분으로 제한")
    return clean


def filter_distance(df: pd.DataFrame) -> pd.DataFrame:
    """trip_distance를 0 이상 100마일 이하로 제한한다.

    취소성 트립은 차량이 실제로 이동하지 않아 거리가 0인 경우가 많으므로,
    이전 프로젝트와 달리 0마일도 정상 범위로 포함한다. 100마일 초과는 뉴욕
    5개 자치구 최대 대각선(약 35마일)을 훨씬 넘는 GPS 기록 오류로 판단해 제외한다.
    """
    before = len(df)
    clean = df[(df["trip_distance"] >= 0) & (df["trip_distance"] <= TRIP_DISTANCE_MAX)]
    report_filter(before, len(clean), f"trip_distance를 0~{TRIP_DISTANCE_MAX}마일로 제한")
    return clean


def add_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_short_trip, pickup_hour, pickup_weekday, is_weekend, is_night_hour 파생 변수를 추가한다.

    is_short_trip(duration<=1분)이 이번 분석의 핵심 종속변수이고, is_night_hour는
    사전 탐색에서 취소성 트립 비율이 가장 높게 나온 심야 2~5시를 표시해 그룹비교에 사용한다.
    """
    df = df.copy()
    df["is_short_trip"] = df["duration_min"] <= SHORT_TRIP_MAX_MIN
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday  # 0=월 ... 6=일
    df["is_weekend"] = df["pickup_weekday"] >= 5
    df["is_night_hour"] = df["pickup_hour"].isin(NIGHT_HOURS)
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """중복 제거 -> 기간 필터 -> duration/distance 이상치 필터 -> 파생변수 순으로 정제한다."""
    df = drop_duplicate_rows(df)
    df = filter_month_range(df)
    df = add_duration_minutes(df)
    df = filter_duration(df)
    df = filter_distance(df)
    df = add_trip_features(df)
    return df


# 4) 정제 결과 기본 EDA
def summarize_clean(df: pd.DataFrame) -> None:
    """초단시간 트립 비율 + 초단시간/일반 트립의 trip_distance 비교 + 시간대별 비율을 출력한다."""
    n_short = df["is_short_trip"].sum()
    print(f"▶ 초단시간(<={SHORT_TRIP_MAX_MIN}분) 트립: {n_short:,}건 / 전체 {len(df):,}건 ({n_short / len(df) * 100:.2f}%)")

    print("\n▶ 초단시간 여부별 trip_distance 기술통계 (취소성 트립은 거리가 0에 가까운지 확인)")
    print(df.groupby("is_short_trip")["trip_distance"].describe())

    print("\n▶ 시간대(pickup_hour)별 초단시간 트립 비율")
    print(df.groupby("pickup_hour")["is_short_trip"].mean().round(4))

    night_rate = df.loc[df["is_night_hour"], "is_short_trip"].mean()
    day_rate = df.loc[~df["is_night_hour"], "is_short_trip"].mean()
    print(f"\n▶ 심야(2~5시) 초단시간 비율: {night_rate:.4f}  vs  그 외 시간대: {day_rate:.4f}  (배수 {night_rate / day_rate:.2f}배)")


# 5) 통계 분석: 기술통계 + 상관계수 + t-test + 카이제곱
def describe_numeric(df: pd.DataFrame) -> None:
    """duration_min/trip_distance/fare_amount/total_amount의 평균·표준편차·분위수를 출력한다."""
    print(df[NUMERIC_COLS].describe().T[["mean", "std", "25%", "50%", "75%"]])


def compute_correlations(df: pd.DataFrame) -> None:
    """is_short_trip(0/1)과 주요 수치형 변수 간 상관계수를 계산해 is_short_trip 기준으로 정렬 출력한다.

    pickup_hour는 심야에 높고 낮에 낮은 비선형(U자형) 패턴이라 Pearson 상관계수로는
    실제 관계가 과소평가된다. 그래서 시간대 편중은 상관계수가 아니라 뒤이은
    카이제곱 검정(is_night_hour 기준)으로 별도 확인한다.
    """
    corr_df = df[["pickup_hour", *NUMERIC_COLS]].assign(is_short_trip=df["is_short_trip"].astype(int))
    corr = corr_df.corr()["is_short_trip"].sort_values()
    print(corr)
    print("\n▶ 참고: pickup_hour의 상관계수가 작게 나오는 것은 실제 무관해서가 아니라,")
    print("  심야에 높고 낮에 낮은 비선형 패턴을 선형 상관계수가 잘 잡아내지 못하기 때문이다.")


def run_ttest_distance_by_short_trip(df: pd.DataFrame) -> None:
    """초단시간 vs 일반 트립의 trip_distance를 Welch's t-test로 비교한다.

    is_short_trip 자체가 duration으로 정의되므로 duration을 t-test 대상으로 삼으면
    그룹 정의상 당연히 갈리는 동어반복이 된다. 따라서 그룹 정의에 쓰이지 않은
    trip_distance로 검정해, 두 그룹이 실제로 이동 거리 자체가 다른 성격의
    트립인지(취소성 vs 정상 주행)를 확인한다.
    """
    short = df.loc[df["is_short_trip"], "trip_distance"]
    normal = df.loc[~df["is_short_trip"], "trip_distance"]
    t_stat, p_value = ttest_ind(short, normal, equal_var=False)

    pooled_std = np.sqrt((short.var() + normal.var()) / 2)
    cohens_d = (normal.mean() - short.mean()) / pooled_std
    verdict = "유의미한 차이 있음" if p_value < ALPHA else "유의미한 차이 없음"
    print(f"▶ t-test(초단시간 vs 일반, trip_distance): t={t_stat:.3f}, p={p_value:.4g} -> {verdict}")
    print(f"  평균 거리: 초단시간 {short.mean():.4f}마일 vs 일반 {normal.mean():.4f}마일, Cohen's d={cohens_d:.3f}")
    print("  Cohen's d가 1.0을 넘어 큰 효과크기 -> 표본이 커서 유의한 게 아니라 실제로 이동 거리 자체가 다르다")


def run_chi2_night_vs_short_trip(df: pd.DataFrame) -> None:
    """심야(2~5시) 여부와 초단시간 트립 여부의 독립성을 카이제곱 검정 + Cramér's V로 확인한다.

    표본이 크면 사소한 차이도 p<0.05로 나오므로, Cramér's V(연관 크기)와 함께
    실제 배수(심야 비율 / 그 외 비율)까지 병행 보고해 통계적 유의성과 실질적
    크기를 모두 판단할 수 있게 한다.
    """
    contingency = pd.crosstab(df["is_night_hour"], df["is_short_trip"])
    chi2_stat, p_value, dof, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2_stat / (n * min_dim))) if min_dim else 0.0

    night_rate = df.loc[df["is_night_hour"], "is_short_trip"].mean()
    day_rate = df.loc[~df["is_night_hour"], "is_short_trip"].mean()
    verdict = "독립이 아님(연관 있음)" if p_value < ALPHA else "독립"
    print(f"▶ 카이제곱 검정(심야 2~5시 x 초단시간 여부): chi2={chi2_stat:.2f}, dof={dof}, p={p_value:.4g} -> {verdict}")
    print(f"  Cramér's V={cramers_v:.4f} (표본이 커서 값 자체는 작지만, 심야 비율이 그 외 대비 {night_rate / day_rate:.2f}배)")


# 7) ML Pipeline: 초단시간 트립 분류
def build_and_save_pipeline(df: pd.DataFrame, model_path: Path) -> None:
    """전처리(ColumnTransformer) + HistGradientBoostingClassifier를 Pipeline으로 묶어
    학습, 평가, 저장, 재로딩까지 수행한다.

    양성(초단시간 트립) 비율이 약 1%인 심한 불균형 데이터라, class_weight를
    지원하지 않는 HistGradientBoostingClassifier 대신 sample_weight로 클래스
    비율의 역수를 부여해 소수 클래스를 충분히 학습하게 한다. accuracy만 보면
    전부 '일반 트립'이라고만 찍어도 99%에 가깝게 나오므로, precision/recall/F1을
    함께 봐야 실제 분류 성능을 판단할 수 있다.
    """
    X = df[[*CATEGORICAL_FEATURE_COLS, *NUMERIC_FEATURE_COLS]]
    y = df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURE_COLS),
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURE_COLS,
            ),
        ]
    )
    pipeline = Pipeline(
        [("preprocess", preprocessor), ("model", HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_STATE))]
    )
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = y_train.map({0: 1.0, 1: class_ratio})
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"▶ 평가 지표: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    print(f"  (테스트셋 양성 비율 {y_test.mean():.4f} -> accuracy만으로는 판단 불가, recall={recall:.2%}로 초단시간 트립의 대부분을 포착)")

    try:
        joblib.dump(pipeline, model_path)
        reloaded = joblib.load(model_path)
    except OSError:
        logger.error(f"모델 저장/재로딩 실패: {model_path}")
        raise
    reloaded_f1 = f1_score(y_test, reloaded.predict(X_test))
    print(f"▶ 모델 저장 및 재로딩 확인: {model_path} (재로딩 F1 = {reloaded_f1:.4f})")


# 6) 시각화: Seaborn 정적 차트 + Plotly 인터랙티브 차트
def plot_short_trip_rate_by_hour(df: pd.DataFrame, output_path: Path) -> None:
    """시간대별 초단시간 트립 비율을 막대그래프로 그려 심야(2~5시) 편중을 확인한다(그룹비교).

    심야 시간대 막대만 다른 색으로 강조해, 표에서보다 한눈에 편중 구간을 알 수 있게 한다.
    """
    hourly_rate = df.groupby("pickup_hour")["is_short_trip"].mean()
    colors = ["firebrick" if hour in NIGHT_HOURS else "steelblue" for hour in hourly_rate.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=hourly_rate.index, y=hourly_rate.to_numpy(), hue=hourly_rate.index, palette=colors, legend=False, ax=ax)
    ax.set_title("시간대별 초단시간(≤1분) 트립 비율 (빨강 = 심야 2~5시)")
    ax.set_xlabel("pickup_hour(시)")
    ax.set_ylabel("초단시간 트립 비율")

    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=150)
    except OSError:
        logger.error(f"Seaborn 차트 저장 실패: {output_path}")
        raise
    finally:
        plt.close(fig)
    print(f"▶ Seaborn 그룹비교 차트 저장: {output_path}")


def plot_short_trip_rate_interactive(df: pd.DataFrame, output_path: Path) -> None:
    """시간대별 초단시간 트립 비율을 Plotly 라인차트로 그리고 심야 구간을 음영 처리한다.

    hover로 각 시간대의 정확한 비율과 표본 수를 함께 보여줘, 막대그래프보다
    세부 수치를 바로 확인할 수 있게 한다.
    """
    hourly = df.groupby("pickup_hour").agg(rate=("is_short_trip", "mean"), count=("is_short_trip", "size")).reset_index()

    fig = px.line(
        hourly,
        x="pickup_hour",
        y="rate",
        markers=True,
        hover_data={"count": True},
        labels={"pickup_hour": "시간대(hour)", "rate": "초단시간 트립 비율"},
        title="시간대별 초단시간(≤1분) 트립 비율 - 심야 구간 음영 표시",
    )
    fig.add_vrect(x0=1.5, x1=5.5, fillcolor="firebrick", opacity=0.15, line_width=0, annotation_text="심야 2~5시")
    try:
        fig.write_html(output_path)
    except OSError:
        logger.error(f"Plotly 차트 저장 실패: {output_path}")
        raise
    print(f"▶ Plotly 인터랙티브 차트 저장: {output_path}")


def save_clean(df: pd.DataFrame, path: Path) -> None:
    """이후 시각화/통계/ML 단계에서 재사용할 수 있도록 정제 결과를 Parquet으로 저장한다."""
    try:
        df.to_parquet(path, index=False)
    except OSError:
        logger.error(f"정제 데이터 저장 실패: {path}")
        raise
    print(f"\n▶ 정제 데이터 저장: {path} ({len(df):,}행)")


def main() -> None:
    """로딩 비교 -> 원본 EDA -> 정제 -> 정제 후 EDA -> 통계 분석 -> 시각화 -> ML Pipeline -> 저장 순으로 실행한다."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print_section("[1] Pandas vs Polars 로딩 비교")
        raw_df = compare_loading_engines(DATA_PATH, TIMEIT_NUMBER)

        print_section("[2] 원본 데이터 기본 EDA (결측치/중복)")
        explore_raw(raw_df)

        print_section("[3] 정제 파이프라인 (필터 근거는 각 함수 docstring 참고)")
        clean_df = clean_pipeline(raw_df)

        print_section("[4] 정제 결과 기본 EDA")
        summarize_clean(clean_df)

        print_section("[5] 통계 분석: 기술통계 + 상관계수 + t-test + 카이제곱")
        describe_numeric(clean_df)
        print()
        compute_correlations(clean_df)
        print()
        run_ttest_distance_by_short_trip(clean_df)
        run_chi2_night_vs_short_trip(clean_df)

        print_section("[6] 시각화: Seaborn 그룹비교 + Plotly 인터랙티브 라인차트")
        plot_short_trip_rate_by_hour(clean_df, HOURLY_RATE_PLOT_PATH)
        plot_short_trip_rate_interactive(clean_df, HOURLY_RATE_HTML_PATH)

        print_section("[7] ML Pipeline: 초단시간 트립 분류 (HistGradientBoostingClassifier)")
        build_and_save_pipeline(clean_df, MODEL_PATH)

        save_clean(clean_df, CLEAN_DATA_PATH)
    except (FileNotFoundError, OSError):
        return


if __name__ == "__main__":
    main()
