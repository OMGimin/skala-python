from __future__ import annotations

from typing import Optional
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.axes import Axes


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int = 30,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """수치형 열의 분포를 히스토그램과 밀도 곡선으로 표시한다."""
    if column not in df.columns:
        raise KeyError(f"존재하지 않는 컬럼입니다: {column}")

    chart = ax or plt.subplots(figsize=(9, 5))[1]
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=chart)
    chart.set_title(title or f"Distribution of {column}")
    chart.set_xlabel(column)
    chart.set_ylabel("Count")
    return chart


def save_distance_comparison(frame: pd.DataFrame, output_path: Path) -> Path:
    """거리 그룹별 팁 금액·팁률 비교 차트를 PNG로 저장한다."""
    required_columns = {"distance_group", "tip_rate", "tip_amount"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"거리 차트 필수 컬럼이 없습니다: {missing_columns}")
    required_groups = {"short_lt_5mi", "long_ge_5mi"}
    missing_groups = sorted(required_groups - set(frame["distance_group"].dropna().unique()))
    if missing_groups:
        raise ValueError(f"거리 차트 필수 그룹이 없습니다: {missing_groups}")
    summary = (
        frame.groupby("distance_group", observed=True)
        .agg(mean_tip_rate=("tip_rate", "mean"), mean_tip_amount=("tip_amount", "mean"))
        .reset_index()
    )
    summary["distance_label"] = summary["distance_group"].map(
        {"short_lt_5mi": "Short (<5 mi)", "long_ge_5mi": "Long (≥5 mi)"}
    )
    label_order = ["Short (<5 mi)", "Long (≥5 mi)"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=summary, x="distance_label", y="mean_tip_rate", order=label_order, ax=axes[0])
    axes[0].set(title="Mean Tip Rate by Distance", xlabel="Distance group", ylabel="Mean tip rate")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].bar_label(axes[0].containers[0], labels=[f"{value:.2%}" for value in summary.set_index("distance_label").loc[label_order, "mean_tip_rate"]], padding=3)
    sns.barplot(data=summary, x="distance_label", y="mean_tip_amount", order=label_order, ax=axes[1])
    axes[1].set(title="Mean Tip Amount by Distance", xlabel="Distance group", ylabel="Mean tip amount ($)")
    axes[1].bar_label(axes[1].containers[0], labels=[f"${value:.2f}" for value in summary.set_index("distance_label").loc[label_order, "mean_tip_amount"]], padding=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_hourly_figure(frame: pd.DataFrame) -> go.Figure:
    """시간대·주중/주말별 평균 팁률 Plotly Figure를 생성한다."""
    required_columns = {"pickup_hour", "is_weekend", "tip_rate", "tip_amount"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"시간대 차트 필수 컬럼이 없습니다: {missing_columns}")
    missing_groups = {False, True} - set(frame["is_weekend"].dropna().unique())
    if missing_groups:
        raise ValueError(f"시간대 차트에 주중·주말 그룹이 모두 필요합니다: {sorted(missing_groups)}")
    summary = (
        frame.groupby(["pickup_hour", "is_weekend"], observed=True)
        .agg(
            mean_tip_rate=("tip_rate", "mean"),
            mean_tip_amount=("tip_amount", "mean"),
            trip_count=("tip_rate", "size"),
        )
        .reset_index()
    )
    summary["day_type"] = summary["is_weekend"].map({False: "Weekday", True: "Weekend"})
    figure = px.line(
        summary,
        x="pickup_hour",
        y="mean_tip_rate",
        color="day_type",
        markers=True,
        hover_data={"trip_count": ":,", "mean_tip_amount": ":.2f"},
        title="Hourly Mean Tip Rate: Weekday vs Weekend",
        labels={"pickup_hour": "Pickup hour", "mean_tip_rate": "Mean tip rate", "day_type": "Day type"},
    )
    figure.update_yaxes(tickformat=".1%")
    return figure


def hourly_interactive_fragment(frame: pd.DataFrame) -> str:
    """Jinja2 보고서에 삽입할 Plotly HTML 조각을 반환한다."""
    return build_hourly_figure(frame).to_html(full_html=False, include_plotlyjs="cdn")


def save_hourly_interactive(frame: pd.DataFrame, output_path: Path) -> Path:
    """시간대·주중/주말별 평균 팁률을 인터랙티브 HTML로 저장한다."""
    figure = build_hourly_figure(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path
