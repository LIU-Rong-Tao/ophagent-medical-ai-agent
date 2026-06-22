"""OphAgent Audit Demo 的关键科研图表。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


NAVY = "#17324D"
TEAL = "#0F766E"
AMBER = "#B7791F"
RED = "#B42318"
MUTED = "#5B6878"
BORDER = "#D8E0EA"
BASELINE = "#8A98A8"
SURFACE = "#FFFFFF"

METHOD_LABELS = {
    "random_gate_only": "Random within gate",
    "random_expected": "Random expectation",
    "gated_severe_prob_mass_only": "Severity-aware ranking",
    "ophagent_combined": "OphAgent combined",
    "expected_gap_only": "Expected grade gap",
    "confidence_only": "1-MSP",
    "confidence_only_1msp": "1-MSP",
    "entropy_only": "Entropy",
}

BACKBONE_LABELS = {
    "convnext_tiny": "ConvNeXt-T",
    "retfound_mae_cfp_official_like": "RETFound-L",
    "swin_tiny": "Swin-T",
    "vit_b_imagenet": "ViT-B IN1K",
    "vit_b_official_like": "ViT-B",
    "vit_l_official_like": "ViT-L",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.edgecolor": BORDER,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": NAVY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_probability_profile(
    probabilities: Mapping[str, float] | Sequence[float],
    *,
    pred_grade: int,
    expected_grade: float | None = None,
    expected_gap: float | None = None,
) -> Figure:
    """绘制 DR 五级概率剖面，强调 Top-1 与 grade 3/4 残余概率。"""

    _style()
    labels = ["0 · No DR", "1 · Mild", "2 · Moderate", "3 · Severe", "4 · PDR"]
    if isinstance(probabilities, Mapping):
        values = np.array(list(probabilities.values()), dtype=float)
    else:
        values = np.asarray(probabilities, dtype=float)
    if len(values) != 5:
        raise ValueError("五级概率剖面必须包含 5 个概率。")

    colors = [BASELINE] * 5
    colors[pred_grade] = NAVY
    for grade in (3, 4):
        if grade != pred_grade:
            colors[grade] = AMBER

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(5)
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, float(values.max()) * 1.16))
    ax.set_xlabel("Model probability")
    ax.set_title("Five-grade probability profile", loc="left", pad=14)
    ax.grid(axis="x", alpha=0.18)

    for bar, value in zip(bars, values):
        ax.text(
            min(value + 0.012, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            color=NAVY,
            fontsize=9,
        )

    severe_mass = float(values[3] + values[4])
    annotation = f"P(3)+P(4) = {severe_mass:.1%}"
    if expected_grade is not None:
        annotation += f"   |   Expected grade = {expected_grade:.2f}"
    if expected_gap is not None:
        annotation += f"   |   Expected gap = {expected_gap:+.2f}"
    ax.text(
        0,
        -0.2,
        annotation,
        transform=ax.transAxes,
        color=AMBER,
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_review_budget_curve(
    frame: pd.DataFrame,
    *,
    method_col: str = "method",
    budget_col: str = "budget",
    recall_col: str = "event_recall",
    highlight_method: str = "gated_severe_prob_mass_only",
    title: str = "Review budget vs event recall",
) -> Figure:
    """绘制方法在不同复核预算下的事件召回曲线。"""

    _style()
    required = {method_col, budget_col, recall_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"复核预算曲线缺少列：{', '.join(sorted(missing))}")

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    method_colors = {
        "random_gate_only": BASELINE,
        "random_expected": BASELINE,
        highlight_method: TEAL,
        "ophagent_combined": NAVY,
        "expected_gap_only": AMBER,
    }
    for method, group in frame.groupby(method_col, sort=False):
        points = group.groupby(budget_col, as_index=False)[recall_col].mean()
        points = points.sort_values(budget_col)
        color = method_colors.get(str(method), MUTED)
        width = 2.8 if method == highlight_method else 1.8
        ax.plot(
            points[budget_col] * 100,
            points[recall_col] * 100,
            marker="o",
            color=color,
            linewidth=width,
        )
        ax.plot([], [], color=color, linewidth=width, label=METHOD_LABELS.get(str(method), str(method)))

    ax.axvline(20, color=AMBER, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.text(20.5, 4, "Top20% primary point", color=AMBER, fontsize=8)
    ax.set_xlabel("Review budget")
    ax.set_ylabel("Event recall")
    ax.set_title(title, loc="left", pad=14)
    ax.set_xlim(0, max(52, float(frame[budget_col].max()) * 100 + 8))
    ax.set_ylim(0, 103)
    ax.set_xticks([5, 10, 20, 30, 40, 50], ["5%", "10%", "20%", "30%", "40%", "50%"])
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(alpha=0.18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, max(1, frame[method_col].nunique())),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def plot_external_dumbbell(
    frame: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    baseline_col: str = "baseline_recall",
    ranked_col: str = "ranked_recall",
) -> Figure:
    """绘制同一 Top-K 预算下随机候选池与概率排序的捕获差异。"""

    _style()
    required = {dataset_col, baseline_col, ranked_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"外部证据哑铃图缺少列：{', '.join(sorted(missing))}")

    data = frame.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    y = np.arange(len(data))
    for index, row in data.iterrows():
        left = float(row[baseline_col]) * 100
        right = float(row[ranked_col]) * 100
        ax.plot([left, right], [index, index], color=BORDER, linewidth=5, zorder=1)
        ax.scatter(left, index, s=90, color=BASELINE, zorder=2)
        ax.scatter(right, index, s=105, color=TEAL, zorder=3)
        ax.text(
            left - 1.4,
            index,
            f"{left:.1f}%",
            ha="right",
            va="center",
            color=BASELINE,
        )
        ax.text(
            right + 1.4,
            index,
            f"{right:.1f}%",
            ha="left",
            va="center",
            color=TEAL,
        )
        ax.text(
            (left + right) / 2,
            index + 0.20,
            f"+{right - left:.1f} pp",
            ha="center",
            color=NAVY,
            fontsize=9,
        )
    ax.set_yticks(y, data[dataset_col].astype(str))
    ax.set_ylim(-0.55, len(data) - 0.35)
    ax.set_xlim(0, 100)
    ax.set_xlabel("VTDR-miss recall at Top20% review budget")
    ax.set_title("Random within gate vs severity-aware ranking", loc="left", pad=14)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    return fig


def plot_residual_risk(
    frame: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    residual_col: str = "residual_fraction",
) -> Figure:
    """绘制排序后仍未进入优先复核区的事件比例。"""

    _style()
    required = {dataset_col, residual_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"残余风险图缺少列：{', '.join(sorted(missing))}")
    data = frame.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    bars = ax.bar(
        data[dataset_col].astype(str),
        data[residual_col].astype(float) * 100,
        color=AMBER,
        width=0.5,
    )
    for bar, value in zip(bars, data[residual_col].astype(float)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 100 + 1.2,
            f"{value:.1%}",
            ha="center",
            color=NAVY,
            fontweight="bold",
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Residual event fraction")
    ax.set_title("Residual events after Top20% review", loc="left", pad=14)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return fig


def plot_metric_rank_matrix(
    frame: pd.DataFrame,
    *,
    rank_col: str,
    dataset_col: str = "dataset",
    backbone_col: str = "backbone",
) -> Figure:
    """绘制 2×6 数据集—骨干模型第一排名矩阵。"""

    _style()
    required = {dataset_col, backbone_col, rank_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"稳健性矩阵缺少列：{', '.join(sorted(missing))}")

    datasets = list(dict.fromkeys(frame[dataset_col].astype(str)))
    backbones = list(dict.fromkeys(frame[backbone_col].astype(str)))
    matrix = np.full((len(datasets), len(backbones)), np.nan)
    for i, dataset in enumerate(datasets):
        for j, backbone in enumerate(backbones):
            row = frame[
                (frame[dataset_col].astype(str) == dataset)
                & (frame[backbone_col].astype(str) == backbone)
            ]
            if not row.empty and pd.notna(row.iloc[0][rank_col]):
                matrix[i, j] = 1.0 if float(row.iloc[0][rank_col]) == 1.0 else 0.0

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([BASELINE, TEAL, AMBER])
    display = np.where(np.isnan(matrix), 2, matrix)
    fig, ax = plt.subplots(figsize=(10, 3.1))
    ax.imshow(display, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(
        np.arange(len(backbones)),
        [BACKBONE_LABELS.get(name, name) for name in backbones],
    )
    ax.set_yticks(np.arange(len(datasets)), datasets)
    ax.set_title("Rank consistency across datasets and backbones", loc="left", pad=14)
    ax.tick_params(axis="x", labelsize=8)
    for i in range(len(datasets)):
        for j in range(len(backbones)):
            text = "Rank 1" if display[i, j] == 1 else ("NA" if display[i, j] == 2 else "Other")
            color = "white" if display[i, j] in (0, 1) else NAVY
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    return fig
