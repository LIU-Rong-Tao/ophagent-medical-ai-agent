"""OphAgent demo 中与界面无关的审计计算。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


RISK_REASON_LABELS = {
    "low_margin_boundary": "前两类概率接近",
    "moderate_margin_boundary": "前两类差距有限",
    "high_entropy": "多个类别概率分散",
    "moderate_entropy": "概率分布较分散",
    "potential_severe_undergrading_signal": "预测较轻但重症概率仍较高",
    "weak_severe_undergrading_signal": "预测较轻且保留一定重症概率",
    "second_choice_more_severe": "第二候选等级更重",
    "weak_second_choice_more_severe": "第二候选偏向更高等级",
    "confident_but_close_decision": "置信度较高但类别边界接近",
    "routine_low_risk": "未触发主要复核信号",
}

def infer_prob_columns(df: pd.DataFrame) -> list[str]:
    """按 DataFrame 原始顺序返回所有 ``prob_*`` 概率列。"""

    columns = [str(column) for column in df.columns if str(column).startswith("prob_")]
    if not columns:
        raise ValueError("未找到 prob_* 概率列。")
    return columns


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列：{', '.join(missing)}")


def _numeric_probabilities(
    df: pd.DataFrame,
    prob_cols: Sequence[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    columns = list(prob_cols or infer_prob_columns(df))
    _require_columns(df, columns)
    try:
        probabilities = df[columns].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prob_* 列必须全部为数值。") from exc
    return columns, probabilities


def validate_probability_columns(
    df: pd.DataFrame,
    prob_cols: Sequence[str] | None = None,
    *,
    sum_tolerance: float = 0.05,
) -> list[str]:
    """验证概率范围、行和与 ``pred_class`` 类别对应关系。"""

    columns, probabilities = _numeric_probabilities(df, prob_cols)
    if probabilities.isna().any().any():
        raise ValueError("prob_* 列不能包含缺失值。")
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError("prob_* 概率必须位于 [0, 1]。")

    row_sums = probabilities.sum(axis=1)
    invalid_sum = (row_sums - 1.0).abs() > sum_tolerance
    if invalid_sum.any():
        indices = ", ".join(map(str, df.index[invalid_sum].tolist()[:5]))
        raise ValueError(f"概率和必须接近 1（允许 ±{sum_tolerance:.2f}）；异常行：{indices}")

    if "pred_class" in df.columns:
        labels = {column[len("prob_") :] for column in columns}
        predictions = df["pred_class"].astype(str)
        unknown = sorted(set(predictions) - labels)
        if unknown:
            raise ValueError(
                "pred_class 必须存在对应 prob_* 类别；未匹配："
                + ", ".join(unknown[:5])
            )
    return columns


def compute_confidence(
    df: pd.DataFrame,
    prob_cols: Sequence[str] | None = None,
) -> pd.Series:
    """返回每行最大类别概率。"""

    _, probabilities = _numeric_probabilities(df, prob_cols)
    return probabilities.max(axis=1).rename("confidence")


def compute_margin(
    df: pd.DataFrame,
    prob_cols: Sequence[str] | None = None,
) -> pd.Series:
    """返回 Top-1 与 Top-2 概率差。"""

    _, probabilities = _numeric_probabilities(df, prob_cols)
    if probabilities.shape[1] < 2:
        raise ValueError("计算 margin 至少需要两个 prob_* 类别。")
    values = np.sort(probabilities.to_numpy(dtype=float), axis=1)
    return pd.Series(values[:, -1] - values[:, -2], index=df.index, name="margin")


def compute_entropy(
    df: pd.DataFrame,
    prob_cols: Sequence[str] | None = None,
    *,
    normalize: bool = True,
) -> pd.Series:
    """返回 Shannon entropy；默认除以 ``log(num_classes)`` 归一化。"""

    _, probabilities = _numeric_probabilities(df, prob_cols)
    class_count = probabilities.shape[1]
    if class_count < 2:
        raise ValueError("计算 entropy 至少需要两个 prob_* 类别。")
    values = probabilities.to_numpy(dtype=float)
    safe = np.clip(values, np.finfo(float).tiny, 1.0)
    entropy = -(values * np.log(safe)).sum(axis=1)
    if normalize:
        entropy = entropy / math.log(class_count)
    name = "entropy_norm" if normalize else "entropy"
    return pd.Series(entropy, index=df.index, name=name)


def compute_top2(
    df: pd.DataFrame,
    prob_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """返回第二高概率类别及其概率。"""

    columns, probabilities = _numeric_probabilities(df, prob_cols)
    if len(columns) < 2:
        raise ValueError("计算 Top-2 至少需要两个 prob_* 类别。")
    values = probabilities.to_numpy(dtype=float)
    order = np.argsort(values, axis=1)
    indices = order[:, -2]
    labels = [columns[index][len("prob_") :] for index in indices]
    probs = values[np.arange(len(values)), indices]
    return pd.DataFrame(
        {"top2_class": labels, "top2_probability": probs},
        index=df.index,
    )


def compute_expected_gap_for_dr(df: pd.DataFrame) -> pd.DataFrame:
    """计算 DR 五级概率分布的期望等级及其相对 Top-1 的等级差。"""

    columns = [f"prob_{grade}" for grade in range(5)]
    _require_columns(df, [*columns, "pred_grade"])
    _, probabilities = _numeric_probabilities(df, columns)
    pred_grade = pd.to_numeric(df["pred_grade"], errors="raise").astype(float)
    expected_grade = probabilities.to_numpy() @ np.arange(5, dtype=float)
    result = df.copy()
    result["expected_grade"] = expected_grade
    result["expected_gap"] = expected_grade - pred_grade.to_numpy()
    return result


def compute_gated_severe_mass_for_dr(df: pd.DataFrame) -> pd.Series:
    """在 ``pred_grade <= 2`` 的记录中返回 ``P(3)+P(4)``，其余置零。"""

    _require_columns(df, ["prob_3", "prob_4", "pred_grade"])
    _, probabilities = _numeric_probabilities(df, ["prob_3", "prob_4"])
    pred_grade = pd.to_numeric(df["pred_grade"], errors="raise").astype(float)
    severe_mass = probabilities.sum(axis=1)
    return severe_mass.where(pred_grade <= 2, 0.0).rename("gated_severe_mass")


def rank_by_score(
    df: pd.DataFrame,
    score_col: str,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    """按风险分数排序并添加从 1 开始的 ``review_rank``。"""

    _require_columns(df, [score_col])
    ranked = df.sort_values(score_col, ascending=ascending, kind="mergesort").copy()
    ranked["review_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def evaluate_topk_capture(
    ranked_df: pd.DataFrame,
    event_col: str,
    budgets: Iterable[float] = (0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """评估已排序记录在不同复核预算下的目标事件捕获与残余风险。"""

    if event_col not in ranked_df.columns:
        raise KeyError(f"缺少后验事件列：{event_col}")
    if ranked_df.empty:
        raise ValueError("不能对空数据计算 Top-K 捕获。")

    event = ranked_df[event_col].astype(bool)
    total = int(event.sum())
    n = len(ranked_df)
    rows: list[dict[str, float | int]] = []
    for budget in budgets:
        if not 0 < float(budget) <= 1:
            raise ValueError("复核预算必须位于 (0, 1]。")
        top_k = min(n, max(1, math.ceil(n * float(budget))))
        captured = int(event.iloc[:top_k].sum())
        reviewed_event_rate = captured / top_k
        base_rate = total / n
        residual = total - captured
        released_n = n - top_k
        rows.append(
            {
                "review_budget": float(budget),
                "top_k": top_k,
                "total_event": total,
                "captured_event": captured,
                "event_recall": captured / total if total else np.nan,
                "event_precision": reviewed_event_rate,
                "lift_vs_random": reviewed_event_rate / base_rate if base_rate else np.nan,
                "random_recall": top_k / n,
                "residual_event_count": residual,
                "residual_event_rate": residual / released_n if released_n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def class_specific_miss_ranking(
    df: pd.DataFrame,
    target_class: str,
) -> pd.DataFrame:
    """在非目标类预测中，按目标类概率由高到低生成漏检复核队列。"""

    _require_columns(df, ["pred_class"])
    probability_column = f"prob_{target_class}"
    _require_columns(df, [probability_column])
    ranked = df[df["pred_class"].astype(str) != str(target_class)].copy()
    ranked["target_probability"] = pd.to_numeric(
        ranked[probability_column],
        errors="raise",
    )
    if "true_class" in ranked.columns:
        ranked["target_event"] = ranked["true_class"].astype(str) == str(target_class)
    return rank_by_score(ranked, "target_probability")


def translate_risk_reasons(value: str | float | None) -> list[str]:
    """把 v0.6.6 风险原因代码转换为面向临床展示的中文短语。"""

    if value is None or pd.isna(value):
        return ["未记录排序原因"]
    codes = [item.strip() for item in str(value).split(";") if item.strip()]
    return [RISK_REASON_LABELS.get(code, code.replace("_", " ")) for code in codes]


def summarize_dr_review_priority(
    *,
    pred_grade: int,
    probabilities: Sequence[float],
) -> dict[str, str | float]:
    """根据 DR 五级输出给出模型审计层的复核优先级摘要。

    该结果是工程化复核排序，不是临床诊断或治疗建议。
    """

    values = np.asarray(probabilities, dtype=float)
    if len(values) != 5:
        raise ValueError("DR 复核优先级需要五级概率。")
    severe_mass = float(values[3] + values[4])
    expected_grade = float(values @ np.arange(5, dtype=float))
    expected_gap = expected_grade - int(pred_grade)
    order = np.sort(values)
    margin = float(order[-1] - order[-2])
    safe = np.clip(values, np.finfo(float).tiny, 1.0)
    entropy_norm = float(-(values * np.log(safe)).sum() / math.log(5))

    if int(pred_grade) <= 2 and severe_mass >= 0.15:
        level = "high"
        label = "优先复核"
        action = "建议进入首批模型结果复核队列"
        summary = (
            f"Top-1 为 {int(pred_grade)} 级，但重症类别（3/4 级）仍保留 "
            f"{severe_mass:.1%} 概率。"
        )
    elif expected_gap >= 0.75 or margin < 0.15 or entropy_norm >= 0.60:
        level = "medium"
        label = "建议关注"
        action = "建议在常规队列中提前查看"
        summary = (
            "模型输出存在类别边界接近或概率分布偏散，"
            "需要结合图像质量与临床信息复核。"
        )
    else:
        level = "routine"
        label = "常规队列"
        action = "按科室常规流程复核"
        summary = (
            "当前输出未触发主要复核信号；该等级只表示排序较后，"
            "不代表病例安全或无需医生判断。"
        )

    return {
        "level": level,
        "label": label,
        "action": action,
        "summary": summary,
        "severe_mass": severe_mass,
        "expected_grade": expected_grade,
        "expected_gap": expected_gap,
        "margin": margin,
        "entropy_norm": entropy_norm,
    }
