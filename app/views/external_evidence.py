"""外部证据页。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.plots import (
    plot_external_dumbbell,
    plot_metric_rank_matrix,
    plot_residual_risk,
    plot_review_budget_curve,
)
from app.ui import (
    metric_card,
    page_header,
    render_boundary,
    render_empty_state,
    render_source_caption,
    section_header,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_ROOT = PROJECT_ROOT / "experiments" / "summary"
BUDGET_PATH = SUMMARY_ROOT / "v0_7_1b" / "v071b_budget_curve.csv"
BOOTSTRAP_PATH = SUMMARY_ROOT / "v0_7_1b" / "v071b_primary_bootstrap_ci.csv"
DATASET_PATH = SUMMARY_ROOT / "v0_7_1b" / "v071b_dataset_event_summary.csv"
RANK_PATH = SUMMARY_ROOT / "v0_7_2" / "v072_method_rank_comparison.csv"
KEY_FINDINGS_PATH = SUMMARY_ROOT / "v0_7_2" / "v072_metric_sensitivity_key_findings.md"


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def external_top20_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    budget = load_csv(str(BUDGET_PATH))
    selected = budget[
        (budget["target"] == "vtdr_miss")
        & (budget["budget"] == 0.2)
        & budget["method"].isin(
            ["random_gate_only", "gated_severe_prob_mass_only"]
        )
    ]
    mean = (
        selected.groupby(["dataset", "method"], as_index=False)[
            ["event_recall", "residual_event_count", "residual_event_rate", "total_event"]
        ]
        .mean()
    )
    pivot = mean.pivot(index="dataset", columns="method", values="event_recall")
    chart = pd.DataFrame(
        {
            "dataset": pivot.index,
            "baseline_recall": pivot["random_gate_only"].to_numpy(),
            "ranked_recall": pivot["gated_severe_prob_mass_only"].to_numpy(),
        }
    ).reset_index(drop=True)
    chart["residual_fraction"] = 1.0 - chart["ranked_recall"]
    return chart, mean


def render_primary_validation() -> None:
    section_header(
        "主验证问题：候选池内部的概率排序有没有超过随机抽取？",
        "VTDR miss 的定义要求真实等级 ≥3 且预测等级 <3，因此目标事件都落在 pred_grade≤2 候选池。"
        "random gate-only 只知道候选池成员身份；gated severe mass 继续按 P(3)+P(4) 排序。",
    )
    missing = [
        path for path in [BUDGET_PATH, BOOTSTRAP_PATH, DATASET_PATH] if not path.exists()
    ]
    if missing:
        render_empty_state("外部主验证文件不完整", "；".join(map(str, missing)))
        return

    chart, top20_mean = external_top20_summary()
    bootstrap = load_csv(str(BOOTSTRAP_PATH))
    dataset_summary = load_csv(str(DATASET_PATH))

    st.pyplot(plot_external_dumbbell(chart), use_container_width=True)
    st.caption(
        "灰点表示在 pred_grade≤2 候选池内随机抽取；青绿点表示在同一候选池、同一 Top20% 总预算下，"
        "按 P(grade 3)+P(grade 4) 从高到低选择。两种方法都不在排序时读取真实标签。"
    )

    for dataset in chart["dataset"]:
        row = chart[chart["dataset"] == dataset].iloc[0]
        ci = bootstrap[bootstrap["dataset"] == dataset].iloc[0]
        sample = dataset_summary[dataset_summary["dataset"] == dataset].iloc[0]
        section_header(dataset)
        cols = st.columns(4, gap="small")
        with cols[0]:
            metric_card(
                "随机候选池",
                f"{row['baseline_recall']:.1%}",
                "候选池内随机抽取的平均事件召回",
                accent="baseline",
            )
        with cols[1]:
            metric_card(
                "概率排序",
                f"{row['ranked_recall']:.1%}",
                "按 grade 3/4 残余概率质量排序",
            )
        with cols[2]:
            metric_card(
                "召回增量",
                f"+{ci['mean_delta_event_recall']:.1%}",
                "95% CI "
                f"[{ci['ci95_low_delta_event_recall']:.1%}, "
                f"{ci['ci95_high_delta_event_recall']:.1%}]",
                accent="amber",
            )
        with cols[3]:
            metric_card(
                "平均减少残余记录",
                f"{ci['mean_residual_event_count_reduction']:.2f}",
                f"{int(sample['n_unique_images'])} 张图像 / "
                f"{int(sample['n_prediction_rows'])} 条 prediction records",
                accent="red",
            )

    chart_col, residual_col = st.columns([1.25, 0.75], gap="large")
    budget = load_csv(str(BUDGET_PATH))
    curve = (
        budget[
            (budget["target"] == "vtdr_miss")
            & budget["method"].isin(
                ["random_gate_only", "gated_severe_prob_mass_only"]
            )
        ]
        .groupby(["dataset", "method", "budget"], as_index=False)["event_recall"]
        .mean()
    )
    dataset_choice = chart_col.selectbox(
        "预算曲线数据集",
        curve["dataset"].unique().tolist(),
        key="external_curve_dataset",
    )
    with chart_col:
        st.pyplot(
            plot_review_budget_curve(
                curve[curve["dataset"] == dataset_choice],
                title=f"{dataset_choice}: review budget vs event recall",
            ),
            use_container_width=True,
        )
    with residual_col:
        st.pyplot(
            plot_residual_risk(chart[["dataset", "residual_fraction"]]),
            use_container_width=True,
        )

    render_boundary(
        "<strong>如何理解：</strong>这组结果支持“完整概率分布中仍保留可利用的排序信息”。"
        "它没有修复外部分类性能，也没有证明临床安全。"
        "prediction records 来自同一图像的多个 backbone，不能当作独立患者。"
    )
    render_source_caption(BUDGET_PATH.relative_to(PROJECT_ROOT))
    render_source_caption(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))

    if st.session_state.get("display_mode") == "研究审计":
        with st.expander("查看 bootstrap 主结果表"):
            st.dataframe(
                bootstrap.rename(
                    columns={
                        "dataset": "外部数据集",
                        "target": "目标事件",
                        "budget": "复核预算",
                        "method": "排序方法",
                        "comparator": "比较基线",
                        "n_bootstrap": "聚类重采样次数",
                        "n_unique_images": "独立图像数",
                        "mean_delta_event_recall": "平均事件召回增量",
                        "ci95_low_delta_event_recall": "召回增量区间下界",
                        "ci95_high_delta_event_recall": "召回增量区间上界",
                        "win_rate_delta_gt_0": "增量大于 0 的比例",
                        "mean_residual_event_count_reduction": "平均残余事件减少",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


def render_metric_sensitivity() -> None:
    section_header(
        "v0.7.2 指标敏感性审计",
        "检查核心排序趋势是否只在 Top20% 单点好看，或能在完整 risk-coverage 评价中保持一致。",
    )
    if not RANK_PATH.exists():
        render_empty_state("缺少 v0.7.2 排名文件", str(RANK_PATH))
        return
    ranks = load_csv(str(RANK_PATH))
    selected = ranks[
        (ranks["event"] == "vtdr_miss")
        & (ranks["method"] == "gated_severe_prob_mass_only")
    ].copy()
    metric_columns = {
        "AURC": "rank_aurc",
        "AUGRC": "rank_augrc",
        "局部 AUGRC（coverage 0.70–0.90）": "rank_partial_augrc_70_90",
        "Top20% 事件召回": "rank_top20_event_recall",
    }
    cols = st.columns(4, gap="small")
    for column, (label, rank_column) in zip(cols, metric_columns.items()):
        first_count = int((selected[rank_column] == 1).sum())
        with column:
            metric_card(
                label,
                f"{first_count}/{len(selected)}",
                "第一或并列第一" if "Top20" in label else "第一排名",
                accent="teal" if first_count == len(selected) else "amber",
            )

    metric_label = st.selectbox("矩阵评价口径", list(metric_columns))
    st.pyplot(
        plot_metric_rank_matrix(selected, rank_col=metric_columns[metric_label]),
        use_container_width=True,
    )
    st.caption(
        "AURC 看选择性风险曲线；AUGRC 更直接累计未被优先复核的目标事件风险；"
        "局部 AUGRC 聚焦 Top10%–Top30% 附近的预算区间。"
    )
    render_boundary(
        "12/12 的曲线指标第一是跨评价口径的一致性描述，不是独立临床效用证明。"
        "局部积分第一也不保证区间内每个具体预算点都第一。"
    )
    render_source_caption(RANK_PATH.relative_to(PROJECT_ROOT))
    if KEY_FINDINGS_PATH.exists() and st.session_state.get("display_mode") == "研究审计":
        with st.expander("查看 v0.7.2 原始 key findings"):
            st.markdown(KEY_FINDINGS_PATH.read_text(encoding="utf-8"))


def render() -> None:
    page_header(
        "外部证据",
        "冻结 APTOS 模型与排序协议后，在 IDRiD 和 MESSIDOR2 上检查危险漏检的复核排序增量与残余风险。",
        "v0.7.1b 主验证 + v0.7.2 指标审计",
    )
    primary_tab, sensitivity_tab = st.tabs(["主验证 v0.7.1b", "稳健性 v0.7.2"])
    with primary_tab:
        render_primary_validation()
    with sensitivity_tab:
        render_metric_sensitivity()
