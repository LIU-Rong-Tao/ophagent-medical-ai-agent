"""Streamlit workspace for generic prediction-table model-error risk audits."""

from __future__ import annotations

import html
from io import BytesIO
import json
from pathlib import Path
import tempfile
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from app.generic_result_audit import (
    ResultTableMapping,
    VariantMapping,
    detect_probability_groups,
    export_audit_result,
    list_excel_sheets,
    normalize_result_table,
    read_result_table,
    run_generic_risk_audit,
    suggest_mapping,
    validate_normalized_predictions,
)


NONE_OPTION = "不使用"
CLASS_ALIAS_CONFIG = Path(__file__).resolve().parents[1] / "configs/result_audit_class_aliases.json"
LEAKAGE_CHECK_LABELS = {
    "duplicate_case_id": "病例标识重复",
    "case_id_cross_split": "病例跨数据划分",
    "label_name_in_metadata": "标识或元数据包含类别名",
    "obvious_answer_columns": "疑似答案字段",
    "metadata_label_mapping": "元数据与标签近确定映射",
    "patient_overlap": "患者级重叠",
    "image_overlap": "图像级重叠",
    "training_test_isolation": "训练与测试流程隔离",
}
REVIEW_SIGNAL_LABELS = {
    "低 confidence": "低置信度",
    "高 entropy": "高归一化熵",
    "低 margin": "低 Top1-Top2 差值",
}


def _resolve_class_aliases(labels: list[str]) -> dict[str, str]:
    """Return display-only aliases when a configured label set matches exactly."""
    try:
        payload = json.loads(CLASS_ALIAS_CONFIG.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return {}
    label_set = {str(label) for label in labels}
    for profile in payload.get("profiles", []):
        match_labels = {str(label) for label in profile.get("match_labels", [])}
        aliases = {str(key): str(value) for key, value in profile.get("aliases", {}).items()}
        if label_set == match_labels and match_labels == set(aliases):
            return aliases
    return {}


def _class_display(label: object, aliases: dict[str, str]) -> str:
    raw = str(label)
    alias = aliases.get(raw)
    return f"{raw}｜{alias}" if alias else raw


def _render_class_key(aliases: dict[str, str]) -> None:
    if not aliases:
        return
    items = "".join(
        f'<span><b>{html.escape(label)}</b>{html.escape(alias)}</span>'
        for label, alias in aliases.items()
    )
    st.markdown(
        f'<div class="hub-class-key"><strong>类别说明</strong>{items}</div>',
        unsafe_allow_html=True,
    )


def _render_audit_process(active_step: int) -> None:
    labels = ["上传表格", "字段映射", "数据校验", "审计结果"]
    items = "".join(
        '<div class="hub-process-item{}"><b>{}</b>{}</div>'.format(
            " active" if index == active_step else "",
            index,
            html.escape(label),
        )
        for index, label in enumerate(labels, start=1)
    )
    st.markdown(f'<div class="hub-process">{items}</div>', unsafe_allow_html=True)


def _render_kpi_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        '<div class="hub-audit-kpi{}">'.format(f" {tone}" if tone else "")
        + f"<span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"
        for label, value, tone in items
    )
    compact = " compact" if len(items) <= 4 else ""
    st.markdown(
        f'<div class="hub-audit-kpis{compact}">{cards}</div>',
        unsafe_allow_html=True,
    )


def _optional_selectbox(label: str, columns: list[str], *, default: str | None, key: str) -> str | None:
    options = [NONE_OPTION, *columns]
    index = options.index(default) if default in options else 0
    value = st.selectbox(label, options, index=index, key=key)
    return None if value == NONE_OPTION else value


def _probability_class_name(column: str) -> str:
    detected = detect_probability_groups([column])
    for group in detected.values():
        for class_name in group:
            return class_name
    return str(column).strip()


def _mapping_is_complete(mapping: ResultTableMapping | None) -> bool:
    if mapping is None or not mapping.case_id_column or not mapping.true_label_column:
        return False
    return bool(mapping.variants) and all(
        variant.prediction_column or variant.probability_columns
        for variant in mapping.variants
    )


def _default_primary_variant(mapping: ResultTableMapping) -> str:
    names = [variant.name for variant in mapping.variants]
    return "base" if "base" in names else names[0]


def _render_auto_mapping_summary(mapping: ResultTableMapping) -> None:
    split_column = mapping.split_column or "未提供"
    st.markdown(
        '<div class="hub-band"><strong>字段自动识别完成，可直接运行审计。</strong>'
        f'　病例标识：{html.escape(mapping.case_id_column)}'
        f'　真实标签：{html.escape(mapping.true_label_column)}'
        f'　数据划分：{html.escape(split_column)}</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for variant in mapping.variants:
        probability_count = len(variant.probability_columns)
        rows.append(
            {
                "预测版本": variant.name,
                "预测标签": variant.prediction_column or "由概率最大值推断",
                "置信度": variant.confidence_column or "由最大概率计算",
                "类别概率": f"{probability_count} 列" if probability_count else "未提供",
                "耗时": variant.latency_column or "未提供",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _mapping_controls(frame: pd.DataFrame, *, file_key: str) -> tuple[ResultTableMapping, str]:
    columns = [str(column) for column in frame.columns]
    suggested = suggest_mapping(frame)
    st.markdown("#### 字段映射确认")
    st.caption("仅在自动识别与实际表结构不一致时手动调整。")
    left, middle, right = st.columns(3, gap="medium")
    with left:
        case_default = suggested.case_id_column if suggested else columns[0]
        case_id = st.selectbox(
            "病例标识列",
            columns,
            index=columns.index(case_default),
            key=f"audit_case::{file_key}",
        )
    with middle:
        truth_default = suggested.true_label_column if suggested else columns[0]
        true_label = st.selectbox(
            "真实标签列",
            columns,
            index=columns.index(truth_default),
            key=f"audit_truth::{file_key}",
        )
    with right:
        split = _optional_selectbox(
            "数据划分列（可选）",
            columns,
            default=suggested.split_column if suggested else None,
            key=f"audit_split::{file_key}",
        )

    suggested_variants = list(suggested.variants) if suggested else [VariantMapping("base")]
    variant_count = int(
        st.number_input(
            "预测版本数量",
            min_value=1,
            max_value=8,
            value=len(suggested_variants),
            step=1,
            key=f"audit_variant_count::{file_key}",
            help="单版本保持 1；存在 TTA、ensemble 或其他输出时再增加。",
        )
    )
    variants: list[VariantMapping] = []
    used_probability_columns: set[str] = set()
    for index in range(variant_count):
        default = suggested_variants[index] if index < len(suggested_variants) else VariantMapping(f"other_{index}")
        probability_count = len(default.probability_columns)
        probability_note = f" · {probability_count} 个概率列" if probability_count else ""
        with st.expander(
            f"预测版本 {index + 1} · {default.name}{probability_note}",
            expanded=index == 0 and probability_count <= 20,
        ):
            name = st.text_input(
                "版本名称",
                value=default.name,
                key=f"audit_variant_name::{file_key}::{index}",
                help="可使用 base、tta、ensemble 或自定义名称。",
            ).strip()
            col1, col2, col3 = st.columns(3, gap="medium")
            with col1:
                prediction = _optional_selectbox(
                    "预测标签列",
                    columns,
                    default=default.prediction_column,
                    key=f"audit_pred::{file_key}::{index}",
                )
            with col2:
                confidence = _optional_selectbox(
                    "置信度列（可选）",
                    columns,
                    default=default.confidence_column,
                    key=f"audit_conf::{file_key}::{index}",
                )
            with col3:
                latency = _optional_selectbox(
                    "耗时列（可选）",
                    columns,
                    default=default.latency_column,
                    key=f"audit_latency::{file_key}::{index}",
                )
            default_probabilities = list(default.probability_columns.values())
            probability_columns = st.multiselect(
                "完整类别概率列（可选）",
                columns,
                default=[column for column in default_probabilities if column in columns],
                key=f"audit_probs::{file_key}::{index}",
                help="没有概率时留空；有概率时应选择当前版本的全部类别概率列。",
            )
            duplicates = used_probability_columns.intersection(probability_columns)
            if duplicates:
                st.warning(f"以下概率列同时分配给多个版本：{', '.join(sorted(duplicates))}")
            used_probability_columns.update(probability_columns)
            probability_mapping = {
                _probability_class_name(column): column for column in probability_columns
            }
            if probability_mapping:
                st.caption(
                    "已识别类别："
                    + "、".join(probability_mapping.keys())
                    + "。类别只来自当前映射，不写入核心代码。"
                )
            variants.append(
                VariantMapping(
                    name=name or f"variant_{index + 1}",
                    prediction_column=prediction,
                    probability_columns=probability_mapping,
                    confidence_column=confidence,
                    latency_column=latency,
                )
            )
    metadata_defaults = [
        column
        for column in columns
        if any(token in column.casefold() for token in ("file", "path", "image", "record", "meta"))
        and column != case_id
    ]
    metadata = st.multiselect(
        "用于标签捷径检查的元数据列（可选）",
        columns,
        default=metadata_defaults,
        key=f"audit_metadata::{file_key}",
        placeholder="不选择元数据字段",
        help="这里只检查结果表中可见的文件名、路径或低基数元数据，不会把这些字段展示到病例清单。",
    )
    primary = st.selectbox(
        "主预测版本",
        [variant.name for variant in variants],
        key=f"audit_primary::{file_key}",
        help="主版本由用户预先指定，系统不会根据当前测试表现自动选择。",
    )
    if len(variants) > 1:
        st.caption("主版本应在查看指标前确定；若根据当前评测结果改选，属于探索性比较。")
    return (
        ResultTableMapping(
            case_id_column=case_id,
            true_label_column=true_label,
            split_column=split,
            variants=tuple(variants),
            metadata_columns=tuple(metadata),
        ),
        primary,
    )


def _zip_directory(directory: Path) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    return buffer.getvalue()


def _render_validation(validation, *, compact: bool = False) -> None:
    summary = validation.summary
    if not compact:
        cards = [
            ("导入记录", f"{summary.get('total_rows', 0):,}", ""),
            ("唯一病例", f"{summary.get('unique_cases', 0):,}", ""),
            ("类别数", f"{summary.get('class_count', 0):,}", ""),
            ("预测版本", f"{summary.get('variant_count', 0):,}", ""),
        ]
        content = "".join(
            '<div class="hub-overview-kpi">'
            f"<span>{label}</span><b>{value}</b><small>导入校验</small></div>"
            for label, value, _ in cards
        )
        st.markdown(f'<div class="hub-overview-grid">{content}</div>', unsafe_allow_html=True)
    if validation.passed:
        detail = (
            f"{summary.get('unique_cases', 0):,} 个唯一病例 · "
            f"{summary.get('class_count', 0):,} 个类别 · "
            f"{summary.get('variant_count', 0):,} 个预测版本"
        )
        st.success(f"数据校验通过：{detail}。")
    for issue in validation.issues:
        message = f"{issue.message}" + (f"（{issue.count} 条）" if issue.count else "")
        if issue.severity == "error":
            st.error(message)
        else:
            st.warning(message)


def _render_overall(audit) -> None:
    summary = audit.summary.iloc[0]
    st.markdown("#### 总体表现")
    st.caption("总体指标用于描述当前主预测版本；分类别结果用于定位薄弱类别，不代表临床后果严重度。")
    pairs = audit.confusion_pairs.copy()
    class_labels = sorted(
        set(pairs.get("true_label", pd.Series(dtype=str)).astype(str))
        | set(pairs.get("predicted_label", pd.Series(dtype=str)).astype(str))
    )
    aliases = _resolve_class_aliases(class_labels)
    _render_class_key(aliases)
    metrics = pd.DataFrame(
        [
            ("Accuracy", summary["accuracy"], "全部病例总体命中"),
            ("Macro Precision", summary["macro_precision"], "各类别等权精确率"),
            ("Macro Recall", summary["macro_recall"], "各类别等权召回率"),
            ("Macro-F1", summary["macro_f1"], "类别均衡综合指标"),
            ("Weighted F1", summary["weighted_f1"], "按类别样本数加权"),
            ("Cohen Kappa", summary["cohen_kappa"], "扣除随机一致性"),
        ],
        columns=["指标", "值", "解释"],
    )
    chart_column, metric_column = st.columns([1.65, 0.85], gap="medium")
    with metric_column:
        st.markdown("##### 总体指标")
        st.caption("先看总体，再定位低表现类别。")
        st.dataframe(
            metrics,
            hide_index=True,
            width="stretch",
            column_config={"值": st.column_config.NumberColumn(format="%.3f")},
        )
    with chart_column:
        if not pairs.empty and int(summary["class_count"]) <= 20:
            st.markdown("##### 混淆矩阵")
            st.caption("纵轴为真实类别，横轴为预测类别；对角线表示预测正确。")
            full_grid = pd.MultiIndex.from_product(
                [class_labels, class_labels], names=["true_label", "predicted_label"]
            ).to_frame(index=False)
            matrix = full_grid.merge(pairs, how="left", on=["true_label", "predicted_label"])
            matrix["count"] = matrix["count"].fillna(0).astype(int)
            matrix["真实类别中文"] = matrix["true_label"].map(aliases).fillna("未配置")
            matrix["预测类别中文"] = matrix["predicted_label"].map(aliases).fillna("未配置")
            base = alt.Chart(matrix).encode(
                x=alt.X(
                    "predicted_label:N",
                    title="预测类别",
                    sort=class_labels,
                    axis=alt.Axis(labelAngle=0, labelLimit=90),
                ),
                y=alt.Y("true_label:N", title="真实类别", sort=class_labels),
            )
            heatmap = base.mark_rect(cornerRadius=2).encode(
                color=alt.Color("count:Q", title="病例数", scale=alt.Scale(scheme="teals")),
                tooltip=[
                    alt.Tooltip("true_label:N", title="真实类别"),
                    alt.Tooltip("真实类别中文:N"),
                    alt.Tooltip("predicted_label:N", title="预测类别"),
                    alt.Tooltip("预测类别中文:N"),
                    alt.Tooltip("count:Q", title="病例数"),
                ],
            )
            labels = base.mark_text(fontSize=12).encode(
                text=alt.Text("count:Q"),
                color=alt.condition(
                    alt.datum.count > matrix["count"].max() * 0.55,
                    alt.value("white"),
                    alt.value("#17324d"),
                ),
            )
            st.altair_chart(
                (heatmap + labels).properties(height=min(500, max(320, 46 * len(class_labels)))),
                width="stretch",
            )
        elif not pairs.empty:
            st.markdown("##### 主要混淆方向")
            all_errors = pairs.loc[pairs["true_label"] != pairs["predicted_label"]].copy()
            focus = st.selectbox(
                "混淆视图",
                ["Top-20 主要混淆", *class_labels],
                key="generic_audit_confusion_focus",
                help="类别较多时先看 Top-20；也可以聚焦一个类别的流入与流出错误。",
            )
            if focus == "Top-20 主要混淆":
                errors = all_errors.nlargest(20, "count").copy()
                st.caption("默认展示错误病例数最多的20个真实类别 → 预测类别方向。")
            else:
                errors = all_errors.loc[
                    all_errors["true_label"].astype(str).eq(focus)
                    | all_errors["predicted_label"].astype(str).eq(focus)
                ].nlargest(20, "count").copy()
                st.caption(f"当前聚焦类别：{focus}，同时展示流出和流入错误。")
            if not errors.empty:
                errors["混淆方向"] = (
                    errors["true_label"].astype(str) + " → " + errors["predicted_label"].astype(str)
                )
                chart = alt.Chart(errors).mark_bar(color="#0f766e").encode(
                    x=alt.X("count:Q", title="错误病例数"),
                    y=alt.Y("混淆方向:N", title=None, sort="-x"),
                    tooltip=["混淆方向:N", alt.Tooltip("count:Q", title="病例数")],
                ).properties(height=min(520, max(280, 25 * len(errors))))
                st.altair_chart(chart, width="stretch")
            else:
                st.info("当前聚焦类别没有可显示的预测错误。")
    st.markdown("#### 分类别表现")
    class_metrics = audit.class_metrics.rename(
        columns={
            "class_label": "类别",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "support": "样本数",
        }
    )
    if aliases:
        class_metrics.insert(
            1,
            "中文说明",
            class_metrics["类别"].astype(str).map(aliases).fillna("未配置"),
        )
    filter_col, sort_col = st.columns([1.3, 0.7], gap="small")
    with filter_col:
        class_query = st.text_input(
            "类别搜索",
            placeholder="输入类别名称或编号",
            key="generic_audit_class_search",
        ).strip()
    with sort_col:
        sort_by = st.selectbox(
            "排序方式",
            ["样本数（高到低）", "F1（低到高）", "Recall（低到高）", "类别名称"],
            key="generic_audit_class_sort",
        )
    if class_query:
        class_metrics = class_metrics.loc[
            class_metrics["类别"].astype(str).str.contains(class_query, case=False, regex=False)
        ]
    if sort_by == "F1（低到高）":
        class_metrics = class_metrics.sort_values(["F1", "样本数"], ascending=[True, False])
    elif sort_by == "Recall（低到高）":
        class_metrics = class_metrics.sort_values(["Recall", "样本数"], ascending=[True, False])
    elif sort_by == "类别名称":
        class_metrics = class_metrics.sort_values("类别", key=lambda values: values.astype(str))
    else:
        class_metrics = class_metrics.sort_values("样本数", ascending=False)
    st.dataframe(
        class_metrics,
        hide_index=True,
        width="stretch",
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _render_risk(audit) -> None:
    cases = audit.case_risk_scores
    if not bool(audit.summary.iloc[0]["has_probabilities"]):
        st.info("当前结果表没有完整类别概率，仅提供分类错误、混淆矩阵和错误病例审计；置信度相关分析不可用。")
        return
    st.markdown("#### 模型输出错误风险")
    st.caption("这里分析误分类、不确定性与固定复核预算下的错误富集；不推断疾病或错误的临床后果。")
    signal_labels = {
        "confidence": "置信度",
        "entropy_normalized": "归一化熵",
        "top1_top2_margin": "Top1-Top2 margin",
    }
    selected_signal = st.selectbox(
        "输出信号",
        list(signal_labels),
        format_func=signal_labels.get,
        key="generic_audit_distribution_signal",
    )
    distribution = cases[[selected_signal, "is_error"]].rename(columns={selected_signal: "value"})
    distribution["预测结果"] = distribution["is_error"].map(
        {False: "预测正确", True: "预测错误"}
    )
    values = distribution["value"].dropna()
    if not values.empty:
        _render_kpi_cards(
            [
                ("最小值", f"{values.min():.3f}", ""),
                ("中位数", f"{values.median():.3f}", ""),
                ("P90", f"{values.quantile(0.90):.3f}", ""),
                ("最大值", f"{values.max():.3f}", ""),
            ]
        )
    chart = (
        alt.Chart(distribution)
        .mark_bar(opacity=0.82)
        .encode(
            x=alt.X(
                "value:Q",
                bin=alt.Bin(maxbins=24),
                title=signal_labels[selected_signal],
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("count():Q", title="病例数"),
            color=alt.Color(
                "预测结果:N",
                scale=alt.Scale(
                    domain=["预测正确", "预测错误"],
                    range=["#2f7895", "#b54747"],
                ),
                title=None,
            ),
            tooltip=["预测结果:N", alt.Tooltip("count():Q", title="病例数")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    if selected_signal == "confidence":
        st.caption(
            "置信度取每个病例所有类别概率中的最大值。横轴固定为0–1，"
            "因此可以直接看到当前结果的实际取值范围。"
        )
    st.markdown("#### 选择性复核模拟（离线研究）")
    st.info(
        "这里模拟按模型不确定性优先复核一部分病例，用于判断有限复核容量能捕获多少预测错误；"
        "不会调用专家模型，也不与在线路由联动。"
    )
    review = audit.review_budget_results.copy()
    if not review.empty:
        signal_options = review["signal"].dropna().astype(str).drop_duplicates().tolist()
        review_signal = st.selectbox(
            "复核排序信号",
            signal_options,
            format_func=lambda value: REVIEW_SIGNAL_LABELS.get(value, value),
            key="generic_audit_review_signal",
            help="在同一组人工复核比例下比较一种排序信号，避免多条曲线挤在一起。",
        )
        display = review.rename(
            columns={
                "signal": "排序信号",
                "review_budget": "人工复核比例",
                "reviewed_cases": "复核病例数",
                "captured_errors": "捕获错误数",
                "error_recall": "错误召回率",
                "review_error_rate": "复核区错误率",
                "remaining_errors": "剩余错误数",
                "enrichment_vs_random": "相对随机抽查倍数",
            }
        )
        display = display.loc[display["排序信号"].astype(str).eq(review_signal)]
        reference_index = (display["人工复核比例"] - 0.20).abs().idxmin()
        reference = display.loc[reference_index]
        st.markdown(
            '<div class="hub-band"><strong>如何解读：</strong>'
            f'按“{html.escape(REVIEW_SIGNAL_LABELS.get(review_signal, review_signal))}”优先复核 '
            f'{float(reference["人工复核比例"]):.0%} 的病例，可捕获 '
            f'{float(reference["错误召回率"]):.1%} 的预测错误；'
            f'仍有 {int(reference["剩余错误数"]):,} 个错误留在未复核病例中。</div>',
            unsafe_allow_html=True,
        )
        table_display = display.drop(columns=["排序信号"]).copy()
        for percentage_column in ["人工复核比例", "错误召回率", "复核区错误率"]:
            table_display[percentage_column] = table_display[percentage_column] * 100
        st.dataframe(
            table_display,
            hide_index=True,
            width="stretch",
            column_config={
                "人工复核比例": st.column_config.NumberColumn(format="%.0f%%"),
                "错误召回率": st.column_config.NumberColumn(format="%.1f%%"),
                "复核区错误率": st.column_config.NumberColumn(format="%.1f%%"),
                "相对随机抽查倍数": st.column_config.NumberColumn(format="%.2fx"),
            },
        )
        selected_curve = audit.risk_coverage.loc[
            audit.risk_coverage["signal"].astype(str).eq(review_signal)
        ]
        curve = (
            alt.Chart(selected_curve)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "coverage:Q",
                    title="未进入复核的病例比例",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "selective_error_rate:Q",
                    title="未复核病例错误率",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.value("#0f766e"),
                tooltip=[
                    alt.Tooltip("coverage:Q", title="未复核病例比例", format=".0%"),
                    alt.Tooltip("selective_error_rate:Q", title="未复核病例错误率", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(curve, width="stretch")
        st.caption(
            "横轴越小表示送去复核的病例越多；纵轴表示剩余未复核病例中的预测错误比例。"
            "这里只评估错误筛选效率，不代表临床处置收益。"
        )
        with st.expander("查看复核模拟的计算口径"):
            st.markdown(
                "- 复核病例数：总病例数 × 人工复核比例，向上取整。\n"
                "- 错误召回率：复核病例中捕获的错误数 ÷ 全部预测错误数。\n"
                "- 复核区错误率：捕获错误数 ÷ 复核病例数。\n"
                "- 相对随机抽查倍数：复核区错误率 ÷ 全体病例错误率。\n"
                "- 未复核病例错误率：剩余错误数 ÷ 未进入复核的病例数。"
            )


def _render_cases(audit) -> None:
    cases = audit.case_risk_scores.copy()
    filter_col, label_col, sort_col = st.columns([0.8, 1.25, 0.95], gap="small")
    with filter_col:
        error_filter = st.selectbox(
            "病例范围",
            ["全部病例", "仅预测错误", "仅高置信错误"],
            key="audit_case_filter",
        )
    risk_options = sorted(
        {
            item
            for value in cases.get("risk_labels", pd.Series(dtype=str)).dropna().astype(str)
            for item in value.split("；")
            if item and item != "常规观察"
        }
    )
    with label_col:
        selected_risks = st.multiselect(
            "输出风险标签",
            risk_options,
            key="generic_audit_case_risks",
            placeholder="全部标签",
        )
    with sort_col:
        case_sort = st.selectbox(
            "排序方式",
            ["优先复核候选", "置信度（低到高）", "Margin（低到高）", "Entropy（高到低）"],
            key="generic_audit_case_sort",
        )
    if error_filter == "仅预测错误":
        cases = cases.loc[cases["is_error"]]
    elif error_filter == "仅高置信错误":
        cases = cases.loc[cases["high_confidence_error"]]
    if selected_risks:
        cases = cases.loc[
            cases["risk_labels"].fillna("").astype(str).map(
                lambda value: any(label in value.split("；") for label in selected_risks)
            )
        ]
    if case_sort == "置信度（低到高）":
        cases = cases.sort_values("confidence", ascending=True, na_position="last")
    elif case_sort == "Margin（低到高）":
        cases = cases.sort_values("top1_top2_margin", ascending=True, na_position="last")
    elif case_sort == "Entropy（高到低）":
        cases = cases.sort_values("entropy_normalized", ascending=False, na_position="last")
    else:
        cases = cases.sort_values(
            ["priority_review_candidate", "high_confidence_error", "is_error", "confidence"],
            ascending=[False, False, False, True],
            na_position="last",
        )

    probability_columns = [
        column for column in cases.columns if str(column).startswith("prob::")
    ]
    if probability_columns:
        def second_choice(row: pd.Series) -> str:
            ranked = sorted(
                (
                    (str(column).removeprefix("prob::"), float(row[column]))
                    for column in probability_columns
                    if pd.notna(row[column])
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            return ranked[1][0] if len(ranked) > 1 else "—"

        cases["top2_label"] = cases.apply(second_choice, axis=1)
    display_columns = [
        "display_case",
        "true_label",
        "y_pred",
        "top2_label",
        "confidence",
        "top1_top2_margin",
        "entropy_normalized",
        "risk_labels",
    ]
    display = cases[[column for column in display_columns if column in cases]].rename(
        columns={
            "display_case": "病例",
            "true_label": "真实类别",
            "y_pred": "Top-1 预测",
            "top2_label": "Top-2 预测",
            "confidence": "置信度",
            "entropy_normalized": "归一化熵",
            "top1_top2_margin": "Top1-Top2 margin",
            "risk_labels": "输出风险标签",
        }
    )
    st.caption(
        "默认使用会话内序号，不展示原始患者标识、文件路径或服务器绝对路径；"
        "完整类别概率仅保留在下载产物中。"
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "置信度": st.column_config.NumberColumn(format="%.3f"),
            "归一化熵": st.column_config.NumberColumn(format="%.3f"),
            "Top1-Top2 margin": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _render_stability(audit) -> None:
    st.markdown("#### 预测版本稳定性")
    st.caption("比较预先指定的主版本与其他版本；变化方向按病例配对统计，不默认假设 TTA 或 ensemble 更优。")
    st.warning("切换或比较主预测版本若基于当前评测结果，属于探索性比较，不代表该版本已被独立验证。")
    display = audit.variant_stability.rename(
        columns={
            "primary_variant": "主版本",
            "comparison_variant": "比较版本",
            "aligned_cases": "对齐病例",
            "prediction_changed": "预测变化",
            "error_to_correct": "错误变正确",
            "correct_to_error": "正确变错误",
            "always_correct": "始终正确",
            "always_error": "始终错误",
            "mean_probability_l1": "平均概率 L1",
            "mean_js_divergence": "平均 JS divergence",
            "primary_accuracy": "主版本 Accuracy",
            "comparison_accuracy": "比较版本 Accuracy",
            "primary_macro_f1": "主版本 Macro-F1",
            "comparison_macro_f1": "比较版本 Macro-F1",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _render_leakage(audit, validation) -> None:
    st.markdown("#### 数据完整性与标签捷径检查")
    st.caption("这里仅检查当前结果表中能够观察到的信号，不会给出“已确认不存在数据泄漏”的结论。")
    status_counts = audit.leakage_checks.get("status", pd.Series(dtype=str)).astype(str).value_counts()
    check_cards = [
        ("未发现明显问题", int(status_counts.get("未发现明显问题", 0)), "结果表可见范围", "success"),
        ("发现可疑风险", int(status_counts.get("发现可疑风险", 0)), "需要人工核对", "error"),
        ("当前无法评估", int(status_counts.get("当前无法评估", 0)), "缺少必要上下文", "unknown"),
    ]
    content = "".join(
        f'<div class="hub-audit-kpi {tone}">'
        f"<span>{label}</span><b>{value}</b><small>{note}</small></div>"
        for label, value, note, tone in check_cards
    )
    st.markdown(
        f'<div class="hub-audit-kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">{content}</div>',
        unsafe_allow_html=True,
    )
    display = audit.leakage_checks.rename(
        columns={
            "check_id": "检查项",
            "status": "状态",
            "evidence_count": "证据数量",
            "explanation": "说明",
        }
    )
    display["检查项"] = display["检查项"].map(LEAKAGE_CHECK_LABELS).fillna(display["检查项"])
    st.dataframe(display, hide_index=True, width="stretch")
    with st.expander("查看数据校验明细"):
        st.json(validation.to_dict(), expanded=False)
    st.info("临床后果风险：尚未评估。当前没有医生确认的严重度规则，本模块不会自行推断疾病或错误的临床后果。")


def _render_results(state: dict) -> None:
    validation = state["validation"]
    audit = state["audit"]
    _render_validation(validation, compact=True)
    summary = audit.summary.iloc[0]
    high_confidence_value = (
        f"{int(summary['high_confidence_error_count']):,}"
        if bool(summary["has_probabilities"])
        else "不可用"
    )
    _render_kpi_cards(
        [
            ("样本数", f"{int(summary['sample_count']):,}", ""),
            ("类别数", f"{int(summary['class_count']):,}", ""),
            ("Accuracy", f"{float(summary['accuracy']):.1%}", ""),
            ("Macro-F1", f"{float(summary['macro_f1']):.1%}", ""),
            ("预测错误", f"{int(summary['error_count']):,}", "error"),
            ("高置信错误", high_confidence_value, "severe"),
        ]
    )
    st.markdown(
        '<div class="hub-band"><strong>产物资格：</strong>可用于离线模型输出错误风险审计。'
        '这不代表 Adapter 已实现、模型可在线推理或可进入路由池。</div>',
        unsafe_allow_html=True,
    )
    tab_names = ["总体表现", "错误风险", "病例清单"]
    if not audit.variant_stability.empty:
        tab_names.append("版本稳定性")
    tab_names.append("数据与泄漏检查")
    tabs = st.tabs(tab_names)
    tab_index = 0
    with tabs[tab_index]:
        _render_overall(audit)
    tab_index += 1
    with tabs[tab_index]:
        _render_risk(audit)
    tab_index += 1
    with tabs[tab_index]:
        _render_cases(audit)
    tab_index += 1
    if not audit.variant_stability.empty:
        with tabs[tab_index]:
            _render_stability(audit)
        tab_index += 1
    with tabs[tab_index]:
        _render_leakage(audit, validation)
    download_left, download_right = st.columns(2, gap="medium")
    download_left.download_button(
        "下载完整审计结果",
        data=state["archive"],
        file_name="ophagent_result_risk_audit.zip",
        mime="application/zip",
        icon=":material/download:",
        width="stretch",
    )
    download_right.download_button(
        "下载校验摘要 JSON",
        data=json.dumps(validation.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="validation.json",
        mime="application/json",
        icon=":material/data_object:",
        width="stretch",
    )


def render_result_table_risk_audit() -> None:
    st.markdown("#### 结果表风险审计")
    st.caption("上传已有预测结果，完成通用的模型输出错误风险审计；不训练模型、不运行 GPU，也不推断临床后果。")
    stored_state = st.session_state.get("generic_result_audit_state")
    uploaded_snapshot = st.session_state.get("generic_result_audit_upload")
    if isinstance(stored_state, dict) and "audit" in stored_state:
        active_step = 4
    elif isinstance(stored_state, dict) and "validation" in stored_state:
        active_step = 3
    elif uploaded_snapshot is not None:
        active_step = 2
    else:
        active_step = 1
    _render_audit_process(active_step)
    has_stored_results = isinstance(stored_state, dict) and "audit" in stored_state
    source_controls = (
        st.expander("更换结果表", expanded=False)
        if has_stored_results
        else st.container()
    )
    with source_controls:
        st.markdown("##### 1. 上传结果表")
        uploaded = st.file_uploader(
            "上传 CSV 或 Excel 结果表",
            type=["csv", "xlsx", "xls"],
            key="generic_result_audit_upload",
            help="文件仅用于当前会话处理；不会写入 Git 仓库。",
        )
        if uploaded is None:
            st.info(
                "支持 CSV、XLSX 与 XLS。上传后系统会自动识别病例 ID、真实标签、"
                "预测标签和概率列，再由你确认映射。"
            )
            return
        payload = uploaded.getvalue()
        file_key = f"{uploaded.name}:{len(payload)}"
        if isinstance(stored_state, dict) and stored_state.get("file_key") != file_key:
            st.session_state.pop("generic_result_audit_state", None)
            stored_state = None
        try:
            sheets = list_excel_sheets(payload, filename=uploaded.name)
            sheet = (
                st.selectbox("Excel Sheet", sheets, key=f"audit_sheet::{file_key}")
                if sheets
                else None
            )
            frame = read_result_table(payload, filename=uploaded.name, sheet_name=sheet)
        except (ValueError, OSError) as exc:
            st.error(f"结果表读取失败：{exc}")
            return
        st.caption(
            f"已读取 {len(frame):,} 行、{len(frame.columns):,} 列。"
            "仅显示少量预览，原始标识不会进入结果卡片。"
        )
        if has_stored_results:
            st.dataframe(frame.head(5), hide_index=True, width="stretch")
        else:
            with st.expander("查看原始表结构与前 5 行"):
                st.dataframe(frame.head(5), hide_index=True, width="stretch")
    has_current_results = (
        isinstance(stored_state, dict)
        and stored_state.get("file_key") == file_key
        and "audit" in stored_state
    )
    controls = (
        st.expander("调整字段映射与分析参数", expanded=False)
        if has_current_results
        else st.container()
    )
    with controls:
        st.markdown("##### 2. 字段识别")
        suggested_mapping = suggest_mapping(frame)
        if _mapping_is_complete(suggested_mapping):
            mapping = suggested_mapping
            primary_variant = _default_primary_variant(mapping)
            _render_auto_mapping_summary(mapping)
            manual_override = st.checkbox(
                "自动识别不正确时，启用手动调整",
                value=False,
                key=f"audit_manual_mapping::{file_key}",
                help="仅在字段识别与实际表结构不一致时使用。",
            )
            if manual_override:
                mapping, primary_variant = _mapping_controls(frame, file_key=file_key)
        else:
            st.warning("自动识别不完整，请补充病例标识、真实标签和预测版本映射。")
            mapping, primary_variant = _mapping_controls(frame, file_key=file_key)
        with st.expander("分析参数", expanded=not has_current_results):
            high_confidence_threshold = st.slider(
                "高置信错误阈值",
                min_value=0.00,
                max_value=0.99,
                value=0.80,
                step=0.01,
                key=f"audit_high_confidence::{file_key}",
                help="仅在存在完整概率时使用；默认 0.80。",
            )
            st.caption("该阈值只定义高置信错误标签，不改变模型预测或总体分类指标。")
        run_requested = st.button(
            "校验并运行风险审计",
            type="primary",
            icon=":material/fact_check:",
            width="stretch",
            key=f"run_generic_result_audit::{file_key}",
        )
    if run_requested:
        with st.spinner("正在校验字段、概率与预测版本，并生成审计结果……"):
            try:
                normalized = normalize_result_table(frame, mapping)
                validation = validate_normalized_predictions(normalized)
                state = {"validation": validation, "file_key": file_key}
                if validation.passed:
                    audit = run_generic_risk_audit(
                        normalized,
                        source=frame,
                        mapping=mapping,
                        primary_variant=primary_variant,
                        high_confidence_threshold=high_confidence_threshold,
                    )
                    output_dir = (
                        Path(tempfile.mkdtemp(prefix="ophagent-risk-audit-")) / "risk_audit"
                    )
                    export_audit_result(
                        output_dir,
                        normalized=normalized,
                        validation=validation,
                        audit=audit,
                    )
                    state.update({"audit": audit, "archive": _zip_directory(output_dir)})
                st.session_state["generic_result_audit_state"] = state
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"审计无法运行：{exc}")
                return
        st.rerun()
    state = st.session_state.get("generic_result_audit_state")
    if isinstance(state, dict) and state.get("file_key") != file_key:
        state = None
    if not state:
        return
    if "audit" not in state:
        _render_validation(state["validation"])
        st.error("存在阻止审计的数据问题，请调整映射或修正结果表后重试。")
        return
    _render_results(state)
