"""Streamlit workspace for generic prediction-table model-error risk audits."""

from __future__ import annotations

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


def _mapping_controls(frame: pd.DataFrame, *, file_key: str) -> tuple[ResultTableMapping, str]:
    columns = [str(column) for column in frame.columns]
    suggested = suggest_mapping(frame)
    st.markdown("#### 字段映射确认")
    st.caption("系统只给出候选映射；运行前请确认病例标识、真实标签和每个预测版本。")
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
        with st.expander(f"预测版本 {index + 1} · {default.name}", expanded=index == 0):
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
        columns = st.columns(4, gap="small")
        columns[0].metric("导入记录", f"{summary.get('total_rows', 0):,}")
        columns[1].metric("唯一病例", f"{summary.get('unique_cases', 0):,}")
        columns[2].metric("类别数", f"{summary.get('class_count', 0):,}")
        columns[3].metric("预测版本", f"{summary.get('variant_count', 0):,}")
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
    st.markdown("#### 总体与分类别表现")
    metrics = pd.DataFrame(
        [
            ("Accuracy", summary["accuracy"]),
            ("Macro Precision", summary["macro_precision"]),
            ("Macro Recall", summary["macro_recall"]),
            ("Macro-F1", summary["macro_f1"]),
            ("Weighted F1", summary["weighted_f1"]),
            ("Cohen Kappa", summary["cohen_kappa"]),
        ],
        columns=["指标", "值"],
    )
    st.dataframe(metrics, hide_index=True, width="stretch", column_config={"值": st.column_config.NumberColumn(format="%.4f")})
    pairs = audit.confusion_pairs.copy()
    if not pairs.empty and int(summary["class_count"]) <= 20:
        chart = alt.Chart(pairs).mark_rect(cornerRadius=2).encode(
            x=alt.X("predicted_label:N", title="预测类别", sort=None),
            y=alt.Y("true_label:N", title="真实类别", sort=None),
            color=alt.Color("count:Q", title="病例数", scale=alt.Scale(scheme="teals")),
            tooltip=["true_label:N", "predicted_label:N", "count:Q"],
        ).properties(height=min(520, max(280, 24 * pairs["true_label"].nunique())))
        st.altair_chart(chart, width="stretch")
    elif not pairs.empty:
        errors = pairs.loc[pairs["true_label"] != pairs["predicted_label"]].nlargest(20, "count").copy()
        st.caption("类别较多，默认展示错误病例数最多的 20 个混淆方向；完整分类别指标见下表。")
        if not errors.empty:
            errors["混淆方向"] = errors["true_label"].astype(str) + " → " + errors["predicted_label"].astype(str)
            chart = alt.Chart(errors).mark_bar(color="#0f766e").encode(
                x=alt.X("count:Q", title="错误病例数"),
                y=alt.Y("混淆方向:N", title=None, sort="-x"),
                tooltip=["混淆方向:N", "count:Q"],
            ).properties(height=min(520, max(260, 24 * len(errors))))
            st.altair_chart(chart, width="stretch")
    class_metrics = audit.class_metrics.rename(
        columns={
            "class_label": "类别",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "support": "样本数",
        }
    )
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
    st.markdown("#### 输出不确定性与高置信错误")
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
    distribution = cases[[selected_signal]].rename(columns={selected_signal: "value"})
    chart = (
        alt.Chart(distribution)
        .mark_bar(opacity=0.86, color="#2383c4")
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=24), title=signal_labels[selected_signal]),
            y=alt.Y("count():Q", title="病例数"),
            tooltip=[alt.Tooltip("count():Q", title="病例数")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    st.markdown("#### 固定复核预算")
    review = audit.review_budget_results.copy()
    if not review.empty:
        display = review.rename(
            columns={
                "signal": "排序信号",
                "review_budget": "复核预算",
                "reviewed_cases": "复核病例数",
                "captured_errors": "捕获错误数",
                "error_recall": "错误召回率",
                "review_error_rate": "复核区错误率",
                "remaining_errors": "剩余错误数",
                "enrichment_vs_random": "相对随机富集倍数",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "复核预算": st.column_config.NumberColumn(format="%.0f%%"),
                "错误召回率": st.column_config.NumberColumn(format="%.1f%%"),
                "复核区错误率": st.column_config.NumberColumn(format="%.1f%%"),
                "相对随机富集倍数": st.column_config.NumberColumn(format="%.2fx"),
            },
        )
        curve = (
            alt.Chart(audit.risk_coverage)
            .mark_line(point=True)
            .encode(
                x=alt.X("coverage:Q", title="保留覆盖率", axis=alt.Axis(format="%")),
                y=alt.Y("selective_error_rate:Q", title="保留病例错误率", axis=alt.Axis(format="%")),
                color=alt.Color("signal:N", title="复核排序信号"),
                tooltip=["signal:N", alt.Tooltip("coverage:Q", format=".0%"), alt.Tooltip("selective_error_rate:Q", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(curve, width="stretch")


def _render_cases(audit) -> None:
    cases = audit.case_risk_scores.copy()
    error_filter = st.selectbox("病例范围", ["全部病例", "仅预测错误", "仅高置信错误"], key="audit_case_filter")
    if error_filter == "仅预测错误":
        cases = cases.loc[cases["is_error"]]
    elif error_filter == "仅高置信错误":
        cases = cases.loc[cases["high_confidence_error"]]
    display_columns = [
        "display_case",
        "true_label",
        "y_pred",
        "confidence",
        "entropy_normalized",
        "top1_top2_margin",
        "risk_labels",
    ]
    display = cases[[column for column in display_columns if column in cases]].rename(
        columns={
            "display_case": "病例",
            "true_label": "真实类别",
            "y_pred": "预测类别",
            "confidence": "置信度",
            "entropy_normalized": "归一化熵",
            "top1_top2_margin": "Top1-Top2 margin",
            "risk_labels": "输出风险标签",
        }
    )
    st.caption("默认使用会话内序号，不展示原始患者标识、文件路径或服务器绝对路径。")
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
    first_row = st.columns(3, gap="small")
    first_row[0].metric("样本数", f"{int(summary['sample_count']):,}")
    first_row[1].metric("类别数", f"{int(summary['class_count']):,}")
    first_row[2].metric("Accuracy", f"{float(summary['accuracy']):.1%}")
    second_row = st.columns(3, gap="small")
    second_row[0].metric("Macro-F1", f"{float(summary['macro_f1']):.1%}")
    second_row[1].metric("错误病例", f"{int(summary['error_count']):,}")
    second_row[2].metric("高置信错误", f"{int(summary['high_confidence_error_count']):,}")
    st.markdown(
        '<div class="hub-band"><strong>产物资格：</strong>可用于离线模型输出错误风险审计。'
        '这不代表 Adapter 已实现、模型可在线推理或可进入路由池。</div>',
        unsafe_allow_html=True,
    )
    tab_names = ["总体表现", "风险审计", "病例清单"]
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
    st.markdown("### 结果表风险审计")
    st.caption("上传已有预测结果，完成通用的模型输出错误风险审计；不训练模型、不运行 GPU，也不推断临床后果。")
    uploaded = st.file_uploader(
        "上传 CSV 或 Excel 结果表",
        type=["csv", "xlsx", "xls"],
        key="generic_result_audit_upload",
        help="文件仅用于当前会话处理；不会写入 Git 仓库。",
    )
    if uploaded is None:
        st.info("上传结果表后，系统会自动识别病例 ID、真实标签、预测标签和概率列，再由你确认映射。")
        return
    payload = uploaded.getvalue()
    file_key = f"{uploaded.name}:{len(payload)}"
    try:
        sheets = list_excel_sheets(payload, filename=uploaded.name)
        sheet = st.selectbox("Excel Sheet", sheets, key=f"audit_sheet::{file_key}") if sheets else None
        frame = read_result_table(payload, filename=uploaded.name, sheet_name=sheet)
    except (ValueError, OSError) as exc:
        st.error(f"结果表读取失败：{exc}")
        return
    st.caption(f"已读取 {len(frame):,} 行、{len(frame.columns):,} 列。仅显示少量预览，原始标识不会进入结果卡片。")
    with st.expander("查看原始表结构与前 5 行"):
        st.dataframe(frame.head(5), hide_index=True, width="stretch")
    mapping, primary_variant = _mapping_controls(frame, file_key=file_key)
    high_confidence_threshold = st.slider(
        "高置信错误阈值",
        min_value=0.00,
        max_value=0.99,
        value=0.80,
        step=0.01,
        key=f"audit_high_confidence::{file_key}",
        help="仅在存在完整概率时使用；默认 0.80。",
    )
    if st.button(
        "校验并运行风险审计",
        type="primary",
        icon=":material/fact_check:",
        width="stretch",
        key=f"run_generic_result_audit::{file_key}",
    ):
        try:
            normalized = normalize_result_table(frame, mapping)
            validation = validate_normalized_predictions(normalized)
            state = {"validation": validation}
            if validation.passed:
                audit = run_generic_risk_audit(
                    normalized,
                    source=frame,
                    mapping=mapping,
                    primary_variant=primary_variant,
                    high_confidence_threshold=high_confidence_threshold,
                )
                output_dir = Path(tempfile.mkdtemp(prefix="ophagent-risk-audit-")) / "risk_audit"
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
    state = st.session_state.get("generic_result_audit_state")
    if not state:
        return
    if "audit" not in state:
        _render_validation(state["validation"])
        st.error("存在阻止审计的数据问题，请调整映射或修正结果表后重试。")
        return
    _render_results(state)
