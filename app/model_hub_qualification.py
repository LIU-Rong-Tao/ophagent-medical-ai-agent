"""Model Hub 中的 Route Qualification Benchmark v1.1 只读视图。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.model_hub_ui import TASK_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = (
    PROJECT_ROOT
    / "experiments/opening_risk_routing_closure/outputs"
    / "route_qualification_benchmark_v1_1"
)
CONTROLLER_ROOT = (
    PROJECT_ROOT
    / "experiments/opening_risk_routing_closure/outputs"
    / "controlled_agent_v2_benchmark"
)
LOCAL_CONTROLLER_ROOT = (
    PROJECT_ROOT
    / "experiments/opening_risk_routing_closure/outputs"
    / "local_controller_benchmark_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _percent(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _metric_cards(values: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        '<div class="hub-mini-stat">'
        f"<span>{html.escape(label)}</span>"
        f"<b>{html.escape(value)}</b>"
        f"<small>{html.escape(note)}</small></div>"
        for label, value, note in values
    )
    st.markdown(
        f'<div class="hub-mini-strip">{cards}</div>',
        unsafe_allow_html=True,
    )


def _render_summary() -> None:
    summary = _read_json(BENCHMARK_ROOT / "benchmark_summary.json")
    manifest = _read_json(BENCHMARK_ROOT / "artifact_manifest.json")
    if not summary:
        st.error("v1.1 Benchmark 产物尚未生成。")
        return
    v1 = dict(summary.get("v1", {}))
    v11 = dict(summary.get("v1_1_leave_one_task_out", {}))
    _metric_cards(
        [
            (
                "正式冻结路由",
                str(summary.get("formal_route_count", "—")),
                f"{summary.get('task_count', '—')} 个任务",
            ),
            (
                "v1.1 错误授予率",
                _percent(v11.get("false_grant_rate")),
                f"v1 为 {_percent(v1.get('false_grant_rate'))}",
            ),
            (
                "v1.1 有益保留率",
                _percent(v11.get("beneficial_route_retention_rate")),
                f"v1 为 {_percent(v1.get('beneficial_route_retention_rate'))}",
            ),
            (
                "Test 反转拦截",
                _percent(v11.get("test_reversal_interception_rate")),
                "仅回顾性研究证据",
            ),
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "版本": "v1",
                "有益路由保留率": v1.get(
                    "beneficial_route_retention_rate"
                ),
                "失效路由拦截率": v1.get(
                    "ineffective_route_interception_rate"
                ),
                "错误授予率": v1.get("false_grant_rate"),
                "错误拒绝率": v1.get("false_rejection_rate"),
                "可执行覆盖率": v1.get("executable_coverage_rate"),
            },
            {
                "版本": "v1.1 · LOTO",
                "有益路由保留率": v11.get(
                    "beneficial_route_retention_rate"
                ),
                "失效路由拦截率": v11.get(
                    "ineffective_route_interception_rate"
                ),
                "错误授予率": v11.get("false_grant_rate"),
                "错误拒绝率": v11.get("false_rejection_rate"),
                "可执行覆盖率": v11.get("executable_coverage_rate"),
            },
        ]
    )
    st.dataframe(
        comparison.style.format(
            {
                "有益路由保留率": "{:.1%}",
                "失效路由拦截率": "{:.1%}",
                "错误授予率": "{:.1%}",
                "错误拒绝率": "{:.1%}",
                "可执行覆盖率": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "v1.1 消除了本批回顾性错误授予，但严格稳定性与域偏移约束使"
        "有益路由保留率降至 20%、可执行覆盖率降至 12.5%。"
        "因此它是安全性优先的改进，不是总体效用提升。"
    )
    explanation = dict(summary.get("v1_failure_explanation", {}))
    with st.expander("为什么 v1 出现 50% 有益保留率和 28.6% 错误授予率"):
        st.write(str(explanation.get("cause", "未记录")))
        st.caption(
            "错误授予："
            + "、".join(explanation.get("false_grants", []))
            + "；错误拒绝："
            + "、".join(explanation.get("false_rejections", []))
        )
    st.caption(
        "Qualification Contract SHA256："
        f"{manifest.get('contract_sha256', '未记录')} · "
        f"构建 commit：{manifest.get('source_commit_sha', '未记录')} · "
        "clinical_route_eligible=false"
    )


def _render_ablation() -> None:
    frame = _read_csv(BENCHMARK_ROOT / "ablation_results.csv")
    if frame.empty:
        st.error("门控规则消融产物缺失。")
        return
    labels = {
        "relative_scout_only": "1 · 仅相对 Scout",
        "plus_best_single": "2 · + 最佳单模型",
        "plus_proxy_net": "3 · + introduced / net",
        "plus_cost_budget": "4 · + 成本 / 预算",
        "plus_stability": "5 · + 稳定性",
        "plus_domain_adaptation": "6 · + 域偏移 / 适配",
        "complete_layered_gate": "7 · 完整分层门控",
    }
    display = frame.copy()
    display["规则族"] = display["rule_family"].map(labels)
    columns = [
        "规则族",
        "granted_routes",
        "beneficial_route_retention_rate",
        "ineffective_route_interception_rate",
        "false_grant_rate",
        "false_rejection_rate",
        "executable_coverage_rate",
    ]
    display = display[columns].rename(
        columns={
            "granted_routes": "授予路由",
            "beneficial_route_retention_rate": "有益保留率",
            "ineffective_route_interception_rate": "失效拦截率",
            "false_grant_rate": "错误授予率",
            "false_rejection_rate": "错误拒绝率",
            "executable_coverage_rate": "可执行覆盖率",
        }
    )
    st.markdown("#### 分层规则族消融")
    st.caption("规则按顺序叠加；不构造事后加权综合分数，也不训练分类器。")
    st.dataframe(
        display.style.format(
            {
                "有益保留率": "{:.1%}",
                "失效拦截率": "{:.1%}",
                "错误授予率": "{:.1%}",
                "错误拒绝率": "{:.1%}",
                "可执行覆盖率": "{:.1%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    chart = display.set_index("规则族")[
        ["有益保留率", "失效拦截率", "错误授予率"]
    ]
    st.line_chart(chart)
    sensitivity = _read_csv(BENCHMARK_ROOT / "sensitivity_results.csv")
    with st.expander("单参数敏感性分析", expanded=False):
        st.dataframe(sensitivity, hide_index=True, width="stretch")


def _render_loto() -> None:
    frame = _read_csv(BENCHMARK_ROOT / "leave_one_task_out_results.csv")
    if frame.empty:
        st.error("留一任务结果产物缺失。")
        return
    folds = frame.loc[
        frame["record_type"].eq("held_out_task_validation_only")
    ].copy()
    folds["留出任务"] = folds["held_out_task_id"].map(
        lambda value: TASK_LABELS.get(str(value), str(value))
    )
    display = folds[
        [
            "留出任务",
            "held_out_not_used_for_thresholds",
            "beneficial_route_retention_rate",
            "ineffective_route_interception_rate",
            "false_grant_rate",
            "false_rejection_rate",
            "test_reversal_interception_rate",
            "risk_tradeoff_limitation_rate",
            "executable_coverage_rate",
            "failure_cases",
        ]
    ].rename(
        columns={
            "held_out_not_used_for_thresholds": "留出任务未参与阈值",
            "beneficial_route_retention_rate": "有益保留率",
            "ineffective_route_interception_rate": "失效拦截率",
            "false_grant_rate": "错误授予率",
            "false_rejection_rate": "错误拒绝率",
            "test_reversal_interception_rate": "Test反转拦截率",
            "risk_tradeoff_limitation_rate": "risk_tradeoff限制率",
            "executable_coverage_rate": "可执行覆盖率",
            "failure_cases": "失败案例",
        }
    )
    st.markdown("#### Leave-one-task-out 回顾性验证")
    st.info(
        "每轮阈值只由其他任务且不依赖留出任务选择源的记录确定；"
        "主结果清空留出任务冻结结果字段，留出任务只评价、不参与规则回写。"
    )
    st.dataframe(
        display.style.format(
            {
                "有益保留率": "{:.1%}",
                "失效拦截率": "{:.1%}",
                "错误授予率": "{:.1%}",
                "错误拒绝率": "{:.1%}",
                "Test反转拦截率": "{:.1%}",
                "risk_tradeoff限制率": "{:.1%}",
                "可执行覆盖率": "{:.1%}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )
    overlay = frame.loc[
        frame["record_type"].eq("post_freeze_safety_overlay")
    ]
    if not overlay.empty:
        item = overlay.iloc[0]
        st.caption(
            "冻结结果仅作为单独的 post-freeze safety overlay："
            f"Test 反转拦截率 "
            f"{float(item['test_reversal_interception_rate']):.1%}；"
            "不计入 Validation-only LOTO 预测。"
        )
    failures = _read_csv(BENCHMARK_ROOT / "failure_case_audit.csv")
    with st.expander("逐路由失败案例审计", expanded=False):
        st.dataframe(failures, hide_index=True, width="stretch")


def _render_evidence_matrix() -> None:
    matrix = _read_csv(
        BENCHMARK_ROOT / "route_qualification_evidence_matrix.csv"
    )
    if matrix.empty:
        st.error("统一证据矩阵缺失。")
        return
    display = matrix[
        [
            "task_id",
            "dataset_id",
            "pairing_id",
            "primary_metric",
            "validation_delta_vs_scout",
            "validation_delta_vs_best_single",
            "validation_corrected",
            "validation_introduced",
            "validation_net",
            "cost_protocol_id",
            "cost_protocol_comparable",
            "stability_source",
            "domain_shift_status",
            "prediction_asset_complete",
            "execution_level",
            "error_codes",
        ]
    ].copy()
    display["task_id"] = display["task_id"].map(
        lambda value: TASK_LABELS.get(str(value), str(value))
    )
    display = display.rename(
        columns={
            "task_id": "任务",
            "dataset_id": "数据集",
            "pairing_id": "冻结路由",
            "primary_metric": "主指标",
            "validation_delta_vs_scout": "相对Scout",
            "validation_delta_vs_best_single": "相对最佳单模型",
            "validation_corrected": "corrected",
            "validation_introduced": "introduced",
            "validation_net": "net",
            "cost_protocol_id": "成本协议",
            "cost_protocol_comparable": "成本可比",
            "stability_source": "稳定性证据",
            "domain_shift_status": "域偏移",
            "prediction_asset_complete": "prediction完整",
            "execution_level": "v1.1资格",
            "error_codes": "限制原因",
        }
    )
    st.markdown("#### 统一研究证据矩阵")
    st.caption(
        "不同成本协议不混合排名；Validation、冻结结果、协议和资产均以 SHA256 绑定。"
    )
    st.dataframe(display, hide_index=True, width="stretch")


def _render_controller_results() -> None:
    st.markdown("#### V1 / V2 与本地控制模型比较")
    rendered = False
    v2_summary = _read_csv(
        CONTROLLER_ROOT / "controller_v1_v2_summary.csv"
    )
    v2_rows = _read_csv(
        CONTROLLER_ROOT / "controller_v1_v2_scenario_results.csv"
    )
    if not v2_summary.empty:
        rendered = True
        st.markdown("##### 规则状态机 V1 / V2")
        st.dataframe(v2_summary, hide_index=True, width="stretch")
        with st.expander("12 个固定脱敏场景", expanded=False):
            st.dataframe(v2_rows, hide_index=True, width="stretch")

    local_rows: list[dict[str, Any]] = []
    for path in sorted(LOCAL_CONTROLLER_ROOT.glob("*_results.json")):
        result = _read_json(path)
        metrics = dict(result.get("metrics", {}))
        local_rows.append(
            {
                "控制器": result.get("controller_label", path.stem),
                "状态": result.get("status", "unknown"),
                "下一动作准确率": metrics.get("next_action_accuracy"),
                "合法动作率": metrics.get("legal_action_rate"),
                "Schema合法率": metrics.get("schema_valid_rate"),
                "任务完成率": metrics.get("task_completion_rate"),
                "不必要Expert提议率": metrics.get(
                    "unnecessary_expert_proposal_rate"
                ),
                "转人工率": metrics.get("human_referral_rate"),
                "平均延迟ms": metrics.get("latency_ms_mean"),
                "峰值显存MiB": metrics.get("peak_vram_allocated_mib"),
                "本地token": metrics.get("local_token_cost"),
                "报告忠实度": metrics.get("report_fidelity_rate"),
                "门控拦截次数": metrics.get("gate_intercept_count"),
            }
        )
    if local_rows:
        rendered = True
        st.markdown("##### 本地 4B / 27B · 零样本与少样本")
        local_frame = pd.DataFrame(local_rows)
        st.dataframe(
            local_frame.style.format(
                {
                    "下一动作准确率": "{:.1%}",
                    "合法动作率": "{:.1%}",
                    "Schema合法率": "{:.1%}",
                    "任务完成率": "{:.1%}",
                    "不必要Expert提议率": "{:.1%}",
                    "转人工率": "{:.1%}",
                    "平均延迟ms": "{:.1f}",
                    "峰值显存MiB": "{:.0f}",
                    "报告忠实度": "{:.1%}",
                },
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )
        st.info(
            "规则控制器仍是正式基线：最终任务完成率 100%，无模型显存与"
            "推理延迟。27B 少样本的提议准确率为 100%，但没有带来最终"
            "任务完成率增益；4B 少样本仍有复杂分歧状态错误。"
        )
    if not rendered:
        st.info("控制器 Benchmark 正在完成内部验收，结果产物生成后在此只读展示。")


def render_qualification_workspace() -> None:
    """Render committed benchmark artifacts without recalculating the gate."""

    st.markdown(
        '<div class="hub-band"><b>Route Qualification Benchmark v1.1</b><br>'
        "同一资格函数服务于证据矩阵、状态机和控制器最终裁决；"
        "本页只读取冻结 Benchmark 产物，不在 UI 重新计算资格。</div>",
        unsafe_allow_html=True,
    )
    summary_tab, ablation_tab, loto_tab, matrix_tab, controller_tab = st.tabs(
        ["版本结论", "规则消融", "留一任务", "证据矩阵", "控制器比较"]
    )
    with summary_tab:
        _render_summary()
    with ablation_tab:
        _render_ablation()
    with loto_tab:
        _render_loto()
    with matrix_tab:
        _render_evidence_matrix()
    with controller_tab:
        _render_controller_results()
    st.caption(
        "所有公开标签均为研究审计或模型输出错误风险代理，不是临床金标准。"
    )
