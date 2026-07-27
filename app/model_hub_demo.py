"""OphAgent 模型中转台入口。"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from app.model_hub_clinical import render_clinical_workspace  # noqa: E402
from app.model_hub_data import load_model_hub_outputs  # noqa: E402
from app.model_hub_engineering import render_engineering_workspace  # noqa: E402
from app.model_hub_index import build_model_hub_index  # noqa: E402
from app.model_hub_ui import (  # noqa: E402
    boundary_notice,
    inject_model_hub_css,
    page_header,
    sidebar_navigation,
)
from app.ui import inject_app_css  # noqa: E402


LEGACY_CONTROLLED_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/v0_8_6_interactive_model_hub/outputs"
)


def model_hub_output_dir() -> Path:
    configured = os.environ.get("OPHAGENT_MODEL_HUB_OUTPUT_DIR", "").strip()
    return Path(configured) if configured else LEGACY_CONTROLLED_OUTPUT_DIR


def _set_workspace(workspace: str) -> None:
    st.session_state["pending_hub_workspace"] = workspace
    st.rerun()


def _model_flag(models, column: str, *, default: bool = False):
    if column not in models.columns:
        return models.index.to_series().map(lambda _: default).astype(bool)
    return models[column].fillna(default).astype(bool)


def _render_overview(data: dict[str, object]) -> None:
    hub_index = data["hub_index"]
    checkpoints = hub_index["checkpoints"]
    task_assets = hub_index["task_assets"]
    route_runs = hub_index["route_runs"]
    protocols = hub_index["protocols"]
    datasets = hub_index["datasets"]
    online_endpoints = hub_index.get("online_endpoints")
    if online_endpoints is None:
        online_endpoints = checkpoints.iloc[0:0].copy()
    values = [
        (
            "基础 Checkpoint",
            len(checkpoints),
            f"{int(checkpoints['runtime_smoke_passed'].sum())} 个 Runtime Smoke 通过",
        ),
        (
            "任务预测资产",
            len(task_assets),
            f"覆盖 {task_assets['task_id'].nunique()} 个任务契约",
        ),
        (
            "数据集",
            datasets["dataset_id"].nunique(),
            f"{len(datasets)} 条实验路径 / 准入记录",
        ),
        (
            "路由结果包",
            len(route_runs),
            f"{len(protocols)} 份当前协议记录",
        ),
        (
            "单病例原图入口",
            len(online_endpoints),
            "与离线 prediction asset 分开统计",
        ),
        (
            "正式路由资格",
            int(task_assets["route_eligible"].sum()),
            "当前研究路由均不自动获得资格",
        ),
    ]
    cards = "".join(
        '<div class="hub-overview-kpi">'
        f"<span>{label}</span><b>{value:,}</b><small>{note}</small></div>"
        for label, value, note in values
    )
    st.markdown(f'<div class="hub-overview-grid">{cards}</div>', unsafe_allow_html=True)
    boundary_notice()
    st.info(
        "当前自动化层级：受控工具链与可追溯工作流，不是可自由规划的自主 Agent。"
        "敏感数据场景支持院内部署，不要求调用公网 API。"
    )
    st.markdown(
        '<div class="hub-section"><h3>中转台工作流</h3>'
        '<p>每一层使用独立证据；前一层通过不会自动推出后一层资格。</p></div>'
        '<div class="hub-flow">'
        '<div class="hub-flow-step"><i>1</i><b>模型资产</b><span>来源、Checkpoint、访问限制与本地完整性</span></div>'
        '<div class="hub-flow-step"><i>2</i><b>任务接入</b><span>Adapter、基础加载、任务适配与标准概率输出</span></div>'
        '<div class="hub-flow-step"><i>3</i><b>研究评测</b><span>统一任务指标、成本、错误风险与数据隔离</span></div>'
        '<div class="hub-flow-step"><i>4</i><b>病例回放与路由解释</b><span>模型调用轨迹、路由依据与研究审计标签隔离</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hub-section"><h3>进入工作区</h3><p>从当前任务继续，不需要在多层页面之间来回寻找入口。</p></div>', unsafe_allow_html=True)
    action_columns = st.columns(4, gap="small")
    actions = [
        ("查看模型资产", "模型资产", ":material/inventory_2:"),
        ("核对任务模型", "任务模型", ":material/model_training:"),
        ("开始研究评测", "研究评测", ":material/query_stats:"),
        ("打开病例回放", "病例回放", ":material/clinical_notes:"),
    ]
    for column, (label, workspace, icon) in zip(action_columns, actions, strict=True):
        with column:
            if st.button(label, icon=icon, width="stretch", key=f"overview::{workspace}"):
                _set_workspace(workspace)

    st.markdown(
        '<div class="hub-section"><h3>任务与数据集覆盖</h3>'
        "<p>数量来自正式 prediction registry 与结果包，不再使用旧快照手写汇总。</p></div>",
        unsafe_allow_html=True,
    )
    task_summary = datasets[
        [
            "dataset_label",
            "experiment_label",
            "task_label",
            "admission_status",
            "prediction_assets",
            "online_case_endpoints",
            "route_runs",
            "frozen_route_runs",
        ]
    ].rename(
        columns={
            "dataset_label": "数据集",
            "experiment_label": "实验路径 / 准入口径",
            "task_label": "任务",
            "admission_status": "当前阶段",
            "prediction_assets": "任务预测资产",
            "online_case_endpoints": "单病例入口",
            "route_runs": "路由结果包",
            "frozen_route_runs": "冻结评估",
        }
    )
    task_summary["使用范围"] = "研究评测（未授予正式路由）"
    st.dataframe(
        task_summary,
        hide_index=True,
        width="stretch",
        column_config={
            "任务预测资产": st.column_config.NumberColumn(format="%d"),
            "单病例入口": st.column_config.NumberColumn(format="%d"),
            "路由结果包": st.column_config.NumberColumn(format="%d"),
            "冻结评估": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="OphAgent 模型中转台", layout="wide", initial_sidebar_state="auto")
    inject_app_css()
    inject_model_hub_css()
    workspace = sidebar_navigation()
    context = "离线审阅 · V1.1" if workspace == "病例回放" else "模型工程 · V1.1"
    page_header(workspace, context=context)
    output_dir = model_hub_output_dir()
    data = load_model_hub_outputs(output_dir, model_hub_root=output_dir.parent)
    data["hub_index"] = build_model_hub_index(PROJECT_ROOT)
    if data["missing"]:
        st.error("受控路由基线产物不完整：" + "、".join(data["missing"]))
        st.code(
            "python scripts/routing/run_controlled_protocol.py --config "
            "python scripts/routing/run_interactive_model_hub.py --config "
            "experiments/opening_risk_routing_closure/configs/protocols/aptos_h100_six_model_pool.yaml"
        )
        st.stop()
    if workspace == "中转台总览":
        _render_overview(data)
    elif workspace in {"模型资产", "任务模型", "研究评测", "任务运行记录"}:
        render_engineering_workspace(data, view=workspace)
    else:
        render_clinical_workspace(data)


if __name__ == "__main__":
    main()
