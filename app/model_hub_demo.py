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
from app.model_hub_ui import (  # noqa: E402
    boundary_notice,
    inject_model_hub_css,
    page_header,
    sidebar_navigation,
    task_label,
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
    models = data["models"]
    pairings = data["pairings"]
    ready = models["compatibility_status"].astype(str).eq("ready_for_pairing")
    online = _model_flag(models, "task_inference_ready")
    completed_pairings = pairings["status"].astype(str).eq("completed")
    values = [
        ("可回放任务模型", int(ready.sum()), "具有当前任务冻结预测或在线输出"),
        ("在线链已验证", int(online.sum()), "Adapter 与任务推理链状态"),
        ("受控评测组合", int(pairings.loc[completed_pairings, "pairing_id"].nunique()), "已完成的路由/专家组合"),
        ("已登记任务", int(models["task_id"].nunique()), "当前统一标签空间"),
    ]
    cards = "".join(
        '<div class="hub-overview-kpi">'
        f"<span>{label}</span><b>{value:,}</b><small>{note}</small></div>"
        for label, value, note in values
    )
    st.markdown(f'<div class="hub-overview-grid">{cards}</div>', unsafe_allow_html=True)
    boundary_notice()
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

    st.markdown('<div class="hub-section"><h3>当前任务覆盖</h3><p>这里只汇总已登记任务模型，不改变模型资格判断。</p></div>', unsafe_allow_html=True)
    task_summary = (
        models.assign(
            可回放=ready,
            在线链=online,
        )
        .groupby("task_id", as_index=False)
        .agg(模型记录=("model_id", "nunique"), 可回放=("可回放", "sum"), 在线链=("在线链", "sum"))
    )
    task_summary["任务"] = task_summary["task_id"].map(task_label)
    task_summary = task_summary[["任务", "模型记录", "可回放", "在线链"]]
    st.dataframe(
        task_summary,
        hide_index=True,
        width="stretch",
        column_config={
            "模型记录": st.column_config.NumberColumn(format="%d"),
            "可回放": st.column_config.NumberColumn(format="%d"),
            "在线链": st.column_config.NumberColumn(format="%d"),
        },
    )


def main() -> None:
    st.set_page_config(page_title="OphAgent 模型中转台", layout="wide", initial_sidebar_state="auto")
    inject_app_css()
    inject_model_hub_css()
    workspace = sidebar_navigation()
    context = "离线审阅 · V1" if workspace == "病例回放" else "模型工程 · v0.8.10"
    page_header(workspace, context=context)
    output_dir = model_hub_output_dir()
    data = load_model_hub_outputs(output_dir, model_hub_root=output_dir.parent)
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
