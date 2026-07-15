"""OphAgent 模型输出后审计演示入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from app.ui import inject_app_css  # noqa: E402
from app.views import (  # noqa: E402
    batch_audit,
    external_evidence,
    overview,
    single_case,
    transfer_sandbox,
)


st.set_page_config(
    page_title="OphAgent Audit Demo",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_app_css()

with st.sidebar:
    st.markdown("## OphAgent Audit Demo")
    st.markdown("**科研证据快照 v0.7.2**")
    st.caption("公共数据回顾性实验")
    st.caption("不用于临床诊断")
    st.divider()
    st.session_state["display_mode"] = st.radio(
        "显示模式",
        ["临床展示", "研究审计"],
        horizontal=False,
        help="两种模式使用同一数据和计算口径，只改变信息层级。",
    )
    if st.session_state["display_mode"] == "临床展示":
        st.markdown(
            '<div class="oa-mode-note">面向临床交流：病例卡、复核优先级、'
            "关键结果和边界说明。</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="oa-mode-note">面向研究复核：展开技术字段、'
            "原始结果表、来源路径与下载入口。</div>",
            unsafe_allow_html=True,
        )

if not hasattr(st, "Page") or not hasattr(st, "navigation"):
    st.error(
        "当前 Streamlit 版本不支持 st.Page + st.navigation。"
        "请使用 requirements.txt 中的 Streamlit 1.45.1 或更新版本。"
    )
    st.stop()

pages = [
    st.Page(overview.render, title="项目概览", url_path="overview", default=True),
    st.Page(single_case.render, title="单病例审计", url_path="single-case"),
    st.Page(batch_audit.render, title="批量复核排序", url_path="batch-audit"),
    st.Page(
        external_evidence.render,
        title="外部证据",
        url_path="external-evidence",
    ),
    st.Page(
        transfer_sandbox.render,
        title="新任务审计沙盒",
        url_path="transfer-sandbox",
    ),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
