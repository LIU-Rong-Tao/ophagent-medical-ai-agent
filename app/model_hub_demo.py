"""OphAgent v0.8.6 模型中转台入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_hub_clinical import render_clinical_workspace
from app.model_hub_data import load_model_hub_outputs
from app.model_hub_engineering import render_engineering_workspace
from app.model_hub_ui import inject_model_hub_css, stat_strip, title_block
from app.ui import inject_app_css


OUTPUT_DIR = PROJECT_ROOT / "experiments/v0_8_6_interactive_model_hub/outputs"


def main() -> None:
    st.set_page_config(page_title="OphAgent 模型中转台", layout="wide", initial_sidebar_state="collapsed")
    inject_app_css()
    inject_model_hub_css()
    title_block()
    data = load_model_hub_outputs(OUTPUT_DIR)
    if data["missing"]:
        st.error("v0.8.6 产物不完整：" + "、".join(data["missing"]))
        st.code(
            "python scripts/routing/run_controlled_protocol.py --config "
            "experiments/v0_8_6_interactive_model_hub/configs/controlled_runner.yaml --resume"
        )
        st.stop()
    stat_strip(data["models"], data["pairings"])
    workspace = st.segmented_control(
        "工作区",
        ["模型工程", "病例回放与路由解释"],
        default="模型工程",
        key="hub_workspace",
    )
    if workspace == "模型工程":
        render_engineering_workspace(data)
    else:
        render_clinical_workspace(data)


if __name__ == "__main__":
    main()
