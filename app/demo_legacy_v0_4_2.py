"""
OphAgent v0.4.2 Streamlit Demo

统一入口：
    streamlit run app/demo.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from agent import AgentInput, run_agent  # noqa: E402
from reasoning.report_generator import generate_report  # noqa: E402


st.set_page_config(
    page_title="OphAgent v0.4.2 Demo",
    layout="wide",
)


CONFIG_PATH = Path("configs/vision_baseline.yaml")

CHECKPOINT_PATH = Path(
    "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth"
)

CHECKPOINT_META_PATH = Path(
    "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/checkpoint_meta.json"
)

CLASS_TO_IDX_PATH = Path(
    "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json"
)

METRICS_PATH = (
    "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/"
    "evaluation/test/metrics.json"
)

REPORT_CONFIG_PATH = Path("configs/report_generation.yaml")

DEMO_SAMPLES_ROOT = Path("demo_samples")

GALLERY_ROOT = Path("docs/gradcam_gallery")


CLASS_DISPLAY_NAMES = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}


CASE_GROUPS = {
    "good_cases": {
        "title": "Good Cases",
        "zh_title": "成功样例",
        "description": "热区与可见异常区域相对一致，适合用于 README / Demo 展示。",
    },
    "failure_cases": {
        "title": "Failure Cases",
        "zh_title": "失败样例",
        "description": "热区偏离可见异常区域，或受到边缘、亮度、背景等因素影响，用于说明 CAM 的局限性。",
    },
    "interesting_cases": {
        "title": "Interesting Cases",
        "zh_title": "分析样例",
        "description": "不是单纯成功或失败，但具有分析价值，适合用于 explainability 讨论。",
    },
}


CASE_NOTES = {
    "cmoderatedr_b9127e38d9b9_overlay.png": "代表性 good case：热区较好覆盖左下方黄白色渗出样区域。",
    "cmoderatedr_d9bbdc33db83_overlay.png": "代表性 good case：对大片可见异常区域有较直观响应。",
    "dseveredr_383e72af1955_overlay.png": "代表性 good case：严重病变样例中热区相对集中。",
    "bmilddr_07929d32b5b3_overlay.png": "failure case：热区与轻度异常区域对应不稳定，提示 mild DR explainability 较难。",
    "eproliferativedr_247e98aba610_overlay.png": "failure case：热区可能受到图像边缘或局部高亮区域影响。",
    "eproliferativedr_bba38f2294a3_overlay.png": "failure case：热区与可见异常区域不够一致。",
    "anodr_c9e697117f3f_overlay.png": "interesting case：No DR 图像无明确异常，但模型仍会给出关注区域。",
    "dseveredr_e93394175a19_overlay.png": "interesting case：严重病变样例中模型关注区域具有一定解释价值，但仍不完全稳定。",
    "eproliferativedr_6c3745a222da_overlay.png": "interesting case：PDR 样例中关注区域偏局部，可用于讨论重症类别解释的不稳定性。",
}


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_size(config: dict) -> int:
    if "image_size" in config:
        return int(config["image_size"])

    if "data" in config and "image_size" in config["data"]:
        return int(config["data"]["image_size"])

    return 224


def get_backbone(config: dict) -> str:
    if "backbone" in config:
        return config["backbone"]

    if "model" in config and "backbone" in config["model"]:
        return config["model"]["backbone"]

    return "convnext_tiny"


def get_num_classes(config: dict, class_to_idx: dict) -> int:
    if "num_classes" in config:
        return int(config["num_classes"])

    if "model" in config and "num_classes" in config["model"]:
        return int(config["model"]["num_classes"])

    return len(class_to_idx)


def infer_ground_truth_from_path(image_path: str) -> str:
    path = Path(image_path)
    parent_name = path.parent.name

    if parent_name in CLASS_DISPLAY_NAMES:
        return parent_name

    return "unknown"


@st.cache_resource
def load_model_cached(
    config_path: str,
    checkpoint_path: str,
    class_to_idx_path: str,
):
    config = load_yaml(Path(config_path))
    class_to_idx = load_json(Path(class_to_idx_path))

    if config is None:
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    if class_to_idx is None:
        raise FileNotFoundError(f"class_to_idx 文件不存在：{class_to_idx_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = get_backbone(config)
    num_classes = get_num_classes(config, class_to_idx)

    model = timm.create_model(
        backbone,
        pretrained=False,
        num_classes=num_classes,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    image_size = get_image_size(config)

    return model, device, idx_to_class, image_size, config


def get_demo_sample_paths():
    if not DEMO_SAMPLES_ROOT.exists():
        return []

    image_paths = []

    for suffix in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(DEMO_SAMPLES_ROOT.rglob(suffix))

    return sorted(image_paths)


def load_report_config() -> dict:
    report_config = load_yaml(REPORT_CONFIG_PATH) or {}
    return report_config.get("report", report_config)


def topk_to_dataframe(topk_predictions):
    rows = []

    for item in topk_predictions:
        rows.append(
            {
                "Rank": item.rank,
                "Prediction": item.display_name,
                "Confidence": float(item.confidence),
            }
        )

    return pd.DataFrame(rows)


def render_findings_panel(findings):
    st.subheader("Structured Findings")

    st.caption(
        "Structured findings are generated from classification priors "
        "and explainability context. "
        "They are not lesion detector outputs."
    )

    generated_findings = getattr(findings, "findings", [])

    if generated_findings:
        for i, finding in enumerate(generated_findings, start=1):
            display_name = getattr(
                finding,
                "display_name",
                f"Finding {i}",
            )

            description = getattr(
                finding,
                "description",
                "",
            )

            confidence = getattr(
                finding,
                "confidence",
                None,
            )

            evidence = getattr(
                finding,
                "evidence",
                [],
            )

            with st.expander(
                f"{i}. {display_name}",
                expanded=True,
            ):
                if confidence is not None:
                    st.write(f"**Confidence:** `{float(confidence):.4f}`")

                if evidence:
                    st.write("**Evidence:**")
                    st.write(", ".join(evidence))

                st.write(description)

    else:
        st.info("No structured findings available.")

    disclaimer = getattr(findings, "disclaimer", None)

    if disclaimer:
        st.warning(disclaimer)


def render_rule_based_report_panel(agent_result):
    st.subheader("Rule-based Report")

    st.caption(
        "Generated by the lightweight agent runner. "
        "Reports are for research/demo use only, not clinical diagnosis."
    )

    report = getattr(agent_result, "report", "")

    if isinstance(report, dict):
        st.json(report)
    else:
        st.markdown(str(report))


def render_openai_report_panel(agent_result, report_config, image_source):
    st.subheader("OpenAI Report")

    st.caption(
        "Click the button to generate an OpenAI report. "
        "The result is cached in the current Streamlit session."
    )

    cache_key = f"openai_report::{image_source}"

    if st.button("Generate OpenAI Report"):
        with st.spinner("Generating OpenAI report..."):
            st.session_state[cache_key] = generate_report(
                agent_result.findings,
                provider_name="openai",
                fallback_provider_name="rule_based",
                provider_config=report_config,
            )

    if cache_key in st.session_state:
        st.markdown(st.session_state[cache_key])
    else:
        st.info("Click the button to generate an OpenAI report.")


def render_metrics_panel():
    st.subheader("Model / Evaluation Info")

    checkpoint_meta = load_json(CHECKPOINT_META_PATH)
    metrics = load_json(METRICS_PATH)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Checkpoint Metadata**")
        st.code(str(CHECKPOINT_PATH))

        if checkpoint_meta:
            st.json(checkpoint_meta)
        else:
            st.info("checkpoint_meta.json not found.")

    with col2:
        st.markdown("**Evaluation Summary**")

        if metrics:
            st.json(metrics)

        elif checkpoint_meta:
            metric_keys = [
                "test_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
            ]

            summary_rows = []

            for key in metric_keys:
                if key in checkpoint_meta:
                    summary_rows.append(
                        {
                            "Metric": key,
                            "Value": checkpoint_meta[key],
                        }
                    )

            if summary_rows:
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No evaluation metrics found.")

        else:
            st.info("No evaluation metrics found.")


def render_gradcam_gallery():
    st.header("Grad-CAM / HiResCAM Gallery")

    st.caption(
        "This gallery uses offline generated CAM artifacts. "
        "It is intended for explainability and failure analysis."
    )

    if not GALLERY_ROOT.exists():
        st.warning(f"Gallery directory not found: {GALLERY_ROOT}")
        return

    group_names = list(CASE_GROUPS.keys())

    selected_group = st.selectbox(
        "Select case group",
        group_names,
        format_func=lambda x: f"{CASE_GROUPS[x]['zh_title']} / {CASE_GROUPS[x]['title']}",
    )

    group_info = CASE_GROUPS[selected_group]
    group_dir = GALLERY_ROOT / selected_group

    st.subheader(f"{group_info['zh_title']} / {group_info['title']}")
    st.write(group_info["description"])

    if not group_dir.exists():
        st.warning(f"Group directory not found: {group_dir}")
        return

    images = sorted(
        [
            p
            for p in group_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )

    if not images:
        st.info("No CAM images found in this group.")
        return

    cols = st.columns(3)

    for i, image_path in enumerate(images):
        with cols[i % 3]:
            st.image(
                str(image_path),
                caption=image_path.name,
                use_container_width=True,
            )

            note = CASE_NOTES.get(image_path.name)

            if note:
                st.caption(note)


st.title("OphAgent v0.4.2")

st.caption(
    "Diabetic Retinopathy Classification + Explainability + Lightweight Agent Runner"
)

st.warning(
    "本 Demo 仅用于科研与工程展示，不用于临床诊断。"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Classification & Report",
        "Grad-CAM Gallery",
        "Model Info",
    ],
)


if page == "Classification & Report":
    st.header("Classification & VL Reasoning Report")

    sample_paths = get_demo_sample_paths()

    input_mode = st.radio(
        "Input Mode",
        ["Use demo sample", "Upload image"],
        horizontal=True,
    )

    selected_image = None
    image_source = "uploaded_image"

    if input_mode == "Use demo sample":
        if not sample_paths:
            st.error(f"No demo samples found under {DEMO_SAMPLES_ROOT}")
            st.stop()

        selected_sample = st.selectbox(
            "Select demo sample",
            sample_paths,
            format_func=lambda p: str(p),
        )

        selected_image = Image.open(selected_sample).convert("RGB")
        image_source = str(selected_sample)

    else:
        uploaded_file = st.file_uploader(
            "Upload fundus image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_source = uploaded_file.name

    if selected_image is None:
        st.info("Please select or upload an image.")
        st.stop()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Input Fundus Image")

        st.image(
            selected_image,
            use_container_width=True,
        )

    try:
        model, device, idx_to_class, image_size, config = load_model_cached(
            str(CONFIG_PATH),
            str(CHECKPOINT_PATH),
            str(CLASS_TO_IDX_PATH),
        )

        report_config = load_report_config()

        agent_result = run_agent(
            AgentInput(
                image=selected_image,
                image_source=image_source,
                top_k=3,
                report_providers=("rule_based",),
                fallback_report_provider="rule_based",
                report_config=report_config,
            ),
            model=model,
            device=device,
            idx_to_class=idx_to_class,
            image_size=image_size,
            class_display_names=CLASS_DISPLAY_NAMES,
        )

        ground_truth_label = infer_ground_truth_from_path(image_source)

        ground_truth_display = CLASS_DISPLAY_NAMES.get(
            ground_truth_label,
            "unknown",
        )

        topk_df = topk_to_dataframe(agent_result.topk)

        with right_col:
            st.subheader("Prediction")

            st.metric(
                "Predicted Class",
                agent_result.predicted_display_name,
            )

            st.metric(
                "Confidence",
                f"{agent_result.confidence:.4f}",
            )

            if ground_truth_label != "unknown":
                st.caption(
                    f"Ground Truth: `{ground_truth_display}` ({ground_truth_label})"
                )

            st.markdown("**Top-3 Predictions**")

            st.dataframe(
                topk_df,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        render_findings_panel(agent_result.findings)

        st.divider()

        st.subheader("VL Reasoning Report")

        tab_rule, tab_openai = st.tabs(
            [
                "Rule-based Report",
                "OpenAI Report",
            ]
        )

        with tab_rule:
            render_rule_based_report_panel(agent_result)

        with tab_openai:
            render_openai_report_panel(
                agent_result=agent_result,
                report_config=report_config,
                image_source=image_source,
            )

    except FileNotFoundError as exc:
        st.error(str(exc))

        st.info(
            "请确认 checkpoint、config、class_to_idx 是否存在。"
            "如果只是查看 gallery，可切换到 Grad-CAM Gallery 页面。"
        )

    except Exception as exc:
        st.exception(exc)


elif page == "Grad-CAM Gallery":
    render_gradcam_gallery()


elif page == "Model Info":
    render_metrics_panel()