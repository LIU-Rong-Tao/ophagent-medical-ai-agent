"""
OphAgent v0.2.2 Streamlit Demo

统一入口：
    streamlit run app/demo.py

功能：
1. 分类预测 Demo
   - 上传眼底图像
   - 使用内置 demo_samples
   - 显示预测类别、confidence、Top-3

2. Lightweight VL Reasoning Report
   - 根据 prediction / confidence / structured findings 生成中文报告
   - 同时展示 rule_based 与 OpenAI provider 输出
   - OpenAI 不可用时自动 fallback，不影响 demo

3. Grad-CAM Gallery
   - 展示离线生成的 Grad-CAM / HiResCAM 样例
   - 用于 explainability showcase

注意：
- 本 Demo 仅用于科研与工程展示
- 不用于临床诊断
"""

import json
import sys
from pathlib import Path

# =====================================================
# 把项目根目录加入 Python 搜索路径
# 避免 streamlit run app/demo.py 时找不到 findings / reasoning
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from findings.finding_generator import generate_case_findings
from reasoning.report_generator import generate_report


# =====================================================
# 页面配置
# =====================================================

st.set_page_config(
    page_title="OphAgent v0.2.2 Demo",
    layout="wide",
)


# =====================================================
# 固定路径配置
# =====================================================

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

METRICS_PATH = Path(
    "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/metrics.json"
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


RAW_CLASS_TO_DISPLAY = {
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


# =====================================================
# 工具函数
# =====================================================

def load_json(path: Path):
    """读取 JSON 文件。"""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path):
    """读取 YAML 文件。"""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_size(config: dict) -> int:
    """从 config 中读取 image_size，兼容不同字段写法。"""
    if "image_size" in config:
        return int(config["image_size"])

    if "data" in config and "image_size" in config["data"]:
        return int(config["data"]["image_size"])

    return 224


def get_backbone(config: dict) -> str:
    """从 config 中读取 backbone，兼容不同字段写法。"""
    if "backbone" in config:
        return config["backbone"]

    if "model" in config and "backbone" in config["model"]:
        return config["model"]["backbone"]

    return "convnext_tiny"


def get_num_classes(config: dict, class_to_idx: dict) -> int:
    """读取类别数。"""
    if "num_classes" in config:
        return int(config["num_classes"])

    if "model" in config and "num_classes" in config["model"]:
        return int(config["model"]["num_classes"])

    return len(class_to_idx)


def build_transform(image_size: int):
    """构建模型输入预处理。"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@st.cache_resource
def load_model_cached(
    config_path: str,
    checkpoint_path: str,
    class_to_idx_path: str,
):
    """缓存加载模型，避免 Streamlit 每次刷新都重新加载 checkpoint。"""

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


def predict_image(model, device, image: Image.Image, image_size: int):
    """对单张图像进行分类预测。"""
    transform = build_transform(image_size)

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    return probs.cpu()


def get_demo_sample_paths():
    """读取 demo_samples 下所有图片路径。"""
    if not DEMO_SAMPLES_ROOT.exists():
        return []

    image_paths = []
    for suffix in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(DEMO_SAMPLES_ROOT.rglob(suffix))

    return sorted(image_paths)


def infer_raw_class_from_path(image_path: Path) -> str:
    """从 demo sample 路径推断 raw_class。上传图像则返回 unknown。"""
    try:
        parent_name = image_path.parent.name
        if parent_name in RAW_CLASS_TO_DISPLAY:
            return parent_name
    except Exception:
        pass

    return "unknown"


def build_topk_dataframe(probs, idx_to_class: dict, k: int = 3):
    """构造 Top-K 结果表。"""
    top_probs, top_indices = torch.topk(probs, k=min(k, len(probs)))

    rows = []
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        raw_class = idx_to_class[idx]
        display_name = CLASS_DISPLAY_NAMES.get(raw_class, raw_class)
        rows.append(
            {
                "Rank": rank,
                "Raw Class": raw_class,
                "Prediction": display_name,
                "Confidence": float(prob),
            }
        )

    return pd.DataFrame(rows)


def build_case_findings_from_prediction(
    image_path: str,
    prediction: str,
    raw_class: str,
    confidence: float,
    topk_df: pd.DataFrame,
):
    """把 Streamlit 分类结果转换为 CaseFindings。"""

    topk_predictions = [
        {row["Prediction"]: float(row["Confidence"])}
        for _, row in topk_df.iterrows()
    ]

    # Streamlit 当前页使用离线 Grad-CAM Gallery，不实时生成 CAM。
    # 因此这里 CAM 字段先置空，报告里只描述分类与可能视觉线索。
    return generate_case_findings(
        image_path=image_path,
        prediction=prediction,
        raw_class=raw_class,
        confidence=confidence,
        topk_predictions=topk_predictions,
        cam_method=None,
        cam_target_layer=None,
        cam_output_path=None,
    )


def render_reasoning_reports(case_findings):
    """
    同时展示 rule_based 和 OpenAI provider 报告。

    OpenAI 不可用时，generate_report 内部会 fallback 到 rule_based，
    保证 Streamlit demo 不崩。
    """

    st.subheader("VL Reasoning Report")
    st.caption(
        "Provider comparison: rule-based fallback vs optional OpenAI report provider. "
        "Reports are for research/demo use only, not clinical diagnosis."
    )

    report_config = load_yaml(REPORT_CONFIG_PATH) or {}
    report_config = report_config.get("report", {})

    tab_rule, tab_openai = st.tabs(
        ["Rule-based Report", "OpenAI Report"]
    )

    with tab_rule:
        rule_report = generate_report(
            case_findings,
            provider_name="rule_based",
            fallback_provider_name="rule_based",
            provider_config=report_config,
        )
        st.markdown(rule_report)

    with tab_openai:
        with st.spinner("Generating OpenAI report..."):
            openai_report = generate_report(
                case_findings,
                provider_name="openai",
                fallback_provider_name="rule_based",
                provider_config=report_config,
            )

        st.markdown(openai_report)
        st.caption(
            "If OpenAI API is unavailable, this tab falls back to the rule-based provider."
        )


def render_metrics_panel():
    """展示训练/评估元信息。"""

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

        # 如果存在 metrics.json，则优先展示
        if metrics:
            st.json(metrics)

        # 否则从 checkpoint_meta 中抽取核心指标
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
    """展示离线 Grad-CAM Gallery。"""

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
            st.image(str(image_path), caption=image_path.name, use_container_width=True)
            note = CASE_NOTES.get(image_path.name)
            if note:
                st.caption(note)


# =====================================================
# 主页面
# =====================================================

st.title("OphAgent v0.2.2")
st.caption(
    "Diabetic Retinopathy Classification + Explainability + Lightweight VL Reasoning"
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
        st.image(selected_image, use_container_width=True)

    try:
        model, device, idx_to_class, image_size, config = load_model_cached(
            str(CONFIG_PATH),
            str(CHECKPOINT_PATH),
            str(CLASS_TO_IDX_PATH),
        )

        probs = predict_image(
            model=model,
            device=device,
            image=selected_image,
            image_size=image_size,
        )

        topk_df = build_topk_dataframe(
            probs=probs,
            idx_to_class=idx_to_class,
            k=3,
        )

        top1 = topk_df.iloc[0]
        prediction = top1["Prediction"]
        raw_class = top1["Raw Class"]
        confidence = float(top1["Confidence"])

        with right_col:
            st.subheader("Prediction")
            st.metric("Predicted Class", prediction)
            st.metric("Confidence", f"{confidence:.4f}")

            st.markdown("**Top-3 Predictions**")
            st.dataframe(
                topk_df,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        case_findings = build_case_findings_from_prediction(
            image_path=image_source,
            prediction=prediction,
            raw_class=raw_class,
            confidence=confidence,
            topk_df=topk_df,
        )

        render_reasoning_reports(case_findings)

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
