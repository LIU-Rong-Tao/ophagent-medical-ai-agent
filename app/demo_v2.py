"""
OphAgent v0.2 Streamlit Demo

功能：
1. 分类预测 Demo
   - 上传眼底图像
   - 使用内置 demo_samples
   - 显示预测类别、confidence、Top-3
   - 显示模型信息与测试集指标

2. Grad-CAM Gallery
   - 展示 v0.2 离线生成的 Grad-CAM / HiResCAM 样例
   - 包含 good_cases / failure_cases / interesting_cases

注意：
- 分类预测是实时推理
- Grad-CAM Gallery 是离线结果展示，不实时计算 Grad-CAM
- 本 Demo 仅用于科研与工程展示，不用于临床诊断
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms


# =====================================================
# 页面配置
# =====================================================

st.set_page_config(
    page_title="OphAgent v0.2 Demo",
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
        "description": "热区与可见病灶区域相对一致，适合用于 README / Demo 展示。",
    },
    "failure_cases": {
        "title": "Failure Cases",
        "zh_title": "失败样例",
        "description": "热区偏离病灶，或受到边缘、亮度、背景等因素影响，用于说明 CAM 的局限性。",
    },
    "interesting_cases": {
        "title": "Interesting Cases",
        "zh_title": "分析样例",
        "description": "不是单纯成功或失败，但具有分析价值，适合用于 explainability 讨论。",
    },
}


CASE_NOTES = {
    "cmoderatedr_b9127e38d9b9_overlay.png": "代表性 good case：热区较好覆盖左下方黄白色渗出样病灶。",
    "cmoderatedr_d9bbdc33db83_overlay.png": "代表性 good case：对大片可见异常区域有较直观响应。",
    "dseveredr_383e72af1955_overlay.png": "代表性 good case：严重病变样例中热区相对集中。",
    "bmilddr_07929d32b5b3_overlay.png": "failure case：热区与轻度病灶对应不稳定，提示 mild DR explainability 较难。",
    "eproliferativedr_247e98aba610_overlay.png": "failure case：热区可能受到图像边缘或局部高亮区域影响。",
    "eproliferativedr_bba38f2294a3_overlay.png": "failure case：热区与可见病灶区域不够一致。",
    "anodr_c9e697117f3f_overlay.png": "interesting case：No DR 图像无明确病灶，但模型仍会给出关注区域。",
    "dseveredr_e93394175a19_overlay.png": "interesting case：严重病变样例中模型关注区域具有一定解释价值，但仍不完全稳定。",
    "eproliferativedr_6c3745a222da_overlay.png": "interesting case：PDR 样例中关注区域偏局部，可用于讨论重症类别解释的不稳定性。",
}


# =====================================================
# 通用工具函数
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
    """
    缓存加载模型，避免 Streamlit 每次刷新都重新加载 checkpoint。
    """

    config = load_yaml(Path(config_path))
    class_to_idx = load_json(Path(class_to_idx_path))

    if config is None:
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    if class_to_idx is None:
        raise FileNotFoundError(f"class_to_idx 文件不存在：{class_to_idx_path}")

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
        image_paths.extend(DEMO_SAMPLES_ROOT.glob(f"*/*{suffix[-4:]}"))

    # 更稳妥地递归读取
    image_paths = []
    for suffix in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(DEMO_SAMPLES_ROOT.rglob(suffix))

    return sorted(image_paths)


def show_model_sidebar():
    """侧边栏展示模型和指标信息。"""

    st.sidebar.markdown("---")
    st.sidebar.subheader("模型信息")

    checkpoint_meta = load_json(CHECKPOINT_META_PATH)

    if checkpoint_meta is not None:
        for key, value in checkpoint_meta.items():
            if isinstance(value, (str, int, float)):
                st.sidebar.write(f"**{key}**：{value}")
    else:
        st.sidebar.info("未找到 checkpoint_meta.json")

# =====================================================
# 分类预测 Demo
# =====================================================

def run_classifier_demo():
    """运行分类预测 Demo。"""

    st.title("OphAgent v0.2 Classifier Demo")

    st.markdown(
        """
本页面用于演示 OphAgent 的糖尿病视网膜病变分类能力。

支持两种输入方式：
- 上传本地眼底图像
- 选择内置 demo_samples 样例
"""
    )

    st.warning(
        """
本 Demo 仅用于科研与工程展示。
模型输出的 softmax confidence 不等同于医学诊断可信度。
"""
    )

    missing_files = []

    for path in [CONFIG_PATH, CHECKPOINT_PATH, CLASS_TO_IDX_PATH]:
        if not path.exists():
            missing_files.append(path)

    if missing_files:
        st.error("缺少运行分类 Demo 所需文件。")
        for path in missing_files:
            st.code(str(path))
        st.info("请从 GitHub Release 下载 checkpoint，并确认 class_to_idx.json 存在。")
        return

    model, device, idx_to_class, image_size, _ = load_model_cached(
        str(CONFIG_PATH),
        str(CHECKPOINT_PATH),
        str(CLASS_TO_IDX_PATH),
    )

    show_model_sidebar()

    input_mode = st.radio(
        "选择输入方式",
        ["使用内置样例", "上传图片"],
        horizontal=True,
    )

    image = None
    true_label = None
    image_name = None

    if input_mode == "使用内置样例":
        sample_paths = get_demo_sample_paths()

        if not sample_paths:
            st.error("未找到 demo_samples 图片。")
            return

        sample_display_names = [
            str(path.relative_to(DEMO_SAMPLES_ROOT)) for path in sample_paths
        ]

        selected_sample = st.selectbox(
            "选择 demo sample",
            sample_display_names,
        )

        selected_path = DEMO_SAMPLES_ROOT / selected_sample
        image = Image.open(selected_path).convert("RGB")
        image_name = selected_path.name

        raw_label = selected_path.parent.name
        true_label = CLASS_DISPLAY_NAMES.get(raw_label, raw_label)

    else:
        uploaded_file = st.file_uploader(
            "上传眼底图像",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    if image is None:
        st.info("请选择或上传一张眼底图像。")
        return

    col_image, col_result = st.columns([1, 1])

    with col_image:
        st.subheader("输入图像")
        st.image(
            image,
            caption=image_name,
            use_container_width=True,
        )

        if true_label is not None:
            st.write(f"真实标签：**{true_label}**")

    probs = predict_image(
        model=model,
        device=device,
        image=image,
        image_size=image_size,
    )

    pred_idx = int(torch.argmax(probs).item())
    pred_prob = float(probs[pred_idx].item())

    pred_raw_label = idx_to_class[pred_idx]
    pred_display_label = CLASS_DISPLAY_NAMES.get(pred_raw_label, pred_raw_label)

    topk = min(3, len(probs))
    top_probs, top_indices = torch.topk(probs, k=topk)

    with col_result:
        st.subheader("预测结果")

        st.metric(
            label="预测类别",
            value=pred_display_label,
        )

        st.metric(
            label="Confidence",
            value=f"{pred_prob:.4f}",
        )

        if true_label is not None:
            is_correct = pred_display_label == true_label
            if is_correct:
                st.success("预测正确")
            else:
                st.error("预测错误")

        st.markdown("### Top-3 结果")

        top_rows = []

        for prob, idx in zip(top_probs, top_indices):
            raw_label = idx_to_class[int(idx.item())]
            display_label = CLASS_DISPLAY_NAMES.get(raw_label, raw_label)

            top_rows.append(
                {
                    "Class": display_label,
                    "Internal Label": raw_label,
                    "Probability": float(prob.item()),
                }
            )

        st.dataframe(
            pd.DataFrame(top_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "说明：Confidence 为 softmax 概率，仅表示模型输出分布，不等同于医学诊断可信度。"
        )


# =====================================================
# Grad-CAM Gallery Demo
# =====================================================

def get_gallery_images(group_name: str):
    """读取指定 case group 下的 overlay 图片。"""

    group_dir = GALLERY_ROOT / group_name

    if not group_dir.exists():
        return []

    image_paths = []

    for suffix in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(group_dir.glob(suffix))

    return sorted(image_paths)


def show_case_group(group_name: str, num_columns: int = 3):
    """以网格形式展示某一类 Grad-CAM 样例。"""

    group_info = CASE_GROUPS[group_name]
    image_paths = get_gallery_images(group_name)

    st.markdown("---")
    st.subheader(f"{group_info['title']}｜{group_info['zh_title']}")
    st.write(group_info["description"])

    if not image_paths:
        st.info(f"未找到 {group_name} 样例。")
        return

    columns = st.columns(num_columns)

    for index, image_path in enumerate(image_paths):
        column = columns[index % num_columns]

        with column:
            image = Image.open(image_path).convert("RGB")

            st.image(
                image,
                caption=image_path.name,
                use_container_width=True,
            )

            note = CASE_NOTES.get(
                image_path.name,
                "该样例用于 Grad-CAM qualitative analysis。",
            )
            st.caption(note)


def run_gradcam_gallery():
    """运行 Grad-CAM Gallery 页面。"""

    st.title("OphAgent v0.2 Grad-CAM Explainability Gallery")

    st.markdown(
        """
本页面展示 OphAgent v0.2 中整理的 Grad-CAM / HiResCAM 可解释性样例。

当前样例均为离线生成并人工筛选的 qualitative examples，
不是实时在线计算 Grad-CAM。
"""
    )

    st.warning(
        """
注意：CAM 热力图仅用于模型行为分析与定性解释，
不等同于医学病灶分割、临床诊断或治疗建议。
"""
    )

    if not GALLERY_ROOT.exists():
        st.error("未找到 Grad-CAM Gallery 目录。")
        st.code(str(GALLERY_ROOT))
        return

    st.markdown(
        """
### 样例划分

- **Good Cases｜成功样例**：热区与可见病灶较一致。
- **Failure Cases｜失败样例**：热区偏离病灶，或受到边缘、亮度、背景等干扰。
- **Interesting Cases｜分析样例**：结果具有分析价值，但不能简单视为成功或失败。
"""
    )

    selected_groups = st.sidebar.multiselect(
        "选择展示类别",
        options=list(CASE_GROUPS.keys()),
        default=list(CASE_GROUPS.keys()),
        format_func=lambda x: f"{CASE_GROUPS[x]['title']}｜{CASE_GROUPS[x]['zh_title']}",
    )

    num_columns = st.sidebar.slider(
        "每行显示图片数量",
        min_value=1,
        max_value=4,
        value=3,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 默认 Explainability 配置")
    st.sidebar.write("Method：`HiResCAM`")
    st.sidebar.write("Target Layer：`stage3`")
    st.sidebar.write("Smoothing：`关闭`")

    st.sidebar.markdown("### Gallery 路径")
    st.sidebar.code(str(GALLERY_ROOT))

    st.sidebar.markdown("### 使用限制")
    st.sidebar.warning("仅用于科研与工程展示，不能用于临床诊断。")

    for group_name in selected_groups:
        show_case_group(
            group_name=group_name,
            num_columns=num_columns,
        )


# =====================================================
# 主入口
# =====================================================

st.sidebar.title("OphAgent v0.2")

demo_mode = st.sidebar.radio(
    "Demo 模式",
    ["分类预测", "Grad-CAM Gallery"],
)

if demo_mode == "分类预测":
    run_classifier_demo()
else:
    run_gradcam_gallery()