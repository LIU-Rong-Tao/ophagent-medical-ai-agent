from pathlib import Path
import json

import yaml
import streamlit as st
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


# =====================================================
# 页面配置
# =====================================================
st.set_page_config(
    page_title="OphAgent v0.1 Demo",
    layout="wide",
)

st.title("OphAgent v0.1 Vision Baseline Demo")

st.markdown(
    """
当前版本用于演示眼底图像 DR 五分类。

> 注意：v0.1.0 仅为视觉 baseline，不是完整医疗 Agent，不能用于真实临床诊断。
"""
)

st.info(
    """
Demo 样例图片从测试集中随机抽取，仅用于功能展示，不代表完整测试集表现。
"""
)


# =====================================================
# 固定路径配置
# =====================================================
CONFIG_PATH = "configs/vision_baseline.yaml"
RUN_DIR = "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42"
CHECKPOINT_PATH = f"{RUN_DIR}/checkpoints/convnext_tiny_best.pth"
CLASS_TO_IDX_PATH = f"{RUN_DIR}/configs/class_to_idx.json"
CHECKPOINT_META_PATH = f"{RUN_DIR}/checkpoints/checkpoint_meta.json"
DEMO_SAMPLE_DIR = "demo_samples"


# =====================================================
# 类别名称映射
# =====================================================
CLASS_DISPLAY_NAMES = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}


# =====================================================
# 工具函数
# =====================================================
def check_required_files() -> None:
    """检查 Demo 运行所需文件是否存在。"""

    required_files = {
        "配置文件": Path(CONFIG_PATH),
        "模型权重": Path(CHECKPOINT_PATH),
        "类别映射文件": Path(CLASS_TO_IDX_PATH),
        "模型元信息文件": Path(CHECKPOINT_META_PATH),
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.exists()
    ]

    if missing_files:
        st.error("缺少 Demo 运行所需文件。")
        st.warning(
            """
请先从 GitHub Release 下载预训练模型权重，并放到 README 中指定的位置。

当前 Demo 默认需要以下文件：
"""
        )

        for item in missing_files:
            st.code(item)

        st.info(
            """
推荐放置路径：

`experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth`

如果你刚 clone 仓库但还没有下载 checkpoint，Demo 无法直接运行。
"""
        )
        st.stop()


def load_config(config_path: str) -> dict:
    """读取 YAML 配置文件。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_mapping(class_to_idx_path: str):
    """读取训练时保存的类别映射。"""

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class

def load_checkpoint_metadata(meta_path: str) -> dict:
    """读取 checkpoint 对应的模型元信息。"""

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_transform(image_size: int):
    """构建与训练阶段一致的图像预处理流程。"""

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
def load_model(
    backbone: str,
    num_classes: int,
    checkpoint_path: str,
    device: str,
):
    """加载 ConvNeXt 分类模型。"""

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

    return model


def predict_image(
    image: Image.Image,
    model,
    transform,
    device: str,
    idx_to_class: dict,
):
    """对单张眼底图像进行推理，并返回 Top-3 概率。"""

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)
        top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()

    raw_class = idx_to_class[pred_idx]
    display_class = CLASS_DISPLAY_NAMES.get(raw_class, raw_class)

    top3_results = []
    for prob, idx in zip(top3_probs[0], top3_indices[0]):
        idx = idx.item()
        raw_name = idx_to_class[idx]
        display_name = CLASS_DISPLAY_NAMES.get(raw_name, raw_name)

        top3_results.append(
            {
                "raw_name": raw_name,
                "display_name": display_name,
                "probability": prob.item(),
            }
        )

    return raw_class, display_class, confidence, top3_results


def get_demo_samples(sample_root: str):
    """读取内置 Demo 样例图片。"""

    sample_root = Path(sample_root)

    if not sample_root.exists():
        return {}

    samples = {}

    for class_dir in sorted(sample_root.iterdir()):
        if class_dir.is_dir():
            image_paths = []

            for suffix in ["*.png", "*.jpg", "*.jpeg"]:
                image_paths.extend(class_dir.glob(suffix))

            if image_paths:
                samples[class_dir.name] = sorted(image_paths)

    return samples


# =====================================================
# 初始化
# =====================================================
check_required_files()

config = load_config(CONFIG_PATH)
class_to_idx, idx_to_class = load_class_mapping(CLASS_TO_IDX_PATH)
checkpoint_meta = load_checkpoint_metadata(CHECKPOINT_META_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = build_transform(config["image_size"])

model = load_model(
    backbone=config["backbone"],
    num_classes=config["num_classes"],
    checkpoint_path=CHECKPOINT_PATH,
    device=device,
)

demo_samples = get_demo_samples(DEMO_SAMPLE_DIR)


# =====================================================
# 侧边栏
# =====================================================
st.sidebar.header("Demo 设置")

input_mode = st.sidebar.radio(
    "选择输入方式",
    ["选择内置样例", "上传图片"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 模型信息")
st.sidebar.write(f"Project: `{checkpoint_meta.get('project', 'Unknown')}`")
st.sidebar.write(f"Version: `{checkpoint_meta.get('version', 'Unknown')}`")
st.sidebar.write(f"Stage: `{checkpoint_meta.get('stage', 'Unknown')}`")
st.sidebar.write(f"Dataset: `{checkpoint_meta.get('dataset', 'Unknown')}`")
st.sidebar.write(f"Backbone: `{checkpoint_meta.get('backbone', config['backbone'])}`")
st.sidebar.write(f"Input Size: `{checkpoint_meta.get('input_size', config['image_size'])}`")
st.sidebar.write(f"Seed: `{checkpoint_meta.get('seed', 'Unknown')}`")
st.sidebar.write(f"Checkpoint: `{checkpoint_meta.get('checkpoint', 'Unknown')}`")
st.sidebar.write(f"Device: `{device}`")

st.sidebar.markdown("### 测试集指标")
st.sidebar.write(f"Accuracy: `{checkpoint_meta.get('test_accuracy', 0) * 100:.2f}%`")
st.sidebar.write(f"Macro Recall: `{checkpoint_meta.get('macro_recall', 0) * 100:.2f}%`")
st.sidebar.write(f"Macro F1: `{checkpoint_meta.get('macro_f1', 0) * 100:.2f}%`")

st.sidebar.markdown("### 使用限制")
st.sidebar.warning("仅用于科研和工程演示，不能用于临床诊断。")


# =====================================================
# 主界面输入
# =====================================================
image = None
true_raw_label = None
image_name = None

if input_mode == "选择内置样例":
    if not demo_samples:
        st.warning("未找到 demo_samples 目录或样例图片。")
    else:
        class_options = list(demo_samples.keys())

        selected_class = st.sidebar.selectbox(
            "选择真实类别",
            class_options,
        )

        image_options = demo_samples[selected_class]

        selected_image_path = st.sidebar.selectbox(
            "选择样例图片",
            image_options,
            format_func=lambda p: p.name,
        )

        image = Image.open(selected_image_path).convert("RGB")
        true_raw_label = selected_class
        image_name = selected_image_path.name

elif input_mode == "上传图片":
    uploaded_file = st.file_uploader(
        "上传眼底图像",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name


# =====================================================
# 推理与显示
# =====================================================
if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入图像")
        st.image(
            image,
            caption=image_name,
            use_container_width=True,
        )

    (
        pred_raw_label,
        pred_display_label,
        confidence,
        top3_results,
    ) = predict_image(
        image=image,
        model=model,
        transform=transform,
        device=device,
        idx_to_class=idx_to_class,
    )

    with col2:
        st.subheader("模型预测结果")

        if true_raw_label is not None:
            true_display_label = CLASS_DISPLAY_NAMES.get(
                true_raw_label,
                true_raw_label,
            )

            st.write(f"真实标签：**{true_display_label}**")
            st.write(f"预测标签：**{pred_display_label}**")

            is_correct = true_raw_label == pred_raw_label

            if is_correct:
                st.success("预测结果：正确")
            else:
                st.error("预测结果：错误")
        else:
            st.write(f"预测标签：**{pred_display_label}**")

        st.write(f"置信度：**{confidence:.4f}**")
        st.caption("Confidence 表示模型 softmax 输出概率，不等同于医学诊断可信度。")

        st.markdown("### Top-3 Prediction Probabilities")

        for rank, item in enumerate(top3_results, start=1):
            st.write(
                f"{rank}. {item['display_name']} "
                f"({item['probability']:.4f})"
            )

        st.caption(f"Internal Label: `{pred_raw_label}`")

    st.markdown("---")
    st.markdown(
        """
### v0.2 计划

- Grad-CAM 可解释性分析
- 病灶热力图可视化
- 模型关注区域展示
"""
    )

else:
    st.info("请在左侧选择内置样例，或上传一张眼底图像。")