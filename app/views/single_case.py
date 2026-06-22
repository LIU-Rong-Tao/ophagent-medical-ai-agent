"""单病例审计页。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from app.audit_core import summarize_dr_review_priority
from app.ui import (
    metric_card,
    page_header,
    render_boundary,
    render_case_card,
    render_empty_state,
    render_probability_profile,
    render_source_caption,
    section_header,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = PROJECT_ROOT / "demo_samples"
GALLERY_ROOT = PROJECT_ROOT / "docs" / "gradcam_gallery"
OFFLINE_PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "aptos_convnext_tiny"
    / "lr1e-4_bs32_seed42"
    / "evaluation"
    / "test"
    / "test_predictions.csv"
)

GRADE_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
GRADE_DISPLAY_LABELS = [
    "0级 · 未见 DR",
    "1级 · 轻度",
    "2级 · 中度",
    "3级 · 重度",
    "4级 · 增殖期",
]
GRADE_CLINICAL_NAMES = [
    "未见糖尿病视网膜病变",
    "轻度糖尿病视网膜病变",
    "中度糖尿病视网膜病变",
    "重度糖尿病视网膜病变",
    "增殖期糖尿病视网膜病变",
]
RAW_TO_DISPLAY = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}

CHECKPOINT_REGISTRY = {
    "ConvNeXt-Tiny · APTOS2019": {
        "backbone": "convnext_tiny",
        "checkpoint_path": (
            "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/"
            "checkpoints/convnext_tiny_best.pth"
        ),
        "config_path": (
            "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/config.json"
        ),
        "class_mapping_path": (
            "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/"
            "configs/class_to_idx.json"
        ),
        "metadata_path": (
            "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/"
            "checkpoints/checkpoint_meta.json"
        ),
        "state_dict_key": None,
        "compatibility": "沿用 v0.4.2 已验证的 ConvNeXt-Tiny 加载路径。",
    }
}


@st.cache_data(show_spinner=False)
def load_offline_predictions() -> pd.DataFrame:
    if not OFFLINE_PREDICTIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(OFFLINE_PREDICTIONS)


def sample_paths() -> list[Path]:
    paths: list[Path] = []
    if DEMO_ROOT.exists():
        for suffix in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(DEMO_ROOT.rglob(suffix))
    return sorted(paths)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def registry_status(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    required_keys = ["checkpoint_path", "config_path", "class_mapping_path"]
    missing = [
        key
        for key in required_keys
        if not (PROJECT_ROOT / str(entry[key])).exists()
    ]
    return not missing, missing


@st.cache_resource(show_spinner=False)
def load_registered_model(
    checkpoint_path: str,
    config_path: str,
    class_mapping_path: str,
):
    import timm
    import torch

    config = load_json(Path(config_path))
    mapping = load_json(Path(class_mapping_path))
    if not config or not mapping:
        raise ValueError("checkpoint 配置或类别映射为空。")
    model = timm.create_model(
        config["backbone"],
        pretrained=False,
        num_classes=int(config["num_classes"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    idx_to_class = {int(index): name for name, index in mapping.items()}
    return model, device, idx_to_class, int(config.get("image_size", 224))


def run_registered_inference(
    image: Image.Image,
    entry: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    from agent.runner import build_transform

    checkpoint = PROJECT_ROOT / entry["checkpoint_path"]
    config = PROJECT_ROOT / entry["config_path"]
    mapping = PROJECT_ROOT / entry["class_mapping_path"]
    model, device, idx_to_class, image_size = load_registered_model(
        str(checkpoint),
        str(config),
        str(mapping),
    )
    tensor = build_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = functional.softmax(model(tensor), dim=1)[0].cpu().tolist()
    display_labels = [
        RAW_TO_DISPLAY.get(idx_to_class[index], idx_to_class[index])
        for index in range(len(probabilities))
    ]
    pred_grade = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    return {
        "labels": display_labels,
        "probabilities": probabilities,
        "pred_grade": pred_grade,
        "source": "在线 checkpoint 推理",
    }


def offline_result_for(image_path: Path) -> dict[str, Any] | None:
    frame = load_offline_predictions()
    if frame.empty:
        return None
    stem = image_path.stem
    match = frame[
        frame["image_path"].astype(str).map(lambda value: Path(value).stem) == stem
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    probabilities = [float(row[f"prob_{label}"]) for label in GRADE_LABELS]
    return {
        "labels": GRADE_LABELS,
        "probabilities": probabilities,
        "pred_grade": int(row["pred_idx"]),
        "source": "已提交的 ConvNeXt-Tiny 测试 prediction record",
    }


def matching_cam(image_path: Path) -> Path | None:
    if not GALLERY_ROOT.exists():
        return None
    matches = list(GALLERY_ROOT.rglob(f"*{image_path.stem}*"))
    return matches[0] if matches else None


def render() -> None:
    page_header(
        "单病例审计",
        "把模型最终判断展开为完整五级概率，形成医生可快速浏览的复核优先级卡。真实标签不参与本页排序。",
        "v0.4.2 推理能力 + v0.7.x 输出审计",
    )

    entry_name = st.selectbox("已验证 checkpoint 注册表", list(CHECKPOINT_REGISTRY))
    entry = CHECKPOINT_REGISTRY[entry_name]
    can_infer, missing_keys = registry_status(entry)
    if can_infer:
        st.success("服务器已找到注册表要求的 checkpoint、配置和类别映射，可按需在线推理。")
    else:
        missing_paths = "、".join(missing_keys)
        st.info(
            f"当前快照缺少 {missing_paths}，页面使用已提交 prediction record 离线展示。"
            "服务器文件齐全后可点击在线推理。"
        )

    input_mode = st.radio(
        "病例来源",
        ["仓库样例", "上传图像"],
        horizontal=True,
    )
    selected_image: Image.Image | None = None
    selected_path: Path | None = None
    uploaded_name = ""
    if input_mode == "仓库样例":
        paths = sample_paths()
        if paths:
            selected_path = st.selectbox(
                "选择眼底图像",
                paths,
                format_func=lambda path: f"{path.parent.name} / {path.name}",
            )
            selected_image = Image.open(selected_path).convert("RGB")
        else:
            render_empty_state("没有仓库样例", f"未在 {DEMO_ROOT} 找到图片。")
    else:
        upload = st.file_uploader("上传 PNG / JPG", type=["png", "jpg", "jpeg"])
        if upload is not None:
            uploaded_name = upload.name
            selected_image = Image.open(upload).convert("RGB")

    if selected_image is None:
        return

    offline_result = offline_result_for(selected_path) if selected_path else None
    run_online = st.button(
        "运行单病例审计",
        type="primary",
        disabled=not can_infer,
        help="只有注册表内且文件完整的 checkpoint 才允许在线推理。",
    )
    result = offline_result
    if run_online:
        with st.spinner("加载 checkpoint 并运行推理..."):
            result = run_registered_inference(selected_image, entry)

    if result is None:
        result = {
            "labels": GRADE_LABELS,
            "probabilities": [0.35, 0.15, 0.30, 0.15, 0.05],
            "pred_grade": 0,
            "source": "教学概率示例，不对应上传图像",
        }

    probabilities = [float(value) for value in result["probabilities"]]
    pred_grade = int(result["pred_grade"])
    order = sorted(range(5), key=probabilities.__getitem__, reverse=True)
    review = summarize_dr_review_priority(
        pred_grade=pred_grade,
        probabilities=probabilities,
    )
    reasons: list[str] = []
    if pred_grade <= 2 and float(review["severe_mass"]) >= 0.15:
        reasons.extend(["预测结果未达重症", "重症类别概率仍偏高"])
    if float(review["margin"]) < 0.15:
        reasons.append("前两类判断接近")
    if float(review["entropy_norm"]) >= 0.60:
        reasons.append("多个类别存在犹豫")
    if not reasons:
        reasons.append("未触发主要输出风险信号")

    image_col, result_col = st.columns([0.8, 1.4], gap="large")
    with image_col:
        section_header("输入图像")
        st.image(selected_image, use_container_width=True)
        st.caption(result["source"])
        if selected_path:
            st.code(selected_path.relative_to(PROJECT_ROOT).as_posix())
        elif uploaded_name:
            st.code(uploaded_name)

    with result_col:
        section_header("模型输出复核卡")
        render_case_card(
            case_id=selected_path.stem if selected_path else uploaded_name or "uploaded_case",
            priority_level=str(review["level"]),
            priority_label=str(review["label"]),
            prediction=f"模型预测：{GRADE_CLINICAL_NAMES[order[0]]}",
            summary=str(review["summary"]),
            action=str(review["action"]),
            reasons=reasons,
        )
        st.caption(
            "这是模型输出审计优先级，不是临床病情分级。医生仍需结合图像质量、病史和检查结果判断。"
        )

    with st.expander("查看模型输出依据", expanded=False):
        render_probability_profile(
            GRADE_DISPLAY_LABELS,
            probabilities,
            pred_grade=pred_grade,
        )
        evidence_cols = st.columns(4, gap="small")
        with evidence_cols[0]:
            metric_card(
                "第一候选",
                GRADE_LABELS[order[0]],
                f"{order[0]}级 · {probabilities[order[0]]:.1%}",
            )
        with evidence_cols[1]:
            metric_card(
                "第二候选",
                GRADE_LABELS[order[1]],
                f"{order[1]}级 · {probabilities[order[1]]:.1%}",
            )
        with evidence_cols[2]:
            metric_card(
                "重症类别概率和",
                f"{float(review['severe_mass']):.1%}",
                "3级与4级概率之和",
                accent="amber",
            )
        with evidence_cols[3]:
            metric_card(
                "期望等级差",
                f"{float(review['expected_gap']):+.2f}",
                "完整概率加权等级 − 第一候选等级",
                accent="amber",
            )
        st.caption(
            "重症类别概率和 = P(Severe NPDR)+P(PDR)；"
            "期望等级差 = 五级概率加权等级 − Top-1 等级。"
        )

    render_boundary(
        "本页主要展示模型输出结构。即使离线记录包含参考标签，主界面也不显示该标签，"
        "以保持预审场景：先排序，后验阶段再评价是否抓到危险事件。"
    )

    section_header("模型追溯")
    metadata_path = PROJECT_ROOT / entry["metadata_path"]
    with st.expander("查看 checkpoint 注册信息与 metadata"):
        st.json(
            {
                "registry": entry,
                "files_complete": can_infer,
                "metadata": load_json(metadata_path),
            }
        )
        st.caption(entry["compatibility"])

    with st.expander("CAM 历史解释模块", expanded=False):
        st.caption("历史解释模块，不参与当前 v0.7.x 主排序结论。")
        cam = matching_cam(selected_path) if selected_path else None
        if cam:
            st.image(str(cam), caption=cam.name, use_container_width=True)
        else:
            render_empty_state(
                "没有匹配 CAM",
                "可使用旧版入口查看完整离线 gallery："
                " streamlit run app/demo_legacy_v0_4_2.py",
            )

    if OFFLINE_PREDICTIONS.exists():
        render_source_caption(OFFLINE_PREDICTIONS.relative_to(PROJECT_ROOT))
