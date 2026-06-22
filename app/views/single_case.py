"""单病例审计页。"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from app.audit_core import summarize_dr_review_priority
from app.checkpoints import (
    ModelArtifact,
    compute_file_sha256,
    discover_model_artifacts,
    resolve_capabilities,
    select_preferred_artifacts,
    summarize_frozen_model_finding,
)
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
FINDING_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "summary"
    / "v0_6_7c"
    / "unified_ranking_method_tradeoff.csv"
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

@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    return pd.read_csv(source)


@st.cache_data(show_spinner=False)
def load_artifact_registry() -> dict[str, ModelArtifact]:
    return select_preferred_artifacts(discover_model_artifacts(PROJECT_ROOT))


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


def load_registered_model(
    artifact: ModelArtifact,
):
    import timm
    import torch

    if not artifact.can_attempt_load:
        raise RuntimeError("当前模型产物不满足在线加载条件。")
    assert artifact.checkpoint_path is not None
    assert artifact.config_path is not None
    assert artifact.class_to_idx_path is not None
    cache_key = (
        artifact.model_key,
        str(artifact.checkpoint_path),
        artifact.checkpoint_mtime_ns,
    )
    active = st.session_state.get("_ophagent_active_model")
    if active and active.get("key") == cache_key:
        return (
            active["model"],
            active["device"],
            active["idx_to_class"],
            active["image_size"],
        )
    if active:
        try:
            active["model"].to("cpu")
        except Exception:
            pass
        st.session_state.pop("_ophagent_active_model", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = load_json(artifact.config_path)
    mapping = load_json(artifact.class_to_idx_path)
    if not config or not mapping:
        raise ValueError("checkpoint 配置或类别映射为空。")
    model = timm.create_model(
        artifact.loader_model_name,
        pretrained=False,
        num_classes=int(config["num_classes"]),
        drop_path_rate=float(config.get("drop_path", 0.0)),
    )
    checkpoint = torch.load(
        artifact.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    model.load_state_dict(checkpoint, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    idx_to_class = {int(index): name for name, index in mapping.items()}
    image_size = int(config.get("image_size", 224))
    st.session_state["_ophagent_active_model"] = {
        "key": cache_key,
        "model": model,
        "device": device,
        "idx_to_class": idx_to_class,
        "image_size": image_size,
    }
    return model, device, idx_to_class, image_size


def run_registered_inference(
    image: Image.Image,
    artifact: ModelArtifact,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    from agent.runner import build_transform

    model, device, idx_to_class, image_size = load_registered_model(artifact)
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
        "source": f"{artifact.display_name} 在线 checkpoint 推理",
    }


def offline_result_for(
    image_path: Path,
    artifact: ModelArtifact,
) -> dict[str, Any] | None:
    if artifact.test_predictions_path is None:
        return None
    frame = load_predictions(str(artifact.test_predictions_path))
    if frame.empty:
        return None
    stem = image_path.stem
    match = frame[
        frame["image_path"].astype(str).map(lambda value: Path(value).stem) == stem
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    named_columns = [f"prob_{label}" for label in GRADE_LABELS]
    numeric_columns = [f"prob_{grade}" for grade in range(5)]
    columns = (
        named_columns
        if all(column in row.index for column in named_columns)
        else numeric_columns
    )
    if not all(column in row.index for column in columns):
        return None
    probabilities = [float(row[column]) for column in columns]
    pred_column = "pred_idx" if "pred_idx" in row.index else "pred_grade"
    return {
        "labels": GRADE_LABELS,
        "probabilities": probabilities,
        "pred_grade": int(row[pred_column]),
        "source": f"{artifact.display_name} 已提交的测试 prediction record",
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

    registry = load_artifact_registry()
    if not registry:
        render_empty_state(
            "未发现已知 APTOS 模型产物",
            "只扫描六个已验证实验根目录；不会自动扫描仓库其他 checkpoint。",
        )
        return
    model_keys = list(registry)
    if st.session_state.get("selected_backbone") not in model_keys:
        st.session_state["selected_backbone"] = model_keys[0]
    selected_model = st.selectbox(
        "当前模型",
        model_keys,
        key="selected_backbone",
        format_func=lambda key: registry[key].display_name,
    )
    artifact = registry[selected_model]
    capabilities = resolve_capabilities(
        artifact.protocol_id,
        artifact.prediction_columns,
    )
    if artifact.can_attempt_load:
        st.success(
            "模型文件静态完整，存在已知加载映射；点击“运行单病例审计”后才会加载权重。"
        )
    elif artifact.test_predictions_path is not None:
        st.info(
            "当前快照未发现可加载 checkpoint，保留该模型的离线 prediction record 展示。"
        )
    else:
        st.info(
            "当前模型缺少在线权重和离线 prediction record，相关模块将显示空状态。"
        )
    load_state = st.session_state.get("_ophagent_model_load_state", {}).get(
        selected_model
    )
    if load_state == "verified":
        st.success("本会话加载验证成功。")
    elif isinstance(load_state, str) and load_state.startswith("failed:"):
        st.error(
            "加载失败，已保留离线预测展示；错误摘要："
            + load_state.removeprefix("failed:")
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

    offline_result = (
        offline_result_for(selected_path, artifact) if selected_path else None
    )
    run_online = st.button(
        "运行单病例审计",
        type="primary",
        disabled=not artifact.can_attempt_load,
        help="自动发现阶段不加载模型；只有点击后才尝试加载当前 checkpoint。",
    )
    result = offline_result
    if run_online:
        with st.spinner("加载 checkpoint 并运行推理..."):
            try:
                result = run_registered_inference(selected_image, artifact)
            except Exception as exc:
                states = dict(
                    st.session_state.get("_ophagent_model_load_state", {})
                )
                states[selected_model] = f"failed:{type(exc).__name__}: {exc}"
                st.session_state["_ophagent_model_load_state"] = states
                st.error(
                    "当前模型加载或推理失败，页面继续使用可用的离线预测记录。"
                )
                result = offline_result
            else:
                states = dict(
                    st.session_state.get("_ophagent_model_load_state", {})
                )
                states[selected_model] = "verified"
                st.session_state["_ophagent_model_load_state"] = states

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
        if result is None:
            render_empty_state(
                "当前模型没有该病例结果",
                "未发现匹配的离线 prediction record，且当前 checkpoint 未成功运行。"
                "不会回退到 ConvNeXt 或教学概率。",
            )
        else:
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
            render_case_card(
                case_id=(
                    selected_path.stem
                    if selected_path
                    else uploaded_name or "uploaded_case"
                ),
                priority_level=str(review["level"]),
                priority_label=str(review["label"]),
                prediction=f"模型预测：{GRADE_CLINICAL_NAMES[order[0]]}",
                summary=str(review["summary"]),
                action=str(review["action"]),
                reasons=reasons,
            )
            st.caption(
                f"当前模型：{artifact.display_name}。这是模型输出审计优先级，"
                "不是临床病情分级。"
            )

    if result is not None:
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
    status_labels = {
        "static_complete": "静态完整",
        "inference_only": "可尝试加载，批量证据不完整",
        "offline_only": "仅可离线展示",
        "artifact_missing": "文件不完整",
        "checkpoint_ambiguous": "权重存在歧义",
    }
    trace_cols = st.columns(3, gap="small")
    with trace_cols[0]:
        metric_card(
            "产物状态",
            status_labels.get(artifact.artifact_status, artifact.artifact_status),
            artifact.experiment_dir.name,
        )
    with trace_cols[1]:
        metric_card(
            "通用概率审计",
            "可计算" if capabilities["supports_probability_audit"] else "不可计算",
            artifact.protocol_id,
        )
    with trace_cols[2]:
        metric_card(
            "DR 等级审计",
            "可计算" if capabilities["supports_ordinal_dr_audit"] else "不适用",
            "由显式协议决定，不按类别数量猜测",
        )
    if st.session_state.get("display_mode") == "研究审计":
        with st.expander("查看 checkpoint / artifact 注册信息"):
            st.json(artifact.to_dict())
            metadata = load_json(artifact.checkpoint_meta_path) if artifact.checkpoint_meta_path else {}
            if metadata:
                st.json(metadata)
            if artifact.checkpoint_path and st.button(
                "按需计算 checkpoint SHA256",
                key=f"sha_{artifact.model_key}",
            ):
                st.code(compute_file_sha256(artifact.checkpoint_path))
        finding = summarize_frozen_model_finding(
            FINDING_PATH,
            backbone=artifact.model_key,
            event="vision_threatening_dr_miss",
            method="gated_severe_prob_mass_only",
        )
        with st.expander("查看当前模型的冻结 finding 事实"):
            st.json(finding)
            st.caption(
                "只报告 v0.6.7c 中的排名、Top20% 捕获率和相对最佳比较方法差值，"
                "不自动解释为科研发现成立。"
            )

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

    if artifact.test_predictions_path is not None:
        render_source_caption(artifact.test_predictions_path.relative_to(PROJECT_ROOT))
