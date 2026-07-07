"""模型中转台共享的中文标签与展示组件。"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from app.ui import COLORS


TASK_LABELS = {
    "aptos_dr_5class": "DR 五级分级",
    "glaucoma_3class": "青光眼三分类",
}

MODEL_LABELS = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_tiny": "Swin-Tiny",
    "vit_b_imagenet": "ViT-B/16（ImageNet）",
    "vit_l_imagenet": "ViT-L/16（ImageNet）",
    "vit_b_official_like": "ViT-B/16（official-like）",
    "retfound_green_linear_probe": "RETFound-Green Linear Probe",
    "retfound_mae_cfp_official_like": "RETFound-MAE（official-like）",
    "retfound_mae_cfp_official_protocol": "RETFound-MAE（官方协议）",
    "convnext_tiny_glaucoma_scout": "青光眼 ConvNeXt-Tiny",
    "retfound_dinov2_glaucoma_expert": "青光眼 RETFound-DINOv2",
}

CANONICAL_BACKBONE_LABELS = {
    "convnext_tiny": "ConvNeXt-Tiny（ImageNet）",
    "swin_tiny": "Swin-Tiny（ImageNet）",
    "vit_b": "ViT-B/16（ImageNet）",
    "vit_l": "ViT-L/16（ImageNet）",
}

FAMILY_LABELS = {
    "convnext": "ConvNeXt",
    "swin": "Swin",
    "vit": "ViT",
    "retfound": "RETFound",
    "mock": "测试模型",
    "other": "其他模型",
}

PRETRAINING_SOURCE_LABELS = {
    "timm_pretrained": "timm 默认自然图像预训练",
    "imagenet": "ImageNet 预训练",
    "imagenet1k": "ImageNet-1K 预训练",
    "imagenet12k_ft_imagenet1k": "ImageNet-12K 预训练 → ImageNet-1K 微调",
    "imagenet21k_ft_imagenet1k": "ImageNet-21K 预训练 → ImageNet-1K 微调",
    "official_protocol": "RETFound 官方协议",
    "official_like": "官方风格复现",
    "retfound_green": "RETFound-Green 预训练",
    "dinov2": "DINOv2 预训练",
    "random_initialization": "随机初始化",
    "unspecified": "来源未登记",
    "unknown": "来源未登记",
    "": "来源未登记",
}

GRADE_LABELS = {
    "aptos_dr_5class": {
        0: "0级 · 未见 DR",
        1: "1级 · 轻度 DR",
        2: "2级 · 中度 DR",
        3: "3级 · 重度 DR",
        4: "4级 · 增殖期 DR",
    },
    "glaucoma_3class": {
        0: "正常对照",
        1: "早期青光眼",
        2: "进展/晚期青光眼",
    },
}


def inject_model_hub_css() -> None:
    st.markdown(
        f"""
        <style>
        .hub-title{{font-size:2rem;font-weight:780;color:{COLORS['navy']};margin:.1rem 0 .3rem;letter-spacing:0}}
        .hub-subtitle{{color:{COLORS['muted']};max-width:1050px;line-height:1.65;margin-bottom:1rem}}
        .hub-strip{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:1rem 0 1.35rem}}
        .hub-stat{{background:#fff;border:1px solid {COLORS['border']};border-top:3px solid {COLORS['teal']};border-radius:6px;padding:.8rem .9rem}}
        .hub-stat span{{display:block;color:{COLORS['muted']};font-size:.76rem}} .hub-stat b{{display:block;color:{COLORS['navy']};font-size:1.5rem;margin-top:.2rem}}
        .hub-band{{background:#fff;border:1px solid {COLORS['border']};border-left:4px solid {COLORS['teal']};padding:.85rem 1rem;border-radius:0 6px 6px 0;line-height:1.65;margin:.4rem 0 1rem}}
        .hub-warning{{background:#fff8e8;border-left:4px solid {COLORS['amber']};padding:.75rem 1rem;color:#5e4617;line-height:1.6}}
        .detail-panel{{background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:1rem}}
        .badge{{display:inline-block;padding:.16rem .48rem;margin:0 .25rem .3rem 0;border-radius:4px;font-size:.72rem;font-weight:650}}
        .badge-live{{background:#e4f5ef;color:#08745f}} .badge-replay{{background:#e8f0fb;color:#245f9e}} .badge-wait{{background:#fff1d8;color:#8a5a0a}}
        .result-card{{background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:.85rem;min-height:92px}}
        .result-card small{{color:{COLORS['muted']}}} .result-card b{{color:{COLORS['navy']};font-size:1.08rem}}
        .proxy-event-cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:.75rem 0}}
        .proxy-event-card{{background:#fff;border:1px solid {COLORS['border']};border-left:4px solid {COLORS['teal']};border-radius:6px;padding:.8rem .9rem}}
        .proxy-event-card-title{{font-weight:750;color:{COLORS['navy']};margin-bottom:.6rem}}
        .proxy-event-flow{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.45rem}}
        .proxy-event-flow span{{background:#f7f9fc;border-radius:5px;padding:.5rem;color:{COLORS['muted']};font-size:.76rem}}
        .proxy-event-flow b{{display:block;color:{COLORS['navy']};font-size:1.15rem;margin-top:.12rem;font-variant-numeric:tabular-nums}}
        .proxy-event-flow small{{display:block;color:{COLORS['muted']};font-size:.72rem}}
        .proxy-event-flow .proxy-residual{{background:#fff1f0;color:#9b2c2c}}
        .proxy-event-flow .proxy-residual b{{color:#9b2c2c}}
        .proxy-table-wrap{{overflow-x:auto;border:1px solid {COLORS['border']};border-radius:6px;background:#fff}}
        .proxy-table{{width:100%;border-collapse:collapse;font-size:.9rem}}
        .proxy-table th{{text-align:left;color:{COLORS['muted']};font-weight:650;background:#f7f9fc;padding:.62rem .75rem;border-bottom:1px solid {COLORS['border']}}}
        .proxy-table td{{padding:.62rem .75rem;border-bottom:1px solid {COLORS['border']}}}
        .proxy-table tbody tr:last-child td{{border-bottom:0}}
        .proxy-event-name{{color:{COLORS['navy']};font-weight:650;white-space:nowrap}}
        .proxy-event-value{{text-align:right;font-variant-numeric:tabular-nums;color:{COLORS['text']}}}
        .proxy-help-icon{{display:inline-flex;align-items:center;justify-content:center;width:1.05rem;height:1.05rem;margin-left:.25rem;border:1px solid {COLORS['muted']};border-radius:50%;color:{COLORS['muted']};font-size:.72rem;font-weight:750;cursor:help;vertical-align:middle}}
        .proxy-help-icon:focus{{outline:2px solid {COLORS['teal']};outline-offset:2px}}
        [data-testid="stMetric"]{{background:#fff;border:1px solid {COLORS['border']};padding:.65rem .8rem;border-radius:6px}}
        @media(max-width:760px){{.hub-strip{{grid-template-columns:1fr 1fr}}.proxy-event-cards{{grid-template-columns:1fr}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def title_block() -> None:
    st.markdown('<div class="hub-title">眼科模型中转台</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hub-subtitle">发现并接入服务器模型，提交适配或训练任务，比较路由模型与专家模型组合，并查看病例路由解释。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hub-warning"><strong>系统边界：</strong>模型工程与研究评测服务于研发验证；病例回放仅展示模型输出，不提供诊断或患者分流决定。</div>',
        unsafe_allow_html=True,
    )


def human_model(value: Any) -> str:
    model_id = str(value)
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    for backbone, label in CANONICAL_BACKBONE_LABELS.items():
        prefix = f"{backbone}_imagenet_"
        if model_id.startswith(prefix) and model_id.endswith("_adapter"):
            task_id = model_id[len(prefix) : -len("_adapter")]
            return f"{label}· {task_label(task_id)}"
    return model_id


def human_family(value: Any) -> str:
    return FAMILY_LABELS.get(str(value).lower(), str(value))


def human_pretraining_source(value: Any) -> str:
    text = str(value or "").strip()
    return PRETRAINING_SOURCE_LABELS.get(text.lower(), text or "来源未登记")


def task_label(value: Any) -> str:
    return TASK_LABELS.get(str(value), str(value))


def grade_label(task_id: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "无结果"
    numeric = int(value)
    return GRADE_LABELS.get(task_id, {}).get(numeric, f"类别 {numeric}")


def source_status(row: pd.Series) -> tuple[str, str]:
    if str(row.get("prediction_source")) in {"adapter", "checkpoint_generated"} and str(
        row.get("adapter_status")
    ) == "completed":
        return "在线推理链已验证", "badge-live"
    if str(row.get("prediction_source")) == "legacy":
        return "仅离线结果可回放", "badge-replay"
    return "暂无可用预测结果", "badge-wait"


def stat_strip(models: pd.DataFrame, pairings: pd.DataFrame) -> None:
    ready = models["compatibility_status"].astype(str).eq("ready_for_pairing")
    values = [
        ("可回放模型", int(ready.sum())),
        ("在线链已验证", int(models["adapter_status"].astype(str).eq("completed").sum())),
        ("受控组合", pairings.loc[pairings["status"].astype(str).eq("completed"), "pairing_id"].nunique()),
        ("已登记任务", models["task_id"].nunique()),
    ]
    cards = "".join(
        f'<div class="hub-stat"><span>{html.escape(label)}</span><b>{value}</b></div>'
        for label, value in values
    )
    st.markdown(f'<div class="hub-strip">{cards}</div>', unsafe_allow_html=True)
