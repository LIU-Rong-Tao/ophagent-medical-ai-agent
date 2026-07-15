"""模型中转台共享的中文标签与展示组件。"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from app.ui import COLORS


HUB_WORKSPACES = {
    "中转台总览": {
        "icon": ":material/dashboard:",
        "group": "workspace",
        "title": "眼科模型中转台",
        "subtitle": "统一查看模型资产、任务接入、研究评测与病例回放状态。",
    },
    "模型资产": {
        "icon": ":material/inventory_2:",
        "group": "workspace",
        "title": "模型资产目录",
        "subtitle": "从模型资产出发，核对来源、Checkpoint、Adapter 与当前任务兼容性。",
    },
    "任务模型": {
        "icon": ":material/model_training:",
        "group": "workspace",
        "title": "任务模型与接入状态",
        "subtitle": "区分离线回放、任务推理与路由资格，避免把“已登记”误认为“可用”。",
    },
    "研究评测": {
        "icon": ":material/query_stats:",
        "group": "workspace",
        "title": "研究评测",
        "subtitle": "比较路由组合，或导入预测结果表开展模型输出错误风险审计。",
    },
    "病例回放": {
        "icon": ":material/clinical_notes:",
        "group": "workspace",
        "title": "病例回放与路由解释",
        "subtitle": "回放模型调用轨迹与路由依据；研究审计标签默认与模型输出隔离。",
    },
    "任务运行记录": {
        "icon": ":material/history:",
        "group": "operations",
        "title": "任务运行记录",
        "subtitle": "查看后台训练与全局扫描任务的状态、曲线、日志和本地产物。",
    },
}


TASK_LABELS = {
    "aptos_dr_5class": "DR 五级分级",
    "glaucoma_3class": "青光眼三分类",
}

MODEL_LABELS = {
    "aptos2019-retfound-cfp-linear-probe-v1": "RETFound CFP · APTOS DR 五分类",
    "aptos2019-retfound-cfp-linear-probe-v2": "RETFound CFP · APTOS DR 五分类 · 标准线性探针 v2",
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


OPHAGENT_MARK_SVG = """
<svg class="hub-brand-svg" viewBox="0 0 128 128" role="img" aria-label="OphAgent Retina Router">
  <defs>
    <linearGradient id="ophagent-mark-bg" x1="18" y1="14" x2="112" y2="118" gradientUnits="userSpaceOnUse">
      <stop stop-color="#1D4ED8"/>
      <stop offset=".52" stop-color="#0F766E"/>
      <stop offset="1" stop-color="#14B8A6"/>
    </linearGradient>
    <radialGradient id="ophagent-mark-glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(82 34) rotate(120) scale(66)">
      <stop stop-color="#FFFFFF" stop-opacity=".55"/>
      <stop offset=".42" stop-color="#FFFFFF" stop-opacity=".08"/>
      <stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="ophagent-mark-eye" x1="24" y1="58" x2="101" y2="72" gradientUnits="userSpaceOnUse">
      <stop stop-color="#E0F2FE"/>
      <stop offset=".5" stop-color="#FFFFFF"/>
      <stop offset="1" stop-color="#CCFBF1"/>
    </linearGradient>
  </defs>
  <rect x="10" y="10" width="108" height="108" rx="28" fill="url(#ophagent-mark-bg)"/>
  <rect x="10" y="10" width="108" height="108" rx="28" fill="url(#ophagent-mark-glow)"/>
  <path d="M25 65C34 50 47 42 64 42C81 42 94 50 103 65C94 80 81 88 64 88C47 88 34 80 25 65Z" fill="url(#ophagent-mark-eye)" fill-opacity=".96"/>
  <path d="M34 65C42 55 52 50 64 50C76 50 86 55 94 65C86 75 76 80 64 80C52 80 42 75 34 65Z" fill="#0F172A" fill-opacity=".14"/>
  <circle cx="64" cy="65" r="15" fill="#0F172A" fill-opacity=".76"/>
  <circle cx="64" cy="65" r="8" fill="#5EEAD4"/>
  <circle cx="60" cy="61" r="3.4" fill="#FFFFFF" fill-opacity=".88"/>
  <path d="M78 52H96C101 52 104 55 104 60V62" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round"/>
  <path d="M78 78H96C101 78 104 75 104 70V68" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round"/>
  <path d="M80 65H108" stroke="#FFFFFF" stroke-width="5" stroke-linecap="round"/>
  <circle cx="104" cy="60" r="6" fill="#F59E0B" stroke="#FFFFFF" stroke-width="3"/>
  <circle cx="108" cy="65" r="6" fill="#60A5FA" stroke="#FFFFFF" stroke-width="3"/>
  <circle cx="104" cy="70" r="6" fill="#34D399" stroke="#FFFFFF" stroke-width="3"/>
  <path d="M33 50C41 33 55 25 74 24" stroke="#A7F3D0" stroke-width="4" stroke-linecap="round" opacity=".7"/>
  <path d="M31 82C43 98 59 104 80 101" stroke="#BFDBFE" stroke-width="4" stroke-linecap="round" opacity=".68"/>
</svg>
"""


def inject_model_hub_css() -> None:
    st.markdown(
        f"""
        <style>
        .hub-brand{{display:flex;align-items:center;gap:.85rem;margin:.1rem 0 .35rem}}
        .hub-brand-mark{{width:3.1rem;height:3.1rem;flex:0 0 auto;border-radius:12px;filter:drop-shadow(0 14px 22px rgba(15,23,42,.16))}}
        .hub-brand-svg{{display:block;width:100%;height:100%}}
        .hub-kicker{{color:{COLORS['teal']};font-size:.76rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin-bottom:.12rem}}
        .hub-title{{font-size:2rem;font-weight:780;color:{COLORS['navy']};margin:0;letter-spacing:0;line-height:1.18}}
        .hub-subtitle{{color:{COLORS['muted']};max-width:1050px;line-height:1.65;margin-bottom:1rem}}
        .hub-strip{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:1rem 0 1.35rem}}
        .hub-stat{{background:#fff;border:1px solid {COLORS['border']};border-top:3px solid {COLORS['teal']};border-radius:6px;padding:.8rem .9rem}}
        .hub-stat span{{display:block;color:{COLORS['muted']};font-size:.76rem}} .hub-stat b{{display:block;color:{COLORS['navy']};font-size:1.5rem;margin-top:.2rem}}
        .hub-mini-strip{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.65rem;margin:.75rem 0 1rem}}
        .hub-mini-stat{{background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:.65rem .75rem;min-height:74px}}
        .hub-mini-stat span{{display:block;color:{COLORS['muted']};font-size:.72rem;font-weight:750;line-height:1.35}}
        .hub-mini-stat b{{display:block;color:{COLORS['navy']};font-size:1.28rem;line-height:1.2;margin-top:.18rem;font-variant-numeric:tabular-nums}}
        .hub-mini-stat small{{display:block;color:{COLORS['muted']};font-size:.7rem;line-height:1.3;margin-top:.1rem}}
        .hub-band{{background:#fff;border:1px solid {COLORS['border']};border-left:4px solid {COLORS['teal']};padding:.85rem 1rem;border-radius:0 6px 6px 0;line-height:1.65;margin:.4rem 0 1rem}}
        .hub-warning{{background:#fff8e8;border-left:4px solid {COLORS['amber']};padding:.75rem 1rem;color:#5e4617;line-height:1.6}}
        .detail-panel{{background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:1rem}}
        .badge{{display:inline-block;padding:.16rem .48rem;margin:0 .25rem .3rem 0;border-radius:4px;font-size:.72rem;font-weight:650}}
        .badge-live{{background:#e4f5ef;color:#08745f}} .badge-replay{{background:#e8f0fb;color:#245f9e}} .badge-wait{{background:#fff1d8;color:#8a5a0a}}
        .hub-chip{{display:inline-flex;align-items:center;gap:.35rem;padding:.16rem .5rem;border-radius:999px;font-size:.72rem;font-weight:750;background:#eef2f7;color:#475569}}
        .hub-chip-teal{{background:#dff7f3;color:{COLORS['teal']}}}
        .hub-chip-blue{{background:#e8f0ff;color:#2563eb}}
        .hub-chip-amber{{background:#fff4dc;color:#b45309}}
        .hub-chip-red{{background:#fff1f0;color:{COLORS['red']}}}
        .hub-selection-note{{display:flex;gap:.55rem;align-items:flex-start;background:#fff7ed;border:1px solid #fed7aa;color:#7c2d12;border-radius:6px;padding:.55rem .65rem;font-size:.78rem;line-height:1.45;margin:.55rem 0;min-height:3.2rem}}
        .hub-selection-note-muted{{background:#f8fafc;border-color:#e6edf5;color:#64748b}}
        .hub-selection-note b{{white-space:nowrap}}
        .hub-card-reason{{background:#f8fafc;border:1px solid #e6edf5;border-radius:6px;padding:.62rem .7rem;color:#465569;font-size:.82rem;line-height:1.5;margin:.55rem 0;min-height:4.2rem}}
        .hub-card-formula{{font-size:.78rem;color:#64748b;line-height:1.55;margin:.25rem 0 .45rem}}
        .hub-formula-line{{display:flex;align-items:center;gap:.42rem;flex-wrap:wrap;margin:.55rem 0 .45rem;color:#475569;font-size:.8rem;line-height:1.5}}
        .hub-formula-label{{font-weight:750;color:{COLORS['navy']}}}
        .hub-formula-pill{{display:inline-flex;align-items:center;gap:.18rem;background:#f8fafc;border:1px solid #e6edf5;border-radius:999px;padding:.12rem .42rem;color:#475569;font-weight:650}}
        .hub-audit-score{{display:flex;gap:.5rem;align-items:center;background:#ecfeff;border:1px solid #bae6fd;color:#155e75;border-radius:6px;padding:.55rem .65rem;font-size:.8rem;line-height:1.45;margin:.55rem 0}}
        .hub-audit-score span{{display:inline-flex;align-items:center}}
        .hub-audit-score b{{font-variant-numeric:tabular-nums;color:#0f766e}}
        .result-card{{background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:.85rem;min-height:92px}}
        .result-card small{{color:{COLORS['muted']}}} .result-card b{{color:{COLORS['navy']};font-size:1.08rem}}
        .case-result-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem;margin:.4rem 0 .8rem}}
        .case-result-card{{background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:.75rem;min-height:104px}}
        .case-result-card small{{display:block;color:{COLORS['muted']};font-weight:700;margin-bottom:.35rem}}
        .case-result-card b{{display:block;color:{COLORS['navy']};font-size:1.02rem;line-height:1.35}}
        .case-result-card span{{display:block;color:{COLORS['muted']};font-size:.74rem;margin-top:.35rem;line-height:1.35}}
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
        .hub-help-icon,.proxy-help-icon{{display:inline-flex;align-items:center;justify-content:center;width:1.05rem;height:1.05rem;margin-left:.2rem;border:2px solid #6b7280;border-radius:50%;color:#6b7280;font-size:.72rem;font-weight:800;line-height:1;cursor:help;vertical-align:middle;background:transparent}}
        .hub-help-icon:focus,.proxy-help-icon:focus{{outline:2px solid {COLORS['teal']};outline-offset:2px}}
        .model-entry-row{{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,.9fr) minmax(0,1.15fr) auto;gap:.75rem;align-items:center;background:#fff;border:1px solid {COLORS['border']};border-radius:6px;padding:.85rem .9rem;margin:.55rem 0}}
        .model-entry-label{{display:block;color:{COLORS['muted']};font-size:.72rem;font-weight:750;margin-bottom:.2rem}}
        .model-entry-title{{display:block;color:{COLORS['navy']};font-weight:780;line-height:1.35}}
        .model-entry-copy{{display:block;color:{COLORS['muted']};font-size:.78rem;line-height:1.4;margin-top:.1rem}}
        .model-detail-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55rem;margin:.75rem 0}}
        .model-detail-cell{{background:#f8fafc;border:1px solid #e6edf5;border-radius:6px;padding:.55rem .65rem;min-height:64px}}
        .model-detail-cell span{{display:block;color:{COLORS['muted']};font-size:.72rem;font-weight:750;line-height:1.35}}
        .model-detail-cell b{{display:block;color:{COLORS['navy']};font-size:.86rem;line-height:1.35;margin-top:.18rem;word-break:break-word}}
        .model-readiness-section{{margin-top:.9rem;padding-top:.85rem;border-top:1px solid #e6edf5}}
        .model-readiness-section:first-of-type{{margin-top:.7rem}}
        .model-readiness-section h4{{margin:0 0 .45rem;color:{COLORS['navy']};font-size:.86rem;line-height:1.35}}
        .case-list-note{{background:#fff;border:1px solid {COLORS['border']};border-left:4px solid {COLORS['teal']};border-radius:0 6px 6px 0;padding:.65rem .8rem;color:{COLORS['muted']};line-height:1.55;margin:.45rem 0 .9rem}}
        [data-testid="stMetric"]{{background:#fff;border:1px solid {COLORS['border']};padding:.65rem .8rem;border-radius:6px}}

        /* v0.8.9 hybrid shell: C navigation, A tables/status, B research surfaces. */
        :root{{--hub-ink:#142033;--hub-muted:#657286;--hub-line:#dce3eb;--hub-soft:#f4f7fa;--hub-teal:#0f766e;--hub-blue:#245f9e;--hub-amber:#b7791f;--hub-red:#b42318}}
        [data-testid="stAppViewContainer"]{{background:#f4f7fa}}
        .main .block-container{{max-width:1480px;padding:1.65rem 2rem 4rem}}
        [data-testid="stSidebar"]{{background:#102033;border-right:1px solid #20344b}}
        [data-testid="stSidebar"]>div:first-child{{padding-top:1rem}}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{{color:#aebccc}}
        [data-testid="stSidebar"] [data-testid="stButton"] button{{justify-content:flex-start;border-radius:5px;border-color:transparent;background:transparent;color:#dbe6f1;box-shadow:none;min-height:2.5rem}}
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover{{background:#1a3047;border-color:#29435e;color:#fff}}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{{background:#e8f5f2;color:#0b655f;border-left:3px solid #22a699;font-weight:750}}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] *{{color:#0b655f!important}}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] *{{color:#dbe6f1!important}}
        .hub-sidebar-brand{{display:flex;align-items:center;gap:.65rem;padding:.2rem .1rem 1rem;border-bottom:1px solid rgba(255,255,255,.12);margin-bottom:.8rem}}
        .hub-sidebar-brand .hub-brand-mark{{width:2.35rem;height:2.35rem;border-radius:9px;filter:none}}
        .hub-sidebar-brand b{{display:block;color:#fff;font-size:1rem;line-height:1.2}}
        .hub-sidebar-brand span{{display:block;color:#90a6bc;font-size:.72rem;margin-top:.18rem}}
        .hub-nav-label{{color:#7990a8;font-size:.67rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin:.8rem .2rem .35rem}}
        .hub-sidebar-boundary{{margin-top:1rem;padding:.7rem .75rem;border:1px solid rgba(148,163,184,.2);border-radius:5px;color:#9fb0c1;font-size:.7rem;line-height:1.5;background:rgba(255,255,255,.025)}}
        .hub-page-head{{display:flex;align-items:flex-end;justify-content:space-between;gap:1.2rem;padding-bottom:1rem;margin-bottom:1.05rem;border-bottom:1px solid var(--hub-line)}}
        .hub-page-eyebrow{{color:var(--hub-teal);font-size:.7rem;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin-bottom:.28rem}}
        .hub-page-title{{font-size:1.85rem!important;font-weight:780;color:var(--hub-ink);line-height:1.2;margin:0;letter-spacing:0!important}}
        .hub-page-copy{{color:var(--hub-muted);font-size:.86rem;line-height:1.55;margin-top:.35rem;max-width:920px}}
        .hub-page-context{{white-space:nowrap;color:#5b687a;font-size:.73rem;background:#fff;border:1px solid var(--hub-line);border-radius:4px;padding:.35rem .55rem}}
        .hub-boundary-compact{{display:flex;gap:.6rem;align-items:flex-start;background:#fdf8ec;border:1px solid #ead7aa;border-left:4px solid var(--hub-amber);padding:.65rem .8rem;color:#624d1d;line-height:1.5;font-size:.78rem;margin:0 0 1rem}}
        .hub-overview-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin:.7rem 0 1.1rem}}
        .hub-overview-kpi{{background:#fff;border:1px solid var(--hub-line);border-radius:5px;padding:.85rem .95rem;min-height:98px}}
        .hub-overview-kpi span{{display:block;color:var(--hub-muted);font-size:.72rem;font-weight:700}}
        .hub-overview-kpi b{{display:block;color:var(--hub-ink);font-size:1.55rem;line-height:1.2;margin:.25rem 0;font-variant-numeric:tabular-nums}}
        .hub-overview-kpi small{{display:block;color:#7c8797;font-size:.7rem;line-height:1.35}}
        .hub-section{{margin:1.25rem 0 .6rem}}
        .hub-section h3{{font-size:1rem;color:var(--hub-ink);margin:0 0 .2rem}}
        .hub-section p{{font-size:.76rem;color:var(--hub-muted);margin:0;line-height:1.5}}
        .hub-flow{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));background:#fff;border:1px solid var(--hub-line);border-radius:5px;overflow:hidden;margin:.6rem 0 1rem}}
        .hub-flow-step{{padding:.85rem .9rem;border-right:1px solid var(--hub-line);min-height:96px}}
        .hub-flow-step:last-child{{border-right:0}}
        .hub-flow-step i{{display:flex;align-items:center;justify-content:center;width:1.4rem;height:1.4rem;border-radius:50%;background:#e5f3f1;color:var(--hub-teal);font-style:normal;font-size:.7rem;font-weight:800;margin-bottom:.45rem}}
        .hub-flow-step b{{display:block;color:var(--hub-ink);font-size:.83rem}}
        .hub-flow-step span{{display:block;color:var(--hub-muted);font-size:.72rem;line-height:1.45;margin-top:.2rem}}
        .hub-panel{{background:#fff;border:1px solid var(--hub-line);border-radius:5px;padding:1rem}}
        .hub-process{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;margin:.25rem 0 1rem;background:#fff;border:1px solid var(--hub-line);border-radius:5px;overflow:hidden}}
        .hub-process-item{{position:relative;padding:.65rem .75rem .65rem 2.2rem;color:#708094;font-size:.72rem;border-right:1px solid var(--hub-line)}}
        .hub-process-item:last-child{{border-right:0}}
        .hub-process-item b{{position:absolute;left:.65rem;top:.55rem;display:flex;align-items:center;justify-content:center;width:1.1rem;height:1.1rem;border-radius:50%;background:#e8edf3;color:#596779;font-size:.62rem}}
        .hub-process-item.active{{background:#edf8f6;color:#0b655f;font-weight:750}}
        .hub-process-item.active b{{background:var(--hub-teal);color:#fff}}
        .hub-audit-kpis{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:.65rem;margin:.75rem 0 1rem}}
        .hub-audit-kpis.compact{{grid-template-columns:repeat(4,minmax(0,1fr))}}
        .hub-audit-kpi{{background:#fff;border:1px solid var(--hub-line);border-top:3px solid #8ca0b5;border-radius:5px;padding:.72rem .78rem;min-height:88px}}
        .hub-audit-kpi span{{display:block;color:var(--hub-muted);font-size:.7rem;font-weight:700}}
        .hub-audit-kpi b{{display:block;color:var(--hub-ink);font-size:1.38rem;margin-top:.25rem;font-variant-numeric:tabular-nums}}
        .hub-audit-kpi small{{display:block;color:#7c8797;font-size:.68rem;line-height:1.35;margin-top:.18rem}}
        .hub-audit-kpi.error{{border-top-color:var(--hub-amber)}}
        .hub-audit-kpi.error b{{color:#9a6700}}
        .hub-audit-kpi.severe{{border-top-color:var(--hub-red)}}
        .hub-audit-kpi.severe b{{color:var(--hub-red)}}
        .hub-audit-kpi.success{{border-top-color:var(--hub-teal)}}
        .hub-audit-kpi.unknown{{border-top-color:#94a3b8}}
        .hub-class-key{{display:flex;flex-wrap:wrap;gap:.45rem .85rem;align-items:center;background:#f7fafc;border:1px solid var(--hub-line);border-radius:5px;padding:.65rem .75rem;margin:.55rem 0 .8rem;color:var(--hub-muted);font-size:.74rem}}
        .hub-class-key>strong{{color:var(--hub-ink);margin-right:.2rem}}
        .hub-class-key span{{display:inline-flex;gap:.3rem;align-items:baseline}}
        .hub-class-key span b{{color:var(--hub-teal);font-variant-numeric:tabular-nums}}
        [data-testid="stDataFrame"]{{border:1px solid var(--hub-line);border-radius:4px;overflow:hidden;background:#fff}}
        [data-testid="stTabs"] [data-baseweb="tab-list"]{{gap:1.15rem;border-bottom:1px solid var(--hub-line)}}
        [data-testid="stTabs"] [data-baseweb="tab"]{{padding:.65rem .05rem;border-radius:0;color:#607085}}
        [data-testid="stTabs"] [aria-selected="true"]{{color:var(--hub-teal);font-weight:750}}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"]{{background:var(--hub-teal)}}
        [data-testid="stSegmentedControl"]{{margin-bottom:.5rem}}
        [data-testid="stExpander"]{{border-color:var(--hub-line);border-radius:5px;background:#fff}}
        [data-testid="stFileUploader"] section{{border:1px dashed #8fa2b6;border-radius:5px;background:#fbfcfe}}
        div[data-testid="stAlert"]{{border-radius:4px}}
        @media(max-width:1100px){{.hub-audit-kpis{{grid-template-columns:repeat(3,minmax(0,1fr))}}.hub-audit-kpis.compact{{grid-template-columns:repeat(2,minmax(0,1fr))}}.hub-overview-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.hub-flow{{grid-template-columns:repeat(2,minmax(0,1fr))}}.hub-flow-step:nth-child(2){{border-right:0}}}}
        @media(max-width:900px){{.hub-page-title{{font-size:1.65rem!important}}.hub-page-copy{{font-size:.8rem}}}}
        @media(max-width:760px){{.main .block-container{{padding:1.15rem .9rem 3rem}}.hub-page-head{{align-items:flex-start;flex-direction:column}}.hub-page-context{{white-space:normal}}.hub-brand{{align-items:flex-start}}.hub-strip,.hub-mini-strip,.hub-overview-grid,.hub-audit-kpis,.hub-audit-kpis.compact{{grid-template-columns:1fr 1fr}}.hub-flow,.hub-process{{grid-template-columns:1fr}}.hub-flow-step,.hub-process-item{{border-right:0;border-bottom:1px solid var(--hub-line)}}.proxy-event-cards{{grid-template-columns:1fr}}.case-result-grid{{grid-template-columns:1fr}}.model-entry-row,.model-detail-grid{{grid-template-columns:1fr}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_navigation() -> str:
    pending = st.session_state.pop("pending_hub_workspace", None)
    if pending in HUB_WORKSPACES:
        st.session_state["hub_primary_workspace"] = pending
    current = str(st.session_state.get("hub_primary_workspace", "中转台总览"))
    if current not in HUB_WORKSPACES:
        current = "中转台总览"
        st.session_state["hub_primary_workspace"] = current

    st.sidebar.markdown(
        '<div class="hub-sidebar-brand">'
        f'<div class="hub-brand-mark">{OPHAGENT_MARK_SVG}</div>'
        '<div><b>OphAgent</b><span>Model Hub · 研究版</span></div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="hub-nav-label">模型工程与研究工作区</div>', unsafe_allow_html=True)
    for name, config in HUB_WORKSPACES.items():
        if config["group"] != "workspace":
            continue
        if st.sidebar.button(
            name,
            icon=str(config["icon"]),
            type="primary" if name == current else "secondary",
            width="stretch",
            key=f"hub_nav::{name}",
        ):
            st.session_state["hub_primary_workspace"] = name
            st.rerun()
    st.sidebar.markdown('<div class="hub-nav-label">运行与产物</div>', unsafe_allow_html=True)
    operations = HUB_WORKSPACES["任务运行记录"]
    if st.sidebar.button(
        "任务运行记录",
        icon=str(operations["icon"]),
        type="primary" if current == "任务运行记录" else "secondary",
        width="stretch",
        key="hub_nav::任务运行记录",
    ):
        st.session_state["hub_primary_workspace"] = "任务运行记录"
        st.rerun()
    st.sidebar.markdown(
        '<div class="hub-sidebar-boundary"><strong>研究使用边界</strong><br>'
        '本系统为研究演示 Demo，不提供任何诊断、治疗或患者分流建议。</div>',
        unsafe_allow_html=True,
    )
    return current


def page_header(workspace: str, *, context: str = "v0.8.9 UI refresh") -> None:
    config = HUB_WORKSPACES[workspace]
    st.markdown(
        '<div class="hub-page-head"><div>'
        '<div class="hub-page-eyebrow">OphAgent Model Hub</div>'
        f'<h1 class="hub-page-title">{html.escape(str(config["title"]))}</h1>'
        f'<div class="hub-page-copy">{html.escape(str(config["subtitle"]))}</div>'
        '</div>'
        f'<div class="hub-page-context">{html.escape(context)}</div></div>',
        unsafe_allow_html=True,
    )


def boundary_notice() -> None:
    st.markdown(
        '<div class="hub-boundary-compact"><strong>研究边界</strong><span>'
        '模型工程与研究评测服务于研发验证；病例回放仅展示模型调用轨迹。'
        '模型错误风险可以由结果表审计，临床后果风险尚未评估。</span></div>',
        unsafe_allow_html=True,
    )


def title_block() -> None:
    st.markdown(
        '<div class="hub-brand">'
        f'<div class="hub-brand-mark">{OPHAGENT_MARK_SVG}</div>'
        '<div><div class="hub-kicker">OphAgent Model Hub</div>'
        '<div class="hub-title">眼科模型中转台</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hub-subtitle">发现并接入服务器模型，提交适配或训练任务，比较路由模型与专家模型组合，并查看病例路由解释。</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hub-warning"><strong>系统边界：</strong>本系统为研究演示 Demo，不提供任何诊断、治疗或患者分流建议。</div>',
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
