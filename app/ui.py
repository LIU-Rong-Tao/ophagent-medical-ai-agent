"""OphAgent Audit Demo 的共用 Streamlit 展示组件。"""

from __future__ import annotations

import html
from pathlib import Path
from collections.abc import Sequence

import streamlit as st

from app.clinical_semantics import ClinicalDisplaySummary


COLORS = {
    "navy": "#17324D",
    "teal": "#0F766E",
    "amber": "#B7791F",
    "red": "#B42318",
    "background": "#F6F8FB",
    "surface": "#FFFFFF",
    "text": "#172033",
    "muted": "#5B6878",
    "border": "#D8E0EA",
    "baseline": "#8A98A8",
}


def inject_app_css() -> None:
    """注入少量稳定 CSS，统一研究型仪表盘的排版。"""

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {COLORS["background"]};
            color: {COLORS["text"]};
        }}
        .main .block-container {{
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3 {{
            color: {COLORS["navy"]};
            letter-spacing: 0;
        }}
        [data-testid="stSidebar"] {{
            background: #EEF3F7;
            border-right: 1px solid {COLORS["border"]};
        }}
        .oa-page-kicker {{
            color: {COLORS["teal"]};
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }}
        .oa-page-subtitle {{
            color: {COLORS["muted"]};
            font-size: 1rem;
            line-height: 1.65;
            margin: .15rem 0 .8rem;
            max-width: 860px;
        }}
        .oa-badge {{
            display: inline-block;
            color: {COLORS["teal"]};
            background: #E8F4F1;
            border: 1px solid #B7DCD5;
            border-radius: 6px;
            padding: .22rem .55rem;
            font-size: .78rem;
            font-weight: 700;
        }}
        .oa-metric {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-top: 3px solid var(--accent, {COLORS["teal"]});
            border-radius: 6px;
            padding: .9rem 1rem;
            min-height: 116px;
        }}
        .oa-metric-label {{
            color: {COLORS["muted"]};
            font-size: .82rem;
            margin-bottom: .25rem;
        }}
        .oa-metric-value {{
            color: {COLORS["navy"]};
            font-size: 1.7rem;
            line-height: 1.15;
            font-weight: 750;
        }}
        .oa-metric-note {{
            color: {COLORS["muted"]};
            font-size: .78rem;
            line-height: 1.45;
            margin-top: .4rem;
        }}
        .oa-empty {{
            border: 1px dashed {COLORS["border"]};
            border-radius: 6px;
            padding: 1.1rem;
            color: {COLORS["muted"]};
            background: {COLORS["surface"]};
        }}
        .oa-source {{
            color: {COLORS["muted"]};
            font-size: .74rem;
            margin-top: .35rem;
            overflow-wrap: anywhere;
        }}
        .oa-boundary {{
            border-left: 4px solid {COLORS["amber"]};
            background: #FFF9ED;
            padding: .8rem 1rem;
            color: #5E4617;
            border-radius: 0 6px 6px 0;
            line-height: 1.6;
        }}
        .oa-flow-step {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            padding: .85rem .8rem;
            min-height: 118px;
        }}
        .oa-flow-index {{
            color: {COLORS["teal"]};
            font-size: .76rem;
            font-weight: 750;
        }}
        .oa-flow-title {{
            color: {COLORS["navy"]};
            font-weight: 750;
            margin: .22rem 0;
        }}
        .oa-flow-copy {{
            color: {COLORS["muted"]};
            font-size: .82rem;
            line-height: 1.5;
        }}
        .oa-case-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-left: 5px solid var(--severity-color, {COLORS["baseline"]});
            border-radius: 7px;
            padding: 1rem 1.05rem;
            min-height: 258px;
        }}
        .oa-case-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .7rem;
            margin-bottom: .7rem;
        }}
        .oa-case-id {{
            color: {COLORS["muted"]};
            font-size: .78rem;
            overflow-wrap: anywhere;
        }}
        .oa-case-model {{
            color: {COLORS["muted"]};
            font-size: .74rem;
            margin-top: .18rem;
            overflow-wrap: anywhere;
        }}
        .oa-priority {{
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            border-radius: 999px;
            padding: .22rem .55rem;
            color: var(--priority-color, {COLORS["amber"]});
            background: var(--priority-bg, #FFF4D8);
            font-size: .78rem;
            font-weight: 750;
            white-space: nowrap;
        }}
        .oa-priority-dot {{
            width: .48rem;
            height: .48rem;
            border-radius: 50%;
            background: var(--priority-color, {COLORS["amber"]});
        }}
        .oa-case-semantic-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 1rem;
            margin-top: .8rem;
            padding-top: .85rem;
            border-top: 1px solid {COLORS["border"]};
        }}
        .oa-case-dimension + .oa-case-dimension {{
            border-left: 1px solid {COLORS["border"]};
            padding-left: 1rem;
        }}
        .oa-dimension-label {{
            color: {COLORS["muted"]};
            font-size: .74rem;
            font-weight: 700;
            margin-bottom: .25rem;
        }}
        .oa-dimension-value {{
            color: var(--dimension-color, {COLORS["navy"]});
            font-size: 1.06rem;
            font-weight: 780;
            line-height: 1.35;
        }}
        .oa-dimension-band {{
            color: var(--dimension-color, {COLORS["navy"]});
            font-size: .79rem;
            font-weight: 720;
            margin-top: .2rem;
        }}
        .oa-dimension-copy {{
            color: {COLORS["text"]};
            font-size: .82rem;
            line-height: 1.55;
            margin-top: .42rem;
        }}
        .oa-case-disclaimer {{
            color: {COLORS["muted"]};
            font-size: .72rem;
            line-height: 1.45;
            margin-top: .7rem;
            padding-top: .65rem;
            border-top: 1px dashed {COLORS["border"]};
        }}
        .oa-reasons {{
            display: flex;
            flex-wrap: wrap;
            gap: .35rem;
            margin-top: .55rem;
        }}
        .oa-reason {{
            border: 1px solid {COLORS["border"]};
            background: #F3F6F9;
            border-radius: 5px;
            color: {COLORS["muted"]};
            padding: .18rem .42rem;
            font-size: .72rem;
        }}
        .oa-probability {{
            display: grid;
            grid-template-columns: minmax(105px, 150px) minmax(120px, 1fr) 58px;
            align-items: center;
            gap: .75rem;
            margin: .58rem 0;
        }}
        .oa-probability-label {{
            color: {COLORS["text"]};
            font-size: .84rem;
        }}
        .oa-probability-track {{
            height: 16px;
            border-radius: 3px;
            background: #E8EDF2;
            overflow: hidden;
        }}
        .oa-probability-fill {{
            height: 100%;
            min-width: 2px;
            border-radius: 3px;
            background: var(--bar-color, {COLORS["baseline"]});
        }}
        .oa-probability-value {{
            color: {COLORS["navy"]};
            font-size: .82rem;
            font-weight: 700;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .oa-mode-note {{
            color: {COLORS["muted"]};
            font-size: .78rem;
            line-height: 1.5;
            margin-top: -.2rem;
        }}
        div[data-testid="stMetric"] {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            padding: .65rem .8rem;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            overflow: hidden;
        }}
        @media (max-width: 640px) {{
            .main .block-container {{
                padding-left: .85rem;
                padding-right: .85rem;
            }}
            .oa-metric-value {{
                font-size: 1.35rem;
            }}
            .oa-flow-step {{
                min-height: auto;
            }}
            .oa-probability {{
                grid-template-columns: 92px minmax(80px, 1fr) 52px;
                gap: .45rem;
            }}
            .oa-case-card {{
                min-height: auto;
            }}
            .oa-case-semantic-grid {{
                grid-template-columns: 1fr;
            }}
            .oa-case-dimension + .oa-case-dimension {{
                border-left: 0;
                border-top: 1px solid {COLORS["border"]};
                padding-left: 0;
                padding-top: .8rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    subtitle: str,
    evidence_badge: str | None = None,
    *,
    kicker: str = "OphAgent",
) -> None:
    """渲染页面标题、定位说明和证据徽标。"""

    badge = f'<span class="oa-badge">{evidence_badge}</span>' if evidence_badge else ""
    st.markdown(
        f"""
        <div class="oa-page-kicker">{kicker}</div>
        <h1>{title}</h1>
        <div class="oa-page-subtitle">{subtitle}</div>
        {badge}
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str | None = None) -> None:
    """渲染紧凑章节标题。"""

    st.subheader(title)
    if description:
        st.caption(description)


def metric_card(
    label: str,
    value: str,
    note: str = "",
    *,
    accent: str = "teal",
) -> None:
    """渲染带语义色的固定尺寸指标卡。"""

    color = COLORS.get(accent, accent)
    st.markdown(
        f"""
        <div class="oa-metric" style="border-top-color:{color}">
          <div class="oa-metric-label">{label}</div>
          <div class="oa-metric-value">{value}</div>
          <div class="oa-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, detail: str) -> None:
    """渲染不会打断页面的空状态。"""

    st.markdown(
        f'<div class="oa-empty"><strong>{title}</strong><br>{detail}</div>',
        unsafe_allow_html=True,
    )


def render_source_caption(path: str | Path) -> None:
    """展示结果的本地来源路径。"""

    st.markdown(
        f'<div class="oa-source">数据来源：<code>{Path(path).as_posix()}</code></div>',
        unsafe_allow_html=True,
    )


def render_boundary(text: str) -> None:
    """突出研究边界，不用整屏 warning 组件。"""

    st.markdown(f'<div class="oa-boundary">{text}</div>', unsafe_allow_html=True)


def flow_step(index: str, title: str, copy: str) -> None:
    """渲染审计链单步。"""

    st.markdown(
        f"""
        <div class="oa-flow-step">
          <div class="oa-flow-index">{index}</div>
          <div class="oa-flow-title">{title}</div>
          <div class="oa-flow-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority_palette(level: str) -> tuple[str, str]:
    """返回复核优先级的前景色与浅背景色。"""

    return {
        "high": (COLORS["red"], "#FDECEA"),
        "medium": (COLORS["amber"], "#FFF4D8"),
        "routine": ("#52667A", "#EEF2F5"),
    }.get(level, (COLORS["baseline"], "#EEF2F5"))


def severity_palette(level: str) -> tuple[str, str]:
    """返回模型预测严重程度的前景色与浅背景色。"""

    return {
        "severe": (COLORS["red"], "#FDECEA"),
        "moderate": (COLORS["amber"], "#FFF4D8"),
        "nonsevere": ("#52667A", "#EEF2F5"),
    }.get(level, (COLORS["baseline"], "#EEF2F5"))


def build_case_card_html(
    *,
    case_id: str,
    clinical_summary: ClinicalDisplaySummary,
    model_context: str = "",
    reasons: Sequence[str] = (),
) -> str:
    """构建病情等级与输出可疑度分栏展示的病例卡 HTML。"""

    priority_color, priority_background = priority_palette(
        clinical_summary.audit_priority_level
    )
    severity_color, _ = severity_palette(clinical_summary.clinical_severity_level)
    reason_html = "".join(
        f'<span class="oa-reason">{html.escape(str(reason))}</span>'
        for reason in reasons[:3]
    )
    model_html = (
        f'<div class="oa-case-model">当前模型：{html.escape(model_context)}</div>'
        if model_context
        else ""
    )
    return f"""
        <div class="oa-case-card"
             style="--severity-color:{severity_color};border-left-color:{severity_color}">
          <div class="oa-case-top">
            <div>
              <div class="oa-case-id">病例记录：{html.escape(str(case_id))}</div>
              {model_html}
            </div>
            <span class="oa-priority"
                  style="color:{priority_color};background:{priority_background}">
              <span class="oa-priority-dot" style="background:{priority_color}"></span>
              {html.escape(clinical_summary.audit_priority_label)}
            </span>
          </div>
          <div class="oa-case-semantic-grid">
            <div class="oa-case-dimension">
              <div class="oa-dimension-label">模型预测等级</div>
              <div class="oa-dimension-value"
                   style="--dimension-color:{severity_color}">
                {html.escape(clinical_summary.predicted_grade_label)}
              </div>
              <div class="oa-dimension-band"
                   style="--dimension-color:{severity_color}">
                {html.escape(clinical_summary.predicted_severity_band)}
              </div>
              <div class="oa-dimension-copy">
                {html.escape(clinical_summary.clinical_message)}
              </div>
            </div>
            <div class="oa-case-dimension">
              <div class="oa-dimension-label">模型输出复核优先级</div>
              <div class="oa-dimension-value"
                   style="--dimension-color:{priority_color}">
                {html.escape(clinical_summary.audit_priority_label)}
              </div>
              <div class="oa-dimension-copy">
                {html.escape(clinical_summary.audit_priority_message)}
              </div>
            </div>
          </div>
          <div class="oa-reasons">{reason_html}</div>
          <div class="oa-case-disclaimer">
            {html.escape(clinical_summary.disclaimer)}
          </div>
        </div>
        """


def render_case_card(
    *,
    case_id: str,
    clinical_summary: ClinicalDisplaySummary,
    model_context: str = "",
    reasons: Sequence[str] = (),
) -> None:
    """渲染面向临床展示的双维度病例复核卡片。"""

    st.markdown(
        build_case_card_html(
            case_id=case_id,
            clinical_summary=clinical_summary,
            model_context=model_context,
            reasons=reasons,
        ),
        unsafe_allow_html=True,
    )


def render_probability_profile(
    labels: Sequence[str],
    probabilities: Sequence[float],
    *,
    pred_grade: int,
) -> None:
    """用浏览器原生 HTML 绘制五级概率条，避免服务器缺少中文字体。"""

    rows: list[str] = []
    for index, (label, probability) in enumerate(zip(labels, probabilities)):
        if index == pred_grade:
            color = COLORS["navy"]
        elif index in (3, 4):
            color = COLORS["amber"]
        else:
            color = COLORS["baseline"]
        rows.append(
            f"""
            <div class="oa-probability">
              <div class="oa-probability-label">{html.escape(str(label))}</div>
              <div class="oa-probability-track">
                <div class="oa-probability-fill"
                     style="width:{max(0.0, min(100.0, float(probability) * 100)):.3f}%;
                            --bar-color:{color}"></div>
              </div>
              <div class="oa-probability-value">{float(probability):.1%}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
