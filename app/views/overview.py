"""项目概览页。"""

from __future__ import annotations

import streamlit as st

from app.ui import (
    flow_step,
    metric_card,
    page_header,
    render_boundary,
    section_header,
)


def render() -> None:
    page_header(
        "OphAgent Audit Demo",
        "在基础模型已经给出预测后，利用完整概率分布安排复核优先级，并用后验标签审计危险漏检与残余风险。",
        "科研证据快照 v0.7.2",
        kicker="眼科 AI 输出后审计",
    )

    cols = st.columns(4, gap="small")
    steps = [
        ("01", "模型输出", "接收预测等级、完整概率分布与模型版本，不改动原诊断模型。"),
        ("02", "风险排序", "排序阶段不读取真实标签，只从输出结构提取复核信号。"),
        ("03", "有限复核", "在固定 10%–30% 预算下，把更值得复核的记录排到前面。"),
        ("04", "后验审计", "获得参考标签后，计算捕获率、残余事件与方法增量。"),
    ]
    for column, step in zip(cols, steps):
        with column:
            flow_step(*step)

    st.markdown("")
    left, right = st.columns(2, gap="large")
    with left:
        section_header("无标签阶段能做什么")
        st.markdown(
            """
            - 生成复核优先队列；
            - 展示置信度、间隔、熵、Top-2 与事件特异概率；
            - 解释某条记录为什么被排到前面；
            - 保留模型、checkpoint 和数据来源追溯。
            """
        )
    with right:
        section_header("有参考标签后才能回答什么")
        st.markdown(
            """
            - 前 20% 复核队列抓到了多少目标事件；
            - 相比随机抽取增加了多少事件召回；
            - 还有多少危险事件没有进入优先复核区；
            - 结论是否跨数据集、骨干模型和评价指标保持一致。
            """
        )

    render_boundary(
        "<strong>关键边界：</strong>没有真实标签时不能计算捕获率和残余风险。"
        "当前结果来自公共数据集的图像级回顾性审计，不是患者级临床验证，也不用于诊断。"
    )

    section_header(
        "为什么从分类准确率继续往下走",
        "平均准确率把所有错误混在一起；眼科筛查更需要单独观察重症被预测为非重症的方向性漏检。",
    )
    metric_cols = st.columns(3, gap="small")
    with metric_cols[0]:
        metric_card(
            "通用错分",
            "pred ≠ true",
            "回答模型是否预测错误，不区分错误方向与临床后果。",
            accent="baseline",
        )
    with metric_cols[1]:
        metric_card(
            "跨级低估代理事件",
            "true − pred ≥ 2",
            "利用 ICDR 五级顺序结构，检查真实等级至少高两级的低估。",
            accent="amber",
        )
    with metric_cols[2]:
        metric_card(
            "VTDR miss 等级代理",
            "true ≥ 3 且 pred < 3",
            "APTOS 没有 DME 标签，因此只覆盖 Severe NPDR / PDR 的 grade-only proxy。",
            accent="red",
        )

    section_header(
        "研究脉络：每一步都由上一轮的反例推动",
        "版本号只用于追溯；真正重要的是每一步解决了哪个可质疑点。",
    )
    timeline = [
        (
            "v0.6.6",
            "先排除标签泄露",
            "不看真实标签，模型输出能否形成预审队列？",
            "输出信号能富集错误；手工 combined 没有稳定超过简单强基线。",
        ),
        (
            "v0.6.7",
            "从所有错误转向危险错误",
            "总体错分不能区分轻症高估和重症低估，因而补充方向敏感事件与复核负担。",
            "同一排序信号不适合解释所有错误类型。",
        ),
        (
            "v0.6.7b–c",
            "解释 combined 的收益来源",
            "用期望等级差和门控重症概率质量与复杂手工规则做消融。",
            "简单、事件特异的严重程度信号更容易解释关键增量。",
        ),
        (
            "v0.6.8–b",
            "检验学习型组合是否更好",
            "逻辑回归按图像分组交叉验证，并补 bootstrap、重复划分和捕获重叠。",
            "学习型分数有竞争力，但没有稳定替代最强简单信号。",
        ),
        (
            "v0.7.0–1b",
            "冻结协议后做外部压力测试",
            "与 random gate-only 比较，区分候选池 gate 与概率排序的额外贡献。",
            "VTDR miss 上，grade 3/4 概率质量在 gate 之外提供额外排序信息。",
        ),
        (
            "v0.7.2",
            "检查结论是否依赖单一指标",
            "补充 AURC、AUGRC、局部 AUGRC 和 Top20% 工作点比较。",
            "核心信号在 12 个外部数据集—骨干模型组合中保持曲线指标第一。",
        ),
    ]
    timeline_columns = st.columns(2, gap="large")
    for index, (version, title, question, conclusion) in enumerate(timeline):
        with timeline_columns[index % 2]:
            with st.container(border=True):
                st.caption(version)
                st.markdown(f"#### {title}")
                st.write(question)
                st.markdown(f"**这一轮得到：** {conclusion}")

    section_header(
        "两个事件特异信号为什么出现",
        "它们不是临床指南终点，而是从五级概率分布提取、用于复核排序的项目派生信号。",
    )
    signal_left, signal_right = st.columns(2, gap="large")
    with signal_left:
        st.markdown("#### 期望等级差")
        st.latex(r"\sum_{k=0}^{4} kP(y=k)-\operatorname{argmax}_kP(y=k)")
        st.write(
            "当 Top-1 给出较轻等级，但其余概率仍明显向更高等级延伸时，"
            "概率加权的期望等级会高于 Top-1。它用于刻画跨级低估。"
        )
        st.caption(
            "例：[0.35, 0.15, 0.30, 0.15, 0.05] 的 Top-1 为 0，"
            "期望等级为 1×0.15 + 2×0.30 + 3×0.15 + 4×0.05 = 1.40。"
        )
    with signal_right:
        st.markdown("#### 门控重症概率质量")
        st.latex(r"\mathbb{1}(\hat y\leq2)\,[P(y=3)+P(y=4)]")
        st.write(
            "先把模型预测为 0–2 级的记录作为 VTDR miss 候选池，再按输出中"
            " grade 3/4 的概率和排序。它回答候选池内部谁更值得优先复核。"
        )
        st.caption(
            "gate 只提供“预测为非重症”这一条信息；P(3)+P(4) 进一步利用完整概率分布，"
            "因此 v0.7.1b 专门用 random gate-only 检查额外增量。"
        )

    section_header("研究依据如何改变了实验设计")
    references = [
        (
            "Hendrycks & Gimpel，ICLR 2017",
            "最大 softmax 概率（MSP）是简单而有竞争力的错误检测基线。"
            "它促使 v0.6.6 将 combined 与 1-MSP、margin、entropy 正面对照。",
            "https://arxiv.org/abs/1610.02136",
        ),
        (
            "Geifman & El-Yaniv，NeurIPS 2017",
            "选择性分类把问题改写为：模型只处理一部分样本，其余交由人工。"
            "OphAgent 因此用复核预算和 risk-coverage，而不只报告单一准确率。",
            "https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks",
        ),
        (
            "Wilkinson et al.，Ophthalmology 2003",
            "ICDR 严重程度量表给出 No、Mild、Moderate、Severe NPDR 与 PDR 的有序等级，"
            "为 APTOS 0–4 的顺序关系提供依据。",
            "https://pubmed.ncbi.nlm.nih.gov/13129861/",
        ),
        (
            "Traub et al.，NeurIPS 2024",
            "指出选择性分类不能只盯一个 coverage 工作点，并讨论 generalized risk。"
            "v0.7.2 因此补充 AUGRC 与局部预算区间审计。",
            "https://openreview.net/forum?id=2TktDpGqNM",
        ),
        (
            "Kwon & Kim，Scientific Reports 2026",
            "临床 deferral 需要同时考虑域移、处理成本与人工负担。"
            "OphAgent 因此同时报告捕获、复核量和未进入优先复核区的残余事件。",
            "https://www.nature.com/articles/s41598-026-40637-w",
        ),
    ]
    for title, relation, link in references:
        with st.expander(title):
            st.write(relation)
            st.markdown(f"[打开原始来源]({link})")

    render_boundary(
        "<strong>当前阶段成果：</strong>项目已经形成从输出信号提取、复核排序、"
        "强基线比较、外部压力测试到残余风险报告的完整审计链。"
        "它仍不足以证明临床效用、工作量下降或安全放行。"
    )
