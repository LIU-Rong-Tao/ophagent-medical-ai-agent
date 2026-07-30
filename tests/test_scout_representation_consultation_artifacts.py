from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "scout_representation_consultation_v0_1"
)
EXPECTED_ROUTES = {
    "aptos_dr_5class::flair__to__swin_tiny",
    "aptos_dr_5class::retfound_green__to__retfound_cfp",
    "aptos_dr_5class::retfound_green__to__swin_tiny",
}


def test_formal_output_set_and_decision_are_frozen() -> None:
    assert {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} == {
        "core_results.csv",
        "risk_budget_comparison.png",
        "research_report.md",
    }
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    assert set(core["route_id"]) == EXPECTED_ROUTES
    assert set(core["study_decision"]) == {"NO_IMPROVEMENT"}
    assert set(core["analysis_split"]) == {
        "development_oof",
        "retrospective_evaluation",
    }


def test_same_budget_comparisons_are_exact_and_expert_free() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    same_budget = core.loc[core["comparison_axis"].eq("same_budget")]
    assert set(same_budget["policy"]) == {
        "entropy",
        "margin",
        "prior_simple_gate",
        "scout_representation",
    }
    counts = same_budget.groupby(
        ["route_id", "analysis_split", "requested_budget"]
    )["selected_n"].nunique()
    assert counts.eq(1).all()
    assert same_budget["incremental_online_encoder_forward_calls"].eq(0).all()
    assert not same_budget[
        "current_case_expert_output_used_for_ranking"
    ].astype(bool).any()
    assert not same_budget[
        "test_used_for_fit_selection_or_threshold"
    ].astype(bool).any()
    forbidden = ("path", "patient", "image_sha", "dataset_id_as_predictor")
    assert not any(
        token in column.lower()
        for column in core.columns
        for token in forbidden
    )


def test_predeclared_decision_failure_is_supported() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    method = core.loc[
        core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq("scout_representation")
        & core["requested_budget"].isin([0.1, 0.2, 0.3])
    ]
    assert len(method) == 9
    dominant = (
        method["delta_corrected_selected"].ge(0)
        & method["delta_introduced_selected"].le(0)
        & method["delta_net_selected"].gt(0)
    )
    assert not dominant.any()
    at_thirty = method.loc[method["requested_budget"].eq(0.3)]
    assert at_thirty["net_difference_ci_lower"].lt(0).all()


def test_report_and_figure_preserve_scope() -> None:
    report = (OUTPUT_DIR / "research_report.md").read_text(encoding="utf-8")
    assert "NO_IMPROVEMENT" in report
    assert "同一次在线 Scout 前向的额外编码器调用为 0" in report
    assert "未读取当前病例 Expert 输出或表征" in report
    assert "不能声称患者级泛化" in report
    assert "停止增加调用前模型复杂度" in report
    with Image.open(OUTPUT_DIR / "risk_budget_comparison.png") as image:
        assert image.width >= 3000
        assert image.height >= 1600
