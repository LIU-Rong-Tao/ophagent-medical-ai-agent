from pathlib import Path

import pandas as pd
import pytest

from app.views.case_detail import (
    attach_posthoc_evidence,
    build_pre_review_case,
    clinical_context_placeholders,
    filter_case_queue,
    initialize_review_capacity,
    index_prediction_records,
    normalize_image_key,
    paginate_case_queue,
    resolve_case_image,
    select_review_capacity,
)


def pre_review_row() -> dict:
    return {
        "case_id": "abc123",
        "image_path": "abc123.png",
        "pred_grade": 2,
        "pred_label": "Moderate DR",
        "top2_grade": 3,
        "top2_label": "Severe DR",
        "confidence": 0.40,
        "top2_confidence": 0.35,
        "margin": 0.05,
        "entropy_norm": 0.8,
        "severe_prob_mass": 0.30,
        "pre_review_risk_score": 8,
        "risk_reasons": "low_margin_boundary;high_entropy",
        "pre_review_risk_level": "high",
        "review_priority_rank": 1,
        "true_grade": 3,
        "general_error": True,
        "vision_threatening_dr_miss": True,
    }


def test_pre_review_case_whitelists_fields_and_excludes_posthoc_labels():
    case = build_pre_review_case(pre_review_row(), backbone="convnext_tiny")

    assert case["connection_key"] == ("convnext_tiny", "abc123.png")
    assert case["pred_grade"] == 2
    assert "true_grade" not in case
    assert "general_error" not in case
    assert "large_undergrading" not in case
    assert "vision_threatening_dr_miss" not in case


def test_posthoc_fields_are_only_added_by_explicit_attachment():
    case = build_pre_review_case(pre_review_row(), backbone="convnext_tiny")
    enriched = attach_posthoc_evidence(
        case,
        {
            "true_grade": 3,
            "general_error": True,
            "large_undergrading": False,
            "vision_threatening_dr_miss": True,
        },
    )

    assert "true_grade" not in case
    assert enriched["true_grade"] == 3
    assert enriched["vision_threatening_dr_miss"] is True


def test_prediction_index_uses_backbone_and_normalized_image_key():
    frame = pd.DataFrame(
        [
            {
                "image_path": "/data/APTOS/test/anodr/ABC123.PNG",
                "pred_idx": 0,
            }
        ]
    )
    index = index_prediction_records(frame, backbone="convnext_tiny")

    assert ("convnext_tiny", "abc123.png") in index
    assert normalize_image_key(r"C:\data\ABC123.PNG") == "abc123.png"


def test_duplicate_prediction_connection_key_is_rejected():
    frame = pd.DataFrame(
        [
            {"image_path": "/a/abc.png", "pred_idx": 0},
            {"image_path": "/b/abc.png", "pred_idx": 1},
        ]
    )

    with pytest.raises(ValueError, match="重复"):
        index_prediction_records(frame, backbone="convnext_tiny")


def test_missing_clinical_context_is_explicitly_marked_unconnected():
    context = clinical_context_placeholders()

    assert context == {
        "视力": "未接入",
        "病史": "未接入",
        "OCT": "未接入",
        "治疗记录": "未接入",
        "随访记录": "未接入",
    }


def test_missing_image_returns_none_without_substituting_demo_sample(tmp_path: Path):
    case = {"image_path": str(tmp_path / "missing.png")}

    assert resolve_case_image(case) is None


def test_case_queue_filter_and_pagination_are_stable():
    frame = pd.DataFrame(
        [
            {"case_id": f"case_{index:02d}", "pre_review_risk_level": level}
            for index, level in enumerate(
                ["high"] * 8 + ["medium"] * 8 + ["low"] * 8
            )
        ]
    )

    filtered = filter_case_queue(frame, priority="优先", search="case_0")
    page, total_pages, page_number = paginate_case_queue(
        filtered,
        page_number=9,
        page_size=4,
    )

    assert len(filtered) == 8
    assert total_pages == 2
    assert page_number == 2
    assert page["case_id"].tolist() == ["case_04", "case_05", "case_06", "case_07"]


def test_review_capacity_top_n_preserves_current_risk_order():
    frame = pd.DataFrame(
        [{"case_id": f"case_{index:02d}"} for index in range(10)]
    )

    selected = select_review_capacity(
        frame,
        capacity=4,
        method="风险 Top N",
        random_seed=42,
    )

    assert selected["case_id"].tolist() == [
        "case_00",
        "case_01",
        "case_02",
        "case_03",
    ]


def test_review_capacity_random_n_is_reproducible_and_not_top_n():
    frame = pd.DataFrame(
        [{"case_id": f"case_{index:02d}"} for index in range(20)]
    )

    first = select_review_capacity(
        frame,
        capacity=5,
        method="随机抽 N",
        random_seed=42,
    )
    second = select_review_capacity(
        frame,
        capacity=5,
        method="随机抽 N",
        random_seed=42,
    )

    assert first["case_id"].tolist() == second["case_id"].tolist()
    assert first["case_id"].tolist() != frame.head(5)["case_id"].tolist()
    assert len(first) == 5


def test_review_capacity_clamps_to_candidate_pool_and_rejects_invalid_values():
    frame = pd.DataFrame([{"case_id": "case_00"}, {"case_id": "case_01"}])

    selected = select_review_capacity(
        frame,
        capacity=20,
        method="风险 Top N",
        random_seed=42,
    )

    assert selected["case_id"].tolist() == ["case_00", "case_01"]
    with pytest.raises(ValueError, match="capacity"):
        select_review_capacity(
            frame,
            capacity=0,
            method="风险 Top N",
            random_seed=42,
        )
    with pytest.raises(ValueError, match="method"):
        select_review_capacity(
            frame,
            capacity=1,
            method="未知方式",
            random_seed=42,
        )


def test_review_capacity_session_defaults_to_fifty_and_clamps_after_filtering():
    state: dict[str, int] = {}

    assert initialize_review_capacity(state, "capacity", pool_size=1100) == 50
    assert state["capacity"] == 50

    state["capacity"] = 25
    assert initialize_review_capacity(state, "capacity", pool_size=300) == 25

    assert initialize_review_capacity(state, "capacity", pool_size=12) == 12
    assert state["capacity"] == 12
