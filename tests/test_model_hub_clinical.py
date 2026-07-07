from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_hub_data import attach_retrospective_evidence, build_online_case_view
from app.model_hub_clinical import filter_case_view, paginate_cases


def fixture_detail() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task_id": ["aptos_dr_5class", "aptos_dr_5class"],
            "image_key": ["case_a", "case_b"],
            "image_path": ["/images/a.png", "/images/b.png"],
            "primary_scout_artifact_id": ["convnext_tiny", "convnext_tiny"],
            "primary_scout_pred_label": [2, 1],
            "scout_pred_labels": ['{"convnext_tiny": 2}', '{"convnext_tiny": 1}'],
            "scout_confidences": ['{"convnext_tiny": 0.4}', '{"convnext_tiny": 0.7}'],
            "routing_score": [0.6, 0.3],
            "is_reviewed_by_expert": [True, False],
            "expert_artifact_ids": ["retfound", "retfound"],
            "expert_pred_label": [3, 1],
            "final_pred_label": [3, 1],
            "final_source": ["expert", "scout"],
            "true_label": [4, 1],
            "was_scout_correct": [False, True],
            "was_final_correct": [False, True],
            "dr_severe_pdr_miss_scout_event": [True, False],
            "dr_severe_pdr_miss_final_residual": [True, False],
        }
    )


def test_online_case_view_uses_a_strict_prediction_only_whitelist() -> None:
    online = build_online_case_view(fixture_detail())

    assert online.attrs["display_scope"] == "online_only"
    assert {"image_key", "primary_scout_pred_label", "expert_pred_label", "routing_score"} <= set(online.columns)
    assert "true_label" not in online.columns
    assert not any("residual" in column or "miss" in column for column in online.columns)
    assert not any(column.startswith("was_") for column in online.columns)


def test_retrospective_evidence_is_attached_only_with_unique_complete_keys() -> None:
    detail = fixture_detail()
    online = build_online_case_view(detail)
    evidence = detail[
        [
            "task_id",
            "image_key",
            "primary_scout_artifact_id",
            "true_label",
            "was_final_correct",
            "dr_severe_pdr_miss_scout_event",
            "dr_severe_pdr_miss_final_residual",
        ]
    ]

    research = attach_retrospective_evidence(online, evidence)

    assert research.attrs["display_scope"] == "research_only"
    assert research["true_label"].tolist() == [4, 1]
    assert "dr_severe_pdr_miss_final_residual" in research.columns


def test_retrospective_evidence_rejects_duplicates_and_missing_matches() -> None:
    detail = fixture_detail()
    online = build_online_case_view(detail)
    evidence = detail[["task_id", "image_key", "primary_scout_artifact_id", "true_label"]]

    with pytest.raises(ValueError, match="重复"):
        attach_retrospective_evidence(online, pd.concat([evidence, evidence.iloc[[0]]], ignore_index=True))

    with pytest.raises(ValueError, match="缺少"):
        attach_retrospective_evidence(online, evidence.iloc[[0]])


def test_case_pagination_covers_the_full_queue_without_top_200_truncation() -> None:
    frame = pd.DataFrame({"image_key": [f"case_{index:03d}" for index in range(253)]})

    page, total_pages = paginate_cases(frame, page=3, page_size=100)

    assert total_pages == 3
    assert len(page) == 53
    assert page.iloc[0]["image_key"] == "case_200"


def test_research_case_filter_selects_reference_label_mismatches() -> None:
    filtered = filter_case_view(
        fixture_detail(),
        ["与参考标签不一致"],
        research_mode=True,
    )

    assert filtered["image_key"].tolist() == ["case_a"]


def test_online_case_filter_does_not_expose_reference_label_filter() -> None:
    filtered = filter_case_view(
        build_online_case_view(fixture_detail()),
        ["与参考标签不一致"],
        research_mode=False,
    )

    assert filtered["image_key"].tolist() == ["case_a", "case_b"]


def test_case_replay_ui_separates_online_and_research_views_without_raw_json() -> None:
    source = (ROOT / "app" / "model_hub_clinical.py").read_text(encoding="utf-8")

    assert ".head(200)" not in source
    assert "路由排名" not in source
    assert "st.json" not in source
    assert "模型结果对照" in source
    assert "模型输出回放" in source
    assert "研究审计" in source
    assert "病例回放与路由解释" in source
    assert '"model_hub_last_cases"' in source
    assert '"model_hub_last_research_cases"' in source
    assert "与参考标签不一致" in source
    assert "background-color:#fff4f2" in source


def test_case_explanation_uses_human_review_wording_without_ai_style_claims() -> None:
    source = (ROOT / "app" / "model_hub_clinical.py").read_text(encoding="utf-8")

    assert "路由依据" in source
    assert "未进入当前策略的专家调用额度" in source
    assert "本页只展示模型调用轨迹" in source
    assert "独立标注或人工复核" in source
    assert "当前没有临床确认标签" not in source
    assert "不声称专家纠正了结果" not in source
