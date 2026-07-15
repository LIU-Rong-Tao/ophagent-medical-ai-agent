from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.generic_result_audit import (
    LEAKAGE_STATUS_SUSPICIOUS,
    LEAKAGE_STATUS_UNKNOWN,
    ResultTableMapping,
    VariantMapping,
    export_audit_result,
    normalize_result_table,
    read_result_table,
    run_generic_risk_audit,
    run_lightweight_leakage_checks,
    suggest_mapping,
    validate_normalized_predictions,
)


def _six_class_frame(*, include_tta: bool = False) -> pd.DataFrame:
    labels = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for index in range(18):
        truth = labels[index % len(labels)]
        prediction = truth if index % 4 else labels[(index + 1) % len(labels)]
        probabilities = {label: 0.02 for label in labels}
        probabilities[prediction] = 0.90
        row = {
            "sample_id": f"case_{index:03d}",
            "true_label": truth,
            "pred_label_no_tta": prediction,
            "confidence_no_tta": 0.90,
            **{f"prob_{label}": value for label, value in probabilities.items()},
        }
        if include_tta:
            tta_prediction = truth if index % 5 else labels[(index + 2) % len(labels)]
            tta_probabilities = {label: 0.02 for label in labels}
            tta_probabilities[tta_prediction] = 0.90
            row.update(
                {
                    "pred_label_tta": tta_prediction,
                    "confidence_tta": 0.90,
                    **{f"prob_{label}_tta": value for label, value in tta_probabilities.items()},
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_six_class_excel_legacy_width_is_detected_without_hardcoded_classes(tmp_path: Path) -> None:
    source = _six_class_frame(include_tta=True)
    path = tmp_path / "six_class.xlsx"
    source.to_excel(path, index=False, sheet_name="predictions")
    loaded = read_result_table(path, filename=path.name, sheet_name="predictions")
    mapping = suggest_mapping(loaded)

    assert mapping is not None
    assert [variant.name for variant in mapping.variants] == ["base", "tta"]
    assert set(mapping.variants[0].probability_columns) == {"A", "B", "C", "D", "E", "F"}
    normalized = normalize_result_table(loaded, mapping)
    validation = validate_normalized_predictions(normalized)
    assert validation.passed
    assert validation.summary["class_count"] == 6
    assert validation.summary["variant_count"] == 2


def test_59_class_result_table_is_supported() -> None:
    class_names = [f"class_{index}" for index in range(59)]
    rows = []
    for index in range(180):
        truth = class_names[index % 59]
        prediction = truth if index % 7 else class_names[(index + 1) % 59]
        probabilities = np.full(59, 0.01 / 58)
        probabilities[class_names.index(prediction)] = 0.99
        rows.append(
            {
                "case_id": f"case-{index}",
                "y_true": truth,
                "y_pred": prediction,
                **{f"probability_{name}": probabilities[position] for position, name in enumerate(class_names)},
            }
        )
    frame = pd.DataFrame(rows)
    mapping = suggest_mapping(frame)
    assert mapping is not None
    normalized = normalize_result_table(frame, mapping)
    validation = validate_normalized_predictions(normalized)
    audit = run_generic_risk_audit(
        normalized,
        source=frame,
        mapping=mapping,
        primary_variant="base",
    )

    assert validation.passed
    assert validation.summary["class_count"] == 59
    assert len(audit.class_metrics) == 59
    assert audit.summary.iloc[0]["has_probabilities"]


def test_csv_and_label_only_predictions_use_degraded_audit() -> None:
    payload = b"record_id,ground_truth,prediction\n1,A,A\n2,B,A\n3,B,B\n"
    frame = read_result_table(payload, filename="predictions.csv")
    mapping = suggest_mapping(frame)
    assert mapping is not None
    normalized = normalize_result_table(frame, mapping)
    validation = validate_normalized_predictions(normalized)
    audit = run_generic_risk_audit(
        normalized,
        source=frame,
        mapping=mapping,
        primary_variant="base",
    )

    assert validation.passed
    assert not audit.summary.iloc[0]["has_probabilities"]
    assert audit.review_budget_results.empty
    assert audit.case_risk_scores["confidence"].isna().all()


def test_invalid_probability_sum_and_prediction_argmax_block_audit() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "true_label": ["A", "B"],
            "y_pred": ["B", "B"],
            "prob_A": [0.8, 0.1],
            "prob_B": [0.3, 0.9],
        }
    )
    mapping = suggest_mapping(frame)
    assert mapping is not None
    validation = validate_normalized_predictions(normalize_result_table(frame, mapping))

    assert not validation.passed
    check_ids = {issue.check_id for issue in validation.issues}
    assert "probability_sum::base" in check_ids
    assert "prediction_argmax::base" in check_ids


def test_duplicate_case_and_cross_split_are_flagged() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["same", "same"],
            "split": ["train", "test"],
            "true_label": ["A", "A"],
            "y_pred": ["A", "A"],
        }
    )
    mapping = suggest_mapping(frame)
    assert mapping is not None
    checks = run_lightweight_leakage_checks(frame, mapping).set_index("check_id")
    validation = validate_normalized_predictions(normalize_result_table(frame, mapping))

    assert checks.loc["duplicate_case_id", "status"] == LEAKAGE_STATUS_SUSPICIOUS
    assert checks.loc["case_id_cross_split", "status"] == LEAKAGE_STATUS_SUSPICIOUS
    assert not validation.passed


def test_label_name_in_filename_is_flagged_and_missing_split_is_not_assessable() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["case-1", "case-2"],
            "filename": ["folder/A/image.png", "folder/B/image.png"],
            "true_label": ["A", "B"],
            "y_pred": ["A", "B"],
        }
    )
    mapping = ResultTableMapping(
        case_id_column="case_id",
        true_label_column="true_label",
        variants=(VariantMapping("base", prediction_column="y_pred"),),
        metadata_columns=("filename",),
    )
    checks = run_lightweight_leakage_checks(frame, mapping).set_index("check_id")

    assert checks.loc["label_name_in_metadata", "status"] == LEAKAGE_STATUS_SUSPICIOUS
    assert checks.loc["case_id_cross_split", "status"] == LEAKAGE_STATUS_UNKNOWN
    assert checks.loc["patient_overlap", "status"] == LEAKAGE_STATUS_UNKNOWN


def test_multiple_variants_report_changes_without_assuming_tta_is_better() -> None:
    frame = _six_class_frame(include_tta=True)
    mapping = suggest_mapping(frame)
    assert mapping is not None
    normalized = normalize_result_table(frame, mapping)
    audit = run_generic_risk_audit(
        normalized,
        source=frame,
        mapping=mapping,
        primary_variant="base",
    )

    stability = audit.variant_stability.iloc[0]
    assert stability["comparison_variant"] == "tta"
    assert stability["prediction_changed"] > 0
    assert "error_to_correct" in audit.variant_stability
    assert "correct_to_error" in audit.variant_stability
    assert {"base", "tta"} == set(audit.summary["variant"])
    assert "comparison_accuracy" in audit.variant_stability


def test_imported_result_never_receives_online_or_route_eligibility(tmp_path: Path) -> None:
    frame = _six_class_frame()
    mapping = suggest_mapping(frame)
    assert mapping is not None
    normalized = normalize_result_table(frame, mapping)
    validation = validate_normalized_predictions(normalized)
    audit = run_generic_risk_audit(
        normalized,
        source=frame,
        mapping=mapping,
        primary_variant="base",
    )
    output = export_audit_result(
        tmp_path / "risk_audit",
        normalized=normalized,
        validation=validation,
        audit=audit,
    )
    validation_payload = (output / "validation.json").read_text(encoding="utf-8")

    assert '"offline_evaluation_eligible": true' in validation_payload
    assert '"adapter_implemented": false' in validation_payload
    assert '"task_inference_ready": false' in validation_payload
    assert '"route_eligible": false' in validation_payload
    assert not (output / "report.html").exists()


def test_ui_source_keeps_feature_inside_research_workspace_and_hides_raw_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    research_source = (root / "app/model_hub_research.py").read_text(encoding="utf-8")
    audit_ui_source = (root / "app/model_hub_result_audit.py").read_text(encoding="utf-8")

    assert '["路由组合评测", "结果表风险审计"]' in research_source
    assert "默认使用会话内序号" in audit_ui_source
    assert "临床后果风险：尚未评估" in audit_ui_source
    assert "字段自动识别完成，可直接运行审计" in audit_ui_source
    assert "自动识别不正确时，启用手动调整" in audit_ui_source
    assert "route_eligible" not in audit_ui_source
    assert "report.html" not in audit_ui_source


def test_frd6_display_aliases_live_in_config_not_core_audit_logic() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/result_audit_class_aliases.json").read_text(encoding="utf-8")
    )
    profiles = {profile["profile_id"]: profile for profile in payload["profiles"]}
    aliases = profiles["frd6_label_names"]["aliases"]

    assert aliases["CSC"] == "中心性浆液性脉络膜视网膜病变"
    assert aliases["VKH"] == "小柳原田病"
    core_source = (root / "app/generic_result_audit.py").read_text(encoding="utf-8")
    assert "中心性浆液性脉络膜视网膜病变" not in core_source
