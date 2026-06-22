from types import SimpleNamespace

from app.inference import InferenceResult
from app.views.single_case import (
    clinical_source_status,
    resolve_display_result,
)


def artifact():
    return SimpleNamespace(
        display_name="ConvNeXt-Tiny",
        protocol_id="dr_icdr_5class_proxy_v1",
    )


def test_online_success_replaces_missing_offline_record_with_probabilities():
    online = InferenceResult(
        ok=True,
        stage="complete",
        pred_grade=2,
        probabilities=[0.1, 0.1, 0.5, 0.2, 0.1],
        confidence=0.5,
        margin=0.3,
        entropy_norm=0.7,
        source="ConvNeXt-Tiny 在线 checkpoint 推理",
        backbone="convnext_tiny",
        labels=["No DR", "Mild DR", "Moderate DR", "Severe DR", "PDR"],
    )

    result = resolve_display_result(None, online)

    assert result is not None
    assert result["pred_grade"] == 2
    assert result["probabilities"][2] == 0.5
    assert "在线 checkpoint 推理" in result["source"]
    assert "本会话加载成功" in clinical_source_status(
        artifact(),
        display_result=result,
        online_result=online,
    )


def test_online_failure_without_offline_record_stays_empty_and_reports_stage():
    online = InferenceResult(
        ok=False,
        stage="load_checkpoint",
        error_type="RuntimeError",
        error_message="missing keys",
        backbone="convnext_tiny",
    )

    result = resolve_display_result(None, online)
    status = clinical_source_status(
        artifact(),
        display_result=result,
        online_result=online,
    )

    assert result is None
    assert "在线推理未完成" in status
    assert "load_checkpoint" in status
    assert "未使用教学概率或模型回退" in status


def test_offline_record_status_does_not_claim_model_was_run():
    offline = {
        "pred_grade": 0,
        "probabilities": [0.9, 0.05, 0.03, 0.01, 0.01],
        "source": "ConvNeXt-Tiny 已提交的测试 prediction record",
    }

    assert resolve_display_result(offline, None) == offline
    status = clinical_source_status(
        artifact(),
        display_result=offline,
        online_result=None,
    )

    assert "冻结 prediction record" in status
    assert "本会话未运行模型" in status
