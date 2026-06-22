from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from app.checkpoints import ModelArtifact
from app.inference import (
    InferenceResult,
    postprocess_probabilities,
    run_single_image_inference,
)


def artifact(tmp_path: Path, **overrides) -> ModelArtifact:
    checkpoint = tmp_path / "model_best.pth"
    config = tmp_path / "config.json"
    mapping = tmp_path / "class_to_idx.json"
    checkpoint.write_bytes(b"weights")
    config.write_text(
        '{"num_classes": 5, "image_size": 224}',
        encoding="utf-8",
    )
    mapping.write_text(
        (
            '{"anodr": 0, "bmilddr": 1, "cmoderatedr": 2, '
            '"dseveredr": 3, "eproliferativedr": 4}'
        ),
        encoding="utf-8",
    )
    base = ModelArtifact(
        model_key="convnext_tiny",
        display_name="ConvNeXt-Tiny",
        protocol_id="dr_icdr_5class_proxy_v1",
        loader_model_name="convnext_tiny",
        experiment_dir=tmp_path,
        checkpoint_path=checkpoint,
        config_path=config,
        class_to_idx_path=mapping,
        env_info_path=None,
        metrics_path=None,
        test_predictions_path=None,
        summary_path=None,
        checkpoint_meta_path=None,
        checkpoint_size=checkpoint.stat().st_size,
        checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
        prediction_csv_sha256=None,
        prediction_columns=(),
        generated_time_or_unknown="unknown",
        commit_or_unknown="unknown",
        num_best_checkpoints=1,
        artifact_status="inference_only",
        static_complete=False,
        can_attempt_load=True,
    )
    return replace(base, **overrides)


def test_unsupported_loader_returns_fixed_error_stage(tmp_path: Path):
    item = artifact(
        tmp_path,
        loader_model_name="",
        can_attempt_load=False,
        checkpoint_path=None,
    )

    result = run_single_image_inference(
        Image.new("RGB", (16, 16)),
        item,
    )

    assert isinstance(result, InferenceResult)
    assert not result.ok
    assert result.stage == "unsupported_loader"
    assert result.error_type == "UnsupportedLoader"
    assert result.probabilities is None


def test_probability_postprocessing_returns_complete_five_grade_metrics():
    result = postprocess_probabilities(
        [0.10, 0.20, 0.40, 0.20, 0.10],
        display_name="ConvNeXt-Tiny",
        backbone="convnext_tiny",
    )

    assert result.ok
    assert result.stage == "complete"
    assert result.pred_grade == 2
    assert result.confidence == pytest.approx(0.40)
    assert result.margin == pytest.approx(0.20)
    assert result.entropy_norm is not None
    assert 0.0 <= result.entropy_norm <= 1.0
    assert result.source == "ConvNeXt-Tiny 在线 checkpoint 推理"
    assert result.to_display_payload()["probabilities"] == pytest.approx(
        [0.10, 0.20, 0.40, 0.20, 0.10]
    )


def test_probability_postprocessing_never_silently_returns_none():
    result = postprocess_probabilities(
        [0.6, 0.4],
        display_name="Broken",
        backbone="broken",
    )

    assert isinstance(result, InferenceResult)
    assert not result.ok
    assert result.stage == "postprocess"
    assert result.error_type == "InvalidProbabilityOutput"


def test_runtime_failure_is_reported_with_the_exact_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    item = artifact(tmp_path)

    def fail_build(*args, **kwargs):
        raise RuntimeError("timm model mismatch")

    monkeypatch.setattr("app.inference._build_model", fail_build)
    result = run_single_image_inference(
        Image.new("RGB", (16, 16)),
        item,
        runtime_loader=lambda: (object(), object(), object()),
    )

    assert not result.ok
    assert result.stage == "build_model"
    assert result.error_type == "RuntimeError"
    assert "timm model mismatch" in result.error_message


def test_checkpoint_failure_is_reported_as_load_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    item = artifact(tmp_path)

    class FakeModel:
        def load_state_dict(self, *args, **kwargs):
            raise RuntimeError("classifier head size mismatch")

    class FakeTimm:
        @staticmethod
        def create_model(*args, **kwargs):
            return FakeModel()

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def load(*args, **kwargs):
            return {"head.weight": object()}

    result = run_single_image_inference(
        Image.new("RGB", (16, 16)),
        item,
        runtime_loader=lambda: (FakeTorch(), FakeTimm(), object()),
    )

    assert not result.ok
    assert result.stage == "load_checkpoint"
    assert result.error_type == "RuntimeError"
    assert "head size mismatch" in result.error_message
