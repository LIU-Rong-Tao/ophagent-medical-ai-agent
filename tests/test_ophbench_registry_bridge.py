from types import SimpleNamespace

from app.ophbench_registry_bridge import (
    BridgeErrorCode,
    ExternalModelStatus,
    SourceMetadata,
    build_external_catalog,
    load_external_catalog,
)


def _model(
    model_id="retfound",
    *,
    adapter_status="not_started",
    runtime_phase="phase1_image_encoder",
    smoke_test_status="not_run",
):
    return SimpleNamespace(
        model_id=model_id,
        model_name="RETFound",
        modalities=["CFP", "OCT"],
        runtime_phase=runtime_phase,
        capabilities=["image_encoding", "classification"],
        verification_status="seed_unverified",
        license_verified=False,
        implementation=SimpleNamespace(
            adapter_status=adapter_status,
            smoke_test_status=smoke_test_status,
            benchmark_status="not_run",
        ),
    )


def _checkpoint(checkpoint_id, *, requires_auth=True, weight_url="https://example.test/w"):
    return SimpleNamespace(
        checkpoint_id=checkpoint_id,
        model_id="retfound",
        checkpoint_name=checkpoint_id,
        modalities=["CFP"],
        weight_url=weight_url,
        access_type="auth_required" if requires_auth else "open",
        requires_auth=requires_auth,
        verification_status="seed_unverified",
    )


def _source():
    return SourceMetadata(
        package_version="0.1.0",
        commit_sha="073194f",
        registry_root="/external/registry",
        loaded_at="2026-07-11T00:00:00Z",
    )


def test_unimplemented_model_is_catalogued_but_not_runnable():
    result = build_external_catalog(
        [_model()],
        [_checkpoint("retfound-cfp"), _checkpoint("retfound-oct")],
        source=_source(),
    )

    entry = result.models[0]
    assert result.model_count == 1
    assert result.checkpoint_count == 2
    assert entry.lifecycle_status is ExternalModelStatus.ADAPTER_UNAVAILABLE
    assert ExternalModelStatus.AUTHENTICATION_REQUIRED in entry.statuses
    assert ExternalModelStatus.TASK_ADAPTATION_REQUIRED in entry.statuses
    assert entry.runnable is False


def test_external_pretraining_checkpoint_is_not_a_task_checkpoint():
    result = build_external_catalog(
        [_model(adapter_status="implemented", smoke_test_status="passed")],
        [_checkpoint("retfound-cfp", requires_auth=False)],
        source=_source(),
    )

    entry = result.models[0]
    assert entry.lifecycle_status is ExternalModelStatus.ADAPTER_READY
    assert entry.task_checkpoint is False
    assert entry.direct_inference_ready is False
    assert ExternalModelStatus.TASK_ADAPTATION_REQUIRED in entry.statuses
    row = result.to_model_hub_rows()[0]
    assert row["source"] == "external_ophbench"
    assert row["target_task_status"] == "blocked"
    assert row["task_checkpoint"] is False


def test_missing_weight_url_is_reported():
    result = build_external_catalog(
        [_model()],
        [_checkpoint("retfound-cfp", weight_url=None)],
        source=_source(),
    )
    assert ExternalModelStatus.WEIGHTS_MISSING in result.models[0].statuses


def test_missing_optional_dependency_returns_structured_error():
    def unavailable():
        raise ModuleNotFoundError("No module named 'ophbench'")

    result = load_external_catalog(dependency_resolver=unavailable)

    assert result.available is False
    assert result.error.code is BridgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert "optional" in result.error.message.lower()


def test_invalid_registry_returns_structured_error():
    def dependency():
        return (lambda _root: (_ for _ in ()).throw(ValueError("invalid yaml"))), _source()

    result = load_external_catalog(dependency_resolver=dependency)

    assert result.available is False
    assert result.error.code is BridgeErrorCode.REGISTRY_INVALID
    assert "invalid yaml" in result.error.detail


def test_retfound_keeps_two_checkpoints_and_authentication_state():
    result = build_external_catalog(
        [_model()],
        [_checkpoint("retfound-cfp"), _checkpoint("retfound-oct")],
        source=_source(),
    )
    entry = result.models[0]
    assert [item.checkpoint_id for item in entry.checkpoints] == [
        "retfound-cfp",
        "retfound-oct",
    ]
    assert all(item.requires_auth for item in entry.checkpoints)
    assert result.source.commit_sha == "073194f"
