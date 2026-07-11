from app.model_providers import (
    BaseAdapterStatus,
    OphBenchProvider,
    SourceAccessStatus,
    TaskCompatibilityStatus,
)


def test_real_ophbench_provider_reads_packaged_registry():
    provider = OphBenchProvider()

    assert provider.is_available() is True
    assert provider.health().metadata["model_count"] == 15
    assert provider.health().metadata["checkpoint_count"] == 27
    assert len(provider.list_models()) == 27
    assert len({record.source_model_id for record in provider.list_models()}) == 15


def test_real_retfound_provider_statuses_are_safe():
    provider = OphBenchProvider()
    checkpoints = provider.list_checkpoints("retfound")

    assert len(checkpoints) == 2
    assert {record.source_checkpoint_id for record in checkpoints} == {
        "retfound-cfp",
        "retfound-oct",
    }
    assert all(
        record.source_access_status is SourceAccessStatus.AUTHENTICATION_REQUIRED
        for record in checkpoints
    )
    assert all(
        record.base_adapter_status is BaseAdapterStatus.NOT_IMPLEMENTED
        for record in checkpoints
    )
    assert all(
        record.task_compatibility_status is TaskCompatibilityStatus.ADAPTATION_REQUIRED
        for record in checkpoints
    )
    assert not any(record.route_eligible for record in checkpoints)
