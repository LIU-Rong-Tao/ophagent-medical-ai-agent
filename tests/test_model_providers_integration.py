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
    statuses = {record.source_checkpoint_id: record.base_adapter_status for record in checkpoints}
    assert statuses == {
        "retfound-cfp": BaseAdapterStatus.SMOKE_TEST_PASSED,
        "retfound-oct": BaseAdapterStatus.NOT_IMPLEMENTED,
    }
    assert all(
        record.task_compatibility_status is TaskCompatibilityStatus.ADAPTATION_REQUIRED
        for record in checkpoints
    )
    assert not any(record.route_eligible for record in checkpoints)


def test_real_flair_metadata_and_checkpoint_are_preserved():
    record = OphBenchProvider().get_model("ophbench::flair::flair-default")

    assert record.model_name == "FLAIR"
    assert record.architecture == "ResNet-50图像编码器 + BioClinicalBERT文本编码器"
    assert record.source_checkpoint_id == "flair-default"
    assert record.checkpoint_name == "Default"
    assert record.paper_url and record.code_url
    assert record.modalities
    assert record.pretraining_strategy
