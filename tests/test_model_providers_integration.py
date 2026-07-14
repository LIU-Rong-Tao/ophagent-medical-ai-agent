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
    assert provider.health().metadata["manifest_checkpoint_count"] == 27


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


def test_final_registry_contract_metadata_is_preserved():
    provider = OphBenchProvider()

    eyeclip = provider.get_model("ophbench::eyeclip::eyeclip-default")
    assert "corneal_photography" in eyeclip.model_modalities
    assert "CT" not in eyeclip.model_modalities
    assert "ViT-B/32" in eyeclip.architecture

    keepfit_modalities = {
        modality
        for record in provider.list_checkpoints("keepfit")
        for modality in record.model_modalities
    }
    assert keepfit_modalities == {"CFP", "FFA", "text"}

    mirage = provider.get_model("ophbench::mirage::mirage-base")
    assert set(mirage.modalities) == {"OCT", "SLO"}

    visionunite = provider.get_model(
        "ophbench::visionunite::visionunite-default"
    )
    assert set(visionunite.modalities) == {"CFP", "text"}

    fmue = provider.get_model("ophbench::fmue::fmue-default")
    assert fmue.artifact_type == "task_checkpoint"


def test_final_registry_statuses_are_checkpoint_scoped_and_conservative():
    records = OphBenchProvider().list_models()

    assert len(records) == 27
    assert all(record.catalog_registered for record in records)
    assert all(record.official_source_verified for record in records)
    assert not any(
        record.checkpoint_verification_status == "seed_unverified"
        for record in records
    )
    assert sum(record.download_status == "downloaded" for record in records) == 19
    assert sum(record.encoder_smoke_passed for record in records) == 1
    assert sum(record.task_inference_ready for record in records) == 0
    assert sum(record.route_eligible for record in records) == 0

    cfp = next(
        record for record in records if record.source_checkpoint_id == "retfound-cfp"
    )
    oct_record = next(
        record for record in records if record.source_checkpoint_id == "retfound-oct"
    )
    assert cfp.adapter_implemented and cfp.encoder_smoke_passed
    assert not cfp.local_asset_verified
    assert not cfp.task_adapted
    assert not cfp.task_inference_ready
    assert not cfp.route_eligible
    assert not oct_record.adapter_implemented
    assert not oct_record.encoder_smoke_passed


def test_visionfm_legacy_assets_are_not_treated_as_locally_verified():
    records = OphBenchProvider().list_checkpoints("visionfm")

    assert len(records) == 8
    assert all(
        record.local_asset_status == "legacy_asset_not_reverified"
        for record in records
    )
    assert not any(record.local_asset_verified for record in records)
    assert not any(record.route_eligible for record in records)


def test_real_flair_metadata_and_checkpoint_are_preserved():
    record = OphBenchProvider().get_model("ophbench::flair::flair-default")

    assert record.model_name == "FLAIR"
    assert record.architecture == "ResNet-50图像编码器 + BioClinicalBERT文本编码器"
    assert record.source_checkpoint_id == "flair-default"
    assert record.checkpoint_name == "Default"
    assert record.paper_url and record.code_url
    assert record.modalities
    assert record.pretraining_strategy
