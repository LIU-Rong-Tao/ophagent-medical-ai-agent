from types import SimpleNamespace

from app.model_providers import (
    BaseAdapterStatus,
    LocalArtifactProvider,
    OphBenchProvider,
    SourceAccessStatus,
    TaskCompatibilityStatus,
    TimmProvider,
    build_provider_catalog,
)


def _snapshot():
    implementation = SimpleNamespace(
        adapter_status="not_started",
        smoke_test_status="not_run",
        benchmark_status="not_run",
    )
    model = SimpleNamespace(
        model_id="retfound",
        model_name="RETFound",
        modalities=["CFP", "OCT"],
        capabilities=["image_encoding", "classification"],
        runtime_phase="phase1_image_encoder",
        verification_status="seed_unverified",
        implementation=implementation,
    )
    checkpoints = [
        SimpleNamespace(
            checkpoint_id="retfound-cfp",
            model_id="retfound",
            checkpoint_name="CFP",
            modalities=["CFP"],
            weight_url="https://example.test/cfp",
            access_type="auth_required",
            requires_auth=True,
            verification_status="seed_unverified",
        ),
        SimpleNamespace(
            checkpoint_id="retfound-oct",
            model_id="retfound",
            checkpoint_name="OCT",
            modalities=["OCT"],
            weight_url="https://example.test/oct",
            access_type="auth_required",
            requires_auth=True,
            verification_status="seed_unverified",
        ),
    ]
    return SimpleNamespace(
        models=(model,),
        checkpoints=tuple(checkpoints),
        model_count=1,
        checkpoint_count=2,
        package_version="0.1.1",
        schema_version="1.0",
        registry_source="package:ophbench/_registry_data",
    )


def test_three_providers_construct_independently():
    timm = TimmProvider([{"model_id": "convnext_tiny", "display_name": "ConvNeXt-Tiny"}])
    ophbench = OphBenchProvider(snapshot_loader=_snapshot)
    local = LocalArtifactProvider(
        [
            {
                "artifact_id": "aptos-convnext-v1",
                "display_name": "APTOS ConvNeXt",
                "task_id": "aptos2019",
                "route_eligible": True,
            }
        ]
    )

    assert timm.is_available() and ophbench.is_available() and local.is_available()
    assert timm.health().provider_id == "timm"
    assert ophbench.health().provider_id == "ophbench"
    assert local.health().provider_id == "local_artifact"


def test_ophbench_status_dimensions_are_independent():
    provider = OphBenchProvider(snapshot_loader=_snapshot)
    records = provider.list_models()

    assert len(records) == 2
    cfp = provider.get_model("ophbench::retfound::retfound-cfp")
    assert cfp.source_access_status is SourceAccessStatus.AUTHENTICATION_REQUIRED
    assert cfp.base_adapter_status is BaseAdapterStatus.NOT_IMPLEMENTED
    assert cfp.task_compatibility_status is TaskCompatibilityStatus.ADAPTATION_REQUIRED
    assert cfp.base_adapter_ready is False
    assert cfp.task_inference_ready is False
    assert cfp.route_eligible is False
    assert cfp.task_checkpoint is False
    assert cfp.runnable is False


def test_ophbench_retfound_lists_two_checkpoints():
    provider = OphBenchProvider(snapshot_loader=_snapshot)
    records = provider.list_checkpoints("retfound")
    assert [record.source_checkpoint_id for record in records] == [
        "retfound-cfp",
        "retfound-oct",
    ]


def test_local_task_artifact_is_route_eligible_and_namespaced():
    provider = LocalArtifactProvider(
        [
            {
                "artifact_id": "aptos2019-retfound-cfp-linear-probe-v1",
                "display_name": "RETFound APTOS linear probe",
                "task_id": "aptos2019",
                "route_eligible": True,
            }
        ]
    )
    record = provider.list_models()[0]
    assert record.unified_model_id == (
        "local_artifact::aptos2019-retfound-cfp-linear-probe-v1"
    )
    assert record.task_checkpoint is True
    assert record.task_inference_ready is True
    assert record.route_eligible is True


def test_unified_ids_do_not_collide_across_providers():
    providers = [
        TimmProvider([{"model_id": "retfound", "display_name": "Generic"}]),
        OphBenchProvider(snapshot_loader=_snapshot),
        LocalArtifactProvider(
            [{"artifact_id": "retfound", "display_name": "Task", "task_id": "aptos2019"}]
        ),
    ]
    catalog = build_provider_catalog(providers)
    ids = [record.unified_model_id for record in catalog.records]
    assert len(ids) == len(set(ids))


def test_missing_ophbench_dependency_does_not_affect_other_providers():
    def missing():
        raise ModuleNotFoundError("No module named 'ophbench'")

    ophbench = OphBenchProvider(snapshot_loader=missing)
    timm = TimmProvider([{"model_id": "convnext_tiny"}])
    local = LocalArtifactProvider([])
    catalog = build_provider_catalog([timm, ophbench, local])

    assert ophbench.is_available() is False
    assert ophbench.health().code == "dependency_unavailable"
    assert {record.provider_id for record in catalog.records} == {"timm"}


def test_registry_error_is_structured_and_not_swallowed():
    def broken():
        raise ValueError("invalid registry")

    provider = OphBenchProvider(snapshot_loader=broken)
    assert provider.is_available() is False
    assert provider.health().code == "registry_invalid"
    assert "invalid registry" in provider.health().detail
