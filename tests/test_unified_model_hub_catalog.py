from types import SimpleNamespace

import pandas as pd

from app.model_hub_data import (
    build_unified_model_catalog,
    route_eligible_model_ids,
)
from app.model_hub_engineering import (
    _format_file_size,
    _runtime_architecture,
    architecture_display,
    filter_global_model_catalog,
    partition_model_catalog,
    source_access_display,
    ui_value,
)
from app.model_providers import OphBenchProvider, TimmProvider


def _models():
    return pd.DataFrame(
        [
            {
                "model_id": "aptos::convnext_task_v1",
                "task_id": "aptos2019",
                "dataset_id": "aptos2019",
                "dataset_display_name": "APTOS 2019",
                "dataset_source": "public",
                "artifact_id": "convnext_task_v1",
                "model_family": "convnext",
                "architecture": "convnext_tiny",
                "label_space": "aptos_5class",
                "n_classes": 5,
                "prediction_source": "adapter",
                "adapter_status": "completed",
                "compatibility_status": "ready_for_pairing",
                "task_checkpoint": True,
                "task_inference_ready": True,
                "route_eligible": True,
                "role_candidates": "scout|expert",
                "pretraining_source": "imagenet1k",
            }
        ]
    )


def _snapshot():
    implementation = SimpleNamespace(
        adapter_status="not_started", smoke_test_status="not_run", benchmark_status="not_run"
    )
    model = SimpleNamespace(
        model_id="retfound",
        model_name="RETFound",
        capabilities=["image_encoding", "classification"],
        implementation=implementation,
    )
    checkpoint = SimpleNamespace(
        checkpoint_id="retfound-cfp",
        model_id="retfound",
        checkpoint_name="CFP",
        modalities=["CFP"],
        weight_url="https://example.test/retfound",
        access_type="auth_required",
        requires_auth=True,
        verification_status="seed_unverified",
    )
    return SimpleNamespace(
        models=(model,),
        checkpoints=(checkpoint,),
        model_count=1,
        checkpoint_count=1,
        package_version="0.1.1",
        schema_version="1.0",
        registry_source="package:ophbench/_registry_data",
    )


def test_unified_catalog_contains_three_provider_sources():
    catalog = build_unified_model_catalog(
        _models(),
        target_task_id="aptos2019",
        recipes=pd.DataFrame(),
        providers=[
            TimmProvider([{"model_id": "convnext_tiny", "display_name": "ConvNeXt-Tiny"}]),
            OphBenchProvider(snapshot_loader=_snapshot),
        ],
    )

    assert set(catalog["provider_id"]) == {"timm", "ophbench", "local_artifact"}
    assert len(catalog["model_id"]) == len(set(catalog["model_id"]))


def test_external_foundation_model_cannot_enter_routing_selection():
    catalog = build_unified_model_catalog(
        _models(),
        target_task_id="aptos2019",
        recipes=pd.DataFrame(),
        providers=[OphBenchProvider(snapshot_loader=_snapshot)],
    )
    external = catalog.loc[catalog["provider_id"].eq("ophbench")].iloc[0]
    local = catalog.loc[catalog["provider_id"].eq("local_artifact")].iloc[0]

    assert external["task_checkpoint"] == False  # noqa: E712
    assert external["task_inference_ready"] == False  # noqa: E712
    assert external["route_eligible"] == False  # noqa: E712
    assert local["task_checkpoint"] == True  # noqa: E712
    assert local["route_eligible"] == True  # noqa: E712
    assert route_eligible_model_ids(catalog) == [local["model_id"]]


def test_provider_filter_and_dependency_health_are_preserved():
    def missing():
        raise ModuleNotFoundError("ophbench unavailable")

    catalog = build_unified_model_catalog(
        _models(),
        target_task_id="aptos2019",
        recipes=pd.DataFrame(),
        providers=[TimmProvider([{"model_id": "vit_base_patch16"}]), OphBenchProvider(missing)],
    )
    filtered = filter_global_model_catalog(catalog, provider="timm")

    assert set(filtered["provider_id"]) == {"timm"}
    health = {item.provider_id: item for item in catalog.attrs["provider_health"]}
    assert health["ophbench"].code == "dependency_unavailable"
    assert health["timm"].available is True


def test_default_layer_contains_only_ready_task_checkpoints():
    catalog = build_unified_model_catalog(
        _models(),
        target_task_id="aptos2019",
        recipes=pd.DataFrame(),
        providers=[OphBenchProvider(snapshot_loader=_snapshot)],
    )
    layers = partition_model_catalog(catalog)

    assert list(layers) == [
        "在线可用任务模型",
        "离线预测回放资产",
        "可适配基础模型",
        "候选基础模型库",
    ]
    assert layers["在线可用任务模型"]["task_checkpoint"].all()
    assert layers["在线可用任务模型"]["task_inference_ready"].all()
    assert not layers["候选基础模型库"]["route_eligible"].any()


def test_frozen_prediction_is_replay_only_not_online() -> None:
    models = _models()
    models["task_inference_ready"] = False
    models["route_eligible"] = False
    catalog = build_unified_model_catalog(
        models,
        target_task_id="aptos2019",
        recipes=pd.DataFrame(),
        providers=[],
    )
    layers = partition_model_catalog(catalog)
    assert layers["在线可用任务模型"].empty
    assert len(layers["离线预测回放资产"]) == 1
    row = layers["离线预测回放资产"].iloc[0]
    assert row["replay_eligible"] == True  # noqa: E712
    assert row["route_eligible"] == False  # noqa: E712


def test_ui_missing_and_unverified_open_values_are_human_readable():
    assert ui_value(float("nan")) == "尚未登记"
    assert ui_value(None) == "尚未登记"
    assert ui_value("missing") == "尚未登记"
    row = pd.Series(
        {"source_access_status": "open", "checkpoint_verification_status": "seed_unverified"}
    )
    assert source_access_display(row) == "登记为开放，尚未核验"
    assert architecture_display("ResNet-50图像编码器 + BioClinicalBERT文本编码器") == (
        "ResNet-50 图像编码器 + BioClinicalBERT 文本编码器"
    )
    assert _format_file_size(40 * 1024) == "40.0 KB"
    assert _runtime_architecture(pd.Series({"model_family": "retfound"})) == (
        "ViT-Large/16 图像编码器"
    )


def test_superseded_task_artifact_cannot_be_reenabled_by_catalog_classification():
    models = _models()
    models["task_inference_ready"] = False
    models["route_eligible"] = False
    models["lifecycle_status"] = "superseded"
    catalog = build_unified_model_catalog(
        models,
        target_task_id="aptos2019",
        recipes=pd.DataFrame(),
        providers=[],
    )
    row = catalog.iloc[0]
    assert row["task_checkpoint"] == True  # noqa: E712
    assert row["task_inference_ready"] == False  # noqa: E712
    assert row["route_eligible"] == False  # noqa: E712
