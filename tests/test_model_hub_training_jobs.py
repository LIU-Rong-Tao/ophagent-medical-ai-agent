from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.datasets.imagefolder_classification import inspect_imagefolder_dataset, should_pin_memory
import app.training_jobs as training_jobs_module
from app.model_hub_inference_jobs import checkpoint_inference_capability, checkpoint_loader_spec
from scripts.training.train_timm_classifier import (
    TrainingConfig,
    build_optimizer,
    build_scheduler,
    compute_class_weights,
    create_timm_model_with_pretrained_fallback,
    extract_checkpoint_state,
    freeze_model_backbone,
    load_training_config,
    publish_training_run,
    summarize_forward_cost,
)
from scripts.routing.timm_adapter_runtime import timm_model_create_kwargs
from app.training_jobs import (
    TrainingRequest,
    archive_training_job,
    build_adaptation_request,
    build_training_context,
    build_retry_request,
    cancel_training_job,
    delete_training_job_and_outputs,
    list_training_jobs,
    prepare_training_subprocess_environment,
    registered_dataset_options,
    read_job_log_tail,
    read_job_status,
    submit_training_job,
    update_job_status,
    validate_training_request,
)
from scripts.training.run_training_job import execute_job
from app.training_config import build_training_draft, compile_effective_config, dump_yaml, load_yaml
from app.training_config import discover_training_recipes


def create_imagefolder(root: Path, classes: tuple[str, ...] = ("class_a", "class_b")) -> None:
    for split in ("train", "val", "test"):
        for class_name in classes:
            directory = root / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{split}_{class_name}.png").write_bytes(b"fixture")


def create_aptos_imagefolder(root: Path) -> None:
    classes = ("anodr", "bmilddr", "cmoderatedr", "dseveredr", "eproliferativedr")
    for split in ("train", "val", "test"):
        for index, class_name in enumerate(classes):
            directory = root / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{split}_{index}.png").write_bytes(b"fixture")


def test_retfound_standard_recipe_is_discoverable_and_compilable(tmp_path: Path) -> None:
    recipe_root = ROOT / "experiments/model_hub/registry/training_recipes"
    recipes = discover_training_recipes(recipe_root)
    base = next(
        item for item in recipes if item["recipe"]["recipe_id"] == "ophbench_retfound_linear_probe_v1"
    )
    checkpoint = tmp_path / "retfound.pth"
    checkpoint.write_bytes(b"checkpoint")
    context = {
        "task_id": "aptos_dr_5class",
        "dataset_id": "APTOS2019",
        "artifact_id": "aptos2019-retfound-cfp-linear-probe-v2",
        "source_artifact_id": "retfound-cfp",
        "source_task_id": "",
        "trainer_adapter": "ophbench_retfound_linear_probe_v1",
        "model_family": "retfound",
        "architecture": "retfound-mae-vit-large-patch16-256",
        "data_root": str(tmp_path / "data"),
        "num_classes": 5,
        "class_to_idx": {
            "anodr": 0,
            "bmilddr": 1,
            "cmoderatedr": 2,
            "dseveredr": 3,
            "eproliferativedr": 4,
        },
        "label_space": "dr_icdr_0_4",
        "label_structure": "ordinal",
        "output_dir": str(tmp_path / "run"),
        "display_metrics": ["accuracy", "macro_f1", "quadratic_kappa"],
        "source_checkpoint_path": str(checkpoint),
        "encoder_checkpoint_sha256": "a" * 64,
        "source_num_classes": 0,
    }
    draft = build_training_draft(base, context)
    effective, report = compile_effective_config(draft, context)

    assert report["trainer_adapter"] == "ophbench_retfound_linear_probe_v1"
    assert effective["classifier"]["c_candidates"] == [0.01, 0.1, 1.0]
    assert effective["classifier"]["max_iter"] == 2000


def test_retfound_standard_job_can_be_submitted_to_background_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "APTOS2019"
    create_aptos_imagefolder(data_root)
    checkpoint = tmp_path / "retfound.pth"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "run-v2"
    base = load_yaml(
        ROOT / "experiments/model_hub/registry/training_recipes/ophbench_retfound_linear_probe_v1.yaml"
    )
    request = TrainingRequest(
        task_id="aptos_dr_5class",
        dataset_id="APTOS2019",
        artifact_id="aptos2019-retfound-cfp-linear-probe-v2",
        source_artifact_id="retfound-cfp",
        model_family="retfound",
        architecture="retfound-mae-vit-large-patch16-256",
        data_root=str(data_root),
        num_classes=5,
        output_dir=str(output_dir),
        recipe_id="ophbench_retfound_linear_probe_v1",
        label_space="dr_icdr_0_4",
        label_structure="ordinal",
        source_checkpoint_path=str(checkpoint),
        encoder_checkpoint_sha256="a" * 64,
        trainer_adapter="ophbench_retfound_linear_probe_v1",
        display_metrics=["accuracy", "macro_f1", "quadratic_kappa"],
        base_recipe=base,
    )
    inspection = inspect_imagefolder_dataset(data_root)
    context = build_training_context(request, inspection)
    request = replace(request, submitted_config=build_training_draft(base, context))
    monkeypatch.setattr(
        "scripts.training.train_ophbench_retfound_linear_probe.strict_preflight",
        lambda config: {"strict_preflight": True},
    )

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr("app.training_jobs.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    job_id = submit_training_job(request, tmp_path / "jobs", tmp_path / "unused.csv")

    assert job_id
    assert (output_dir / "base_recipe.yaml").is_file()
    assert (output_dir / "submitted_config.yaml").is_file()
    assert (output_dir / "effective_config.yaml").is_file()
    assert (output_dir / "validation_report.json").is_file()


def create_recipe_registry(path: Path) -> Path:
    path.write_text(
        "recipe_id,model_family,architecture,image_size,batch_size,num_epochs,learning_rate,pretrained,weight_decay,label_smoothing,num_workers,enabled\n"
        "convnext_screening,convnext,convnext_tiny,224,2,1,0.0001,0,0.01,0.0,0,1\n",
        encoding="utf-8",
    )
    return path


def make_request(tmp_path: Path, *, recipe_id: str = "convnext_screening") -> TrainingRequest:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    return TrainingRequest(
        task_id="mock_task",
        dataset_id="mock_dataset",
        artifact_id="convnext_tiny_mock",
        model_family="convnext",
        architecture="convnext_tiny",
        data_root=str(data_root),
        num_classes=2,
        output_dir=str(tmp_path / "trained_model"),
        recipe_id=recipe_id,
        label_structure="nominal",
    )


def test_imagefolder_inspection_reports_mapping_and_split_sizes(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)

    inspection = inspect_imagefolder_dataset(data_root)

    assert inspection.class_to_idx == {"class_a": 0, "class_b": 1}
    assert inspection.split_sizes == {"train": 2, "val": 2, "test": 2}


def test_imagefolder_inspection_rejects_inconsistent_classes(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    (data_root / "test" / "class_b" / "test_class_b.png").unlink()
    (data_root / "test" / "class_b").rmdir()

    with pytest.raises(ValueError, match="类别目录不一致"):
        inspect_imagefolder_dataset(data_root)


def test_pinned_memory_is_disabled_for_explicit_cpu_training() -> None:
    assert not should_pin_memory("cpu", cuda_available=True)
    assert not should_pin_memory("auto", cuda_available=False)
    assert should_pin_memory("auto", cuda_available=True)
    assert should_pin_memory("cuda:1", cuda_available=True)


def test_training_config_is_task_aware_and_rejects_class_count_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "task_id": "glaucoma_3class",
                "dataset_id": "Glaucoma_fundus",
                "data_root": str(data_root),
                "architecture": "convnext_tiny",
                "num_classes": 3,
                "image_size": 224,
                "batch_size": 2,
                "num_epochs": 1,
                "learning_rate": 0.0001,
                "pretrained": False,
                "seed": 42,
                "output_dir": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_classes"):
        load_training_config(config_path)


def test_training_config_rejects_existing_nonempty_output(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "data_root": str(data_root),
                "architecture": "convnext_tiny",
                "num_classes": 2,
                "image_size": 224,
                "batch_size": 2,
                "num_epochs": 1,
                "learning_rate": 0.0001,
                "pretrained": False,
                "seed": 42,
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="输出目录"):
        load_training_config(config_path)


def test_training_config_rejects_unknown_device(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "data_root": str(data_root),
                "architecture": "convnext_tiny",
                "num_classes": 2,
                "image_size": 64,
                "batch_size": 2,
                "num_epochs": 1,
                "learning_rate": 0.0001,
                "pretrained": False,
                "seed": 42,
                "device": "tpu",
                "output_dir": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="device"):
        load_training_config(config_path)


def test_training_config_loads_nested_effective_yaml_from_unified_run_bundle(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    output_dir = tmp_path / "run"
    context = {
        "task_id": "mock_task",
        "dataset_id": "mock_dataset",
        "artifact_id": "convnext_adapter",
        "source_artifact_id": "convnext_tiny",
        "source_task_id": "source_task",
        "trainer_adapter": "timm_imagefolder_v1",
        "model_family": "convnext",
        "architecture": "convnext_tiny",
        "data_root": str(data_root),
        "num_classes": 2,
        "class_to_idx": {"class_a": 0, "class_b": 1},
        "label_space": "mock_labels",
        "label_structure": "nominal",
        "output_dir": str(output_dir),
        "display_metrics": ["accuracy", "macro_f1"],
    }
    base = load_yaml(ROOT / "experiments/model_hub/registry/training_recipes/timm_full_train.yaml")
    draft = build_training_draft(base, context)
    draft["optimizer"]["name"] = "sgd"
    draft["scheduler"]["name"] = "step"
    draft["training"]["grad_accum_steps"] = 2
    effective, _report = compile_effective_config(draft, context)
    config_path = dump_yaml(output_dir / "configs" / "effective_config.yaml", effective)

    config = load_training_config(config_path)

    assert config.optimizer_name == "sgd"
    assert config.scheduler_name == "step"
    assert config.grad_accum_steps == 2
    assert config.amp is True
    assert config.save_best_by == "macro_f1"


def test_training_request_requires_enabled_matching_recipe(tmp_path: Path) -> None:
    recipes = create_recipe_registry(tmp_path / "recipes.csv")

    with pytest.raises(ValueError, match="recipe"):
        validate_training_request(make_request(tmp_path, recipe_id="missing"), recipes)


def test_registered_dataset_options_only_exposes_registered_preflight_ready_roots(tmp_path: Path) -> None:
    glaucoma_root = tmp_path / "glaucoma"
    create_imagefolder(glaucoma_root, classes=("normal", "early", "advanced"))
    registry = tmp_path / "tasks.csv"
    pd.DataFrame(
        [
            {
                "task_id": "glaucoma_3class",
                "dataset_id": "glaucoma",
                "data_root": str(glaucoma_root),
                "num_classes": 3,
                "label_structure": "nominal",
                "enabled": True,
            },
            {
                "task_id": "aptos_dr_5class",
                "dataset_id": "aptos",
                "data_root": str(tmp_path / "aptos"),
                "num_classes": 5,
                "label_structure": "ordinal",
                "enabled": True,
            },
        ]
    ).to_csv(registry, index=False)

    options = registered_dataset_options(registry, "glaucoma_3class")

    assert options["dataset_id"].tolist() == ["glaucoma"]
    assert options.iloc[0]["availability_status"] == "ready"
    assert options.iloc[0]["class_to_idx"] == {"advanced": 0, "early": 1, "normal": 2}
    assert options.iloc[0]["split_sizes"] == {"train": 3, "val": 3, "test": 3}


def test_registered_dataset_options_blocks_missing_or_invalid_registered_root(tmp_path: Path) -> None:
    registry = tmp_path / "tasks.csv"
    pd.DataFrame(
        [
            {
                "task_id": "glaucoma_3class",
                "dataset_id": "glaucoma",
                "data_root": str(tmp_path / "missing"),
                "num_classes": 3,
                "label_structure": "nominal",
                "enabled": True,
            }
        ]
    ).to_csv(registry, index=False)

    options = registered_dataset_options(registry, "glaucoma_3class")

    assert options.iloc[0]["availability_status"] == "blocked"
    assert "不存在" in options.iloc[0]["availability_reason"]


def test_cross_task_adaptation_uses_timm_pretrained_instead_of_task_checkpoint(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "vit_dr.pth"
    source_checkpoint.write_bytes(b"checkpoint")
    source_model = pd.Series(
        {
            "artifact_id": "vit_b_imagenet",
            "model_family": "vit",
            "architecture": "vit_base_patch16_224",
            "task_id": "aptos_dr_5class",
            "n_classes": 5,
            "checkpoint_path": str(source_checkpoint),
            "checkpoint_status": "found",
        }
    )
    target_task = pd.Series(
        {
            "task_id": "glaucoma_3class",
            "dataset_id": "Glaucoma_fundus",
            "num_classes": 3,
            "label_structure": "nominal",
        }
    )

    request = build_adaptation_request(
        source_model,
        target_task,
        data_root=str(tmp_path / "glaucoma"),
        recipe_id="vit_base_screening",
        output_dir=str(tmp_path / "output"),
    )

    assert request.source_artifact_id == "vit_b_imagenet"
    assert request.artifact_id == "vit_b_imagenet_glaucoma_3class_adapter"
    assert request.architecture == "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    assert request.task_id == "glaucoma_3class"
    assert request.dataset_id == "Glaucoma_fundus"
    assert request.num_classes == 3
    assert request.artifact_id != request.source_artifact_id
    assert request.source_checkpoint_path == ""
    assert request.source_num_classes == 0
    assert request.initialization_source == "timm_pretrained"


@pytest.mark.parametrize(
    (
        "source_artifact_id",
        "model_family",
        "architecture",
        "target_task_id",
        "expected_artifact_id",
        "expected_architecture",
    ),
    [
        (
            "convnext_tiny_glaucoma_scout",
            "convnext",
            "convnext_tiny",
            "aptos_dr_5class",
            "convnext_tiny_imagenet_aptos_dr_5class_adapter",
            "convnext_tiny.in12k_ft_in1k",
        ),
        (
            "swin_tiny__aptos_dr_5class__adapter",
            "swin",
            "swin_tiny_patch4_window7_224",
            "glaucoma_3class",
            "swin_tiny_imagenet_glaucoma_3class_adapter",
            "swin_tiny_patch4_window7_224.ms_in1k",
        ),
        (
            "vit_b_imagenet",
            "vit",
            "vit_base_patch16_224",
            "glaucoma_3class",
            "vit_b_imagenet_glaucoma_3class_adapter",
            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        ),
        (
            "vit_l_official_like",
            "vit",
            "vit_large_patch16_224",
            "aptos_dr_5class",
            "vit_l_imagenet_aptos_dr_5class_adapter",
            "vit_large_patch16_224.augreg_in21k_ft_in1k",
        ),
    ],
)
def test_timm_pretrained_adaptation_name_uses_backbone_not_source_task(
    tmp_path: Path,
    source_artifact_id: str,
    model_family: str,
    architecture: str,
    target_task_id: str,
    expected_artifact_id: str,
    expected_architecture: str,
) -> None:
    source_model = pd.Series(
        {
            "artifact_id": source_artifact_id,
            "model_family": model_family,
            "architecture": architecture,
            "task_id": "source_task",
            "n_classes": 5,
            "checkpoint_path": str(tmp_path / "source.pth"),
            "checkpoint_status": "found",
        }
    )
    target_task = pd.Series(
        {
            "task_id": target_task_id,
            "dataset_id": "target_dataset",
            "num_classes": 3,
            "label_structure": "nominal",
        }
    )

    request = build_adaptation_request(
        source_model,
        target_task,
        data_root=str(tmp_path / "data"),
        recipe_id="screening",
        output_dir=str(tmp_path / "output"),
        initialization_source="timm_pretrained",
    )

    assert request.artifact_id == expected_artifact_id
    assert request.architecture == expected_architecture
    assert request.source_checkpoint_path == ""


def test_same_task_fresh_finetune_defaults_to_timm_pretrained(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "convnext_dr.pth"
    source_checkpoint.write_bytes(b"checkpoint")
    source_model = pd.Series(
        {
            "artifact_id": "convnext_tiny",
            "model_family": "convnext",
            "architecture": "convnext_tiny",
            "task_id": "aptos_dr_5class",
            "n_classes": 5,
            "checkpoint_path": str(source_checkpoint),
            "checkpoint_status": "found",
        }
    )
    target_task = pd.Series(
        {
            "task_id": "aptos_dr_5class",
            "dataset_id": "APTOS2019",
            "num_classes": 5,
            "label_structure": "ordinal",
        }
    )

    request = build_adaptation_request(
        source_model,
        target_task,
        data_root=str(tmp_path / "aptos"),
        recipe_id="convnext_screening",
        output_dir=str(tmp_path / "output"),
    )

    assert request.initialization_source == "timm_pretrained"
    assert request.source_checkpoint_path == ""
    assert request.source_num_classes == 0


def test_registered_checkpoint_requires_explicit_initialization_choice(tmp_path: Path) -> None:
    source_checkpoint = tmp_path / "convnext_dr.pth"
    source_checkpoint.write_bytes(b"checkpoint")
    source_model = pd.Series(
        {
            "artifact_id": "convnext_tiny",
            "model_family": "convnext",
            "architecture": "convnext_tiny",
            "task_id": "aptos_dr_5class",
            "n_classes": 5,
            "checkpoint_path": str(source_checkpoint),
            "checkpoint_status": "found",
        }
    )
    target_task = pd.Series(
        {
            "task_id": "aptos_dr_5class",
            "dataset_id": "APTOS2019",
            "num_classes": 5,
            "label_structure": "ordinal",
        }
    )

    request = build_adaptation_request(
        source_model,
        target_task,
        data_root=str(tmp_path / "aptos"),
        recipe_id="convnext_screening",
        output_dir=str(tmp_path / "output"),
        initialization_source="registered_checkpoint",
    )

    assert request.initialization_source == "registered_checkpoint"
    assert request.source_checkpoint_path == str(source_checkpoint)
    assert request.source_num_classes == 5


def test_training_parameter_overrides_are_written_to_generated_config(tmp_path: Path) -> None:
    recipes = create_recipe_registry(tmp_path / "recipes.csv")
    request = make_request(tmp_path)
    request = TrainingRequest(
        **{
            **request.__dict__,
            "training_overrides": {
                "image_size": 256,
                "batch_size": 8,
                "num_epochs": 12,
                "learning_rate": 0.0003,
                "weight_decay": 0.02,
                "label_smoothing": 0.05,
                "num_workers": 2,
                "seed": 7,
                "device": "cpu",
            },
        }
    )

    preflight = validate_training_request(request, recipes)

    assert preflight.generated_config["image_size"] == 256
    assert preflight.generated_config["batch_size"] == 8
    assert preflight.generated_config["num_epochs"] == 12
    assert preflight.generated_config["learning_rate"] == pytest.approx(0.0003)
    assert preflight.generated_config["weight_decay"] == pytest.approx(0.02)
    assert preflight.generated_config["label_smoothing"] == pytest.approx(0.05)
    assert preflight.generated_config["num_workers"] == 2
    assert preflight.generated_config["seed"] == 7
    assert preflight.generated_config["device"] == "cpu"


def test_training_wizard_is_yaml_driven_and_cross_task_safe() -> None:
    source = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert "训练初始化" in source
    assert "默认从 timm 登记的原始自然图像预训练权重开始" in source
    assert "跨疾病 checkpoint 迁移（研究实验）" in source
    assert "只有明确选择现有 checkpoint 时" in source
    assert "完整训练配置（YAML）" in source
    assert "保存为新 recipe" in source
    assert "导出命令" in source
    assert "effective_config.yaml" in source
    assert "st.number_input" not in source


def test_training_request_rejects_missing_declared_source_checkpoint(tmp_path: Path) -> None:
    recipes = create_recipe_registry(tmp_path / "recipes.csv")
    request = make_request(tmp_path)
    request = TrainingRequest(
        **{
            **request.__dict__,
            "source_checkpoint_path": str(tmp_path / "missing.pth"),
            "source_num_classes": 2,
            "initialization_source": "registered_checkpoint",
        }
    )

    with pytest.raises(ValueError, match="源 checkpoint"):
        validate_training_request(request, recipes)


def test_forward_cost_summary_uses_repeated_full_pass_timings() -> None:
    summary = summarize_forward_cost(
        total_forward_ms=[10.0, 12.0, 11.0],
        n_images=10,
        artifact_id="vit_glaucoma",
        task_id="glaucoma_3class",
        dataset_id="Glaucoma_fundus",
        model_family="vit",
        batch_size=2,
        device="cuda:0",
        checkpoint_mb=100.0,
        peak_allocated_memory_mb=500.0,
    )

    assert summary["timed_runs"] == 3
    assert summary["median_ms_per_image"] == pytest.approx(1.1)
    assert summary["estimated_forward_ms_per_image"] == pytest.approx(1.1)
    assert summary["timing_scope"] == "forward_only"


def test_checkpoint_state_extraction_supports_common_wrappers_and_module_prefix() -> None:
    state = extract_checkpoint_state(
        {"state_dict": {"module.layer.weight": "weight", "module.layer.bias": "bias"}}
    )

    assert state == {"layer.weight": "weight", "layer.bias": "bias"}


def test_optimizer_scheduler_and_freeze_settings_are_executable() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 2)

        def get_classifier(self):
            return self.head

    model = TinyModel()
    config = SimpleNamespace(
        optimizer_name="sgd",
        learning_rate=0.01,
        weight_decay=0.001,
        optimizer_momentum=0.8,
        scheduler_name="step",
        scheduler_warmup_epochs=0,
        scheduler_step_size=2,
        scheduler_gamma=0.5,
        scheduler_minimum_learning_rate=0.0,
        num_epochs=5,
    )

    freeze_model_backbone(model)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    assert not model.backbone.weight.requires_grad
    assert model.head.weight.requires_grad
    assert optimizer.__class__.__name__ == "SGD"
    assert scheduler.__class__.__name__ == "StepLR"


def test_auto_class_weights_use_inverse_training_frequency() -> None:
    torch = pytest.importorskip("torch")
    weights = compute_class_weights([0, 0, 0, 1], num_classes=2, device=torch.device("cpu"))

    assert weights.tolist() == pytest.approx([2 / 3, 2.0])


def test_publish_training_run_writes_manifest_and_registration_record(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "glaucoma" / "vit" / "run-1"
    checkpoint = output / "checkpoints" / "best.pth"
    predictions = output / "evaluation" / "test" / "test_predictions.csv"
    metrics_path = output / "evaluation" / "test" / "metrics.json"
    config_path = output / "configs" / "config.json"
    class_map_path = output / "configs" / "class_to_idx.json"
    for path in (checkpoint, predictions, metrics_path, config_path, class_map_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    cost_path = output / "forward_cost_summary.csv"
    pd.DataFrame([{"estimated_forward_ms_per_image": 1.2}]).to_csv(cost_path, index=False)
    config = TrainingConfig(
        task_id="glaucoma_3class",
        dataset_id="Glaucoma_fundus",
        data_root=str(tmp_path / "dataset"),
        architecture="vit_base_patch16_224",
        num_classes=3,
        image_size=224,
        batch_size=2,
        num_epochs=1,
        learning_rate=0.0001,
        pretrained=True,
        seed=42,
        output_dir=str(output),
        artifact_id="vit_b_imagenet__glaucoma_3class__adapter",
        source_artifact_id="vit_b_imagenet",
        model_family="vit",
        label_space="glaucoma_normal_early_advanced",
        initialization_source="timm_pretrained",
    )

    manifest_path, registration_path = publish_training_run(
        config,
        checkpoint_path=checkpoint,
        predictions_path=predictions,
        metrics_path=metrics_path,
        config_path=config_path,
        class_mapping_path=class_map_path,
        cost_path=cost_path,
        metrics={"accuracy": 0.8, "macro_f1": 0.75, "qwk": None},
    )

    registration = pd.read_csv(registration_path)
    manifest = pd.read_csv(manifest_path)
    assert registration.loc[0, "artifact_id"] == "vit_b_imagenet__glaucoma_3class__adapter"
    assert registration.loc[0, "source_artifact_id"] == "vit_b_imagenet"
    assert registration.loc[0, "prediction_source"] == "adapter"
    assert registration.loc[0, "adapter_status"] == "completed"
    assert manifest["published_path"].str.contains("runtime").sum() == 0
    assert manifest["published_path"].str.contains("work/").sum() == 0
    assert (output / "run_manifest.yaml").is_file()


def test_submit_training_job_writes_stable_records_and_never_uses_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipes = create_recipe_registry(tmp_path / "recipes.csv")
    request = make_request(tmp_path)
    launched: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("app.training_jobs.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "app.training_jobs.prepare_training_subprocess_environment",
        lambda: ({"SAFE_ENV": "1"}, ["已忽略不可用代理"]),
    )
    job_id = submit_training_job(request, tmp_path / "jobs", recipes)
    job_dir = tmp_path / "jobs" / job_id

    assert (job_dir / "request.json").is_file()
    assert (job_dir / "generated_config.json").is_file()
    assert (job_dir / "command.json").is_file()
    assert read_job_status(job_dir)["status"] == "queued"
    assert launched["kwargs"]["shell"] is False
    assert launched["kwargs"]["env"] == {"SAFE_ENV": "1"}
    assert read_job_status(job_dir)["startup_warnings"] == ["已忽略不可用代理"]
    assert isinstance(launched["command"], list)
    assert "run_training_job.py" in " ".join(launched["command"])
    assert not list(job_dir.glob("*.tmp"))


def test_config_driven_job_writes_unified_run_config_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dataset"
    create_imagefolder(data_root)
    output_dir = tmp_path / "runs" / "mock_task" / "convnext_adapter" / "run-1"
    base_recipe = load_yaml(
        ROOT / "experiments/model_hub/registry/training_recipes/timm_quick_smoke.yaml"
    )
    context = {
        "task_id": "mock_task",
        "dataset_id": "mock_dataset",
        "artifact_id": "convnext_adapter",
        "source_artifact_id": "convnext_tiny",
        "source_task_id": "source_task",
        "trainer_adapter": "timm_imagefolder_v1",
        "model_family": "convnext",
        "architecture": "convnext_tiny",
        "data_root": str(data_root),
        "num_classes": 2,
        "class_to_idx": {"class_a": 0, "class_b": 1},
        "label_space": "mock_labels",
        "label_structure": "nominal",
        "output_dir": str(output_dir),
        "display_metrics": ["accuracy", "macro_f1"],
    }
    submitted = build_training_draft(base_recipe, context)
    request = TrainingRequest(
        task_id="mock_task",
        dataset_id="mock_dataset",
        artifact_id="convnext_adapter",
        source_artifact_id="convnext_tiny",
        source_task_id="source_task",
        model_family="convnext",
        architecture="convnext_tiny",
        data_root=str(data_root),
        num_classes=2,
        output_dir=str(output_dir),
        recipe_id="timm_quick_smoke",
        label_space="mock_labels",
        base_recipe=base_recipe,
        submitted_config=submitted,
        display_metrics=["accuracy", "macro_f1"],
    )

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr("app.training_jobs.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    job_id = submit_training_job(request, tmp_path / "runtime" / "jobs", tmp_path / "unused.csv")

    configs = output_dir / "configs"
    assert (configs / "base_recipe.yaml").is_file()
    assert (configs / "submitted_config.yaml").is_file()
    assert (configs / "effective_config.yaml").is_file()
    assert (configs / "validation_report.json").is_file()
    job_status = read_job_status(tmp_path / "runtime" / "jobs" / job_id)
    assert job_status["effective_config_path"] == str((configs / "effective_config.yaml").resolve())
    assert not (tmp_path / "runtime" / "jobs" / job_id / "generated_config.json").exists()


def test_config_driven_retry_updates_locked_output_path(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    submitted = {
        "runtime": {"device": "auto"},
        "output": {"run_dir": request.output_dir},
    }
    request = TrainingRequest(
        **{
            **request.__dict__,
            "training_overrides": {"device": "auto"},
            "submitted_config": submitted,
        }
    )

    retry = build_retry_request(
        request,
        request.output_dir + "-retry",
        device_override="cuda:2",
    )

    assert retry.output_dir.endswith("-retry")
    assert retry.submitted_config["output"]["run_dir"] == retry.output_dir
    assert retry.submitted_config["runtime"]["device"] == "cuda:2"
    assert retry.training_overrides["device"] == "cuda:2"
    assert request.submitted_config["output"]["run_dir"] != retry.output_dir
    assert request.submitted_config["runtime"]["device"] == "auto"
    assert request.training_overrides["device"] == "auto"


def test_job_status_transitions_are_atomic(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    update_job_status(job_dir, "queued")
    update_job_status(job_dir, "running", pid=123)
    update_job_status(job_dir, "succeeded", output_dir="model-output")

    status = read_job_status(job_dir)
    assert status["status"] == "succeeded"
    assert status["output_dir"] == "model-output"
    assert not list(job_dir.glob("*.tmp"))


def test_training_wrapper_records_failure_without_hiding_error(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    config_path = job_dir / "generated_config.json"
    config_path.write_text("{}", encoding="utf-8")
    update_job_status(job_dir, "queued")

    def fail_training(_config_path: Path) -> Path:
        raise RuntimeError("checkpoint load failed")

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        execute_job(job_dir, training_callable=fail_training)

    status = read_job_status(job_dir)
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert "checkpoint load failed" in status["error_message"]


def test_training_wrapper_uses_registered_effective_config_path(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    effective = tmp_path / "run" / "configs" / "effective_config.yaml"
    effective.parent.mkdir(parents=True)
    effective.write_text("schema_version: 1\n", encoding="utf-8")
    update_job_status(job_dir, "queued", effective_config_path=str(effective))
    received: list[Path] = []

    def record_training(config_path: Path) -> Path:
        received.append(config_path)
        return tmp_path / "run"

    execute_job(job_dir, training_callable=record_training)

    assert received == [effective]


def test_cancel_training_job_terminates_recorded_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    update_job_status(job_dir, "running", pid=9876)
    terminated: list[int] = []
    monkeypatch.setattr("app.training_jobs._terminate_pid", lambda pid: terminated.append(pid))

    cancel_training_job(job_dir)

    assert terminated == [9876]
    assert read_job_status(job_dir)["status"] == "cancelled"


def test_job_listing_and_log_tail_are_stable(tmp_path: Path) -> None:
    first = tmp_path / "jobs" / "20260701-a"
    second = tmp_path / "jobs" / "20260701-b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    output = tmp_path / "model-output"
    output.mkdir()
    update_job_status(first, "succeeded", artifact_id="model_a", output_dir=str(output))
    update_job_status(second, "running", artifact_id="model_b", pid=22, output_dir=str(tmp_path / "missing"))
    (second / "train.log").write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    jobs = list_training_jobs(tmp_path / "jobs")

    assert [job["job_id"] for job in jobs] == ["20260701-b", "20260701-a"]
    assert jobs[0]["output_exists"] is False
    assert jobs[1]["output_exists"] is True
    assert read_job_log_tail(second, max_lines=2) == "line-2\nline-3"


def test_archived_terminal_job_is_hidden_by_default_and_can_be_restored(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "20260701-a"
    job_dir.mkdir(parents=True)
    update_job_status(job_dir, "succeeded", artifact_id="model_a")

    archive_training_job(job_dir, archived=True)

    assert list_training_jobs(tmp_path / "jobs") == []
    archived = list_training_jobs(tmp_path / "jobs", include_archived=True)
    assert archived[0]["archived"] is True

    archive_training_job(job_dir, archived=False)

    restored = list_training_jobs(tmp_path / "jobs")
    assert restored[0]["archived"] is False


def test_running_job_cannot_be_archived(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    update_job_status(job_dir, "running", pid=123)

    with pytest.raises(ValueError, match="运行中的任务"):
        archive_training_job(job_dir, archived=True)


def test_delete_terminal_job_removes_its_record_and_controlled_outputs(tmp_path: Path) -> None:
    jobs_root = tmp_path / "runtime" / "training_jobs"
    runs_root = tmp_path / "runs" / "training"
    job_dir = jobs_root / "job-a"
    output_dir = runs_root / "task-a" / "model-a" / "run-a"
    job_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoint.pth").write_bytes(b"weights")
    update_job_status(job_dir, "succeeded", output_dir=str(output_dir))

    delete_training_job_and_outputs(job_dir, jobs_root=jobs_root, runs_root=runs_root)

    assert not job_dir.exists()
    assert not output_dir.exists()


def test_delete_job_refuses_output_outside_controlled_training_root(tmp_path: Path) -> None:
    jobs_root = tmp_path / "runtime" / "training_jobs"
    runs_root = tmp_path / "runs" / "training"
    job_dir = jobs_root / "job-a"
    outside = tmp_path / "do-not-delete"
    job_dir.mkdir(parents=True)
    outside.mkdir()
    update_job_status(job_dir, "failed", output_dir=str(outside))

    with pytest.raises(ValueError, match="受控训练目录"):
        delete_training_job_and_outputs(job_dir, jobs_root=jobs_root, runs_root=runs_root)

    assert job_dir.exists()
    assert outside.exists()


def test_delete_job_with_already_missing_output_still_removes_record(tmp_path: Path) -> None:
    jobs_root = tmp_path / "runtime" / "training_jobs"
    runs_root = tmp_path / "runs" / "training"
    job_dir = jobs_root / "job-a"
    missing_output = runs_root / "task-a" / "missing-run"
    job_dir.mkdir(parents=True)
    update_job_status(job_dir, "cancelled", output_dir=str(missing_output))

    delete_training_job_and_outputs(job_dir, jobs_root=jobs_root, runs_root=runs_root)

    assert not job_dir.exists()


def test_training_environment_drops_only_unreachable_loopback_proxy() -> None:
    environment, warnings = prepare_training_subprocess_environment(
        {
            "HTTP_PROXY": "http://127.0.0.1:7897",
            "HTTPS_PROXY": "http://127.0.0.1:7897",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "KEEP_ME": "yes",
        },
        proxy_reachable=lambda _value: False,
    )

    assert "HTTP_PROXY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert environment["KEEP_ME"] == "yes"
    assert len(warnings) == 1
    assert "127.0.0.1:7897" in warnings[0]


def test_training_environment_keeps_reachable_loopback_proxy() -> None:
    environment, warnings = prepare_training_subprocess_environment(
        {"HTTP_PROXY": "http://127.0.0.1:7897"},
        proxy_reachable=lambda _value: True,
    )

    assert environment["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert warnings == []


def test_timm_pretrained_loader_falls_back_to_official_url_after_hub_failure() -> None:
    calls: list[dict[str, object]] = []

    class FakeCfg:
        url = "https://official.example/model.pth"

    class FakeTimm:
        @staticmethod
        def get_pretrained_cfg(_architecture):
            return FakeCfg()

        @staticmethod
        def create_model(_architecture, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("hub metadata failed")
            return "loaded-from-official-url"

    model = create_timm_model_with_pretrained_fallback(
        FakeTimm,
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=3,
    )

    assert model == "loaded-from-official-url"
    assert calls[0] == {"pretrained": True, "num_classes": 3}
    assert calls[1]["pretrained_cfg_overlay"] == {"hf_hub_id": None}


def test_training_progress_reads_epoch_history_and_summary_from_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    logs = run_dir / "logs"
    evaluation = run_dir / "evaluation" / "test"
    logs.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": 1.2,
                "val_loss": 1.0,
                "val_accuracy": 0.7,
                "val_macro_f1": 0.6,
                "val_qwk": 0.5,
                "learning_rate": 0.0001,
            }
        ]
    ).to_csv(logs / "train_log.csv", index=False)
    (logs / "summary.json").write_text(
        json.dumps({"best_epoch": 1, "best_validation_score": 0.6, "total_train_time_sec": 12.5}),
        encoding="utf-8",
    )
    (evaluation / "metrics.json").write_text(
        json.dumps({"accuracy": 0.72, "macro_f1": 0.63, "qwk": 0.55}),
        encoding="utf-8",
    )

    progress = training_jobs_module.load_training_progress(run_dir)

    assert progress["history"].loc[0, "epoch"] == 1
    assert progress["summary"]["best_epoch"] == 1
    assert progress["test_metrics"]["macro_f1"] == 0.63


def test_known_retfound_checkpoint_loader_is_resolved_automatically(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-best.pth"
    checkpoint.write_bytes(b"fixture")
    row = pd.Series(
        {
            "artifact_id": "retfound_mae_cfp_official_protocol",
            "model_family": "retfound",
            "architecture": "retfound_mae_cfp",
            "checkpoint_path": str(checkpoint),
        }
    )

    spec = checkpoint_loader_spec(row)
    supported, reason = checkpoint_inference_capability(row)

    assert spec == {
        "loader_id": "retfound_mae_cfp_timm_v1",
        "architecture": "vit_large_patch16_224",
        "checkpoint_key": "model",
        "global_pool": "avg",
        "norm": "imagenet",
        "allow_argparse_namespace": True,
    }
    assert supported is True
    assert reason == "可从已登记 checkpoint 重新生成评测输出"


def test_unknown_specialist_checkpoint_still_requires_loader_registration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    row = pd.Series(
        {
            "artifact_id": "retfound_dinov2_glaucoma_expert",
            "model_family": "retfound",
            "architecture": "retfound_dinov2",
            "checkpoint_path": str(checkpoint),
        }
    )

    assert checkpoint_loader_spec(row) is None
    assert checkpoint_inference_capability(row) == (
        False,
        "当前模型尚未登记可执行的 checkpoint 推理 Loader",
    )


def test_retfound_loader_passes_global_pool_to_timm_model() -> None:
    job = pd.Series({"num_classes": 5, "global_pool": "avg"})

    assert timm_model_create_kwargs(job) == {
        "pretrained": False,
        "num_classes": 5,
        "global_pool": "avg",
    }
