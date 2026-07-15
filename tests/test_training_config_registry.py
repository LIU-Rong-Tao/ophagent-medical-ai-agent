from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository-root bootstrap above must run before these imports.
from app.training_config import (  # noqa: E402
    TrainingConfigError,
    build_official_trial_recipe,
    build_training_draft,
    compile_effective_config,
    discover_official_recipe_profiles,
    discover_training_recipes,
    dump_yaml,
    expand_official_recipe_trials,
    flatten_effective_config,
    load_yaml,
    match_official_recipe_profiles,
    parse_yaml_text,
    save_training_recipe,
)


def context(tmp_path: Path) -> dict:
    return {
        "task_id": "glaucoma_3class",
        "dataset_id": "Glaucoma_fundus",
        "artifact_id": "convnext_tiny__glaucoma_3class__adapter",
        "source_artifact_id": "convnext_tiny",
        "source_task_id": "aptos_dr_5class",
        "trainer_adapter": "timm_imagefolder_v1",
        "model_family": "convnext",
        "architecture": "convnext_tiny",
        "data_root": str(tmp_path / "dataset"),
        "num_classes": 3,
        "class_to_idx": {"normal": 0, "early": 1, "advanced": 2},
        "label_space": "glaucoma_normal_early_advanced",
        "label_structure": "nominal",
        "output_dir": str(tmp_path / "run"),
        "display_metrics": ["accuracy", "macro_f1"],
    }


def base_recipe() -> dict:
    return {
        "schema_version": 1,
        "recipe": {
            "recipe_id": "timm_full_train",
            "display_name": "通用全量微调",
            "trainer_adapter": "timm_imagefolder_v1",
            "supported_model_families": ["convnext", "swin", "vit"],
        },
        "model": {"freeze_backbone": False},
        "training": {
            "epochs": 20,
            "batch_size": 16,
            "image_size": 224,
            "amp": True,
            "grad_accum_steps": 1,
            "seed": 42,
            "early_stopping_patience": 5,
        },
        "optimizer": {"name": "adamw", "learning_rate": 0.0001, "weight_decay": 0.01, "momentum": 0.9},
        "scheduler": {"name": "cosine", "warmup_epochs": 2, "minimum_learning_rate": 0.000001, "step_size": 10, "gamma": 0.1},
        "loss": {"name": "cross_entropy", "label_smoothing": 0.1, "class_weights": None},
        "augmentation": {
            "random_resized_crop": False,
            "horizontal_flip_probability": 0.5,
            "rotation_degrees": 10.0,
            "color_jitter": 0.0,
        },
        "evaluation": {"save_best_by": "macro_f1"},
        "runtime": {"device": "auto", "num_workers": 4},
    }


def official_profile() -> dict:
    return {
        "schema_version": 1,
        "profile": {
            "profile_id": "convnext_tiny_official_anchor",
            "display_name": "ConvNeXt-Tiny 官方协议锚点",
            "model_family": "convnext",
            "supported_architectures": ["convnext_tiny"],
            "trainer_adapter": "timm_imagefolder_v1",
            "source": {
                "title": "ConvNeXt official training instructions",
                "url": "https://github.com/facebookresearch/ConvNeXt",
                "config_url": "https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md",
                "scope": "ImageNet fine-tuning",
                "status": "official_repository",
            },
            "translation": {
                "status": "task_adapted_subset",
                "base_recipe_id": "timm_full_train",
                "note": "使用当前 trainer 可执行字段转译，不声称逐项复现原仓库。",
                "unsupported_fields": ["drop_path_rate", "layer_decay"],
                "executable_overrides": {
                    "training.epochs": 30,
                    "optimizer.learning_rate": 0.00005,
                    "optimizer.weight_decay": 0.00000001,
                },
            },
        },
        "search_plan": {
            "plan_id": "fixed_budget_lr_wd_6trial",
            "display_name": "固定预算 6 次验证集搜索",
            "selection_split": "val",
            "selection_metric": "task_default",
            "final_seeds": [42, 3407, 2026],
            "grid": {
                "optimizer.learning_rate": [0.000025, 0.00005, 0.0001],
                "optimizer.weight_decay": [0.00000001, 0.01],
            },
        },
    }


def test_recipe_registry_discovers_versioned_yaml_templates(tmp_path: Path) -> None:
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    dump_yaml(recipe_dir / "timm_full_train.yaml", base_recipe())

    recipes = discover_training_recipes(recipe_dir)

    assert [recipe["recipe"]["recipe_id"] for recipe in recipes] == ["timm_full_train"]
    assert load_yaml(recipe_dir / "timm_full_train.yaml")["schema_version"] == 1


def test_official_profile_registry_matches_exact_architecture(tmp_path: Path) -> None:
    profile_dir = tmp_path / "official_profiles"
    profile_dir.mkdir()
    dump_yaml(profile_dir / "convnext_tiny.yaml", official_profile())

    profiles = discover_official_recipe_profiles(profile_dir)

    assert [item["profile"]["profile_id"] for item in profiles] == ["convnext_tiny_official_anchor"]
    assert len(match_official_recipe_profiles(profiles, "convnext", "convnext_tiny")) == 1
    assert match_official_recipe_profiles(profiles, "convnext", "convnext_base") == []
    assert match_official_recipe_profiles(profiles, "swin", "convnext_tiny") == []


def test_official_profile_expands_six_trials_without_creating_six_recipes() -> None:
    trials = expand_official_recipe_trials(official_profile())

    assert len(trials) == 6
    assert {trial["profile_id"] for trial in trials} == {"convnext_tiny_official_anchor"}
    assert {trial["trial_id"] for trial in trials} == {
        "trial_01",
        "trial_02",
        "trial_03",
        "trial_04",
        "trial_05",
        "trial_06",
    }
    assert {trial["optimizer.learning_rate"] for trial in trials} == {0.000025, 0.00005, 0.0001}
    assert {trial["optimizer.weight_decay"] for trial in trials} == {0.00000001, 0.01}
    assert all(trial["selection_split"] == "val" for trial in trials)
    assert all(trial["final_seed_count"] == 3 for trial in trials)


def test_official_trial_recipe_translates_one_selected_trial_into_executable_recipe() -> None:
    translated = build_official_trial_recipe(base_recipe(), official_profile(), "trial_06")

    assert translated["recipe"]["recipe_id"] == "convnext_tiny_official_anchor__trial_06"
    assert translated["recipe"]["recipe_kind"] == "research_trial"
    assert translated["recipe"]["evaluation_role"] == "validation_search"
    assert translated["training"]["epochs"] == 30
    assert translated["optimizer"]["learning_rate"] == 0.0001
    assert translated["optimizer"]["weight_decay"] == 0.01


def test_reference_only_official_profile_cannot_be_translated_for_execution() -> None:
    profile = official_profile()
    profile["profile"]["translation"]["status"] = "reference_only_until_adapter_parity"

    with pytest.raises(TrainingConfigError, match="不允许生成可执行 trial"):
        build_official_trial_recipe(base_recipe(), profile, "trial_01")


def test_training_draft_injects_locked_context_and_cross_task_timm_initialization(tmp_path: Path) -> None:
    draft = build_training_draft(base_recipe(), context(tmp_path))

    assert draft["identity"]["task_id"] == "glaucoma_3class"
    assert draft["data"]["class_to_idx"] == {"normal": 0, "early": 1, "advanced": 2}
    assert draft["model"]["initialization"]["source"] == "timm_pretrained"
    assert draft["recipe"]["display_name"] == "通用全量微调"
    assert draft["model"]["initialization"]["checkpoint_path"] is None
    assert draft["output"]["run_dir"] == str(tmp_path / "run")


def test_compile_effective_config_rejects_unknown_fields_and_locked_field_changes(tmp_path: Path) -> None:
    expected = context(tmp_path)
    draft = build_training_draft(base_recipe(), expected)
    draft["training"]["mystery_option"] = True

    with pytest.raises(TrainingConfigError, match="未知字段"):
        compile_effective_config(draft, expected)

    draft = build_training_draft(base_recipe(), expected)
    draft["data"]["num_classes"] = 5

    with pytest.raises(TrainingConfigError, match="锁定字段"):
        compile_effective_config(draft, expected)


def test_compile_effective_config_rejects_unsupported_optimizer_and_metric(tmp_path: Path) -> None:
    expected = context(tmp_path)
    draft = build_training_draft(base_recipe(), expected)
    draft["optimizer"]["name"] = "lion"

    with pytest.raises(TrainingConfigError, match="optimizer"):
        compile_effective_config(draft, expected)

    draft = build_training_draft(base_recipe(), expected)
    draft["evaluation"]["save_best_by"] = "qwk"

    with pytest.raises(TrainingConfigError, match="save_best_by"):
        compile_effective_config(draft, expected)


def test_flatten_effective_config_is_the_only_trainer_input_shape(tmp_path: Path) -> None:
    expected = context(tmp_path)
    effective, report = compile_effective_config(build_training_draft(base_recipe(), expected), expected)

    flat = flatten_effective_config(effective)

    assert report["ok"] is True
    assert effective["recipe"]["recipe_id"] == "timm_full_train"
    assert effective["recipe"]["display_name"] == "通用全量微调"
    assert flat["task_id"] == "glaucoma_3class"
    assert flat["initialization_source"] == "timm_pretrained"
    assert flat["optimizer_name"] == "adamw"
    assert flat["scheduler_name"] == "cosine"
    assert flat["amp"] is True
    assert flat["save_best_by"] == "macro_f1"


def test_yaml_editor_payload_can_be_promoted_to_new_recipe_without_locked_context(tmp_path: Path) -> None:
    submitted = build_training_draft(base_recipe(), context(tmp_path))
    parsed = parse_yaml_text(dump_yaml(tmp_path / "submitted.yaml", submitted).read_text(encoding="utf-8"))

    saved = save_training_recipe(tmp_path / "recipes", parsed, "custom_convnext_v1")
    recipe = load_yaml(saved)

    assert recipe["recipe"]["recipe_id"] == "custom_convnext_v1"
    assert "identity" not in recipe
    assert "data" not in recipe
    assert "output" not in recipe
    assert "architecture" not in recipe["model"]
    with pytest.raises(TrainingConfigError, match="已存在"):
        save_training_recipe(tmp_path / "recipes", parsed, "custom_convnext_v1")
