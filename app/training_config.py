"""模型中转台训练 recipe、YAML 配置编译与严格校验。"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import re
from typing import Any

import yaml


class TrainingConfigError(ValueError):
    """训练配置无法安全执行。"""


TOP_LEVEL_FIELDS = {
    "schema_version",
    "identity",
    "recipe",
    "data",
    "model",
    "training",
    "optimizer",
    "scheduler",
    "loss",
    "augmentation",
    "evaluation",
    "runtime",
    "output",
}
SECTION_FIELDS = {
    "identity": {"task_id", "dataset_id", "artifact_id", "source_artifact_id", "source_task_id", "trainer_adapter"},
    "recipe": {
        "recipe_id",
        "display_name",
        "recipe_kind",
        "evaluation_role",
        "source_profile_id",
        "trial_id",
        "trainer_adapter",
        "supported_model_families",
        "description",
    },
    "data": {"root", "num_classes", "class_to_idx", "label_space", "label_structure"},
    "model": {"family", "architecture", "initialization", "freeze_backbone"},
    "training": {"epochs", "batch_size", "image_size", "amp", "grad_accum_steps", "seed", "early_stopping_patience"},
    "optimizer": {"name", "learning_rate", "weight_decay", "momentum"},
    "scheduler": {"name", "warmup_epochs", "minimum_learning_rate", "step_size", "gamma"},
    "loss": {"name", "label_smoothing", "class_weights"},
    "augmentation": {"random_resized_crop", "horizontal_flip_probability", "rotation_degrees", "color_jitter"},
    "evaluation": {"save_best_by", "metrics"},
    "runtime": {"device", "num_workers"},
    "output": {"run_dir"},
}
INITIALIZATION_FIELDS = {"source", "checkpoint_path", "source_num_classes"}
SUPPORTED_OPTIMIZERS = {"adamw", "adam", "sgd"}
SUPPORTED_SCHEDULERS = {"none", "cosine", "step"}
OFFICIAL_PROFILE_REQUIRED_FIELDS = {
    "profile_id",
    "display_name",
    "model_family",
    "supported_architectures",
    "trainer_adapter",
    "source",
    "translation",
}


def load_yaml(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingConfigError(f"YAML 根节点必须是 mapping：{path}")
    return payload


def parse_yaml_text(text: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TrainingConfigError(f"YAML 解析失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingConfigError("YAML 根节点必须是 mapping")
    return payload


def dump_yaml(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def discover_training_recipes(root: Path | str) -> list[dict[str, Any]]:
    directory = Path(root)
    if not directory.is_dir():
        return []
    recipes = [load_yaml(path) for path in sorted(directory.glob("*.y*ml"))]
    seen: set[str] = set()
    for recipe in recipes:
        recipe_id = str(recipe.get("recipe", {}).get("recipe_id", "")).strip()
        if not recipe_id:
            raise TrainingConfigError("recipe 缺少 recipe.recipe_id")
        if recipe_id in seen:
            raise TrainingConfigError(f"recipe_id 重复：{recipe_id}")
        seen.add(recipe_id)
    return recipes


def discover_official_recipe_profiles(root: Path | str) -> list[dict[str, Any]]:
    """读取模型专属的官方协议档案，与可直接提交的工程 recipe 分开。"""

    directory = Path(root)
    if not directory.is_dir():
        return []
    profiles = [load_yaml(path) for path in sorted(directory.glob("*.y*ml"))]
    seen: set[str] = set()
    for payload in profiles:
        if payload.get("schema_version") != 1:
            raise TrainingConfigError("官方协议档案仅支持 schema_version=1")
        profile = payload.get("profile")
        if not isinstance(profile, dict):
            raise TrainingConfigError("官方协议档案缺少 profile mapping")
        missing = OFFICIAL_PROFILE_REQUIRED_FIELDS - set(profile)
        if missing:
            raise TrainingConfigError(f"官方协议档案缺少字段：{sorted(missing)}")
        profile_id = str(profile["profile_id"]).strip()
        if not profile_id:
            raise TrainingConfigError("官方协议档案缺少 profile_id")
        if profile_id in seen:
            raise TrainingConfigError(f"profile_id 重复：{profile_id}")
        seen.add(profile_id)
        architectures = profile.get("supported_architectures")
        if not isinstance(architectures, list) or not architectures:
            raise TrainingConfigError(f"{profile_id} 必须声明 supported_architectures")
        source = profile.get("source")
        if not isinstance(source, dict) or not str(source.get("url", "")).strip():
            raise TrainingConfigError(f"{profile_id} 必须记录官方来源 URL")
        search_plan = payload.get("search_plan")
        if not isinstance(search_plan, dict):
            raise TrainingConfigError(f"{profile_id} 缺少 search_plan")
        if str(search_plan.get("selection_split", "")) != "val":
            raise TrainingConfigError(f"{profile_id} 的参数搜索只允许使用 val split")
        expand_official_recipe_trials(payload)
    return profiles


def match_official_recipe_profiles(
    profiles: list[dict[str, Any]],
    model_family: str,
    architecture: str,
) -> list[dict[str, Any]]:
    """精确按模型族和 architecture 匹配，防止把同族协议错用到其他规模模型。"""

    family = str(model_family).strip()
    arch = str(architecture).strip()
    return [
        payload
        for payload in profiles
        if str(payload["profile"].get("model_family", "")).strip() == family
        and arch in [str(value).strip() for value in payload["profile"].get("supported_architectures", [])]
    ]


def expand_official_recipe_trials(profile_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """将一个官方协议档案展开为固定预算 trial，但不创建多份 recipe 文件。"""

    profile = profile_payload.get("profile")
    search_plan = profile_payload.get("search_plan")
    if not isinstance(profile, dict) or not isinstance(search_plan, dict):
        raise TrainingConfigError("官方协议档案必须包含 profile 和 search_plan")
    grid = search_plan.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise TrainingConfigError(f"{profile.get('profile_id', '未知 profile')} 缺少非空 search_plan.grid")
    paths = list(grid)
    value_lists: list[list[Any]] = []
    for path in paths:
        values = grid[path]
        if not isinstance(values, list) or not values:
            raise TrainingConfigError(f"search_plan.grid.{path} 必须是非空列表")
        value_lists.append(values)
    combinations = list(product(*value_lists))
    expected_trials = int(search_plan.get("trial_budget", len(combinations)))
    if len(combinations) != expected_trials:
        raise TrainingConfigError(
            f"search_plan 展开为 {len(combinations)} 次，与 trial_budget={expected_trials} 不一致"
        )
    final_seeds = search_plan.get("final_seeds", [])
    if not isinstance(final_seeds, list) or not final_seeds:
        raise TrainingConfigError("search_plan.final_seeds 必须是非空列表")
    rows = []
    for index, values in enumerate(combinations, start=1):
        row = {
            "profile_id": str(profile.get("profile_id", "")),
            "plan_id": str(search_plan.get("plan_id", "")),
            "trial_id": f"trial_{index:02d}",
            "selection_split": str(search_plan.get("selection_split", "val")),
            "selection_metric": str(search_plan.get("selection_metric", "task_default")),
            "final_seed_count": len(final_seeds),
        }
        row.update(dict(zip(paths, values)))
        rows.append(row)
    return rows


def _set_dotted_path(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = str(dotted_path).split(".")
    if len(parts) < 2:
        raise TrainingConfigError(f"配置覆盖路径必须为 section.field：{dotted_path}")
    current: dict[str, Any] = payload
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise TrainingConfigError(f"配置覆盖路径不存在：{dotted_path}")
        current = child
    if parts[-1] not in current:
        raise TrainingConfigError(f"配置覆盖字段不存在：{dotted_path}")
    current[parts[-1]] = deepcopy(value)


def build_official_trial_recipe(
    engineering_recipe: dict[str, Any],
    profile_payload: dict[str, Any],
    trial_id: str,
) -> dict[str, Any]:
    """将选定的官方协议锚点 trial 转译为当前 trainer 可预检的单份 recipe。"""

    profile = profile_payload.get("profile")
    if not isinstance(profile, dict):
        raise TrainingConfigError("官方协议档案缺少 profile")
    translation = profile.get("translation")
    if not isinstance(translation, dict):
        raise TrainingConfigError("官方协议档案缺少 translation")
    if str(translation.get("status", "")) != "task_adapted_subset":
        raise TrainingConfigError("当前官方协议档案不允许生成可执行 trial")
    expected_base = str(translation.get("base_recipe_id", "")).strip()
    actual_base = str(engineering_recipe.get("recipe", {}).get("recipe_id", "")).strip()
    if expected_base != actual_base:
        raise TrainingConfigError(f"官方协议档案需要基础 recipe={expected_base}，实际为 {actual_base}")
    matching_trial = next(
        (row for row in expand_official_recipe_trials(profile_payload) if row["trial_id"] == str(trial_id)),
        None,
    )
    if matching_trial is None:
        raise TrainingConfigError(f"未找到 official trial：{trial_id}")

    translated = deepcopy(engineering_recipe)
    for path, value in dict(translation.get("executable_overrides", {})).items():
        _set_dotted_path(translated, path, value)
    for path in profile_payload["search_plan"]["grid"]:
        _set_dotted_path(translated, path, matching_trial[path])

    profile_id = str(profile["profile_id"])
    translated["recipe"].update(
        {
            "recipe_id": f"{profile_id}__{trial_id}",
            "display_name": f"{profile.get('display_name', profile_id)} / {trial_id}",
            "recipe_kind": "research_trial",
            "evaluation_role": "validation_search",
            "source_profile_id": profile_id,
            "trial_id": str(trial_id),
            "description": str(translation.get("note", "")),
        }
    )
    return translated


def save_training_recipe(
    root: Path | str,
    submitted: dict[str, Any],
    recipe_id: str,
) -> Path:
    normalized_id = str(recipe_id).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", normalized_id):
        raise TrainingConfigError("recipe_id 只能使用 3-64 位小写字母、数字、下划线或连字符")
    _reject_unknown_fields(submitted)
    recipe_metadata = deepcopy(submitted["recipe"])
    recipe_metadata["recipe_id"] = normalized_id
    payload = {
        "schema_version": 1,
        "recipe": recipe_metadata,
        "model": {"freeze_backbone": bool(submitted["model"].get("freeze_backbone", False))},
        "training": deepcopy(submitted["training"]),
        "optimizer": deepcopy(submitted["optimizer"]),
        "scheduler": deepcopy(submitted["scheduler"]),
        "loss": deepcopy(submitted["loss"]),
        "augmentation": deepcopy(submitted["augmentation"]),
        "evaluation": {"save_best_by": submitted["evaluation"].get("save_best_by")},
        "runtime": deepcopy(submitted["runtime"]),
    }
    target = Path(root) / f"{normalized_id}.yaml"
    if target.exists():
        raise TrainingConfigError(f"recipe 已存在，拒绝覆盖：{target}")
    return dump_yaml(target, payload)


def _initialization_from_context(context: dict[str, Any]) -> dict[str, Any]:
    cross_task = str(context.get("source_task_id", "")) != str(context["task_id"])
    if cross_task:
        return {"source": "timm_pretrained", "checkpoint_path": None, "source_num_classes": 0}
    checkpoint_path = str(context.get("source_checkpoint_path", "") or "").strip()
    if checkpoint_path:
        return {
            "source": "registered_checkpoint",
            "checkpoint_path": checkpoint_path,
            "source_num_classes": int(context.get("source_num_classes", 0) or 0),
        }
    return {"source": "timm_pretrained", "checkpoint_path": None, "source_num_classes": 0}


def build_training_draft(base_recipe: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    draft = deepcopy(base_recipe)
    draft["schema_version"] = 1
    draft["identity"] = {
        "task_id": str(context["task_id"]),
        "dataset_id": str(context["dataset_id"]),
        "artifact_id": str(context["artifact_id"]),
        "source_artifact_id": str(context.get("source_artifact_id", "")),
        "source_task_id": str(context.get("source_task_id", "")),
        "trainer_adapter": str(context["trainer_adapter"]),
    }
    draft["data"] = {
        "root": str(context["data_root"]),
        "num_classes": int(context["num_classes"]),
        "class_to_idx": deepcopy(context["class_to_idx"]),
        "label_space": str(context.get("label_space", "")),
        "label_structure": str(context.get("label_structure", "nominal")),
    }
    model = deepcopy(draft.get("model", {}))
    model.update(
        {
            "family": str(context["model_family"]),
            "architecture": str(context["architecture"]),
            "initialization": _initialization_from_context(context),
        }
    )
    draft["model"] = model
    evaluation = deepcopy(draft.get("evaluation", {}))
    evaluation["metrics"] = list(context.get("display_metrics", ["accuracy", "macro_f1"]))
    draft["evaluation"] = evaluation
    draft["output"] = {"run_dir": str(context["output_dir"])}
    return draft


def _require_mapping(payload: dict[str, Any], section: str) -> dict[str, Any]:
    value = payload.get(section)
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{section} 必须是 mapping")
    return value


def _reject_unknown_fields(payload: dict[str, Any]) -> None:
    unknown_top = set(payload) - TOP_LEVEL_FIELDS
    if unknown_top:
        raise TrainingConfigError(f"未知字段：{sorted(unknown_top)}")
    for section, allowed in SECTION_FIELDS.items():
        mapping = _require_mapping(payload, section)
        unknown = set(mapping) - allowed
        if unknown:
            raise TrainingConfigError(f"{section} 包含未知字段：{sorted(unknown)}")
    initialization = _require_mapping(_require_mapping(payload, "model"), "initialization")
    unknown_init = set(initialization) - INITIALIZATION_FIELDS
    if unknown_init:
        raise TrainingConfigError(f"model.initialization 包含未知字段：{sorted(unknown_init)}")


def _assert_locked_fields(submitted: dict[str, Any], context: dict[str, Any]) -> None:
    expected = build_training_draft(
        {
            "schema_version": 1,
            "recipe": deepcopy(submitted["recipe"]),
            "model": {"freeze_backbone": submitted["model"].get("freeze_backbone", False)},
            "training": deepcopy(submitted["training"]),
            "optimizer": deepcopy(submitted["optimizer"]),
            "scheduler": deepcopy(submitted["scheduler"]),
            "loss": deepcopy(submitted["loss"]),
            "augmentation": deepcopy(submitted["augmentation"]),
            "evaluation": {"save_best_by": submitted["evaluation"].get("save_best_by")},
            "runtime": deepcopy(submitted["runtime"]),
        },
        context,
    )
    checks = {
        "identity": (submitted["identity"], expected["identity"]),
        "data": (submitted["data"], expected["data"]),
        "model.family": (submitted["model"].get("family"), expected["model"]["family"]),
        "model.architecture": (submitted["model"].get("architecture"), expected["model"]["architecture"]),
        "model.initialization": (submitted["model"].get("initialization"), expected["model"]["initialization"]),
        "output": (submitted["output"], expected["output"]),
        "evaluation.metrics": (submitted["evaluation"].get("metrics"), expected["evaluation"]["metrics"]),
    }
    changed = [name for name, (actual, wanted) in checks.items() if actual != wanted]
    if changed:
        raise TrainingConfigError(f"系统锁定字段不能修改：{changed}")


def _positive_number(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{field} 必须是数值")
    if value < 0 or (not allow_zero and value == 0):
        operator = "非负" if allow_zero else "大于 0"
        raise TrainingConfigError(f"{field} 必须{operator}")
    return float(value)


def compile_effective_config(
    submitted: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if submitted.get("schema_version") != 1:
        raise TrainingConfigError("仅支持 schema_version=1")
    _reject_unknown_fields(submitted)
    _assert_locked_fields(submitted, context)
    recipe = submitted["recipe"]
    adapter = str(recipe.get("trainer_adapter", ""))
    if adapter != "timm_imagefolder_v1" or submitted["identity"]["trainer_adapter"] != adapter:
        raise TrainingConfigError(f"当前不支持 trainer_adapter={adapter}")
    supported_families = recipe.get("supported_model_families")
    if not isinstance(supported_families, list) or str(context["model_family"]) not in supported_families:
        raise TrainingConfigError("recipe 不支持当前模型家族")

    training = submitted["training"]
    for field in ("epochs", "batch_size", "image_size", "grad_accum_steps"):
        _positive_number(training.get(field), f"training.{field}")
    _positive_number(training.get("seed"), "training.seed", allow_zero=True)
    _positive_number(training.get("early_stopping_patience"), "training.early_stopping_patience", allow_zero=True)
    if not isinstance(training.get("amp"), bool):
        raise TrainingConfigError("training.amp 必须是布尔值")

    optimizer = submitted["optimizer"]
    optimizer_name = str(optimizer.get("name", "")).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise TrainingConfigError(f"不支持 optimizer={optimizer_name}；可选 {sorted(SUPPORTED_OPTIMIZERS)}")
    _positive_number(optimizer.get("learning_rate"), "optimizer.learning_rate")
    _positive_number(optimizer.get("weight_decay"), "optimizer.weight_decay", allow_zero=True)
    _positive_number(optimizer.get("momentum"), "optimizer.momentum", allow_zero=True)

    scheduler = submitted["scheduler"]
    scheduler_name = str(scheduler.get("name", "")).lower()
    if scheduler_name not in SUPPORTED_SCHEDULERS:
        raise TrainingConfigError(f"不支持 scheduler={scheduler_name}；可选 {sorted(SUPPORTED_SCHEDULERS)}")
    for field in ("warmup_epochs", "minimum_learning_rate", "gamma"):
        _positive_number(scheduler.get(field), f"scheduler.{field}", allow_zero=True)
    _positive_number(scheduler.get("step_size"), "scheduler.step_size")
    if int(scheduler.get("warmup_epochs", 0)) >= int(training["epochs"]):
        raise TrainingConfigError("scheduler.warmup_epochs 必须小于 training.epochs")

    loss = submitted["loss"]
    if str(loss.get("name", "")).lower() != "cross_entropy":
        raise TrainingConfigError("当前仅支持 loss=cross_entropy")
    smoothing = _positive_number(loss.get("label_smoothing"), "loss.label_smoothing", allow_zero=True)
    if smoothing >= 1:
        raise TrainingConfigError("loss.label_smoothing 必须小于 1")
    class_weights = loss.get("class_weights")
    if class_weights not in (None, "auto"):
        if not isinstance(class_weights, list) or len(class_weights) != int(context["num_classes"]):
            raise TrainingConfigError("loss.class_weights 必须为 null、auto 或与类别数等长的列表")
        for index, weight in enumerate(class_weights):
            _positive_number(weight, f"loss.class_weights[{index}]")

    augmentation = submitted["augmentation"]
    if not isinstance(augmentation.get("random_resized_crop"), bool):
        raise TrainingConfigError("augmentation.random_resized_crop 必须是布尔值")
    flip = _positive_number(augmentation.get("horizontal_flip_probability"), "augmentation.horizontal_flip_probability", allow_zero=True)
    if flip > 1:
        raise TrainingConfigError("augmentation.horizontal_flip_probability 不能大于 1")
    for field in ("rotation_degrees", "color_jitter"):
        _positive_number(augmentation.get(field), f"augmentation.{field}", allow_zero=True)

    evaluation = submitted["evaluation"]
    metrics = list(context.get("display_metrics", []))
    save_best_by = str(evaluation.get("save_best_by", ""))
    if save_best_by not in metrics and save_best_by != "val_loss":
        raise TrainingConfigError(f"evaluation.save_best_by={save_best_by} 不在任务注册指标 {metrics} 中")
    runtime = submitted["runtime"]
    device = str(runtime.get("device", ""))
    if device not in {"auto", "cpu"} and not device.startswith("cuda"):
        raise TrainingConfigError("runtime.device 必须为 auto、cpu 或 cuda[:n]")
    _positive_number(runtime.get("num_workers"), "runtime.num_workers", allow_zero=True)

    effective = deepcopy(submitted)
    effective["optimizer"]["name"] = optimizer_name
    effective["scheduler"]["name"] = scheduler_name
    effective["loss"]["name"] = "cross_entropy"
    report = {
        "ok": True,
        "schema_version": 1,
        "trainer_adapter": adapter,
        "recipe_id": str(recipe.get("recipe_id", "")),
        "locked_fields_verified": True,
        "warnings": [],
    }
    return effective, report


def flatten_effective_config(config: dict[str, Any]) -> dict[str, Any]:
    identity = config["identity"]
    data = config["data"]
    model = config["model"]
    initialization = model["initialization"]
    training = config["training"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    loss = config["loss"]
    augmentation = config["augmentation"]
    evaluation = config["evaluation"]
    runtime = config["runtime"]
    return {
        "task_id": identity["task_id"],
        "dataset_id": identity["dataset_id"],
        "artifact_id": identity["artifact_id"],
        "source_artifact_id": identity.get("source_artifact_id", ""),
        "trainer_adapter": identity["trainer_adapter"],
        "data_root": data["root"],
        "architecture": model["architecture"],
        "model_family": model["family"],
        "num_classes": int(data["num_classes"]),
        "label_space": data.get("label_space", ""),
        "label_structure": data.get("label_structure", "nominal"),
        "image_size": int(training["image_size"]),
        "batch_size": int(training["batch_size"]),
        "num_epochs": int(training["epochs"]),
        "learning_rate": float(optimizer["learning_rate"]),
        "pretrained": initialization["source"] == "timm_pretrained",
        "seed": int(training["seed"]),
        "output_dir": config["output"]["run_dir"],
        "weight_decay": float(optimizer["weight_decay"]),
        "label_smoothing": float(loss["label_smoothing"]),
        "num_workers": int(runtime["num_workers"]),
        "device": runtime["device"],
        "initialization_source": initialization["source"],
        "source_checkpoint_path": initialization.get("checkpoint_path") or "",
        "source_num_classes": int(initialization.get("source_num_classes") or 0),
        "freeze_backbone": bool(model["freeze_backbone"]),
        "optimizer_name": optimizer["name"],
        "optimizer_momentum": float(optimizer["momentum"]),
        "scheduler_name": scheduler["name"],
        "scheduler_warmup_epochs": int(scheduler["warmup_epochs"]),
        "scheduler_minimum_learning_rate": float(scheduler["minimum_learning_rate"]),
        "scheduler_step_size": int(scheduler["step_size"]),
        "scheduler_gamma": float(scheduler["gamma"]),
        "amp": bool(training["amp"]),
        "grad_accum_steps": int(training["grad_accum_steps"]),
        "early_stopping_patience": int(training["early_stopping_patience"]),
        "class_weights": loss["class_weights"],
        "random_resized_crop": bool(augmentation["random_resized_crop"]),
        "horizontal_flip_probability": float(augmentation["horizontal_flip_probability"]),
        "rotation_degrees": float(augmentation["rotation_degrees"]),
        "color_jitter": float(augmentation["color_jitter"]),
        "save_best_by": evaluation["save_best_by"],
    }
