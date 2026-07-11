"""Optional bridge from the external ophbench registry into OphAgent.

This module is metadata-only. It does not download weights, instantiate adapters, mutate the
existing Model Hub catalog, or claim that a foundation checkpoint is a task checkpoint.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable


class ExternalModelStatus(str, Enum):
    CATALOG_ONLY = "catalog_only"
    WEIGHTS_MISSING = "weights_missing"
    AUTHENTICATION_REQUIRED = "authentication_required"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    ADAPTER_READY = "adapter_ready"
    TASK_ADAPTATION_REQUIRED = "task_adaptation_required"
    DIRECT_INFERENCE_READY = "direct_inference_ready"


class BridgeErrorCode(str, Enum):
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    REGISTRY_INVALID = "registry_invalid"


@dataclass(frozen=True)
class BridgeError:
    code: BridgeErrorCode
    message: str
    detail: str


@dataclass(frozen=True)
class SourceMetadata:
    package_version: str
    commit_sha: str
    registry_root: str
    loaded_at: str


@dataclass(frozen=True)
class ExternalCheckpoint:
    checkpoint_id: str
    checkpoint_name: str
    modalities: tuple[str, ...]
    weight_url: str | None
    access_type: str
    requires_auth: bool
    verification_status: str
    task_checkpoint: bool = False


@dataclass(frozen=True)
class ExternalModel:
    model_id: str
    model_name: str
    modalities: tuple[str, ...]
    runtime_phase: str
    capabilities: tuple[str, ...]
    verification_status: str
    license_verified: bool
    adapter_status: str
    smoke_test_status: str
    benchmark_status: str
    checkpoints: tuple[ExternalCheckpoint, ...]
    statuses: tuple[ExternalModelStatus, ...]
    lifecycle_status: ExternalModelStatus
    runnable: bool
    task_checkpoint: bool = False
    direct_inference_ready: bool = False


@dataclass(frozen=True)
class BridgeResult:
    available: bool
    models: tuple[ExternalModel, ...] = field(default_factory=tuple)
    source: SourceMetadata | None = None
    error: BridgeError | None = None

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def checkpoint_count(self) -> int:
        return sum(len(model.checkpoints) for model in self.models)

    def to_model_hub_rows(self) -> list[dict[str, Any]]:
        """Return an isolated data interface for future Model Hub presentation.

        Rows remain blocked for target-task inference because v0.1 contains only upstream
        foundation checkpoints and no OphAgent task head.
        """

        return [
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "source": "external_ophbench",
                "modalities": list(model.modalities),
                "runtime_phase": model.runtime_phase,
                "external_statuses": [status.value for status in model.statuses],
                "checkpoint_count": len(model.checkpoints),
                "target_task_status": "blocked",
                "target_task_reason": (
                    "External foundation checkpoint; an OphAgent task adapter or trained task "
                    "head is required before direct inference."
                ),
                "task_checkpoint": False,
                "direct_inference_ready": False,
                "runnable": model.runnable,
            }
            for model in self.models
        ]


DependencyResolver = Callable[[], tuple[Callable[[Path], Any], SourceMetadata]]


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _implementation_value(model: Any, name: str, default: str) -> str:
    implementation = _value(model, "implementation", {})
    return str(_value(implementation, name, default))


def _checkpoint(record: Any) -> ExternalCheckpoint:
    return ExternalCheckpoint(
        checkpoint_id=str(_value(record, "checkpoint_id")),
        checkpoint_name=str(_value(record, "checkpoint_name")),
        modalities=tuple(str(value) for value in _value(record, "modalities", [])),
        weight_url=_value(record, "weight_url"),
        access_type=str(_value(record, "access_type", "unknown")),
        requires_auth=bool(_value(record, "requires_auth", False)),
        verification_status=str(_value(record, "verification_status", "seed_unverified")),
    )


def build_external_catalog(
    models: Iterable[Any],
    checkpoints: Iterable[Any],
    *,
    source: SourceMetadata,
) -> BridgeResult:
    checkpoint_groups: dict[str, list[ExternalCheckpoint]] = defaultdict(list)
    for record in checkpoints:
        checkpoint_groups[str(_value(record, "model_id"))].append(_checkpoint(record))

    external_models = []
    for model in sorted(models, key=lambda item: str(_value(item, "model_id"))):
        model_id = str(_value(model, "model_id"))
        model_checkpoints = tuple(
            sorted(checkpoint_groups.pop(model_id, []), key=lambda item: item.checkpoint_id)
        )
        adapter_status = _implementation_value(model, "adapter_status", "not_started")
        smoke_status = _implementation_value(model, "smoke_test_status", "not_run")
        benchmark_status = _implementation_value(model, "benchmark_status", "not_run")
        adapter_ready = adapter_status == "implemented" and smoke_status == "passed"
        lifecycle_status = (
            ExternalModelStatus.ADAPTER_READY
            if adapter_ready
            else ExternalModelStatus.ADAPTER_UNAVAILABLE
        )
        statuses = {lifecycle_status, ExternalModelStatus.TASK_ADAPTATION_REQUIRED}
        if not adapter_ready:
            statuses.add(ExternalModelStatus.CATALOG_ONLY)
        if not model_checkpoints or any(not checkpoint.weight_url for checkpoint in model_checkpoints):
            statuses.add(ExternalModelStatus.WEIGHTS_MISSING)
        if any(checkpoint.requires_auth for checkpoint in model_checkpoints):
            statuses.add(ExternalModelStatus.AUTHENTICATION_REQUIRED)
        external_models.append(
            ExternalModel(
                model_id=model_id,
                model_name=str(_value(model, "model_name")),
                modalities=tuple(str(value) for value in _value(model, "modalities", [])),
                runtime_phase=str(_value(model, "runtime_phase", "catalog_only")),
                capabilities=tuple(str(value) for value in _value(model, "capabilities", [])),
                verification_status=str(
                    _value(model, "verification_status", "seed_unverified")
                ),
                license_verified=bool(_value(model, "license_verified", False)),
                adapter_status=adapter_status,
                smoke_test_status=smoke_status,
                benchmark_status=benchmark_status,
                checkpoints=model_checkpoints,
                statuses=tuple(sorted(statuses, key=lambda status: status.value)),
                lifecycle_status=lifecycle_status,
                runnable=adapter_ready,
            )
        )
    if checkpoint_groups:
        unknown = ", ".join(sorted(checkpoint_groups))
        raise ValueError(f"Checkpoints reference unknown external model IDs: {unknown}")
    return BridgeResult(available=True, models=tuple(external_models), source=source)


def _git_commit(package_root: Path) -> str:
    configured = os.environ.get("OPHBENCH_COMMIT_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _resolve_ophbench_dependency() -> tuple[Callable[[Path], Any], SourceMetadata]:
    module = importlib.import_module("ophbench")
    loader_module = importlib.import_module("ophbench.registry.loader")
    package_root = Path(module.__file__).resolve().parent.parent
    registry_root = Path(os.environ.get("OPHBENCH_REGISTRY_ROOT", package_root / "registry"))
    try:
        version = importlib.metadata.version("ophthalmic-foundation-model-benchmark")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")
    source = SourceMetadata(
        package_version=str(version),
        commit_sha=_git_commit(package_root),
        registry_root=str(registry_root),
        loaded_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
    )
    return loader_module.load_registry, source


def load_external_catalog(
    registry_root: Path | None = None,
    *,
    dependency_resolver: DependencyResolver = _resolve_ophbench_dependency,
) -> BridgeResult:
    try:
        loader, source = dependency_resolver()
    except (ImportError, ModuleNotFoundError) as exc:
        return BridgeResult(
            available=False,
            error=BridgeError(
                code=BridgeErrorCode.DEPENDENCY_UNAVAILABLE,
                message=(
                    "The optional ophbench registry dependency is unavailable. Install a "
                    "Python-compatible ophthalmic-foundation-model-benchmark package to enable "
                    "the external catalog."
                ),
                detail=str(exc),
            ),
        )
    selected_root = Path(registry_root or source.registry_root)
    source = SourceMetadata(
        package_version=source.package_version,
        commit_sha=source.commit_sha,
        registry_root=str(selected_root),
        loaded_at=source.loaded_at,
    )
    try:
        models, checkpoints = loader(selected_root)
        return build_external_catalog(models, checkpoints, source=source)
    except Exception as exc:
        return BridgeResult(
            available=False,
            source=source,
            error=BridgeError(
                code=BridgeErrorCode.REGISTRY_INVALID,
                message="The external ophbench registry could not be loaded or validated.",
                detail=str(exc),
            ),
        )
