from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .base import BaseModelProvider
from .records import (
    BaseAdapterStatus,
    ProviderHealth,
    SourceAccessStatus,
    TaskCompatibilityStatus,
    UnifiedModelRecord,
)


def _public_snapshot_loader():
    from ophbench import load_registry

    return load_registry()


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


class OphBenchProvider(BaseModelProvider):
    provider_id = "ophbench"

    def __init__(self, snapshot_loader: Callable[[], Any] = _public_snapshot_loader):
        self._snapshot_loader = snapshot_loader
        self._loaded = False
        self._records: tuple[UnifiedModelRecord, ...] = ()
        self._health = ProviderHealth(
            self.provider_id, False, "not_loaded", "ophbench provider has not been loaded"
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            snapshot = self._snapshot_loader()
            self._records = self._convert(snapshot)
            self._health = ProviderHealth(
                self.provider_id,
                True,
                "available",
                "ophbench registry available",
                metadata={
                    "package_version": str(snapshot.package_version),
                    "schema_version": str(snapshot.schema_version),
                    "registry_source": str(snapshot.registry_source),
                    "model_count": int(snapshot.model_count),
                    "checkpoint_count": int(snapshot.checkpoint_count),
                },
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self._health = ProviderHealth(
                self.provider_id,
                False,
                "dependency_unavailable",
                "Optional ophbench dependency is unavailable.",
                detail=str(exc),
            )
        except Exception as exc:
            self._health = ProviderHealth(
                self.provider_id,
                False,
                "registry_invalid",
                "OphBench registry could not be loaded or validated.",
                detail=str(exc),
            )

    def _convert(self, snapshot: Any) -> tuple[UnifiedModelRecord, ...]:
        checkpoints_by_model = defaultdict(list)
        for checkpoint in snapshot.checkpoints:
            checkpoints_by_model[str(_value(checkpoint, "model_id"))].append(checkpoint)
        records = []
        for model in snapshot.models:
            model_id = str(_value(model, "model_id"))
            checkpoints = sorted(
                checkpoints_by_model[model_id],
                key=lambda item: str(_value(item, "checkpoint_id")),
            )
            for checkpoint in checkpoints:
                records.append(self._record(model, checkpoint, snapshot))
        return tuple(sorted(records, key=lambda record: record.unified_model_id))

    def _record(self, model: Any, checkpoint: Any, snapshot: Any) -> UnifiedModelRecord:
        model_id = str(_value(model, "model_id"))
        checkpoint_id = str(_value(checkpoint, "checkpoint_id"))
        implementation = _value(model, "implementation", {})
        adapter = str(_value(implementation, "adapter_status", "not_started"))
        smoke = str(_value(implementation, "smoke_test_status", "not_run"))
        if adapter == "failed" or smoke == "failed":
            adapter_status = BaseAdapterStatus.FAILED
        elif adapter == "implemented" and smoke == "passed":
            adapter_status = BaseAdapterStatus.SMOKE_TEST_PASSED
        elif adapter == "implemented":
            adapter_status = BaseAdapterStatus.IMPLEMENTED_NOT_TESTED
        else:
            adapter_status = BaseAdapterStatus.NOT_IMPLEMENTED
        weight_url = _value(checkpoint, "weight_url")
        if not weight_url:
            access_status = SourceAccessStatus.WEIGHTS_MISSING
        elif bool(_value(checkpoint, "requires_auth", False)):
            access_status = SourceAccessStatus.AUTHENTICATION_REQUIRED
        elif str(_value(checkpoint, "access_type", "open")) == "unavailable":
            access_status = SourceAccessStatus.UNAVAILABLE
        else:
            access_status = SourceAccessStatus.OPEN
        base_ready = adapter_status is BaseAdapterStatus.SMOKE_TEST_PASSED
        return UnifiedModelRecord(
            provider_id=self.provider_id,
            source_model_id=model_id,
            source_checkpoint_id=checkpoint_id,
            unified_model_id=f"ophbench::{model_id}::{checkpoint_id}",
            display_name=f"{_value(model, 'model_name')} {_value(checkpoint, 'checkpoint_name')}",
            family_id=model_id,
            modalities=tuple(str(value) for value in _value(checkpoint, "modalities", [])),
            capabilities=tuple(str(value) for value in _value(model, "capabilities", [])),
            source_access_status=access_status,
            base_adapter_status=adapter_status,
            task_compatibility_status=TaskCompatibilityStatus.ADAPTATION_REQUIRED,
            base_adapter_ready=base_ready,
            task_inference_ready=False,
            route_eligible=False,
            task_checkpoint=False,
            target_task_id=None,
            provenance={
                "provider": self.provider_id,
                "package_version": str(snapshot.package_version),
                "schema_version": str(snapshot.schema_version),
                "registry_source": str(snapshot.registry_source),
                "verification_status": str(
                    _value(checkpoint, "verification_status", "seed_unverified")
                ),
            },
        )

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._health.available

    def list_models(self) -> tuple[UnifiedModelRecord, ...]:
        self._ensure_loaded()
        return self._records

    def list_checkpoints(self, model_id: str) -> tuple[UnifiedModelRecord, ...]:
        return tuple(record for record in self.list_models() if record.source_model_id == model_id)

    def get_model(self, unified_model_id: str) -> UnifiedModelRecord:
        return next(
            record for record in self.list_models() if record.unified_model_id == unified_model_id
        )

    def health(self) -> ProviderHealth:
        self._ensure_loaded()
        return self._health
