from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .base import BaseModelProvider
from .records import (
    BaseAdapterStatus,
    ProviderHealth,
    SourceAccessStatus,
    TaskCompatibilityStatus,
    UnifiedModelRecord,
)


class LocalArtifactProvider(BaseModelProvider):
    provider_id = "local_artifact"

    def __init__(self, artifacts: Iterable[Mapping[str, Any]]):
        self._records = tuple(self._convert(item) for item in artifacts)

    def _convert(self, item: Mapping[str, Any]) -> UnifiedModelRecord:
        artifact_id = str(item["artifact_id"])
        ready = bool(item.get("route_eligible", False))
        return UnifiedModelRecord(
            provider_id=self.provider_id,
            source_model_id=str(item.get("base_model_id", artifact_id)),
            source_checkpoint_id=artifact_id,
            unified_model_id=f"local_artifact::{artifact_id}",
            display_name=str(item.get("display_name", artifact_id)),
            family_id=str(item.get("family_id", item.get("base_model_id", artifact_id))),
            modalities=tuple(item.get("modalities", ("CFP",))),
            capabilities=tuple(item.get("capabilities", ("classification",))),
            source_access_status=SourceAccessStatus.OPEN,
            base_adapter_status=BaseAdapterStatus.SMOKE_TEST_PASSED,
            task_compatibility_status=(
                TaskCompatibilityStatus.DIRECT_INFERENCE
                if ready
                else TaskCompatibilityStatus.OFFLINE_REPLAY
            ),
            base_adapter_ready=True,
            task_inference_ready=ready,
            route_eligible=ready,
            task_checkpoint=True,
            target_task_id=str(item.get("task_id")) if item.get("task_id") else None,
            provenance={"provider": self.provider_id, **dict(item.get("provenance", {}))},
        )

    def is_available(self) -> bool:
        return True

    def list_models(self) -> tuple[UnifiedModelRecord, ...]:
        return self._records

    def list_checkpoints(self, model_id: str) -> tuple[UnifiedModelRecord, ...]:
        return tuple(record for record in self._records if record.source_model_id == model_id)

    def get_model(self, unified_model_id: str) -> UnifiedModelRecord:
        return next(record for record in self._records if record.unified_model_id == unified_model_id)

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            self.provider_id, True, "available", "local artifact provider available"
        )
