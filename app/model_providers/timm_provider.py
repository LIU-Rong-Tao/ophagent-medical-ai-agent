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


class TimmProvider(BaseModelProvider):
    provider_id = "timm"

    def __init__(self, inventory: Iterable[Mapping[str, Any]]):
        self._records = tuple(self._convert(item) for item in inventory)

    def _convert(self, item: Mapping[str, Any]) -> UnifiedModelRecord:
        model_id = str(item["model_id"])
        return UnifiedModelRecord(
            provider_id=self.provider_id,
            source_model_id=model_id,
            source_checkpoint_id=None,
            unified_model_id=f"timm::{model_id}",
            display_name=str(item.get("display_name", model_id)),
            family_id=str(item.get("family_id", model_id)),
            modalities=tuple(item.get("modalities", ("CFP",))),
            capabilities=tuple(item.get("capabilities", ("classification",))),
            source_access_status=SourceAccessStatus.OPEN,
            base_adapter_status=BaseAdapterStatus.SMOKE_TEST_PASSED,
            task_compatibility_status=TaskCompatibilityStatus.ADAPTATION_REQUIRED,
            base_adapter_ready=True,
            task_inference_ready=False,
            route_eligible=False,
            task_checkpoint=False,
            target_task_id=None,
            provenance={"provider": "timm", **dict(item.get("provenance", {}))},
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
        return ProviderHealth(self.provider_id, True, "available", "timm provider available")
