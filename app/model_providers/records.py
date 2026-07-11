from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceAccessStatus(str, Enum):
    OPEN = "open"
    AUTHENTICATION_REQUIRED = "authentication_required"
    WEIGHTS_MISSING = "weights_missing"
    UNAVAILABLE = "unavailable"


class BaseAdapterStatus(str, Enum):
    NOT_IMPLEMENTED = "not_implemented"
    IMPLEMENTED_NOT_TESTED = "implemented_not_tested"
    SMOKE_TEST_PASSED = "smoke_test_passed"
    FAILED = "failed"


class TaskCompatibilityStatus(str, Enum):
    METADATA_ONLY = "metadata_only"
    ADAPTATION_REQUIRED = "adaptation_required"
    OFFLINE_REPLAY = "offline_replay"
    DIRECT_INFERENCE = "direct_inference"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ProviderHealth:
    provider_id: str
    available: bool
    code: str
    message: str
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedModelRecord:
    provider_id: str
    source_model_id: str
    source_checkpoint_id: str | None
    unified_model_id: str
    display_name: str
    family_id: str
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    source_access_status: SourceAccessStatus
    base_adapter_status: BaseAdapterStatus
    task_compatibility_status: TaskCompatibilityStatus
    base_adapter_ready: bool
    task_inference_ready: bool
    route_eligible: bool
    task_checkpoint: bool
    target_task_id: str | None
    provenance: dict[str, Any]

    @property
    def runnable(self) -> bool:
        """Deprecated compatibility alias for task-level inference readiness."""

        return self.task_inference_ready
