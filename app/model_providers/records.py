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
    model_name: str = ""
    year: int | None = None
    venue: str = ""
    model_category: str = ""
    architecture: str = ""
    pretraining_data_summary: str = ""
    pretraining_strategy: str = ""
    reported_summary: str = ""
    paper_url: str = ""
    code_url: str = ""
    runtime_phase: str = ""
    verification_status: str = ""
    license: str = ""
    license_verified: bool = False
    checkpoint_name: str = ""
    checkpoint_provider: str = ""
    weight_url: str = ""
    access_type: str = ""
    requires_auth: bool = False
    framework: str = ""
    input_size: str = ""
    normalization: str = ""
    embedding_dim: int | None = None
    sha256: str = ""
    last_verified: str = ""
    checkpoint_verification_status: str = ""
    artifact_type: str = ""
    model_modalities: tuple[str, ...] = ()
    catalog_registered: bool = False
    official_source_verified: bool = False
    download_status: str = ""
    local_integrity_status: str = ""
    provider_verification_status: str = ""
    runtime_status: str = ""
    local_asset_status: str = ""
    upstream_license_status: str = ""
    preprocessing_verification_status: str = ""
    local_asset_verified: bool = False
    adapter_implemented: bool = False
    encoder_smoke_passed: bool = False
    feature_output_verified: bool = False
    task_adapted: bool = False
    unified_evaluation_complete: bool = False
    cost_verified: bool = False

    @property
    def runnable(self) -> bool:
        """Deprecated compatibility alias for task-level inference readiness."""

        return self.task_inference_ready
