from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import csv
import logging
from pathlib import Path
from typing import Any

from .base import BaseModelProvider
from .records import (
    BaseAdapterStatus,
    ProviderHealth,
    SourceAccessStatus,
    TaskCompatibilityStatus,
    UnifiedModelRecord,
)


LOGGER = logging.getLogger(__name__)


def _public_snapshot_loader():
    from ophbench import load_registry

    return load_registry()


def _public_manifest_loader() -> dict[str, dict[str, str]]:
    import ophbench

    manifest_path = (
        Path(ophbench.__file__).resolve().parents[1] / "catalog" / "download_manifest.csv"
    )
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_checkpoint = {str(row["checkpoint_id"]): row for row in rows}
    if len(rows) != 27 or len(by_checkpoint) != 27:
        raise ValueError(
            f"OphBench download manifest must contain 27 unique checkpoints: {manifest_path}"
        )
    return by_checkpoint


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


class OphBenchProvider(BaseModelProvider):
    provider_id = "ophbench"

    def __init__(
        self,
        snapshot_loader: Callable[[], Any] = _public_snapshot_loader,
        manifest_loader: Callable[[], dict[str, dict[str, str]]] | None = None,
    ):
        self._snapshot_loader = snapshot_loader
        self._manifest_loader = (
            _public_manifest_loader
            if manifest_loader is None and snapshot_loader is _public_snapshot_loader
            else manifest_loader
        )
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
            manifest = self._manifest_loader() if self._manifest_loader else {}
            self._records = self._convert(snapshot, manifest)
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
                    "manifest_checkpoint_count": len(manifest),
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
            LOGGER.exception("OphBench registry provider failed to load")
            self._health = ProviderHealth(
                self.provider_id,
                False,
                "registry_invalid",
                "OphBench registry could not be loaded or validated.",
                detail=str(exc),
            )

    def _convert(
        self, snapshot: Any, manifest: dict[str, dict[str, str]]
    ) -> tuple[UnifiedModelRecord, ...]:
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
                checkpoint_id = str(_value(checkpoint, "checkpoint_id"))
                records.append(
                    self._record(
                        model,
                        checkpoint,
                        snapshot,
                        manifest.get(checkpoint_id, {}),
                    )
                )
        return tuple(sorted(records, key=lambda record: record.unified_model_id))

    def _record(
        self,
        model: Any,
        checkpoint: Any,
        snapshot: Any,
        manifest: dict[str, str],
    ) -> UnifiedModelRecord:
        model_id = str(_value(model, "model_id"))
        checkpoint_id = str(_value(checkpoint, "checkpoint_id"))
        implementation = _value(model, "implementation", {})
        adapter = str(_value(implementation, "adapter_status", "not_started"))
        smoke = str(_value(implementation, "smoke_test_status", "not_run"))
        verification = _value(checkpoint, "verification", {}) or {}
        adapter_verification = str(_value(verification, "adapter", "pending"))
        feature_verification = str(_value(verification, "feature_output", "pending"))
        preprocessing_verification = str(
            _value(verification, "preprocessing", "pending")
        )
        license_verification = str(_value(verification, "license", "pending"))
        checkpoint_verification = str(
            _value(checkpoint, "verification_status", "seed_unverified")
        )
        if adapter_verification == "blocked" or (
            adapter_verification == "verified" and (adapter == "failed" or smoke == "failed")
        ):
            adapter_status = BaseAdapterStatus.FAILED
        elif (
            adapter_verification == "verified"
            and feature_verification == "verified"
            and adapter == "implemented"
            and smoke == "passed"
        ):
            adapter_status = BaseAdapterStatus.SMOKE_TEST_PASSED
        elif adapter_verification == "verified":
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
        official_source_verified = (
            manifest.get("source_provenance_status") == "official_source_verified"
        )
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
                    checkpoint_verification
                ),
            },
            model_name=str(_value(model, "model_name", "")),
            year=_value(model, "year"),
            venue=str(_value(model, "venue", "")),
            model_category=str(_value(model, "model_category", "")),
            architecture=str(_value(model, "architecture", "")),
            pretraining_data_summary=str(_value(model, "pretraining_data_summary", "")),
            pretraining_strategy=str(_value(model, "pretraining_strategy", "")),
            reported_summary=str(_value(model, "reported_summary", "")),
            paper_url=str(_value(model, "paper_url", "") or ""),
            code_url=str(_value(model, "code_url", "") or ""),
            runtime_phase=str(_value(model, "runtime_phase", "")),
            verification_status=str(_value(model, "verification_status", "")),
            license=str(
                _value(checkpoint, "license", "")
                or manifest.get("upstream_weight_license", "")
                or _value(model, "license", "")
                or ""
            ),
            license_verified=bool(_value(model, "license_verified", False)),
            checkpoint_name=str(_value(checkpoint, "checkpoint_name", "")),
            checkpoint_provider=str(_value(checkpoint, "provider", "")),
            weight_url=str(weight_url or ""),
            access_type=str(_value(checkpoint, "access_type", "")),
            requires_auth=bool(_value(checkpoint, "requires_auth", False)),
            framework=str(_value(checkpoint, "framework", "") or ""),
            input_size=str(_value(checkpoint, "input_size", "") or ""),
            normalization=str(_value(checkpoint, "normalization", "") or ""),
            embedding_dim=_value(checkpoint, "embedding_dim"),
            sha256=str(
                _value(checkpoint, "sha256", "") or manifest.get("sha256", "") or ""
            ),
            last_verified=str(_value(checkpoint, "last_verified", "") or ""),
            checkpoint_verification_status=checkpoint_verification,
            artifact_type=str(_value(checkpoint, "artifact_type", "") or ""),
            model_modalities=tuple(
                str(value) for value in _value(model, "modalities", [])
            ),
            catalog_registered=True,
            official_source_verified=official_source_verified,
            download_status=str(manifest.get("download_status", "not_reported")),
            local_integrity_status=str(
                manifest.get("local_integrity_status", "not_reported")
            ),
            provider_verification_status=str(
                manifest.get("provider_integrity_status", "not_reported")
            ),
            runtime_status=str(manifest.get("runtime_status", "not_reported")),
            local_asset_status=str(
                manifest.get("local_asset_status", "not_reported")
            ),
            upstream_license_status=license_verification,
            preprocessing_verification_status=preprocessing_verification,
            local_asset_verified=False,
            adapter_implemented=adapter_status
            in {
                BaseAdapterStatus.IMPLEMENTED_NOT_TESTED,
                BaseAdapterStatus.SMOKE_TEST_PASSED,
            },
            encoder_smoke_passed=base_ready,
            feature_output_verified=feature_verification == "verified",
            task_adapted=False,
            unified_evaluation_complete=False,
            cost_verified=False,
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
