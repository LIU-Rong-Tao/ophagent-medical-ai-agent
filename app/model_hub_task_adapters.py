"""Task Profile adapters used by the Model Hub tool layer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.model_hub_index import build_task_profile_index
from app.orchestration_contracts import ConfiguredTaskAdapter, TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    PROJECT_ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols"
    / "model_hub_task_profiles_v2.json"
)


@dataclass(frozen=True)
class ModelHubTaskAdapter:
    """Inject task-specific labels, risk semantics, and tool assets."""

    configured: ConfiguredTaskAdapter
    risk_proxy_name: str
    risk_proxy_definition: str
    tool_contract: dict[str, Any]

    @property
    def spec(self) -> TaskSpec:
        return self.configured.spec

    def validate_metadata(
        self,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        return self.configured.validate_metadata(metadata)

    def risk_summary(
        self,
        probabilities: tuple[float, ...],
    ) -> dict[str, Any]:
        result = self.configured.risk_summary(probabilities)
        return {
            **result,
            "name": self.risk_proxy_name,
            "definition": self.risk_proxy_definition,
        }

    def as_legacy_contract(self) -> dict[str, Any]:
        """Conservative compatibility projection for existing UI helpers."""

        return {
            "task_name": self.spec.report_label,
            "class_order": list(range(len(self.spec.label_space))),
            "class_labels": list(self.spec.label_space),
            "modality": self.spec.modality,
            "risk_proxy": self.spec.risk_semantics,
            **self.tool_contract,
        }


def load_task_adapters(
    path: Path = PROFILE_PATH,
) -> dict[str, ModelHubTaskAdapter]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = build_task_profile_index(PROJECT_ROOT).set_index("task_id")
    adapters: dict[str, ModelHubTaskAdapter] = {}
    for row in payload.get("profiles", []):
        if not isinstance(row, dict):
            continue
        task_id = str(row["task_id"])
        profile = profiles.loc[task_id]
        spec = TaskSpec(
            task_id=task_id,
            dataset_id=str(profile["dataset_id"]),
            modality=str(profile["modality"]),
            label_space=tuple(
                str(value)
                for value in json.loads(str(profile["label_space"]))
            ),
            primary_metric=str(profile["primary_metric"]),
            risk_semantics=str(profile["risk_semantics"]),
            report_label=str(profile["report_label"]),
            risk_positive_class_ids=tuple(
                int(value)
                for value in json.loads(
                    str(profile["risk_positive_class_ids"])
                )
            ),
            adaptation_type=str(profile["adaptation_type"]),
        )
        adapters[task_id] = ModelHubTaskAdapter(
            configured=ConfiguredTaskAdapter(spec),
            risk_proxy_name=str(row["risk_proxy_name"]),
            risk_proxy_definition=str(row["risk_proxy_definition"]),
            tool_contract=dict(row["tool_contract"]),
        )
    return adapters


TASK_ADAPTERS = load_task_adapters()


def task_adapter_for(task_id: str) -> ModelHubTaskAdapter:
    try:
        return TASK_ADAPTERS[task_id]
    except KeyError as exc:
        raise KeyError(f"unregistered Task Profile: {task_id}") from exc


TASK_CONTRACTS = {
    task_id: adapter.as_legacy_contract()
    for task_id, adapter in TASK_ADAPTERS.items()
}
