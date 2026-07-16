"""Export checked runtime YAML contracts into audit-friendly CSV summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs/model_runtime"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/model_runtime_smoke"


def _value(data: dict[str, Any], *keys: str, default: Any = "not_provided") -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None or current == "":
        return default
    if isinstance(current, (dict, list)):
        return json.dumps(current, ensure_ascii=False, sort_keys=True)
    return current


def load_contracts() -> list[dict[str, Any]]:
    contracts = []
    for path in sorted(CONFIG_ROOT.glob("*/*.yaml")):
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        contract["contract_path"] = path.relative_to(PROJECT_ROOT).as_posix()
        contracts.append(contract)
    return contracts


def runtime_rows(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for contract in contracts:
        rows.append(
            {
                "model_id": _value(contract, "model_id"),
                "checkpoint_id": _value(contract, "checkpoint_id"),
                "artifact_type": _value(contract, "artifact_type"),
                "modalities": _value(contract, "modalities"),
                "official_repo": _value(contract, "official_source", "repository"),
                "official_commit_or_tag": _value(
                    contract, "official_source", "commit_or_tag"
                ),
                "model_constructor": _value(contract, "model", "constructor"),
                "checkpoint_container": _value(contract, "checkpoint", "container"),
                "state_dict_key": _value(contract, "checkpoint", "state_dict_key"),
                "strict_load": _value(contract, "checkpoint", "strict_load"),
                "key_transform": _value(contract, "checkpoint", "key_transform"),
                "input_size": _value(contract, "preprocessing", "input_size"),
                "resize": _value(contract, "preprocessing", "resize"),
                "crop": _value(contract, "preprocessing", "crop"),
                "normalization": _value(contract, "preprocessing", "normalization"),
                "tokenizer": _value(contract, "preprocessing", "tokenizer"),
                "prompt_template": _value(contract, "preprocessing", "prompt_template"),
                "input_contract": _value(contract, "forward", "input_contract"),
                "forward_method": _value(contract, "forward", "method"),
                "output_node": _value(contract, "forward", "output_node"),
                "expected_output_shape": _value(contract, "forward", "expected_shape"),
                "output_semantics": _value(contract, "forward", "output_semantics"),
                "official_inference_script": _value(contract, "downstream", "script"),
                "official_downstream_script": _value(contract, "downstream", "script"),
                "official_downstream_config": _value(contract, "downstream", "config"),
                "protocol_verification_status": _value(
                    contract, "evidence", "verification_status"
                ),
                "source_path": _value(contract, "evidence", "source_path"),
                "source_symbol_or_config_key": _value(
                    contract, "evidence", "source_symbol_or_config_key"
                ),
                "conflicts": _value(contract, "evidence", "conflicts"),
                "unresolved_items": _value(contract, "evidence", "unresolved_items"),
                "contract_path": contract["contract_path"],
            }
        )
    return rows


def transfer_rows(contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for contract in contracts:
        common = {
            "model_id": _value(contract, "model_id"),
            "checkpoint_id": _value(contract, "checkpoint_id"),
        }
        rows.append(
            {
                **common,
                "track": "native_official_track",
                "dataset": "not_provided",
                "split_method": "not_provided",
                "patient_level_split": "not_provided",
                "preprocessing": _value(contract, "preprocessing"),
                "classifier_head": "not_provided",
                "frozen_or_finetuned": "not_provided",
                "optimizer": "not_provided",
                "learning_rate": "not_provided",
                "weight_decay": "not_provided",
                "batch_size": "not_provided",
                "epochs": "not_provided",
                "scheduler": "not_provided",
                "augmentations": "not_provided",
                "seed": "not_provided",
                "checkpoint_selection": "not_provided",
                "evaluation_metrics": "not_provided",
                "source_evidence": _value(contract, "downstream", "script"),
                "verification_status": _value(
                    contract, "downstream", "official_protocol_status"
                ),
            }
        )
        rows.append(
            {
                **common,
                "track": "unified_transfer_track",
                "dataset": "not_defined",
                "split_method": "not_defined",
                "patient_level_split": "not_defined",
                "preprocessing": "project_protocol_pending",
                "classifier_head": "not_defined",
                "frozen_or_finetuned": "not_defined",
                "optimizer": "not_defined",
                "learning_rate": "not_defined",
                "weight_decay": "not_defined",
                "batch_size": "not_defined",
                "epochs": "not_defined",
                "scheduler": "not_defined",
                "augmentations": "not_defined",
                "seed": "not_defined",
                "checkpoint_selection": "not_defined",
                "evaluation_metrics": "not_defined",
                "source_evidence": "OphAgent unified protocol has not been frozen",
                "verification_status": "not_provided",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    contracts = load_contracts()
    if not contracts:
        raise RuntimeError("No runtime contract YAML files found")
    _write_csv(OUTPUT_ROOT / "official_runtime_contracts.csv", runtime_rows(contracts))
    _write_csv(OUTPUT_ROOT / "official_transfer_protocols.csv", transfer_rows(contracts))
    print(f"Exported {len(contracts)} runtime contracts and {len(contracts) * 2} tracks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
