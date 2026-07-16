from __future__ import annotations

import csv
from pathlib import Path
import sys
import types
import zipfile

import pandas as pd
import yaml

import app.model_asset_runtime as runtime
from app.model_asset_runtime import build_asset_readiness, run_probe_worker


class _Record:
    def __init__(self, checkpoint_id: str, *, adapter: bool = False):
        self.source_checkpoint_id = checkpoint_id
        self.modalities = ("CFP",)
        self.framework = "PyTorch"
        self.adapter_implemented = adapter
        self.encoder_smoke_passed = False


def _manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = [
        "model_id",
        "checkpoint_id",
        "filename",
        "size_bytes",
        "artifact_type",
        "local_integrity_status",
        "source_provenance_status",
        "sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_readiness_requires_current_file_and_adapter_for_runtime_smoke(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    checkpoint = root / "retfound" / "retfound-cfp" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test")
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            {
                "model_id": "retfound",
                "checkpoint_id": "retfound-cfp",
                "filename": "model.pth",
                "size_bytes": "4",
                "artifact_type": "foundation_encoder",
                "local_integrity_status": "local_size_sha256_and_non_html_verified",
                "source_provenance_status": "official_source_verified",
                "sha256": "abc",
            }
        ],
    )
    ready = build_asset_readiness(
        root,
        manifest_path=manifest,
        records=[_Record("retfound-cfp", adapter=True)],
    ).iloc[0]

    assert ready["local_asset_exists"]
    assert ready["size_matches_manifest"]
    assert ready["registry_sha256_evidence"]
    assert ready["sha256_rechecked_this_run"] == False  # noqa: E712
    assert ready["runtime_smoke_eligible"]
    assert ready["task_inference_ready"] == False  # noqa: E712
    assert ready["route_eligible"] == False  # noqa: E712


def test_numeric_or_registry_status_never_substitutes_for_current_file(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            {
                "model_id": "missing",
                "checkpoint_id": "missing-default",
                "filename": "missing.safetensors",
                "size_bytes": "10",
                "artifact_type": "foundation_encoder",
                "local_integrity_status": "local_size_sha256_and_non_html_verified",
                "source_provenance_status": "official_source_verified",
            }
        ],
    )
    row = build_asset_readiness(
        tmp_path / "assets", manifest_path=manifest, records=[]
    ).iloc[0]

    assert row["local_asset_exists"] == False  # noqa: E712
    assert row["probe_eligible"] == False  # noqa: E712
    assert "本机资产不存在" in row["blocked_reason"]


def test_zip_probe_reads_structure_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("weights/config.json", "{}")
        handle.writestr("weights/model.bin", "payload")
    result = run_probe_worker({"asset_path": str(archive), "probe_profile": "zip_structure"})

    assert result["status"] == "passed"
    assert result["achieved_stage"] == "asset_probe_passed"
    assert result["asset_probe_passed"] is True
    assert result["runtime_smoke_passed"] is False
    assert result["details"]["member_count"] == 2
    assert result["task_inference_ready"] is False


def test_safetensors_probe_reports_tensor_shapes(tmp_path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    checkpoint = tmp_path / "model.safetensors"
    save_file({"encoder.weight": torch.ones(2, 3)}, checkpoint)
    result = run_probe_worker(
        {"asset_path": str(checkpoint), "probe_profile": "safetensors_structure"}
    )

    assert result["status"] == "passed"
    assert result["achieved_stage"] == "asset_probe_passed"
    assert result["details"]["tensor_count"] == 1
    assert result["route_eligible"] is False


def test_latest_result_contract_does_not_grant_online_qualification() -> None:
    result = pd.DataFrame(
        [{"status": "passed", "task_inference_ready": False, "route_eligible": False}]
    )
    assert not result["task_inference_ready"].any()
    assert not result["route_eligible"].any()


def test_visionunite_resource_audit_is_not_runtime_eligible(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    checkpoint = root / "visionunite" / "visionunite-default" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test")
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            {
                "model_id": "visionunite",
                "checkpoint_id": "visionunite-default",
                "filename": "model.pth",
                "size_bytes": "4",
                "artifact_type": "multimodal_full_model",
                "local_integrity_status": "local_size_sha256_and_non_html_verified",
                "source_provenance_status": "official_source_verified",
            }
        ],
    )
    row = build_asset_readiness(root, manifest_path=manifest, records=[]).iloc[0]

    assert row["probe_profile"] == "visionunite_resource_audit"
    assert row["probe_eligible"]
    assert row["runtime_smoke_eligible"] == False  # noqa: E712
    assert row["task_inference_ready"] == False  # noqa: E712
    assert row["route_eligible"] == False  # noqa: E712


def test_runtime_stage_evidence_never_grants_route_qualification(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test")
    monkeypatch.setitem(
        runtime.RUNTIME_HANDLERS,
        "test_runtime",
        lambda path, spec: {
            "source_commit": "abc123",
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 2],
        },
    )
    result = run_probe_worker(
        {
            "asset_path": str(checkpoint),
            "probe_profile": "test_runtime",
            "device": "cpu",
        }
    )

    assert result["runtime_smoke_passed"] is True
    assert len(result["details"]["stage_evidence"]) == 8
    assert result["details"]["process_peak_vram_bytes"] == 0
    assert result["task_inference_ready"] is False
    assert result["route_eligible"] is False


def test_runtime_peak_memory_uses_the_resolved_cuda_device(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"test")
    observed = {"selected": [], "reset": [], "read": []}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_device(device) -> None:
            observed["selected"].append(str(device))

        @staticmethod
        def reset_peak_memory_stats(device) -> None:
            observed["reset"].append(str(device))

        @staticmethod
        def max_memory_allocated(device) -> int:
            observed["read"].append(str(device))
            return 1234

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(), device=lambda value: value
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        runtime.RUNTIME_HANDLERS,
        "cuda_device_test",
        lambda path, spec: {"source_commit": "abc123"},
    )

    result = run_probe_worker(
        {
            "asset_path": str(checkpoint),
            "probe_profile": "cuda_device_test",
            "device": "cuda:3",
        }
    )

    assert observed == {
        "selected": ["cuda:3"],
        "reset": ["cuda:3"],
        "read": ["cuda:3"],
    }
    assert result["details"]["process_peak_vram_bytes"] == 1234


def test_checked_runtime_contract_outputs_cover_all_local_assets() -> None:
    project_root = Path(__file__).resolve().parents[1]
    contract_paths = sorted((project_root / "configs/model_runtime").glob("*/*.yaml"))
    assert len(contract_paths) == 19
    allowed_statuses = {
        "verified",
        "partially_verified",
        "ambiguous",
        "conflicting_sources",
        "not_provided",
    }
    for path in contract_paths:
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert contract["evidence"]["verification_status"] in allowed_statuses

    with (project_root / "artifacts/model_runtime_smoke/official_runtime_contracts.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        runtime_rows = list(csv.DictReader(handle))
    with (project_root / "artifacts/model_runtime_smoke/official_transfer_protocols.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        transfer_rows = list(csv.DictReader(handle))
    assert len(runtime_rows) == 19
    assert len(transfer_rows) == 38
    assert {row["track"] for row in transfer_rows} == {
        "native_official_track",
        "unified_transfer_track",
    }
