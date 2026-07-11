from pathlib import Path

import pandas as pd
import pytest

from scripts.training.train_ophbench_retfound_linear_probe import (
    REGISTRATION_PROVENANCE_FIELDS,
    STANDARD_RUN_FILES,
    validate_standard_run,
)


def _standard_run(root: Path) -> Path:
    root.mkdir()
    for name in STANDARD_RUN_FILES - {"registration_record.csv"}:
        (root / name).write_text("fixture", encoding="utf-8")
    record = {field: "fixture" for field in REGISTRATION_PROVENANCE_FIELDS}
    record.update(
        {
            "artifact_id": "aptos2019-retfound-cfp-linear-probe-v2",
            "task_checkpoint": True,
            "task_inference_ready": True,
            "route_eligible": True,
        }
    )
    pd.DataFrame([record]).to_csv(root / "registration_record.csv", index=False)
    return root


def test_standard_run_requires_complete_files_and_provenance(tmp_path: Path):
    run = _standard_run(tmp_path / "run")
    assert validate_standard_run(run) == {
        "ok": True,
        "artifact_id": "aptos2019-retfound-cfp-linear-probe-v2",
        "files": 10,
    }


def test_standard_run_rejects_missing_manifest(tmp_path: Path):
    run = _standard_run(tmp_path / "run")
    (run / "dataset_manifest.json").unlink()
    with pytest.raises(ValueError, match="dataset_manifest.json"):
        validate_standard_run(run)
