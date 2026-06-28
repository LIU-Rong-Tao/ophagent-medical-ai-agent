from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "routing" / "run_controlled_protocol.py"


def write_config(tmp_path: Path, *, stages: list[dict], mode: str = "exploratory", publish=None) -> Path:
    config = {
        "protocol_id": "fixture_protocol",
        "mode": mode,
        "selection_split": "test" if mode == "exploratory" else "val",
        "evaluation_split": "test",
        "output_dir": str(tmp_path / "published"),
        "stages": stages,
        "publish": {"artifacts": publish or []},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_runner(config: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(config), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def counter_command(counter: Path, output: Path) -> list[str]:
    code = (
        "from pathlib import Path; "
        f"counter=Path({str(counter)!r}); output=Path({str(output)!r}); "
        "n=int(counter.read_text()) if counter.exists() else 0; "
        "counter.write_text(str(n+1)); output.parent.mkdir(parents=True, exist_ok=True); "
        "output.write_text('ok')"
    )
    return ["{python}", "-c", code]


def test_dry_run_does_not_execute_or_write_state(tmp_path: Path):
    marker = tmp_path / "marker.txt"
    output = tmp_path / "stage.csv"
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "routing",
                "kind": "routing",
                "command": counter_command(marker, output),
                "outputs": [str(output)],
            }
        ],
    )

    result = run_runner(config, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "PLANNED" in result.stdout
    assert not marker.exists()
    assert not output.exists()
    assert not (tmp_path / "published" / ".controlled_runner_state.json").exists()


def test_dry_run_allows_input_declared_by_earlier_stage(tmp_path: Path):
    intermediate = tmp_path / "intermediate.csv"
    final = tmp_path / "final.csv"
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "prepare",
                "kind": "prediction",
                "command": ["{python}", "-c", "pass"],
                "outputs": [str(intermediate)],
            },
            {
                "id": "evaluate",
                "kind": "routing",
                "depends_on": ["prepare"],
                "command": ["{python}", "-c", "pass"],
                "inputs": [str(intermediate)],
                "outputs": [str(final)],
            },
        ],
    )

    result = run_runner(config, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[PLANNED]") == 2


def test_resume_skips_unchanged_stage_and_reruns_after_input_change(tmp_path: Path):
    input_file = tmp_path / "input.csv"
    counter = tmp_path / "counter.txt"
    output = tmp_path / "stage.csv"
    input_file.write_text("v1", encoding="utf-8")
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "routing",
                "kind": "routing",
                "command": counter_command(counter, output),
                "inputs": [str(input_file)],
                "outputs": [str(output)],
            }
        ],
    )

    first = run_runner(config, "--resume")
    second = run_runner(config, "--resume")
    input_file.write_text("v2-with-different-size", encoding="utf-8")
    third = run_runner(config, "--resume")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "SKIPPED" in second.stdout
    assert third.returncode == 0, third.stderr
    assert counter.read_text(encoding="utf-8") == "2"


def test_resume_reruns_when_declared_output_was_modified(tmp_path: Path):
    counter = tmp_path / "counter.txt"
    output = tmp_path / "stage.csv"
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "routing",
                "kind": "routing",
                "command": counter_command(counter, output),
                "outputs": [str(output)],
            }
        ],
    )

    assert run_runner(config, "--resume").returncode == 0
    output.write_text("tampered", encoding="utf-8")
    rerun = run_runner(config, "--resume")

    assert rerun.returncode == 0, rerun.stderr
    assert counter.read_text(encoding="utf-8") == "2"


def test_force_stage_accepts_stage_kind(tmp_path: Path):
    counter = tmp_path / "counter.txt"
    output = tmp_path / "stage.csv"
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "evaluate-routing",
                "kind": "routing",
                "command": counter_command(counter, output),
                "outputs": [str(output)],
            }
        ],
    )

    assert run_runner(config, "--resume").returncode == 0
    forced = run_runner(config, "--resume", "--force-stage", "routing")

    assert forced.returncode == 0, forced.stderr
    assert counter.read_text(encoding="utf-8") == "2"


def test_missing_training_stage_requires_explicit_flag(tmp_path: Path):
    counter = tmp_path / "counter.txt"
    output = tmp_path / "checkpoint.pth"
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "train-scout",
                "kind": "train",
                "command": counter_command(counter, output),
                "outputs": [str(output)],
            }
        ],
    )

    blocked = run_runner(config, "--resume")
    assert blocked.returncode != 0
    assert "--train-missing" in blocked.stderr
    assert not counter.exists()

    allowed = run_runner(config, "--resume", "--train-missing")
    assert allowed.returncode == 0, allowed.stderr
    assert output.exists()


def test_final_mode_rejects_same_selection_and_evaluation_split(tmp_path: Path):
    config = write_config(tmp_path, stages=[], mode="final")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["selection_split"] = "test"
    config.write_text(json.dumps(payload), encoding="utf-8")

    result = run_runner(config, "--dry-run")

    assert result.returncode != 0
    assert "selection_split" in result.stderr
    assert "evaluation_split" in result.stderr


def test_publish_writes_canonical_artifacts_manifest_and_report(tmp_path: Path):
    raw_baseline = tmp_path / "raw" / "baseline.csv"
    raw_routing = tmp_path / "raw" / "routing.csv"
    raw_cases = tmp_path / "raw" / "cases.csv"
    code = (
        "from pathlib import Path; "
        f"files={[str(raw_baseline), str(raw_routing), str(raw_cases)]!r}; "
        "[(Path(p).parent.mkdir(parents=True, exist_ok=True), "
        "Path(p).write_text('name,value\\nrow,1\\n')) for p in files]"
    )
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "produce-results",
                "kind": "routing",
                "command": ["{python}", "-c", code],
                "outputs": [str(raw_baseline), str(raw_routing), str(raw_cases)],
            }
        ],
        publish=[
            {"name": "model_baselines", "source": str(raw_baseline), "target": "model_baselines.csv"},
            {"name": "routing_results", "source": str(raw_routing), "target": "routing_results.csv"},
            {"name": "case_audit", "source": str(raw_cases), "target": "case_audit.csv"},
        ],
    )

    result = run_runner(config)
    published = tmp_path / "published"

    assert result.returncode == 0, result.stderr
    assert (published / "model_baselines.csv").exists()
    assert (published / "routing_results.csv").exists()
    assert (published / "case_audit.csv").exists()
    assert (published / "artifact_manifest.csv").exists()
    assert (published / "report.html").exists()
    assert "exploratory" in (published / "report.html").read_text(encoding="utf-8")
    with (published / "artifact_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows
    assert manifest_rows[0]["created_at_utc"]
    assert manifest_rows[0]["reused_or_generated"] == "generated"


def test_repository_v082c_profile_has_a_valid_dry_run():
    config = ROOT / "experiments" / "v0_8_3_controlled_runner" / "configs" / "v082c_dr_replay.yaml"

    result = run_runner(config, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "[DRY-RUN COMPLETE]" in result.stdout
