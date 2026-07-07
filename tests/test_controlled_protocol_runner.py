from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "routing" / "run_controlled_protocol.py"


def write_config(
    tmp_path: Path,
    *,
    stages: list[dict],
    mode: str = "exploratory",
    publish=None,
    extra: dict | None = None,
) -> Path:
    config = {
        "protocol_id": "fixture_protocol",
        "mode": mode,
        "selection_split": "test" if mode == "exploratory" else "val",
        "evaluation_split": "test",
        "output_dir": str(tmp_path / "published"),
        "stages": stages,
        "publish": {"artifacts": publish or []},
    }
    config.update(extra or {})
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


def test_publish_can_preserve_custom_report(tmp_path: Path):
    source_report = tmp_path / "custom_report.html"
    source_report.write_text(
        "<html><body>interactive replay; not formal model selection</body></html>",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "report",
                "kind": "report",
                "command": ["{python}", "-c", "pass"],
                "inputs": [str(source_report)],
                "outputs": [str(source_report)],
            }
        ],
        extra={
            "publish": {
                "generate_runner_report": False,
                "report": "report.html",
                "artifacts": [
                    {
                        "name": "interactive_report",
                        "source": str(source_report),
                        "target": "report.html",
                    }
                ],
            }
        },
    )

    result = run_runner(config, "--resume")

    assert result.returncode == 0, result.stderr
    published = tmp_path / "published" / "report.html"
    assert "interactive replay" in published.read_text(encoding="utf-8")
    assert (tmp_path / "published" / "artifact_manifest.csv").exists()


def test_publish_can_hide_ephemeral_source_paths_from_manifest(tmp_path: Path):
    source = tmp_path / "work" / "artifact.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("value\n1\n", encoding="utf-8")
    config = write_config(
        tmp_path,
        stages=[
            {
                "id": "publishable",
                "kind": "report",
                "command": ["{python}", "-c", "pass"],
                "inputs": [str(source)],
                "outputs": [str(source)],
            }
        ],
        extra={
            "publish": {
                "stable_manifest_paths": True,
                "artifacts": [
                    {
                        "name": "artifact",
                        "source": str(source),
                        "target": "artifact.csv",
                    }
                ],
            }
        },
    )

    result = run_runner(config, "--resume")

    assert result.returncode == 0, result.stderr
    manifest = tmp_path / "published" / "artifact_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    published_path = str(tmp_path / "published" / "artifact.csv")
    assert rows[0]["source_path"] == published_path
    assert rows[0]["published_path"] == published_path
    assert "work" not in manifest.read_text(encoding="utf-8-sig")


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
    report_text = (published / "report.html").read_text(encoding="utf-8")
    assert "exploratory" in report_text
    assert "探索性结果" in report_text
    assert "流水线阶段" in report_text
    with (published / "artifact_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows
    assert manifest_rows[0]["created_at_utc"]
    assert manifest_rows[0]["reused_or_generated"] == "generated"


def test_publish_rewrites_intermediate_work_paths_in_csv_and_html(tmp_path: Path):
    work_dir = tmp_path / "work" / "activation"
    source_prefix = str(tmp_path / "work" / "activation")
    target_prefix = str(tmp_path / "published")
    work_dir.mkdir(parents=True)
    csv_names = [
        "adapter_job_summary.csv",
        "adapter_manifest.csv",
        "adapter_vs_legacy_prediction_check.csv",
        "routing_replay_summary.csv",
    ]
    for name in csv_names:
        (work_dir / name).write_text(
            "job_id,predictions_path\n"
            f"job,{source_prefix}/onboarded_models/job/predictions.csv\n",
            encoding="utf-8",
        )
    raw_html = work_dir / "summary.html"
    raw_html.write_text(
        f'<a href="{source_prefix}/onboarded_models/job/predictions.csv">result</a>',
        encoding="utf-8",
    )
    publish = [
        {"name": Path(name).stem, "source": str(work_dir / name), "target": name}
        for name in csv_names
    ]
    publish.append({"name": "summary_html", "source": str(raw_html), "target": "summary.html"})
    config = write_config(
        tmp_path,
        stages=[],
        publish=publish,
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["publish"]["path_rewrites"] = [
        {"source_prefix": source_prefix, "target_prefix": target_prefix}
    ]
    config.write_text(json.dumps(payload), encoding="utf-8")

    result = run_runner(config)

    assert result.returncode == 0, result.stderr
    for path in [*(tmp_path / "published" / name for name in csv_names), tmp_path / "published" / "summary.html"]:
        text = path.read_text(encoding="utf-8-sig")
        assert source_prefix not in text
        assert target_prefix in text


def test_publish_enriches_baselines_and_routing_with_forward_cost(tmp_path: Path):
    raw_baseline = tmp_path / "raw" / "baseline.csv"
    raw_routing = tmp_path / "raw" / "routing.csv"
    registry = tmp_path / "registry.csv"
    scout_cost = tmp_path / "scout_cost.csv"
    expert_cost = tmp_path / "expert_cost.csv"

    raw_baseline.parent.mkdir(parents=True)
    raw_baseline.write_text(
        "name,accuracy,macro_f1,qwk\n"
        "scout,0.8,0.7,0.75\n"
        "expert,0.9,0.8,0.85\n",
        encoding="utf-8",
    )
    raw_routing.write_text(
        "protocol_family,protocol_name,role,ms_per_image,accuracy,macro_f1,qwk\n"
        "dense_baseline,dense_expert,dense_expert_reference,4.0,0.9,0.8,0.85\n"
        "single_scout,scout_to_expert,main,2.0,0.88,0.79,0.83\n",
        encoding="utf-8",
    )
    scout_cost.write_text(
        "model_name,mean_ms_per_image,images_per_second,pytorch_peak_allocated_mem_mb,checkpoint_mb,batch_size,device,cost_note\n"
        "scout,1.0,1000,512,100,32,cuda,forward-only scout benchmark\n",
        encoding="utf-8",
    )
    expert_cost.write_text(
        "model_name,mean_ms_per_image,images_per_second,pytorch_peak_allocated_mem_mb,checkpoint_mb,batch_size,device,cost_note\n"
        "expert,4.0,250,1024,500,32,cuda,forward-only expert benchmark\n",
        encoding="utf-8",
    )
    registry.write_text(
        "model_name,role_hint,cost_csv,enabled\n"
        f"scout,scout,{scout_cost},1\n"
        f"expert,expert,{expert_cost},1\n",
        encoding="utf-8",
    )

    config = write_config(
        tmp_path,
        stages=[],
        publish=[
            {"name": "model_baselines", "source": str(raw_baseline), "target": "model_baselines.csv"},
            {"name": "routing_results", "source": str(raw_routing), "target": "routing_results.csv"},
        ],
        extra={
            "risk_metric_profile": "generic_multiclass",
            "cost_enrichment": {
                "model_registry": str(registry),
                "expert_reference_model": "expert",
            },
        },
    )

    result = run_runner(config)
    assert result.returncode == 0, result.stderr

    with (tmp_path / "published" / "model_baselines.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        baselines = list(csv.DictReader(handle))
    assert baselines[0]["estimated_forward_ms_per_image"] == "1.0"
    assert float(baselines[0]["images_per_second"]) == 1000.0
    assert baselines[0]["relative_forward_cost_vs_fastest_model"] == "1.0"
    assert baselines[0]["relative_forward_cost_vs_expert"] == "0.25"
    assert baselines[0]["accuracy"] == "0.8"
    assert baselines[1]["relative_forward_cost_vs_expert"] == "1.0"
    assert baselines[1]["timing_source"] == str(expert_cost)

    with (tmp_path / "published" / "routing_results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        routing = list(csv.DictReader(handle))
    assert routing[0]["estimated_forward_ms_per_image"] == "4.0"
    assert routing[0]["relative_forward_cost_vs_dense_expert"] == "1.0"
    assert routing[0]["forward_cost_reduction_vs_dense_expert"] == "0.0"
    assert routing[1]["relative_forward_cost_vs_dense_expert"] == "0.5"
    assert routing[1]["forward_cost_reduction_vs_dense_expert"] == "0.5"
    assert routing[1]["accuracy"] == "0.88"

    report = (tmp_path / "published" / "report.html").read_text(encoding="utf-8")
    assert "estimated forward-only cost" in report
    assert "image decoding" in report
    assert "估算的仅前向传播成本" in report


def test_generic_risk_profile_rejects_dr_specific_columns(tmp_path: Path):
    raw_routing = tmp_path / "routing.csv"
    raw_routing.write_text(
        "protocol_name,accuracy,severe_pdr_miss_event_recall_fixed_pool\n"
        "glaucoma_route,0.85,0.9\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        stages=[],
        publish=[
            {"name": "routing_results", "source": str(raw_routing), "target": "routing_results.csv"}
        ],
        extra={"risk_metric_profile": "generic_multiclass"},
    )

    result = run_runner(config)

    assert result.returncode != 0
    assert "DR-specific" in result.stderr


def test_forward_only_scope_shows_chinese_notice_without_legacy_cost_enrichment(tmp_path: Path):
    raw_routing = tmp_path / "routing.csv"
    raw_routing.write_text(
        "protocol_name,cost_status,estimated_forward_ms_per_image\n"
        "scout_to_expert,estimated_from_measured_models,2.0\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        stages=[],
        publish=[
            {"name": "routing_results", "source": str(raw_routing), "target": "routing_results.csv"}
        ],
        extra={"risk_metric_profile": "generic_multiclass", "cost_scope": "forward_only"},
    )

    result = run_runner(config)

    assert result.returncode == 0, result.stderr
    report = (tmp_path / "published" / "report.html").read_text(encoding="utf-8")
    assert "成本口径" in report
    assert "仅前向传播成本" in report


def test_task_agnostic_routing_does_not_add_empty_dense_reference_columns(tmp_path: Path):
    raw_routing = tmp_path / "routing.csv"
    raw_routing.write_text(
        "protocol_name,cost_status,estimated_forward_ms_per_image,"
        "relative_forward_cost_vs_expert_only,forward_cost_reduction_vs_expert_only\n"
        "scout_to_expert,estimated_from_measured_models,2.0,0.5,0.5\n",
        encoding="utf-8",
    )
    config = write_config(
        tmp_path,
        stages=[],
        publish=[
            {"name": "routing_results", "source": str(raw_routing), "target": "routing_results.csv"}
        ],
        extra={"risk_metric_profile": "generic_multiclass", "cost_scope": "forward_only"},
    )

    result = run_runner(config)

    assert result.returncode == 0, result.stderr
    with (tmp_path / "published" / "routing_results.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])
    assert "relative_forward_cost_vs_expert_only" in fieldnames
    assert "relative_forward_cost_vs_dense_expert" not in fieldnames
    assert "forward_cost_reduction_vs_dense_expert" not in fieldnames


def test_report_previews_all_controlled_routing_rows(tmp_path: Path):
    raw_routing = tmp_path / "routing.csv"
    rows = "".join(f"route_{index},0.{index}\n" for index in range(12))
    raw_routing.write_text("protocol_name,accuracy\n" + rows, encoding="utf-8")
    config = write_config(
        tmp_path,
        stages=[],
        publish=[
            {"name": "routing_results", "source": str(raw_routing), "target": "routing_results.csv"}
        ],
        extra={"risk_metric_profile": "generic_multiclass"},
    )

    result = run_runner(config)

    assert result.returncode == 0, result.stderr
    report = (tmp_path / "published" / "report.html").read_text(encoding="utf-8")
    assert "route_11" in report


def test_repository_v082c_profile_has_a_valid_dry_run():
    config = ROOT / "experiments" / "v0_8_3_controlled_runner" / "configs" / "v082c_dr_replay.yaml"

    result = run_runner(config, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "[DRY-RUN COMPLETE]" in result.stdout
