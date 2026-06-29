#!/usr/bin/env python3
"""Run an existing controlled-protocol pipeline from one declarative config.

The runner deliberately does not implement model evaluation or routing math. Each
stage invokes an existing repository script and declares its inputs and outputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ".controlled_runner_state.json"
DR_RISK_COLUMN_MARKERS = (
    "large_undergrading",
    "referable_miss",
    "severe_pdr_miss",
    "vtdr",
)


class RunnerError(RuntimeError):
    pass


@dataclass
class StageResult:
    stage_id: str
    kind: str
    status: str
    duration_sec: float
    fingerprint: str
    command: str
    outputs: list[Path]


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RunnerError("YAML config requires PyYAML; install project requirements first") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RunnerError("config root must be a mapping")
    return data


def validate_config(config: dict[str, Any]) -> None:
    for key in ("protocol_id", "mode", "selection_split", "evaluation_split", "stages"):
        if key not in config:
            raise RunnerError(f"missing config field: {key}")

    mode = str(config["mode"])
    if mode not in {"exploratory", "final"}:
        raise RunnerError("mode must be exploratory or final")
    if mode == "final" and config["selection_split"] == config["evaluation_split"]:
        raise RunnerError(
            "final mode requires different selection_split and evaluation_split; "
            f"both are {config['selection_split']!r}"
        )

    risk_profile = str(config.get("risk_metric_profile", "unspecified"))
    if risk_profile not in {
        "unspecified",
        "generic_multiclass",
        "dr_icdr_5class",
        "custom",
    }:
        raise RunnerError(f"unsupported risk_metric_profile: {risk_profile}")

    seen: set[str] = set()
    for raw in config["stages"]:
        if not isinstance(raw, dict):
            raise RunnerError("every stage must be a mapping")
        stage_id = str(raw.get("id", "")).strip()
        if not stage_id:
            raise RunnerError("every stage requires a non-empty id")
        if stage_id in seen:
            raise RunnerError(f"duplicated stage id: {stage_id}")
        for dep in raw.get("depends_on", []):
            if dep not in seen:
                raise RunnerError(f"stage {stage_id} depends on unknown or later stage: {dep}")
        command = raw.get("command")
        if not isinstance(command, list) or not command:
            raise RunnerError(f"stage {stage_id} requires a non-empty command list")
        seen.add(stage_id)


def resolve_path(value: str, *, repo_root: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else repo_root / path


def expand_paths(values: list[str], *, repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        if any(char in expanded for char in "*?["):
            pattern = Path(expanded)
            if pattern.is_absolute():
                parent = Path(pattern.anchor)
                relative = str(pattern)[len(pattern.anchor) :].lstrip("\\/")
                matches = sorted(parent.glob(relative))
            else:
                matches = sorted(repo_root.glob(expanded))
            paths.extend(matches)
        else:
            paths.append(resolve_path(expanded, repo_root=repo_root))
    return paths


def file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    signature: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_file() and stat.st_size <= 5 * 1024 * 1024:
        signature["sha256"] = sha256_file(path)
    return signature


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_records(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enabled_registry_row(row: dict[str, str]) -> bool:
    return str(row.get("enabled", "1")).strip().lower() not in {"0", "false", "no"}


def load_model_costs(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str | None]:
    settings = config.get("cost_enrichment", {})
    registry_value = settings.get("model_registry")
    if not registry_value:
        return {}, None

    registry_path = resolve_path(str(registry_value), repo_root=REPO_ROOT)
    if not registry_path.exists():
        raise RunnerError(f"cost model registry is missing: {registry_path}")

    _, registry_rows = read_csv_records(registry_path)
    costs: dict[str, dict[str, Any]] = {}
    source_cache: dict[Path, list[dict[str, str]]] = {}
    expert_candidates: list[str] = []

    for registry_row in registry_rows:
        if not enabled_registry_row(registry_row):
            continue
        model_name = str(registry_row.get("model_name", "")).strip()
        cost_value = str(registry_row.get("cost_csv", "")).strip()
        if not model_name or not cost_value:
            continue
        if "expert" in str(registry_row.get("role_hint", "")).lower():
            expert_candidates.append(model_name)

        cost_path = resolve_path(cost_value, repo_root=REPO_ROOT)
        if not cost_path.exists():
            continue
        if cost_path not in source_cache:
            _, source_cache[cost_path] = read_csv_records(cost_path)
        matches = [row for row in source_cache[cost_path] if row.get("model_name") == model_name]
        if len(matches) != 1:
            continue

        row = matches[0]
        estimate = optional_float(
            row.get("mean_ms_per_image") or row.get("estimated_forward_ms_per_image")
        )
        if estimate is None or estimate <= 0:
            continue
        throughput = optional_float(row.get("images_per_second")) or (1000.0 / estimate)
        costs[model_name] = {
            "estimated_forward_ms_per_image": estimate,
            "images_per_second": throughput,
            "peak_allocated_memory_mb": optional_float(
                row.get("pytorch_peak_allocated_mem_mb") or row.get("peak_allocated_memory_mb")
            ),
            "checkpoint_mb": optional_float(row.get("checkpoint_mb")),
            "batch_size": row.get("batch_size", ""),
            "device": row.get("device", ""),
            "timing_source": str(cost_path),
            "timing_scope": row.get("cost_note", "single-GPU forward-only benchmark"),
        }

    explicit_expert = str(settings.get("expert_reference_model", "")).strip()
    expert_reference = explicit_expert or (
        expert_candidates[0] if len(set(expert_candidates)) == 1 else None
    )
    if expert_reference and expert_reference not in costs:
        expert_reference = None
    return costs, expert_reference


def enrich_model_baselines(path: Path, config: dict[str, Any]) -> None:
    fieldnames, rows = read_csv_records(path)
    if not rows:
        return
    model_column = "name" if "name" in fieldnames else "model_name" if "model_name" in fieldnames else None
    if model_column is None:
        return

    costs, expert_reference = load_model_costs(config)
    estimates = [
        cost["estimated_forward_ms_per_image"]
        for row in rows
        if (cost := costs.get(str(row.get(model_column, "")))) is not None
    ]
    fastest = min(estimates) if estimates else None
    expert_cost = (
        costs[expert_reference]["estimated_forward_ms_per_image"]
        if expert_reference in costs
        else None
    )
    appended = [
        "estimated_forward_ms_per_image",
        "images_per_second",
        "relative_forward_cost_vs_fastest_model",
        "relative_forward_cost_vs_expert",
        "peak_allocated_memory_mb",
        "checkpoint_mb",
        "batch_size",
        "device",
        "timing_source",
        "timing_scope",
    ]
    for name in appended:
        if name not in fieldnames:
            fieldnames.append(name)

    for row in rows:
        cost = costs.get(str(row.get(model_column, "")))
        if cost is None:
            for name in appended:
                row.setdefault(name, "")
            continue
        estimate = cost["estimated_forward_ms_per_image"]
        row.update(cost)
        row["relative_forward_cost_vs_fastest_model"] = (
            estimate / fastest if fastest else ""
        )
        row["relative_forward_cost_vs_expert"] = (
            estimate / expert_cost if expert_cost else ""
        )
    write_csv_records(path, fieldnames, rows)


def validate_risk_columns(fieldnames: list[str], config: dict[str, Any]) -> None:
    if str(config.get("risk_metric_profile", "unspecified")) != "generic_multiclass":
        return
    offending = [
        name for name in fieldnames if any(marker in name.lower() for marker in DR_RISK_COLUMN_MARKERS)
    ]
    if offending:
        raise RunnerError(
            "generic_multiclass output contains DR-specific risk columns: " + ", ".join(offending)
        )


def enrich_routing_results(path: Path, config: dict[str, Any]) -> None:
    fieldnames, rows = read_csv_records(path)
    validate_risk_columns(fieldnames, config)
    if not rows:
        return

    appended = [
        "estimated_forward_ms_per_image",
        "relative_forward_cost_vs_dense_expert",
        "forward_cost_reduction_vs_dense_expert",
    ]
    for name in appended:
        if name not in fieldnames:
            fieldnames.append(name)

    estimates: list[float | None] = []
    for row in rows:
        estimate = optional_float(
            row.get("estimated_forward_ms_per_image")
            or row.get("ms_per_image")
            or row.get("online_no_cache_ms_per_image")
        )
        estimates.append(estimate)
        row["estimated_forward_ms_per_image"] = estimate if estimate is not None else ""

    dense_candidates = [
        estimate
        for row, estimate in zip(rows, estimates)
        if estimate is not None and str(row.get("role", "")) == "dense_expert_reference"
    ]
    dense_reference = dense_candidates[0] if len(dense_candidates) == 1 else None
    for row, estimate in zip(rows, estimates):
        if estimate is None or not dense_reference:
            row["relative_forward_cost_vs_dense_expert"] = ""
            row["forward_cost_reduction_vs_dense_expert"] = ""
            continue
        ratio = estimate / dense_reference
        row["relative_forward_cost_vs_dense_expert"] = ratio
        row["forward_cost_reduction_vs_dense_expert"] = 1.0 - ratio
    write_csv_records(path, fieldnames, rows)


def normalize_published_artifact(name: str, target: Path, config: dict[str, Any]) -> None:
    if target.suffix.lower() != ".csv":
        return
    if name == "model_baselines":
        enrich_model_baselines(target, config)
    elif name == "routing_results":
        enrich_routing_results(target, config)


def render_command(tokens: list[Any], *, config_path: Path, output_dir: Path) -> list[str]:
    values = {
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
        "config_dir": str(config_path.parent),
        "output_dir": str(output_dir),
    }
    return [str(token).format(**values) for token in tokens]


def stage_fingerprint(
    stage: dict[str, Any],
    *,
    command: list[str],
    inputs: list[Path],
    config_hash: str,
) -> str:
    payload = {
        "stage_id": stage["id"],
        "kind": stage.get("kind", "command"),
        "command": command,
        "inputs": [file_signature(path) for path in inputs],
        "config_hash": config_hash,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RunnerError(f"cannot read runner state: {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("stages", {}), dict):
        raise RunnerError(f"invalid runner state: {path}")
    data.setdefault("stages", {})
    return data


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def output_paths(stage: dict[str, Any]) -> list[Path]:
    return expand_paths([str(x) for x in stage.get("outputs", [])], repo_root=REPO_ROOT)


def outputs_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def missing_inputs(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def is_forced(stage: dict[str, Any], force_values: set[str]) -> bool:
    return str(stage["id"]) in force_values or str(stage.get("kind", "command")) in force_values


def run_stage(
    stage: dict[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    config_hash: str,
    state: dict[str, Any],
    resume: bool,
    force_values: set[str],
    train_missing: bool,
    dry_run: bool,
    planned_available: set[Path],
) -> StageResult:
    stage_id = str(stage["id"])
    kind = str(stage.get("kind", "command"))
    inputs = expand_paths([str(x) for x in stage.get("inputs", [])], repo_root=REPO_ROOT)
    outputs = output_paths(stage)
    command = render_command(stage["command"], config_path=config_path, output_dir=output_dir)
    fingerprint = stage_fingerprint(stage, command=command, inputs=inputs, config_hash=config_hash)
    command_text = subprocess.list2cmdline(command)
    forced = is_forced(stage, force_values)
    ready = outputs_exist(outputs)
    current_output_signatures = [file_signature(path) for path in outputs] if ready else []

    if dry_run:
        missing = [path for path in missing_inputs(inputs) if path not in planned_available]
        state_label = "BLOCKED" if missing else "PLANNED"
        print(f"[{state_label}] {stage_id} ({kind})")
        if missing:
            for path in missing:
                print(f"  missing input: {path}")
        print(f"  command: {command_text}")
        return StageResult(stage_id, kind, state_label, 0.0, fingerprint, command_text, outputs)

    if kind == "train" and not train_missing:
        if ready and not forced:
            print(f"[SKIPPED] {stage_id}: existing training outputs")
            return StageResult(stage_id, kind, "skipped", 0.0, fingerprint, command_text, outputs)
        raise RunnerError(
            f"stage {stage_id} is a training stage with missing or forced outputs; "
            "rerun with --train-missing to permit training"
        )

    previous = state["stages"].get(stage_id, {})
    if (
        resume
        and not forced
        and ready
        and previous.get("fingerprint") == fingerprint
        and previous.get("outputs") == current_output_signatures
    ):
        print(f"[SKIPPED] {stage_id}: fingerprint unchanged")
        return StageResult(stage_id, kind, "skipped", 0.0, fingerprint, command_text, outputs)

    if resume and not forced and ready and stage.get("reuse_existing", False) and not previous:
        print(f"[SKIPPED] {stage_id}: adopted existing outputs")
        state["stages"][stage_id] = {
            "fingerprint": fingerprint,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "adopted": True,
            "outputs": current_output_signatures,
        }
        return StageResult(stage_id, kind, "adopted", 0.0, fingerprint, command_text, outputs)

    missing = missing_inputs(inputs)
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise RunnerError(f"stage {stage_id} has missing inputs:\n{joined}")

    print(f"[RUNNING] {stage_id} ({kind})")
    started = time.perf_counter()
    cwd = resolve_path(str(stage.get("cwd", REPO_ROOT)), repo_root=REPO_ROOT)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in stage.get("env", {}).items()})
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    duration = time.perf_counter() - started
    if completed.returncode != 0:
        raise RunnerError(f"stage {stage_id} failed with exit code {completed.returncode}")

    missing_outputs = [path for path in outputs if not path.exists()]
    if missing_outputs:
        joined = "\n".join(f"  - {path}" for path in missing_outputs)
        raise RunnerError(f"stage {stage_id} completed but declared outputs are missing:\n{joined}")

    state["stages"][stage_id] = {
        "fingerprint": fingerprint,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": duration,
        "outputs": [file_signature(path) for path in outputs],
    }
    print(f"[DONE] {stage_id}: {duration:.2f}s")
    return StageResult(stage_id, kind, "executed", duration, fingerprint, command_text, outputs)


def publish_artifacts(
    config: dict[str, Any],
    *,
    output_dir: Path,
    config_path: Path,
    config_hash: str,
    stage_results: list[StageResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = config.get("publish", {}).get("artifacts", [])
    rows: list[dict[str, Any]] = []
    created_at = datetime.now(timezone.utc).isoformat()

    for item in artifacts:
        name = str(item["name"])
        source = resolve_path(str(item["source"]), repo_root=REPO_ROOT)
        target = output_dir / str(item.get("target", source.name))
        required = bool(item.get("required", True))
        if not source.exists():
            if required:
                raise RunnerError(f"publish source is missing for {name}: {source}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        normalize_published_artifact(name, target, config)
        stat = target.stat()
        producer_status = "external"
        for result in stage_results:
            if any(source.resolve() == output.resolve() for output in result.outputs):
                producer_status = "generated" if result.status == "executed" else "reused"
                break
        rows.append(
            {
                "protocol_id": config["protocol_id"],
                "mode": config["mode"],
                "artifact_name": name,
                "source_path": str(source),
                "published_path": str(target),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(target),
                "config_sha256": config_hash,
                "created_at_utc": created_at,
                "reused_or_generated": producer_status,
            }
        )

    manifest = output_dir / "artifact_manifest.csv"
    fieldnames = [
        "protocol_id",
        "mode",
        "artifact_name",
        "source_path",
        "published_path",
        "size_bytes",
        "mtime_utc",
        "sha256",
        "config_sha256",
        "created_at_utc",
        "reused_or_generated",
    ]
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_html_report(
        output_dir / str(config.get("publish", {}).get("report", "report.html")),
        config=config,
        config_path=config_path,
        rows=rows,
        stage_results=stage_results,
    )


def csv_preview(path: Path, limit: int = 8) -> tuple[list[str], list[list[str]]]:
    if path.suffix.lower() != ".csv":
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = []
        for index, row in enumerate(reader):
            rows.append(row)
            if index >= limit:
                break
    if not rows:
        return [], []
    return rows[0], rows[1:]


def write_html_report(
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    rows: list[dict[str, Any]],
    stage_results: list[StageResult],
) -> None:
    warning = ""
    if config["mode"] == "exploratory":
        warning = (
            "<div class='warning'><strong>探索性结果（exploratory）：</strong>选择与评估可能使用"
            "同一数据划分，不得将本报告表述为无偏的最终评估。</div>"
        )

    cost_notice = ""
    if config.get("cost_enrichment"):
        cost_notice = (
            "<div class='notice'><strong>成本口径：</strong>估算的仅前向传播成本"
            "（estimated forward-only cost）。不包括图像解码（image decoding）、预处理、"
            "磁盘与网络 I/O、主机到设备传输、排队、模型加载、服务开销、后处理和临床"
            "工作流耗时。</div>"
        )

    stage_html = "".join(
        "<tr>"
        f"<td>{html.escape(result.stage_id)}</td>"
        f"<td>{html.escape(result.kind)}</td>"
        f"<td>{html.escape(result.status)}</td>"
        f"<td>{result.duration_sec:.2f}</td>"
        "</tr>"
        for result in stage_results
    )

    artifact_sections = []
    artifact_labels = {
        "model_baselines": "模型基线",
        "routing_results": "路由结果",
        "risk_results": "风险结果",
        "case_audit": "病例审计",
    }
    for row in rows:
        artifact_path = Path(row["published_path"])
        headers, preview = csv_preview(artifact_path)
        table = ""
        if headers:
            head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
            body = "".join(
                "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"
                for values in preview
            )
            table = f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        artifact_sections.append(
            f"<section><h2>{html.escape(artifact_labels.get(str(row['artifact_name']), str(row['artifact_name'])))}"
            f" <small>({html.escape(str(row['artifact_name']))})</small></h2>"
            f"<p><code>{html.escape(str(artifact_path))}</code></p>{table}</section>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(str(config['protocol_id']))} 受控协议报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;margin:0;background:#f4f7fa;color:#14253d}}
main{{max-width:1180px;margin:0 auto;padding:36px 24px 64px}}
h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:20px;margin-top:32px}}
.meta{{color:#607086;margin-bottom:24px}} .warning{{background:#fff4dc;border-left:4px solid #c88012;padding:14px 16px;margin:22px 0}}
.notice{{background:#eaf4f3;border-left:4px solid #0f8178;padding:14px 16px;margin:22px 0}}
table{{border-collapse:collapse;width:100%;background:white}} th,td{{padding:9px 11px;border:1px solid #d9e1ea;text-align:left;font-size:13px;white-space:nowrap}}
th{{background:#eaf0f6}} .table-wrap{{overflow:auto;border:1px solid #d9e1ea}} code{{font-family:Consolas,monospace}}
</style>
</head>
<body><main>
<h1>{html.escape(str(config['protocol_id']))} 受控协议报告</h1>
<div class="meta">运行模式={html.escape(str(config['mode']))} | 选择数据划分={html.escape(str(config['selection_split']))} | 评估数据划分={html.escape(str(config['evaluation_split']))}<br>配置文件=<code>{html.escape(str(config_path))}</code></div>
{warning}
{cost_notice}
<section><h2>流水线阶段</h2><table><thead><tr><th>阶段</th><th>类型</th><th>状态</th><th>耗时（秒）</th></tr></thead><tbody>{stage_html}</tbody></table></section>
{''.join(artifact_sections)}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Controlled protocol JSON/YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writes")
    parser.add_argument("--resume", action="store_true", help="Reuse stages with matching fingerprints")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="Force a stage id or stage kind; repeat for multiple values",
    )
    parser.add_argument(
        "--train-missing",
        action="store_true",
        help="Permit stages with kind=train to execute when outputs are missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        validate_config(config)
        config_hash = sha256_file(config_path)
        output_dir = resolve_path(str(config.get("output_dir", "outputs/controlled_protocol")), repo_root=REPO_ROOT)
        state_path = output_dir / STATE_FILE
        state = load_state(state_path) if args.resume and not args.dry_run else {"stages": {}}
        state.update(
            {
                "protocol_id": config["protocol_id"],
                "config_sha256": config_hash,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        results: list[StageResult] = []
        planned_available: set[Path] = set()
        for stage in config["stages"]:
            result = run_stage(
                stage,
                config_path=config_path,
                output_dir=output_dir,
                config_hash=config_hash,
                state=state,
                resume=args.resume,
                force_values=set(args.force_stage),
                train_missing=args.train_missing,
                dry_run=args.dry_run,
                planned_available=planned_available,
            )
            results.append(result)
            planned_available.update(result.outputs)
            if not args.dry_run and result.status in {"executed", "adopted"}:
                write_state(state_path, state)

        if args.dry_run:
            blocked = [result for result in results if result.status == "BLOCKED"]
            if blocked:
                raise RunnerError(f"dry-run found {len(blocked)} blocked stage(s)")
            print("[DRY-RUN COMPLETE] no commands executed and no files written")
            return 0

        publish_artifacts(
            config,
            output_dir=output_dir,
            config_path=config_path,
            config_hash=config_hash,
            stage_results=results,
        )
        write_state(state_path, state)
        print(f"[COMPLETE] published controlled protocol results to {output_dir}")
        return 0
    except RunnerError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
