#!/usr/bin/env python3
"""Aggregate registered scout/expert artifacts into one protocol report.

This script is intentionally deterministic. It does not train models, run
checkpoint inference, or invent metrics. It only reads existing task outputs
declared by registries and publishes unified tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUTHY = {"1", "true", "yes", "y", "on"}
FALSY = {"0", "false", "no", "n", "off"}
DR_RISK_MARKERS = (
    "risk",
    "event",
    "dangerous",
    "large_undergrading",
    "undergrading",
    "referable_miss",
    "severe_pdr_miss",
    "vtdr",
    "miss",
    "residual",
    "recall",
    "precision",
    "lift",
    "captured",
    "auto_released",
)
NON_RISK_COLUMNS = {
    "accuracy",
    "macro_f1",
    "macro_auroc_ovr",
    "macro_aupr_ovr",
    "qwk",
    "n_error",
    "budget",
    "policy",
    "selected_n",
    "ms_per_image",
    "estimated_forward_ms_per_image",
}


class RegistryProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProtocolResult:
    output_dir: Path
    files: list[Path]


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project env includes PyYAML
            raise RegistryProtocolError("读取 YAML 配置需要 PyYAML") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RegistryProtocolError("protocol config 根节点必须是 mapping")
    return data


def resolve_path(value: str | Path, *, config_dir: Path) -> Path:
    raw = str(value).strip()
    if not raw:
        raise RegistryProtocolError("路径字段不能为空")
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if expanded.is_absolute():
        return expanded
    config_relative = config_dir / expanded
    if config_relative.exists():
        return config_relative
    return REPO_ROOT / expanded


def enabled(row: pd.Series | dict[str, Any]) -> bool:
    value = str(row.get("enabled", "true")).strip().lower()
    if value in FALSY:
        return False
    return True


def read_registry(path: Path, required_columns: set[str], *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise RegistryProtocolError(f"找不到 {name}：{path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = required_columns - set(frame.columns)
    if missing:
        raise RegistryProtocolError(f"{name} 缺少字段：{sorted(missing)}")
    return frame.loc[frame.apply(enabled, axis=1)].copy()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def parse_float_set(value: str) -> set[float]:
    output: set[float] = set()
    for item in parse_list(value):
        try:
            output.add(float(item))
        except ValueError as exc:
            raise RegistryProtocolError(f"budget 不是数字：{item}") from exc
    return output


def truthy_string(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def read_source_table(task: pd.Series, filename: str) -> pd.DataFrame:
    source_dir = Path(str(task["source_output_dir"]))
    path = source_dir / filename
    return read_optional_csv(path)


def infer_source_protocol(frame: pd.DataFrame, fallback: str) -> str:
    if "protocol_id" in frame.columns and not frame.empty:
        values = [value for value in frame["protocol_id"].astype(str).unique() if value]
        if values:
            return values[0]
    return fallback


def add_task_context(frame: pd.DataFrame, task: pd.Series, source_protocol: str) -> pd.DataFrame:
    output = frame.copy()
    context = {
        "task_id": str(task["task_id"]),
        "disease_family": str(task["disease_family"]),
        "dataset_id": str(task["dataset_id"]),
        "source_output_dir": str(task["source_output_dir"]),
        "source_protocol": source_protocol,
    }
    for key, value in context.items():
        if key not in output.columns:
            output.insert(0, key, value)
        else:
            output[key] = output[key].where(output[key].astype(str) != "", value)
    return output


def concat_schema(frames: list[pd.DataFrame], fallback_columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=fallback_columns)
    columns: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    normalized = []
    for frame in frames:
        copy = frame.copy()
        for column in columns:
            if column not in copy.columns:
                copy[column] = ""
        normalized.append(copy[columns])
    return pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame(columns=columns)


def has_dr_risk_columns(frame: pd.DataFrame) -> bool:
    return any(any(marker in column for marker in DR_RISK_MARKERS) for column in frame.columns)


def risk_columns_from_routing(frame: pd.DataFrame) -> list[str]:
    output: list[str] = []
    for column in frame.columns:
        normalized = column.lower()
        if normalized in NON_RISK_COLUMNS:
            continue
        if any(marker in normalized for marker in DR_RISK_MARKERS):
            output.append(column)
    return output


def extract_embedded_risk_results(
    routing: pd.DataFrame,
    task: pd.Series,
    source_protocol: str,
) -> pd.DataFrame:
    risk_columns = risk_columns_from_routing(routing)
    if routing.empty or not risk_columns:
        return pd.DataFrame()
    context_columns = [
        column
        for column in [
            "protocol_id",
            "task_id",
            "protocol_family",
            "protocol_name",
            "role",
            "scouts",
            "experts",
            "scout_artifact",
            "expert_artifact",
            "budget",
            "policy",
            "selected_n",
        ]
        if column in routing.columns
    ]
    output = routing[context_columns + risk_columns].copy()
    output["risk_source_mode"] = str(task["risk_source_mode"])
    output["source_schema"] = str(task["source_schema"])
    return add_task_context(output, task, source_protocol)


def filter_protocol_rows(routing: pd.DataFrame, protocol: pd.Series) -> pd.DataFrame:
    if routing.empty:
        return routing.copy()

    frame = routing.copy()
    budgets = parse_float_set(str(protocol["budgets"]))
    policies = set(parse_list(str(protocol["routing_policies"])))

    if "budget" in frame.columns:
        numeric_budget = pd.to_numeric(frame["budget"], errors="coerce")
        frame = frame.loc[numeric_budget.apply(lambda value: any(abs(value - b) < 1e-9 for b in budgets))]
    if policies and "policy" in frame.columns:
        frame = frame.loc[frame["policy"].astype(str).isin(policies)]

    scout = str(protocol["scout_artifact_id"])
    expert = str(protocol["expert_artifact_id"])
    if "scout_artifact" in frame.columns:
        frame = frame.loc[frame["scout_artifact"].astype(str) == scout]
    elif "scouts" in frame.columns:
        frame = frame.loc[
            frame["scouts"].astype(str).apply(lambda value: scout in parse_model_set(value))
        ]
    if "expert_artifact" in frame.columns:
        frame = frame.loc[frame["expert_artifact"].astype(str) == expert]
    elif "experts" in frame.columns:
        frame = frame.loc[
            frame["experts"].astype(str).apply(lambda value: expert in parse_model_set(value))
        ]

    return frame.copy()


def parse_model_set(value: str) -> set[str]:
    pieces: set[str] = set()
    for block in str(value).replace(",", "+").split("+"):
        block = block.strip()
        if block:
            pieces.add(block)
    return pieces


def deployable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "non_deployable" in output.columns:
        output = output.loc[~output["non_deployable"].apply(truthy_string)]
    if "method_kind" in output.columns:
        output = output.loc[output["method_kind"].astype(str).str.lower() != "oracle"]
    if "role" in output.columns:
        output = output.loc[output["role"].astype(str).str.lower() != "oracle"]
    return output.copy()


def best_row_by_macro_f1(frame: pd.DataFrame) -> pd.Series | None:
    deployable = deployable_rows(frame)
    if deployable.empty or "macro_f1" not in deployable.columns:
        return None
    scores = pd.to_numeric(deployable["macro_f1"], errors="coerce")
    if scores.dropna().empty:
        return None
    return deployable.loc[scores.idxmax()]


def validate_registries(
    tasks: pd.DataFrame,
    models: pd.DataFrame,
    routes: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    task_ids = set(tasks["task_id"].astype(str))
    model_ids = set(models["artifact_id"].astype(str))
    model_by_id = models.set_index("artifact_id", drop=False)

    for _, task in tasks.iterrows():
        if str(task["risk_mode"]) not in {"none", "dr_risk_events"}:
            raise RegistryProtocolError(f"不支持的 risk_mode：{task['risk_mode']}")
        if str(task["risk_source_mode"]) not in {
            "empty_schema",
            "standard_file",
            "embedded_in_routing",
        }:
            raise RegistryProtocolError(f"不支持的 risk_source_mode：{task['risk_source_mode']}")
        source_dir = Path(str(task["source_output_dir"]))
        if not source_dir.exists():
            raise RegistryProtocolError(f"{task['task_id']} 的 source_output_dir 不存在：{source_dir}")
        for filename in ("model_baselines.csv", "routing_results.csv"):
            if not (source_dir / filename).exists():
                raise RegistryProtocolError(f"{task['task_id']} 缺少源产物：{filename}")

    for _, model in models.iterrows():
        if str(model["task_id"]) not in task_ids:
            raise RegistryProtocolError(f"{model['artifact_id']} 引用了不存在的 task：{model['task_id']}")
        roles = set(parse_list(str(model["role_candidates"])))
        if not roles or not roles <= {"scout", "expert"}:
            raise RegistryProtocolError(f"{model['artifact_id']} 的 role_candidates 无效：{model['role_candidates']}")

    for _, cost in costs.iterrows():
        if str(cost["artifact_id"]) not in model_ids:
            raise RegistryProtocolError(f"cost_registry 引用了不存在的模型：{cost['artifact_id']}")
        if str(cost["cost_scope"]) not in {"forward_only", "missing"}:
            raise RegistryProtocolError(f"cost_scope 只能是 forward_only 或 missing：{cost['cost_scope']}")

    for _, route in routes.iterrows():
        task_id = str(route["task_id"])
        scout = str(route["scout_artifact_id"])
        expert = str(route["expert_artifact_id"])
        if task_id not in task_ids:
            raise RegistryProtocolError(f"route_protocols 引用了不存在的 task：{task_id}")
        if scout not in model_ids:
            raise RegistryProtocolError(f"route_protocols 引用了不存在的 scout 模型：{scout}")
        if expert not in model_ids:
            raise RegistryProtocolError(f"route_protocols 引用了不存在的 expert 模型：{expert}")
        if str(model_by_id.loc[scout, "task_id"]) != task_id:
            raise RegistryProtocolError(f"{scout} 与 route task_id 不一致：{task_id}")
        if str(model_by_id.loc[expert, "task_id"]) != task_id:
            raise RegistryProtocolError(f"{expert} 与 route task_id 不一致：{task_id}")
        scout_roles = set(parse_list(str(model_by_id.loc[scout, "role_candidates"])))
        expert_roles = set(parse_list(str(model_by_id.loc[expert, "role_candidates"])))
        if "scout" not in scout_roles:
            raise RegistryProtocolError(f"{scout} 不支持作为 scout")
        if "expert" not in expert_roles:
            raise RegistryProtocolError(f"{expert} 不支持作为 expert")


def build_summary(
    tasks: pd.DataFrame,
    routes: pd.DataFrame,
    routing_by_task: dict[str, pd.DataFrame],
    risk_by_task: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    task_by_id = tasks.set_index("task_id", drop=False)
    rows: list[dict[str, Any]] = []
    for _, route in routes.iterrows():
        task = task_by_id.loc[str(route["task_id"])]
        filtered = filter_protocol_rows(routing_by_task[str(route["task_id"])], route)
        best = best_row_by_macro_f1(filtered)
        risk_frame = risk_by_task.get(str(route["task_id"]), pd.DataFrame())
        has_risk = str(task["risk_mode"]) == "dr_risk_events" and (
            has_dr_risk_columns(filtered) or not risk_frame.empty
        )
        has_cost = False
        if not filtered.empty and "cost_status" in filtered.columns:
            has_cost = any(
                value not in {"", "missing"}
                for value in filtered["cost_status"].astype(str).str.strip()
            )
        row = {
            "protocol_id": route["protocol_id"],
            "task_id": route["task_id"],
            "disease_family": task["disease_family"],
            "dataset_id": task["dataset_id"],
            "source_schema": task["source_schema"],
            "risk_source_mode": task["risk_source_mode"],
            "scout_artifact_id": route["scout_artifact_id"],
            "expert_artifact_id": route["expert_artifact_id"],
            "enabled": route.get("enabled", "true"),
            "n_routing_rows": len(filtered),
            "best_non_oracle_policy_by_macro_f1": "",
            "best_non_oracle_budget": "",
            "best_non_oracle_accuracy": "",
            "best_non_oracle_macro_f1": "",
            "best_non_oracle_cost_status": "",
            "best_non_oracle_estimated_forward_ms_per_image": "",
            "has_risk_events": str(bool(has_risk)).lower(),
            "has_cost_profile": str(bool(has_cost)).lower(),
            "notes": route.get("notes", ""),
        }
        if best is not None:
            row.update(
                {
                    "best_non_oracle_policy_by_macro_f1": best.get("policy", ""),
                    "best_non_oracle_budget": best.get("budget", ""),
                    "best_non_oracle_accuracy": best.get("accuracy", ""),
                    "best_non_oracle_macro_f1": best.get("macro_f1", ""),
                    "best_non_oracle_cost_status": best.get("cost_status", ""),
                    "best_non_oracle_estimated_forward_ms_per_image": best.get(
                        "estimated_forward_ms_per_image", ""
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest(output_dir: Path, protocol_id: str, files: list[Path]) -> pd.DataFrame:
    created = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        rows.append(
            {
                "protocol_id": protocol_id,
                "artifact_name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
                "created_at_utc": created,
            }
        )
    return pd.DataFrame(rows)


def write_html_report(
    path: Path,
    *,
    protocol_id: str,
    tasks: pd.DataFrame,
    models: pd.DataFrame,
    routes: pd.DataFrame,
    summary: pd.DataFrame,
    files: list[Path],
) -> None:
    def table(frame: pd.DataFrame, limit: int = 8) -> str:
        if frame.empty:
            return "<p>无记录。</p>"
        preview = frame.head(limit)
        headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in preview.columns)
        rows = []
        for _, row in preview.iterrows():
            rows.append(
                "<tr>"
                + "".join(html.escape(str(row.get(col, ""))) for col in preview.columns)
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                + "</tr>"
            )
        # Rebuild cells after escaping row values to avoid over-escaping tags.
        rows = []
        for _, row in preview.iterrows():
            cells = "".join(
                f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in preview.columns
            )
            rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    file_items = "".join(f"<li>{html.escape(str(file.name))}</li>" for file in files)
    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>v0.8.5 模型注册协议运行摘要</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 32px; color: #142033; }}
    h1, h2 {{ color: #10233f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dee8; padding: 7px 9px; text-align: left; }}
    th {{ background: #eef3f8; }}
    .notice {{ background: #fff8e8; border-left: 4px solid #b7791f; padding: 12px 14px; margin: 16px 0; }}
    .metric {{ display: inline-block; margin-right: 24px; padding: 10px 14px; background: #f4f8fb; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>v0.8.5 模型注册协议运行摘要</h1>
  <p>协议：<strong>{html.escape(protocol_id)}</strong></p>
  <p>当前版本定位：可插拔初筛模型（scout）/专家模型（expert）注册协议。</p>
  <div>
    <span class="metric">已注册任务：{len(tasks)}</span>
    <span class="metric">已注册模型：{len(models)}</span>
    <span class="metric">已启用路由协议：{len(routes)}</span>
  </div>
  <div class="notice">
    本版本是注册表级接入（registry-level plug-in）/产物级接入（artifact-level plug-in）。
    它不是适配器级接入（adapter-level plug-in），不会自动从 checkpoint 推理；
    也不是训练级接入（training-level plug-in），不会自动训练或微调模型。
    forward-only cost（仅前向传播计算成本）不是真实部署端到端延迟，也不替代临床判断。
  </div>
  <h2>输出文件</h2>
  <ul>{file_items}</ul>
  <h2>任务注册表</h2>
  {table(tasks)}
  <h2>模型注册表</h2>
  {table(models)}
  <h2>路由协议摘要</h2>
  {table(summary)}
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def load_all_registries(config_path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    config_dir = config_path.parent
    tasks = read_registry(
        resolve_path(config["task_registry"], config_dir=config_dir),
        {
            "task_id",
            "disease_family",
            "dataset_id",
            "label_space",
            "num_classes",
            "risk_mode",
            "source_schema",
            "risk_source_mode",
            "source_output_dir",
            "enabled",
            "notes",
        },
        name="task_registry.csv",
    )
    models = read_registry(
        resolve_path(config["model_registry"], config_dir=config_dir),
        {
            "artifact_id",
            "task_id",
            "model_family",
            "role_candidates",
            "prediction_source",
            "baseline_source",
            "cost_profile_id",
            "cost_source",
            "source_version",
            "enabled",
            "notes",
        },
        name="model_registry.csv",
    )
    routes = read_registry(
        resolve_path(config["route_protocols"], config_dir=config_dir),
        {
            "protocol_id",
            "task_id",
            "scout_artifact_id",
            "expert_artifact_id",
            "routing_policies",
            "budgets",
            "enabled",
            "notes",
        },
        name="route_protocols.csv",
    )
    costs = read_registry(
        resolve_path(config["cost_registry"], config_dir=config_dir),
        {
            "artifact_id",
            "cost_profile_id",
            "cost_scope",
            "cost_source",
            "device",
            "precision",
            "batch_size",
            "enabled",
            "notes",
        },
        name="cost_registry.csv",
    )
    return tasks, models, routes, costs


def run_protocol(
    config_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    dry_run: bool = False,
) -> ProtocolResult:
    config_path = Path(config_path)
    config = load_yaml_or_json(config_path)
    protocol_id = str(config.get("protocol_id", "")).strip()
    if not protocol_id:
        raise RegistryProtocolError("protocol.yaml 缺少 protocol_id")
    if output_dir is None:
        if "output_dir" not in config:
            raise RegistryProtocolError("protocol.yaml 缺少 output_dir")
        output_path = resolve_path(config["output_dir"], config_dir=config_path.parent)
    else:
        output_path = Path(output_dir)

    tasks, models, routes, costs = load_all_registries(config_path, config)
    validate_registries(tasks, models, routes, costs)
    if dry_run:
        return ProtocolResult(output_dir=output_path, files=[])

    baseline_frames: list[pd.DataFrame] = []
    routing_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    case_frames: list[pd.DataFrame] = []
    routing_by_task: dict[str, pd.DataFrame] = {}
    risk_by_task: dict[str, pd.DataFrame] = {}

    for _, task in tasks.iterrows():
        baseline = read_source_table(task, "model_baselines.csv")
        routing = read_source_table(task, "routing_results.csv")
        risk = read_source_table(task, "risk_results.csv")
        case_audit = read_source_table(task, "case_audit.csv")
        source_protocol = infer_source_protocol(routing, str(task["source_output_dir"]))
        baseline_frames.append(add_task_context(baseline, task, source_protocol))
        routing_with_context = add_task_context(routing, task, source_protocol)
        routing_frames.append(routing_with_context)
        if not risk.empty:
            risk_copy = risk.copy()
            risk_copy["risk_source_mode"] = str(task["risk_source_mode"])
            risk_copy["source_schema"] = str(task["source_schema"])
            risk_frames.append(add_task_context(risk_copy, task, source_protocol))
        elif str(task["risk_source_mode"]) == "embedded_in_routing":
            embedded = extract_embedded_risk_results(routing, task, source_protocol)
            if not embedded.empty:
                risk_frames.append(embedded)
        if not case_audit.empty:
            case_frames.append(add_task_context(case_audit, task, source_protocol))
        routing_by_task[str(task["task_id"])] = routing
        risk_by_task[str(task["task_id"])] = risk

    registered_tasks = tasks.copy()
    registered_models = models.copy()
    summary = build_summary(tasks, routes, routing_by_task, risk_by_task)
    model_baselines = concat_schema(
        baseline_frames,
        ["task_id", "disease_family", "dataset_id", "source_output_dir", "source_protocol"],
    )
    routing_results = concat_schema(
        routing_frames,
        ["task_id", "disease_family", "dataset_id", "source_output_dir", "source_protocol"],
    )
    risk_results = concat_schema(
        risk_frames,
        ["task_id", "disease_family", "dataset_id", "source_output_dir", "source_protocol"],
    )
    case_audit = concat_schema(
        case_frames,
        ["task_id", "disease_family", "dataset_id", "source_output_dir", "source_protocol"],
    )

    output_path.mkdir(parents=True, exist_ok=True)
    files = [
        output_path / "registered_tasks.csv",
        output_path / "registered_models.csv",
        output_path / "route_protocol_summary.csv",
        output_path / "model_baselines_all.csv",
        output_path / "routing_results_all.csv",
        output_path / "risk_results_all.csv",
        output_path / "case_audit_all.csv",
    ]
    write_csv(files[0], registered_tasks)
    write_csv(files[1], registered_models)
    write_csv(files[2], summary)
    write_csv(files[3], model_baselines)
    write_csv(files[4], routing_results)
    write_csv(files[5], risk_results)
    write_csv(files[6], case_audit)

    report_name = str(config.get("report", "summary.html"))
    report_path = output_path / report_name
    write_html_report(
        report_path,
        protocol_id=protocol_id,
        tasks=registered_tasks,
        models=registered_models,
        routes=routes,
        summary=summary,
        files=files,
    )
    files.append(report_path)
    manifest_path = output_path / "artifact_manifest.csv"
    write_csv(manifest_path, build_manifest(output_path, protocol_id, files))
    files.append(manifest_path)
    return ProtocolResult(output_dir=output_path, files=files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_protocol(args.config, output_dir=args.output_dir, dry_run=args.dry_run)
    except RegistryProtocolError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"[DRY-RUN] v0.8.5 registry protocol valid: {args.config}")
    else:
        print(f"[DONE] wrote v0.8.5 registry protocol outputs: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
