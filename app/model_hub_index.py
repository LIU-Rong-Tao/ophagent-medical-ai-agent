"""Read-only projection of registered Model Hub assets and experiment evidence."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from app.checkpoints import discover_model_artifacts, select_preferred_artifacts
from app.model_providers import OphBenchProvider


TASK_LABELS = {
    "aptos_dr_5class": "APTOS DR 五级分级",
    "deepdrid_dr_5class_external": "DR 五级分级",
    "deepdrid_dr_5class_native": "DR 五级分级",
    "glaucoma_3class": "青光眼三分类",
    "glaucoma_binary": "RIM-ONE 青光眼二分类",
    "trhd59_observed_label": "TRHD59 59 类观测标签",
}

DATASET_LABELS = {
    "aptos_dr_5class": "APTOS2019",
    "deepdrid_dr_5class_external": "DeepDRiD",
    "deepdrid_dr_5class_native": "DeepDRiD",
    "glaucoma_3class": "Glaucoma_fundus",
    "glaucoma_binary": "RIM-ONE DL",
    "trhd59_observed_label": "TRHD59",
}

EXPERIMENT_LABELS = {
    "aptos_dr_5class": "原生任务适配",
    "deepdrid_dr_5class_external": "APTOS 模型冻结迁移",
    "deepdrid_dr_5class_native": "DeepDRiD 原生任务适配",
    "glaucoma_3class": "清理口径",
    "glaucoma_binary": "原生任务适配",
    "trhd59_observed_label": "canonical 观测标签",
}

DATASET_IDS = {
    "aptos_dr_5class": "aptos2019",
    "deepdrid_dr_5class_external": "deepdrid",
    "deepdrid_dr_5class_native": "deepdrid",
    "glaucoma_3class": "glaucoma_fundus",
    "glaucoma_binary": "rim_one_dl",
    "trhd59_observed_label": "trhd59_canonical",
}

TASK_ADAPTATION_TYPES = {
    "aptos_dr_5class": "task_native",
    "deepdrid_dr_5class_external": "frozen_external_transfer",
    "deepdrid_dr_5class_native": "task_native_adaptation",
    "glaucoma_3class": "task_native",
    "glaucoma_binary": "task_native_adaptation",
    "trhd59_observed_label": "exploratory_weak_label",
}

TASK_RISK_CLASS_IDS = {
    "aptos_dr_5class": [3, 4],
    "deepdrid_dr_5class_external": [3, 4],
    "deepdrid_dr_5class_native": [3, 4],
    "glaucoma_3class": [1, 2],
    "glaucoma_binary": [1],
    "trhd59_observed_label": [],
}

JOB_GROUP_LABELS = {
    "asset_smoke_jobs": "模型资产 Smoke",
    "global_scan_jobs": "旧版全局候选扫描",
    "training_jobs": "训练适配",
    "aptos_frozen_encoder_jobs": "APTOS 冻结编码器适配",
    "exploratory_test_jobs": "APTOS 探索性全池扫描",
    "inference_jobs": "DeepDRiD 批量推理",
    "trhd59_observed_label_jobs": "TRHD59 首次协议任务",
    "trhd59_observed_label_v2_jobs": "TRHD59 正式任务",
}

CURRENT_PROTOCOL_PATTERNS = (
    "experiments/opening_risk_routing_closure/configs/protocols/*protocol*.json",
    "experiments/model_hub/tasks/**/*protocol*.json",
    "experiments/model_hub/tasks/**/*locked_test*.yaml",
)


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value or "").strip()).expanduser()
    return path if path.is_absolute() else project_root / path


def _column(
    frame: pd.DataFrame,
    name: str,
    default: object,
) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _safe_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_label(task_id: str) -> str:
    return TASK_LABELS.get(task_id, task_id)


def _dataset_label(task_id: str) -> str:
    return DATASET_LABELS.get(task_id, task_id)


def _latest_smoke_records(project_root: Path) -> dict[str, dict[str, Any]]:
    root = project_root / "experiments/model_hub/runtime/asset_smoke_jobs"
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    if not root.exists():
        return {}
    for path in root.glob("*/checkpoints/*.json"):
        payload = _safe_json(path)
        checkpoint_id = str(payload.get("checkpoint_id") or path.stem)
        timestamp = path.stat().st_mtime_ns
        if checkpoint_id not in latest or timestamp > latest[checkpoint_id][0]:
            latest[checkpoint_id] = (timestamp, payload)
    return {checkpoint_id: payload for checkpoint_id, (_, payload) in latest.items()}


def build_checkpoint_index(project_root: Path) -> pd.DataFrame:
    """Return all OphBench checkpoints with the latest local Smoke evidence."""

    provider = OphBenchProvider()
    records = provider.list_models()
    smoke = _latest_smoke_records(project_root)
    rows: list[dict[str, Any]] = []
    for record in records:
        row = asdict(record)
        checkpoint_id = str(record.source_checkpoint_id or "")
        evidence = smoke.get(checkpoint_id, {})
        smoke_status = str(evidence.get("status") or "not_smoked")
        row.update(
            {
                "checkpoint_id": checkpoint_id,
                "model_id": record.source_model_id,
                "model_name": record.model_name or record.display_name,
                "smoke_status": smoke_status,
                "asset_probe_passed": bool(evidence.get("asset_probe_passed", False)),
                "runtime_smoke_passed": bool(
                    evidence.get("runtime_smoke_passed", False)
                ),
                "smoke_blocker": str(
                    evidence.get("error_detail")
                    or evidence.get("details", {}).get("failure_summary")
                    or ""
                ),
                "smoke_completed_at": str(evidence.get("completed_at_utc") or ""),
                "modalities_text": " / ".join(record.modalities),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_registries(project_root: Path) -> list[Path]:
    roots = [
        project_root / "experiments",
        project_root / "experiments/model_hub/runs",
    ]
    paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*prediction_assets.csv"):
            key = str(path.resolve())
            paths[key] = path
    return sorted(paths.values(), key=lambda value: _relative(project_root, value))


def build_task_asset_index(project_root: Path) -> pd.DataFrame:
    """Load every formal task prediction registry without copying predictions."""

    frames: list[pd.DataFrame] = []
    required = {
        "task_id",
        "artifact_id",
        "validation_prediction_path",
        "test_prediction_path",
    }
    for path in _prediction_registries(project_root):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            continue
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["registry_path"] = _relative(project_root, path)
        frame["dataset_label"] = frame["task_id"].astype(str).map(_dataset_label)
        frame["task_label"] = frame["task_id"].astype(str).map(_task_label)
        frame["dataset_id"] = frame["task_id"].astype(str).map(DATASET_IDS)
        frame["experiment_label"] = frame["task_id"].astype(str).map(EXPERIMENT_LABELS)
        frame["validation_asset_exists"] = frame["validation_prediction_path"].map(
            lambda value: _resolve(project_root, value).is_file()
        )
        frame["test_asset_exists"] = frame["test_prediction_path"].map(
            lambda value: _resolve(project_root, value).is_file()
        )
        frame["offline_evaluation_complete"] = (
            frame["validation_asset_exists"] & frame["test_asset_exists"]
        )
        frame["offline_batch_inference_ready"] = (
            _column(frame, "current_run_reproducible", False)
            .fillna(False)
            .astype(bool)
            & _column(frame, "checkpoint_path", "").map(
                lambda value: _resolve(project_root, value).is_file()
            )
        )
        frame["online_case_inference_ready"] = (
            _column(frame, "task_inference_ready", False).fillna(False).astype(bool)
        )
        frame["route_eligible"] = (
            _column(frame, "route_eligible", False).fillna(False).astype(bool)
        )
        forward_cost = pd.to_numeric(
            _column(frame, "forward_cost_ms_per_image", float("nan")),
            errors="coerce",
        )
        frame["forward_only_cost_available"] = forward_cost.notna() & forward_cost.ge(0)
        frame["total_cost_evidence_status"] = _column(
            frame, "cost_status", "unmeasured"
        ).fillna("unmeasured")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _route_result_paths(project_root: Path) -> list[Path]:
    roots = [
        project_root / "experiments",
        project_root / "experiments/model_hub/runs",
    ]
    paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("pairing_results.csv"):
            paths[str(path.resolve())] = path
    return sorted(paths.values(), key=lambda value: _relative(project_root, value))


def _route_stage(relative_path: str, config: dict[str, Any]) -> str:
    text = relative_path.lower()
    design = str(config.get("evaluation_design", "")).lower()
    if "exploratory" in text or "exploratory" in design:
        return "探索性全池扫描"
    if "test_locked" in text or "locked_test" in text or "frozen" in text:
        return "冻结结果集评估"
    if "external_transfer" in text:
        return "冻结迁移评估"
    if "validation" in text:
        return "Validation 候选扫描"
    return "历史受控评测"


def build_route_run_index(project_root: Path) -> pd.DataFrame:
    """Index existing route result packages; metrics remain in their source CSV."""

    rows: list[dict[str, Any]] = []
    for path in _route_result_paths(project_root):
        output_dir = path.parent
        relative_path = _relative(project_root, output_dir)
        try:
            pairings = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            continue
        if pairings.empty:
            continue
        config_path = output_dir / "run_config.yaml"
        config = _safe_yaml(config_path)
        snapshot_path = output_dir / "model_hub_snapshot.csv"
        model_count = 0
        if snapshot_path.is_file():
            snapshot = pd.read_csv(snapshot_path, usecols=lambda column: column in {"artifact_id"})
            model_count = int(snapshot["artifact_id"].nunique()) if not snapshot.empty else 0
        task_id = str(
            pairings["task_id"].dropna().astype(str).iloc[0]
            if "task_id" in pairings and pairings["task_id"].notna().any()
            else ""
        )
        status = (
            "completed"
            if (output_dir / "artifact_manifest.csv").is_file()
            else "incomplete"
        )
        rows.append(
            {
                "run_id": relative_path.replace("/", "::"),
                "task_id": task_id,
                "task_label": _task_label(task_id),
                "dataset_label": _dataset_label(task_id),
                "stage": _route_stage(relative_path, config),
                "evaluation_design": str(config.get("evaluation_design") or ""),
                "protocol_version": str(
                    config.get("protocol_version")
                    or config.get("protocol_id")
                    or ""
                ),
                "protocol_sha256": str(config.get("protocol_sha256") or ""),
                "model_count": model_count,
                "result_rows": len(pairings),
                "pairing_count": int(
                    pairings["pairing_id"].nunique()
                    if "pairing_id" in pairings
                    else len(pairings)
                ),
                "route_eligible": bool(config.get("route_eligible", False)),
                "status": status,
                "result_path": relative_path,
                "_pairing_results_path": str(path),
                "_snapshot_path": str(snapshot_path),
            }
        )
    return pd.DataFrame(rows)


def build_protocol_index(project_root: Path) -> pd.DataFrame:
    paths: dict[str, Path] = {}
    for pattern in CURRENT_PROTOCOL_PATTERNS:
        for path in project_root.glob(pattern):
            paths[str(path.resolve())] = path
    rows: list[dict[str, Any]] = []
    for path in sorted(paths.values(), key=lambda value: _relative(project_root, value)):
        payload = _safe_json(path) if path.suffix == ".json" else _safe_yaml(path)
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            pool = payload.get("model_pool", {})
            if isinstance(pool, dict):
                task_id = str(pool.get("task_id") or "")
        rows.append(
            {
                "protocol_name": path.stem,
                "task_id": task_id,
                "task_label": _task_label(task_id) if task_id else "跨任务/确认性协议",
                "protocol_id": str(
                    payload.get("protocol_id")
                    or payload.get("protocol_version")
                    or path.stem
                ),
                "selection_split": str(payload.get("selection_split") or ""),
                "test_split": str(payload.get("test_split") or ""),
                "route_eligible": bool(payload.get("route_eligible", False)),
                "path": _relative(project_root, path),
            }
        )
    return pd.DataFrame(rows)


def _job_status(path: Path) -> tuple[str, dict[str, Any]]:
    fallback: dict[str, Any] = {}
    for name in ("status.json", "progress.json", "job.json", "summary.json"):
        payload = _safe_json(path / name)
        if not payload:
            continue
        fallback = payload
        status = str(payload.get("status") or payload.get("job_status") or "")
        if status:
            return status, payload
    return "unknown", fallback


def build_job_index(project_root: Path) -> pd.DataFrame:
    root = project_root / "experiments/model_hub/runtime"
    rows: list[dict[str, Any]] = []
    for group, label in JOB_GROUP_LABELS.items():
        group_root = root / group
        if not group_root.exists():
            continue
        for path in sorted(group_root.iterdir()):
            if not path.is_dir():
                continue
            status, payload = _job_status(path)
            progress = payload.get("progress", payload.get("completed", ""))
            rows.append(
                {
                    "job_id": path.name,
                    "job_type": label,
                    "job_group": group,
                    "status": status,
                    "progress": progress,
                    "created_at": str(
                        payload.get("created_at")
                        or payload.get("created_at_utc")
                        or payload.get("started_at")
                        or ""
                    ),
                    "updated_at": str(
                        payload.get("updated_at")
                        or payload.get("completed_at")
                        or payload.get("completed_at_utc")
                        or ""
                    ),
                    "path": _relative(project_root, path),
                }
            )
    return pd.DataFrame(rows)


def build_online_endpoint_index(project_root: Path) -> pd.DataFrame:
    """List existing single-case raw-image endpoints without loading a model."""

    rows: list[dict[str, Any]] = []
    artifacts = select_preferred_artifacts(discover_model_artifacts(project_root))
    for key, artifact in artifacts.items():
        if not artifact.can_attempt_load:
            continue
        rows.append(
            {
                "endpoint_id": key,
                "task_id": "aptos_dr_5class",
                "task_label": _task_label("aptos_dr_5class"),
                "display_name": artifact.display_name,
                "checkpoint_exists": bool(
                    artifact.checkpoint_path is not None
                    and artifact.checkpoint_path.is_file()
                ),
                "qualification": "online_case_inference_ready",
                "route_eligible": False,
            }
        )
    return pd.DataFrame(rows)


def build_dataset_index(
    project_root: Path,
    task_assets: pd.DataFrame,
    route_runs: pd.DataFrame,
    online_endpoints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not task_assets.empty:
        for (task_id, dataset_label), assets in task_assets.groupby(
            ["task_id", "dataset_label"], dropna=False
        ):
            matching_routes = (
                route_runs.loc[
                    route_runs["task_id"].astype(str).eq(str(task_id))
                ]
                if "task_id" in route_runs.columns
                else pd.DataFrame()
            )
            rows.append(
                {
                    "task_id": str(task_id),
                    "task_label": _task_label(str(task_id)),
                    "dataset_id": DATASET_IDS.get(str(task_id), str(task_id)),
                    "dataset_label": str(dataset_label),
                    "experiment_label": EXPERIMENT_LABELS.get(
                        str(task_id), "任务评测"
                    ),
                    "admission_status": "实验资产已登记",
                    "prediction_assets": int(assets["artifact_id"].nunique()),
                    "validation_assets": int(assets["validation_asset_exists"].sum()),
                    "test_assets": int(assets["test_asset_exists"].sum()),
                    "route_runs": len(matching_routes),
                    "frozen_route_runs": int(
                        matching_routes["stage"]
                        .astype(str)
                        .str.contains("冻结结果集")
                        .sum()
                        if "stage" in matching_routes.columns
                        else 0
                    ),
                    "route_eligible": bool(assets["route_eligible"].any()),
                    "online_case_endpoints": int(
                        online_endpoints["task_id"].astype(str).eq(str(task_id)).sum()
                        if online_endpoints is not None
                        and "task_id" in online_endpoints.columns
                        else 0
                    ),
                    "qualification_note": "离线任务评测证据；不自动授予在线推理或正式路由资格",
                }
            )

    coverage_path = (
        project_root
        / "experiments/opening_risk_routing_closure/model_hub_coverage_matrix.csv"
    )
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path).fillna("")
        refuge = coverage.loc[
            coverage["dataset_id"].astype(str).eq("REFUGE")
            & coverage["current_qualification"]
            .astype(str)
            .eq("admitted_for_labeled_train_validation_only")
        ]
        if not refuge.empty:
            rows.append(
                {
                    "task_id": "glaucoma_binary",
                    "task_label": "REFUGE 青光眼二分类",
                    "dataset_id": "refuge",
                    "dataset_label": "REFUGE",
                    "experiment_label": "数据准入",
                    "admission_status": "仅准入有标签 train/validation",
                    "prediction_assets": 0,
                    "validation_assets": 0,
                    "test_assets": 0,
                    "route_runs": 0,
                    "frozen_route_runs": 0,
                    "route_eligible": False,
                    "online_case_endpoints": 0,
                    "qualification_note": (
                        "third_party_mirror；官方 onsite Test 无公开诊断标签，"
                        "不得用于排行榜复现"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _route_participants(route_runs: pd.DataFrame) -> set[tuple[str, str]]:
    participants: set[tuple[str, str]] = set()
    if route_runs.empty:
        return participants
    for _, run in route_runs.iterrows():
        task_id = str(run.get("task_id") or "")
        snapshot_path = Path(str(run.get("_snapshot_path") or ""))
        if snapshot_path.is_file():
            try:
                snapshot = pd.read_csv(
                    snapshot_path,
                    usecols=lambda column: column == "artifact_id",
                )
            except (OSError, pd.errors.ParserError, UnicodeError):
                snapshot = pd.DataFrame()
            if "artifact_id" in snapshot:
                participants.update(
                    (task_id, str(artifact_id))
                    for artifact_id in snapshot["artifact_id"].dropna().unique()
                )
                continue
        results_path = Path(str(run.get("_pairing_results_path") or ""))
        if not results_path.is_file():
            continue
        try:
            pairings = pd.read_csv(
                results_path,
                usecols=lambda column: column
                in {
                    "scout_artifact_ids",
                    "expert_artifact_id",
                    "scout_ids",
                    "active_expert_ids",
                },
            )
        except (OSError, pd.errors.ParserError, UnicodeError):
            continue
        for column in pairings.columns:
            for value in pairings[column].dropna().astype(str):
                participants.update(
                    (task_id, artifact_id)
                    for artifact_id in value.split("|")
                    if artifact_id
                )
    return participants


def build_model_hub_index(project_root: Path) -> dict[str, pd.DataFrame]:
    """Build one in-memory projection over existing registries and artifacts."""

    checkpoints = build_checkpoint_index(project_root)
    task_assets = build_task_asset_index(project_root)
    route_runs = build_route_run_index(project_root)
    participants = _route_participants(route_runs)
    if not task_assets.empty:
        task_assets["research_route_participated"] = task_assets.apply(
            lambda row: (str(row["task_id"]), str(row["artifact_id"])) in participants,
            axis=1,
        )
        task_assets["research_route_status"] = task_assets.apply(
            lambda row: (
                "已参与历史研究路由"
                if bool(row["research_route_participated"])
                else (
                    "可参与 Validation 研究选择"
                    if bool(row.get("validation_selection_eligible", False))
                    else "仅任务评测"
                )
            ),
            axis=1,
        )
        task_assets["formal_route_status"] = task_assets["route_eligible"].map(
            lambda eligible: "已授予" if bool(eligible) else "未授予（研究评测）"
        )
    protocols = build_protocol_index(project_root)
    jobs = build_job_index(project_root)
    online_endpoints = build_online_endpoint_index(project_root)
    datasets = build_dataset_index(
        project_root,
        task_assets,
        route_runs,
        online_endpoints,
    )
    task_profiles = build_task_profile_index(project_root)
    model_capabilities = build_model_capability_index(
        task_assets,
        online_endpoints,
    )
    return {
        "checkpoints": checkpoints,
        "task_assets": task_assets,
        "route_runs": route_runs,
        "protocols": protocols,
        "jobs": jobs,
        "datasets": datasets,
        "online_endpoints": online_endpoints,
        "task_profiles": task_profiles,
        "model_capabilities": model_capabilities,
    }


def build_task_profile_index(project_root: Path) -> pd.DataFrame:
    """Project versioned Task Profiles from the qualification contract."""

    contract_dir = (
        project_root
        / "experiments/opening_risk_routing_closure/configs/protocols"
    )
    contract_path = contract_dir / "route_qualification_contract_v1_1.json"
    if not contract_path.is_file():
        contract_path = contract_dir / "route_qualification_contract_v1.json"
    contract = _safe_json(contract_path)
    policies = contract.get("task_policies", {})
    rows: list[dict[str, Any]] = []
    for task_id, policy_value in policies.items():
        policy = dict(policy_value)
        class_order = policy.get("class_order", [])
        if isinstance(class_order, str) and ".." in class_order:
            start, end = class_order.split("..", maxsplit=1)
            class_order = list(range(int(start), int(end) + 1))
        labels = policy.get("class_labels")
        if not isinstance(labels, list):
            labels = [str(value) for value in class_order]
        rows.append(
            {
                "task_id": task_id,
                "dataset_id": DATASET_IDS.get(task_id, task_id),
                "modality": str(policy.get("modality", "CFP")),
                "label_space": json.dumps(labels, ensure_ascii=False),
                "primary_metric": str(policy.get("primary_metric", "")),
                "risk_semantics": str(policy.get("proxy_semantics", "")),
                "risk_positive_class_ids": json.dumps(
                    TASK_RISK_CLASS_IDS.get(task_id, []),
                    ensure_ascii=False,
                ),
                "report_label": TASK_LABELS.get(task_id, task_id),
                "adaptation_type": TASK_ADAPTATION_TYPES.get(
                    task_id,
                    "task_adapter_required",
                ),
                "task_spec_version": "ophagent.task_spec.v1",
                "source_contract": _relative(project_root, contract_path),
            }
        )
    return pd.DataFrame(rows)


def _cost_protocol_id(row: pd.Series) -> str:
    registered = str(row.get("cost_protocol_id", "") or "").strip()
    if registered not in {"", "nan", "None"}:
        return registered
    scope = str(row.get("cost_scope", "")).lower()
    status = str(row.get("cost_status", "")).lower()
    if status == "measured" and "h100" in scope and "forward" in scope:
        if "batch32" in scope:
            return "h100_fp32_forward_only_batch32_split5_v1"
        if "batch16" in scope:
            return "h100_fp32_forward_only_batch1_batch16_w10_r30_v1"
    return "cost_protocol_unavailable"


def build_model_capability_index(
    task_assets: pd.DataFrame,
    online_endpoints: pd.DataFrame,
) -> pd.DataFrame:
    """Project model capabilities without granting route qualification."""

    if task_assets.empty:
        return pd.DataFrame()
    online_pairs = set()
    if (
        not online_endpoints.empty
        and {"task_id", "artifact_id"}.issubset(online_endpoints.columns)
    ):
        online_pairs = set(
            zip(
                online_endpoints["task_id"].astype(str),
                online_endpoints["artifact_id"].astype(str),
                strict=False,
            )
        )
    rows: list[dict[str, Any]] = []
    for _, row in task_assets.iterrows():
        task_id = str(row.get("task_id", ""))
        artifact_id = str(row.get("artifact_id", ""))
        prediction_available = any(
            str(row.get(column, "") or "").strip()
            not in {"", "nan", "None"}
            for column in (
                "validation_prediction_path",
                "test_prediction_path",
            )
        )
        reproducible = bool(row.get("current_run_reproducible", False))
        adapter_type = str(row.get("adapter_type", ""))
        batch_ready = reproducible and adapter_type in {
            "timm_classifier",
            "retfound_classifier",
            "preti_classifier",
        }
        online_ready = (task_id, artifact_id) in online_pairs or bool(
            row.get("online_case_inference_ready", False)
        )
        cost = pd.to_numeric(
            pd.Series([row.get("forward_cost_ms_per_image")]),
            errors="coerce",
        ).iloc[0]
        rows.append(
            {
                "task_id": task_id,
                "artifact_id": artifact_id,
                "adapter_type": adapter_type,
                "prediction_asset_available": prediction_available,
                "offline_batch_inference_ready": batch_ready,
                "online_case_inference_ready": online_ready,
                "cost_protocol_id": _cost_protocol_id(row),
                "cost_ms_per_image": (
                    float(cost) if pd.notna(cost) else None
                ),
                "qualification_status": str(
                    row.get("qualification_status", "")
                ),
                "model_capability_version": (
                    "ophagent.model_capability.v1"
                ),
            }
        )
    return pd.DataFrame(rows)
