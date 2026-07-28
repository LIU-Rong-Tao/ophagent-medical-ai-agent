"""Deterministic qualification gate for Model Hub research routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml


CONTRACT_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "route_qualification_contract_v1.json"
)
OUTPUT_RELATIVE_DIR = (
    "experiments/opening_risk_routing_closure/outputs/route_qualification_gate_v1"
)

EXECUTION_LEVELS = (
    "blocked",
    "research_replay_only",
    "research_case_simulation",
    "deployment_candidate",
    "clinical_route_eligible",
)
EVIDENCE_LABELS = (
    "beneficial",
    "risk_tradeoff",
    "ineffective",
    "unstable",
)

GATE_MESSAGES = {
    "RQ_OK": "路由通过当前研究场景门禁",
    "RQ_SAME_SCOUT_EXPERT": "Scout 与 Expert 不能是同一模型",
    "RQ_TASK_MISMATCH": "模型资产与路由任务不一致",
    "RQ_LABEL_SPACE_MISMATCH": "标签空间与任务契约不一致",
    "RQ_CLASS_ORDER_MISMATCH": "类别顺序与任务契约不一致",
    "RQ_PREDICTION_ASSET_INVALID": "Prediction Asset 缺失或未通过结构检查",
    "RQ_PROTOCOL_NOT_FROZEN": "路由协议未冻结或结果状态不完整",
    "RQ_SELECTION_NOT_VALIDATION": "候选不是仅由 Validation 选择",
    "RQ_VALIDATION_EVIDENCE_MISSING": "缺少可还原的 Validation 路由证据",
    "RQ_PRIMARY_METRIC_UNAVAILABLE": "任务主指标缺失，仅能使用登记的次指标作限制性分析",
    "RQ_STABILITY_EVIDENCE_MISSING": "缺少 bootstrap 或稳定性证据",
    "RQ_VALIDATION_INEFFECTIVE": "Validation 未显示相对 Scout 的主指标增量",
    "RQ_PROXY_RISK_TRADEOFF": "路由引入了标签依赖代理事件",
    "RQ_FROZEN_REVERSAL": "冻结结果相对 Validation 发生主指标或代理净收益反转",
    "RQ_COST_EVIDENCE_INCOMPLETE": "统一 H100 forward-only 成本证据不完整",
    "RQ_ONLINE_ENTRY_UNAVAILABLE": "所需模型没有完整的单病例原图推理入口",
    "RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED": "当前项目禁止自动授予临床路由资格",
}

HARD_BLOCK_CODES = {
    "RQ_SAME_SCOUT_EXPERT",
    "RQ_TASK_MISMATCH",
    "RQ_LABEL_SPACE_MISMATCH",
    "RQ_CLASS_ORDER_MISMATCH",
    "RQ_PREDICTION_ASSET_INVALID",
}


@dataclass(frozen=True)
class RouteQualificationRequest:
    task_id: str
    pairing_id: str
    scout_artifact_ids: tuple[str, ...]
    expert_artifact_id: str
    request_scope: str = "cached_prediction_replay"
    task_matches: bool = True
    label_space_matches: bool = True
    class_order_matches: bool = True
    prediction_assets_valid: bool = True
    cost_protocol_complete: bool = False
    all_models_online_case_ready: bool = False
    protocol_frozen: bool = False
    selection_split: str = ""
    validation_main_metric_delta: float | None = None
    validation_delta_vs_best_single: float | None = None
    validation_corrected: float | None = None
    validation_introduced: float | None = None
    validation_net: float | None = None
    stability_ci_lower: float | None = None
    frozen_main_metric_delta: float | None = None
    frozen_corrected: float | None = None
    frozen_introduced: float | None = None
    frozen_net: float | None = None
    primary_metric_available: bool = True
    requested_budget: float | None = None
    expected_cost_ms_per_image: float | None = None
    protocol_sha256: str = ""
    input_asset_fingerprint: str = ""


@dataclass(frozen=True)
class RouteQualificationDecision:
    execution_level: str
    evidence_label: str
    allow_cached_replay: bool
    allow_case_simulation: bool
    allow_new_case_route: bool
    clinical_route_eligible: bool
    human_confirmation_required: bool
    error_codes: tuple[str, ...]
    gate_trace: tuple[dict[str, Any], ...]
    contract_sha256: str
    evidence_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def route_qualification_decision_from_dict(
    payload: dict[str, Any],
) -> RouteQualificationDecision:
    """Restore a JSON-safe decision without recalculating its evidence."""

    return RouteQualificationDecision(
        execution_level=str(payload["execution_level"]),
        evidence_label=str(payload["evidence_label"]),
        allow_cached_replay=bool(payload["allow_cached_replay"]),
        allow_case_simulation=bool(payload["allow_case_simulation"]),
        allow_new_case_route=bool(payload["allow_new_case_route"]),
        clinical_route_eligible=bool(payload["clinical_route_eligible"]),
        human_confirmation_required=bool(payload["human_confirmation_required"]),
        error_codes=tuple(str(value) for value in payload["error_codes"]),
        gate_trace=tuple(dict(value) for value in payload["gate_trace"]),
        contract_sha256=str(payload["contract_sha256"]),
        evidence_fingerprint=str(payload["evidence_fingerprint"]),
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=256)
def _file_sha256(path_text: str, modified_ns: int, size: int) -> str:
    del modified_ns, size
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256(str(path), stat.st_mtime_ns, stat.st_size)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_route_qualification_contract(
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    path = project_root / CONTRACT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, file_sha256(path)


def evaluate_route_qualification(
    request: RouteQualificationRequest,
    *,
    contract: dict[str, Any],
    contract_sha256: str | None = None,
) -> RouteQualificationDecision:
    """Apply asset, research-evidence, and execution gates in that order."""

    codes: list[str] = []
    trace: list[dict[str, Any]] = []
    scouts = tuple(value for value in request.scout_artifact_ids if value)

    asset_checks = {
        "scout_expert_distinct": request.expert_artifact_id not in scouts,
        "task_matches": request.task_matches,
        "label_space_matches": request.label_space_matches,
        "class_order_matches": request.class_order_matches,
        "prediction_assets_valid": request.prediction_assets_valid,
    }
    asset_codes = {
        "scout_expert_distinct": "RQ_SAME_SCOUT_EXPERT",
        "task_matches": "RQ_TASK_MISMATCH",
        "label_space_matches": "RQ_LABEL_SPACE_MISMATCH",
        "class_order_matches": "RQ_CLASS_ORDER_MISMATCH",
        "prediction_assets_valid": "RQ_PREDICTION_ASSET_INVALID",
    }
    for name, passed in asset_checks.items():
        trace.append({"layer": "asset_task", "check": name, "passed": passed})
        if not passed:
            codes.append(asset_codes[name])

    validation_delta = request.validation_main_metric_delta
    corrected = request.validation_corrected or 0.0
    introduced = request.validation_introduced or 0.0
    net = request.validation_net
    if validation_delta is None:
        evidence_label = "unstable"
        codes.append("RQ_VALIDATION_EVIDENCE_MISSING")
    elif validation_delta <= 0:
        evidence_label = "ineffective"
        codes.append("RQ_VALIDATION_INEFFECTIVE")
    elif introduced > 0 or (net is not None and net < 0):
        evidence_label = "risk_tradeoff"
        codes.append("RQ_PROXY_RISK_TRADEOFF")
    else:
        evidence_label = "beneficial"

    if not request.protocol_frozen:
        codes.append("RQ_PROTOCOL_NOT_FROZEN")
    if request.selection_split not in {"validation", "val"}:
        codes.append("RQ_SELECTION_NOT_VALIDATION")
    if not request.primary_metric_available:
        codes.append("RQ_PRIMARY_METRIC_UNAVAILABLE")
    if request.stability_ci_lower is None:
        codes.append("RQ_STABILITY_EVIDENCE_MISSING")
    elif request.stability_ci_lower <= 0 and evidence_label == "beneficial":
        evidence_label = "unstable"

    frozen_metric_reversal = (
        validation_delta is not None
        and validation_delta > 0
        and request.frozen_main_metric_delta is not None
        and request.frozen_main_metric_delta <= 0
    )
    frozen_proxy_reversal = (
        net is not None
        and net >= 0
        and request.frozen_net is not None
        and request.frozen_net < 0
    )
    if frozen_metric_reversal or frozen_proxy_reversal:
        evidence_label = "unstable"
        codes.append("RQ_FROZEN_REVERSAL")

    trace.append(
        {
            "layer": "research_evidence",
            "check": "validation_and_frozen_evidence",
            "passed": evidence_label in {"beneficial", "risk_tradeoff"},
            "evidence_label": evidence_label,
            "validation_delta": validation_delta,
            "corrected": corrected,
            "introduced": introduced,
            "net": net,
            "frozen_metric_reversal": frozen_metric_reversal,
            "frozen_proxy_reversal": frozen_proxy_reversal,
        }
    )

    if not request.cost_protocol_complete:
        codes.append("RQ_COST_EVIDENCE_INCOMPLETE")
    if not request.all_models_online_case_ready:
        codes.append("RQ_ONLINE_ENTRY_UNAVAILABLE")
    codes.append("RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED")
    codes = list(dict.fromkeys(codes))

    hard_blocked = any(code in HARD_BLOCK_CODES for code in codes)
    research_ready = (
        not hard_blocked
        and request.protocol_frozen
        and request.selection_split in {"validation", "val"}
        and evidence_label in {"beneficial", "risk_tradeoff"}
    )
    deployment_ready = (
        research_ready
        and evidence_label == "beneficial"
        and introduced == 0
        and request.cost_protocol_complete
        and request.all_models_online_case_ready
        and request.primary_metric_available
        and request.stability_ci_lower is not None
        and request.stability_ci_lower > 0
        and "RQ_FROZEN_REVERSAL" not in codes
    )

    if hard_blocked:
        execution_level = "blocked"
    elif request.request_scope == "new_case" and deployment_ready:
        execution_level = "deployment_candidate"
    elif research_ready and request.request_scope != "new_case":
        execution_level = "research_case_simulation"
    else:
        execution_level = "research_replay_only"

    trace.append(
        {
            "layer": "execution",
            "check": "allowed_execution_level",
            "passed": execution_level != "blocked",
            "execution_level": execution_level,
            "clinical_route_eligible": False,
        }
    )
    fingerprint_payload = {
        "request": asdict(request),
        "contract_id": contract.get("protocol_id"),
    }
    contract_hash = contract_sha256 or _sha256_bytes(
        _stable_json(contract).encode("utf-8")
    )
    return RouteQualificationDecision(
        execution_level=execution_level,
        evidence_label=evidence_label,
        allow_cached_replay=not hard_blocked,
        allow_case_simulation=execution_level in {
            "research_case_simulation",
            "deployment_candidate",
        },
        allow_new_case_route=execution_level == "deployment_candidate",
        clinical_route_eligible=False,
        human_confirmation_required=True,
        error_codes=tuple(codes or ["RQ_OK"]),
        gate_trace=tuple(trace),
        contract_sha256=contract_hash,
        evidence_fingerprint=_sha256_bytes(
            _stable_json(fingerprint_payload).encode("utf-8")
        ),
    )


def _route_signature(row: pd.Series) -> tuple[str, str, str, float]:
    scouts = "|".join(
        sorted(
            value
            for value in str(row.get("scout_artifact_ids", "")).split("|")
            if value
        )
    )
    return (
        scouts,
        str(row.get("expert_artifact_id", "")),
        str(row.get("routing_policy", "")),
        round(float(row.get("requested_budget", 0.0)), 6),
    )


def _is_validation_run(row: pd.Series) -> bool:
    return "validation" in str(row.get("result_path", "")).lower()


def _is_formal_frozen_run(row: pd.Series) -> bool:
    text = (
        f"{row.get('result_path', '')} "
        f"{row.get('evaluation_design', '')} "
        f"{row.get('stage', '')}"
    ).lower()
    if "exploratory" in text or "protocol_invalid" in text:
        return False
    return any(
        marker in text
        for marker in (
            "test_locked",
            "locked_test",
            "frozen_official",
            "frozen_external",
            "冻结结果集",
        )
    )


def _scout_metric(frame: pd.DataFrame, row: pd.Series, metric: str) -> float | None:
    direct = f"scout_only_{metric}"
    if direct in row.index:
        value = _finite(row.get(direct))
        if value is not None:
            return value
    matching = frame.loc[
        frame["pairing_id"].astype(str).eq(str(row.get("pairing_id", "")))
        & frame["evaluation_kind"].astype(str).eq("scout_only")
    ]
    if matching.empty or metric not in matching:
        return None
    return _finite(matching.iloc[0].get(metric))


def _resolve_metric(
    row: pd.Series,
    policy: dict[str, Any],
) -> tuple[str, float | None, bool]:
    primary = str(policy["primary_metric"])
    primary_value = _finite(row.get(primary))
    if primary_value is not None:
        return primary, primary_value, True
    fallback = str(policy.get("fallback_metric") or "")
    return fallback, _finite(row.get(fallback)), False


def _resolve_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value or "").strip())
    return path if path.is_absolute() else project_root / path


def _asset_evidence(
    project_root: Path,
    task_assets: pd.DataFrame,
    *,
    task_id: str,
    artifact_ids: tuple[str, ...],
    n_classes: int,
) -> dict[str, Any]:
    rows = task_assets.loc[
        task_assets["task_id"].astype(str).eq(task_id)
        & task_assets["artifact_id"].astype(str).isin(artifact_ids)
    ].copy()
    task_matches = len(rows["artifact_id"].unique()) == len(set(artifact_ids))
    prediction_valid = task_matches
    class_order_matches = task_matches
    fingerprints: list[str] = []
    for _, row in rows.iterrows():
        for column in ("validation_prediction_path", "test_prediction_path"):
            path = _resolve_path(project_root, row.get(column))
            if not path.is_file():
                prediction_valid = False
                continue
            try:
                header = pd.read_csv(path, nrows=0)
            except (OSError, pd.errors.ParserError, UnicodeError):
                prediction_valid = False
                continue
            required = {f"prob_{index}" for index in range(n_classes)}
            if not required.issubset(header.columns):
                class_order_matches = False
            fingerprints.append(f"{column}:{file_sha256(path)}")
    costs = pd.to_numeric(
        rows.get(
            "forward_cost_ms_per_image",
            pd.Series(np.nan, index=rows.index),
        ),
        errors="coerce",
    )
    cost_status = rows.get("cost_status", pd.Series("", index=rows.index)).astype(str)
    cost_scope = rows.get("cost_scope", pd.Series("", index=rows.index)).astype(str)
    cost_complete = bool(
        task_matches
        and costs.notna().all()
        and costs.ge(0).all()
        and cost_status.eq("measured").all()
        and cost_scope.str.contains("H100", case=False).all()
    )
    online = rows.get(
        "online_case_inference_ready", pd.Series(False, index=rows.index)
    ).fillna(False)
    return {
        "task_matches": task_matches,
        "label_space_matches": task_matches,
        "class_order_matches": class_order_matches,
        "prediction_assets_valid": prediction_valid,
        "cost_protocol_complete": cost_complete,
        "all_models_online_case_ready": bool(task_matches and online.astype(bool).all()),
        "input_asset_fingerprint": _sha256_bytes(
            "\n".join(sorted(fingerprints)).encode("utf-8")
        ),
    }


def _stability_interval(
    robustness: pd.DataFrame,
    *,
    task_id: str,
    pairing_id: str,
    metric: str,
) -> tuple[float | None, float | None]:
    if robustness.empty:
        return None, None
    rows = robustness.loc[
        robustness["analysis_section"].astype(str).eq("route_metric_ci")
        & robustness["task_id"].astype(str).eq(task_id)
        & robustness["split"].astype(str).eq("val")
        & robustness["candidate_id"].astype(str).eq(pairing_id)
        & robustness["estimate"].astype(str).eq("route_minus_scout")
        & robustness["metric"].astype(str).eq(metric)
    ]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    return _finite(row.get("ci_lower")), _finite(row.get("ci_upper"))


def _protocol_sha(run: pd.Series, result_path: Path) -> str:
    config_path = result_path.parent / "run_config.yaml"
    if config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        registered = str(payload.get("protocol_sha256") or "")
        if registered:
            return registered
        return file_sha256(config_path)
    return str(run.get("protocol_sha256") or "")


def _as_number(value: Any) -> float | None:
    return _finite(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_route_qualification_matrix(project_root: Path) -> pd.DataFrame:
    """Project formal frozen routes onto their predeclared Validation evidence."""

    from app.model_hub_index import build_model_hub_index

    contract, contract_sha = load_route_qualification_contract(project_root)
    task_policies = dict(contract["task_policies"])
    index = build_model_hub_index(project_root)
    route_runs = index["route_runs"]
    task_assets = index["task_assets"]
    robustness_path = (
        project_root / "experiments/opening_risk_routing_closure/dual_task_robustness.csv"
    )
    robustness = (
        pd.read_csv(robustness_path)
        if robustness_path.is_file()
        else pd.DataFrame()
    )
    rows: list[dict[str, Any]] = []

    for task_id, task_runs in route_runs.groupby("task_id"):
        policy = task_policies.get(str(task_id))
        if not policy:
            continue
        validation_records: dict[
            tuple[str, str, str, float], tuple[pd.Series, pd.DataFrame, pd.Series]
        ] = {}
        validation_best_single: dict[str, float] = {}
        for _, run in task_runs.iterrows():
            if not _is_validation_run(run):
                continue
            result_path = Path(str(run["_pairing_results_path"]))
            frame = pd.read_csv(result_path)
            for metric in {
                str(policy["primary_metric"]),
                str(policy.get("fallback_metric") or ""),
            } - {""}:
                singles = frame.loc[
                    frame["evaluation_kind"].astype(str).eq("scout_only")
                ]
                values = pd.to_numeric(singles.get(metric), errors="coerce")
                if values.notna().any():
                    validation_best_single[metric] = float(values.max())
            for _, route in frame.loc[
                frame["evaluation_kind"].astype(str).eq("routed")
            ].iterrows():
                validation_records[_route_signature(route)] = (route, frame, run)

        for _, frozen_run in task_runs.iterrows():
            if not _is_formal_frozen_run(frozen_run):
                continue
            frozen_path = Path(str(frozen_run["_pairing_results_path"]))
            frozen_frame = pd.read_csv(frozen_path)
            frozen_routes = frozen_frame.loc[
                frozen_frame["evaluation_kind"].astype(str).eq("routed")
                & frozen_frame["status"].astype(str).eq("completed")
            ]
            for _, frozen in frozen_routes.iterrows():
                validation_entry = validation_records.get(_route_signature(frozen))
                validation = validation_entry[0] if validation_entry else None
                validation_frame = validation_entry[1] if validation_entry else None
                validation_run = validation_entry[2] if validation_entry else None
                metric, validation_value, primary_available = (
                    _resolve_metric(validation, policy)
                    if validation is not None
                    else (
                        str(policy["primary_metric"]),
                        None,
                        False,
                    )
                )
                frozen_metric = _finite(frozen.get(metric))
                validation_scout = (
                    _scout_metric(validation_frame, validation, metric)
                    if validation is not None and validation_frame is not None
                    else None
                )
                frozen_scout = _scout_metric(frozen_frame, frozen, metric)
                best_single = validation_best_single.get(metric)
                stability_lower, stability_upper = _stability_interval(
                    robustness,
                    task_id=str(task_id),
                    pairing_id=str(frozen["pairing_id"]),
                    metric=metric,
                )
                scouts = tuple(
                    value
                    for value in str(frozen["scout_artifact_ids"]).split("|")
                    if value
                )
                expert = str(frozen["expert_artifact_id"])
                assets = _asset_evidence(
                    project_root,
                    task_assets,
                    task_id=str(task_id),
                    artifact_ids=(*scouts, expert),
                    n_classes=int(policy["n_classes"]),
                )
                protocol_frozen = (
                    str(frozen_run.get("status", "")) == "completed"
                    and "protocol_invalid"
                    not in str(frozen.get("result_semantics", "")).lower()
                )
                request = RouteQualificationRequest(
                    task_id=str(task_id),
                    pairing_id=str(frozen["pairing_id"]),
                    scout_artifact_ids=scouts,
                    expert_artifact_id=expert,
                    request_scope="cached_prediction_replay",
                    protocol_frozen=protocol_frozen,
                    selection_split="validation" if validation is not None else "",
                    validation_main_metric_delta=(
                        validation_value - validation_scout
                        if validation_value is not None
                        and validation_scout is not None
                        else None
                    ),
                    validation_delta_vs_best_single=(
                        validation_value - best_single
                        if validation_value is not None and best_single is not None
                        else None
                    ),
                    validation_corrected=(
                        _as_number(validation.get("dangerous_corrected"))
                        if validation is not None
                        else None
                    ),
                    validation_introduced=(
                        _as_number(validation.get("dangerous_introduced"))
                        if validation is not None
                        else None
                    ),
                    validation_net=(
                        _as_number(validation.get("net_dangerous_reduction"))
                        if validation is not None
                        else None
                    ),
                    stability_ci_lower=stability_lower,
                    frozen_main_metric_delta=(
                        frozen_metric - frozen_scout
                        if frozen_metric is not None and frozen_scout is not None
                        else None
                    ),
                    frozen_corrected=_as_number(frozen.get("dangerous_corrected")),
                    frozen_introduced=_as_number(frozen.get("dangerous_introduced")),
                    frozen_net=_as_number(frozen.get("net_dangerous_reduction")),
                    primary_metric_available=primary_available,
                    requested_budget=_as_number(frozen.get("requested_budget")),
                    expected_cost_ms_per_image=_as_number(
                        (
                            validation.get("estimated_total_compute_ms_per_image")
                            if validation is not None
                            else frozen.get("estimated_total_compute_ms_per_image")
                        )
                    ),
                    protocol_sha256=_protocol_sha(frozen_run, frozen_path),
                    **assets,
                )
                decision = evaluate_route_qualification(
                    request,
                    contract=contract,
                    contract_sha256=contract_sha,
                )
                validation_result_sha = (
                    file_sha256(Path(str(validation_run["_pairing_results_path"])))
                    if validation_run is not None
                    else ""
                )
                frozen_favorable = bool(
                    request.frozen_main_metric_delta is not None
                    and request.frozen_main_metric_delta > 0
                    and (
                        request.frozen_net is None
                        or request.frozen_net >= 0
                    )
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "pairing_id": request.pairing_id,
                        "scout_artifact_ids": "|".join(request.scout_artifact_ids),
                        "expert_artifact_id": request.expert_artifact_id,
                        "routing_policy": str(frozen["routing_policy"]),
                        "requested_budget": request.requested_budget,
                        "primary_metric": str(policy["primary_metric"]),
                        "metric_used": metric,
                        "primary_metric_available": primary_available,
                        "validation_main_metric": validation_value,
                        "validation_delta_vs_scout": (
                            request.validation_main_metric_delta
                        ),
                        "validation_delta_vs_best_single": (
                            request.validation_delta_vs_best_single
                        ),
                        "validation_corrected": request.validation_corrected,
                        "validation_introduced": request.validation_introduced,
                        "validation_net": request.validation_net,
                        "stability_ci_lower": stability_lower,
                        "stability_ci_upper": stability_upper,
                        "frozen_main_metric": frozen_metric,
                        "frozen_delta_vs_scout": request.frozen_main_metric_delta,
                        "frozen_corrected": request.frozen_corrected,
                        "frozen_introduced": request.frozen_introduced,
                        "frozen_net": request.frozen_net,
                        "frozen_favorable": frozen_favorable,
                        "evidence_label": decision.evidence_label,
                        "execution_level": decision.execution_level,
                        "allow_cached_replay": decision.allow_cached_replay,
                        "allow_case_simulation": decision.allow_case_simulation,
                        "allow_new_case_route": decision.allow_new_case_route,
                        "clinical_route_eligible": False,
                        "expected_cost_ms_per_image": (
                            request.expected_cost_ms_per_image
                        ),
                        "cost_protocol_complete": request.cost_protocol_complete,
                        "all_models_online_case_ready": (
                            request.all_models_online_case_ready
                        ),
                        "error_codes": "|".join(decision.error_codes),
                        "protocol_sha256": request.protocol_sha256,
                        "validation_result_sha256": validation_result_sha,
                        "frozen_result_sha256": file_sha256(frozen_path),
                        "input_asset_fingerprint": (
                            request.input_asset_fingerprint
                        ),
                        "qualification_contract_sha256": contract_sha,
                        "source_commit_sha": _git_commit(project_root),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["task_id", "pairing_id"], ignore_index=True)


def route_qualification_request_from_row(
    row: pd.Series,
    *,
    request_scope: str = "cached_prediction_replay",
) -> RouteQualificationRequest:
    return RouteQualificationRequest(
        task_id=str(row.get("task_id", "")),
        pairing_id=str(row.get("pairing_id", "")),
        scout_artifact_ids=tuple(
            value
            for value in str(row.get("scout_artifact_ids", "")).split("|")
            if value
        ),
        expert_artifact_id=str(row.get("expert_artifact_id", "")),
        request_scope=request_scope,
        task_matches=True,
        label_space_matches=True,
        class_order_matches=True,
        prediction_assets_valid=_as_bool(
            row.get("allow_cached_replay", False)
        ),
        cost_protocol_complete=_as_bool(
            row.get("cost_protocol_complete", False)
        ),
        all_models_online_case_ready=_as_bool(
            row.get("all_models_online_case_ready", False)
        ),
        protocol_frozen=bool(
            str(row.get("protocol_sha256", "") or "").strip()
            not in {"", "nan", "None"}
        ),
        selection_split=(
            "validation"
            if _finite(row.get("validation_main_metric")) is not None
            else ""
        ),
        validation_main_metric_delta=_finite(
            row.get("validation_delta_vs_scout")
        ),
        validation_delta_vs_best_single=_finite(
            row.get("validation_delta_vs_best_single")
        ),
        validation_corrected=_finite(row.get("validation_corrected")),
        validation_introduced=_finite(row.get("validation_introduced")),
        validation_net=_finite(row.get("validation_net")),
        stability_ci_lower=_finite(row.get("stability_ci_lower")),
        frozen_main_metric_delta=_finite(row.get("frozen_delta_vs_scout")),
        frozen_corrected=_finite(row.get("frozen_corrected")),
        frozen_introduced=_finite(row.get("frozen_introduced")),
        frozen_net=_finite(row.get("frozen_net")),
        primary_metric_available=_as_bool(
            row.get("primary_metric_available", False)
        ),
        requested_budget=_finite(row.get("requested_budget")),
        expected_cost_ms_per_image=_finite(
            row.get("expected_cost_ms_per_image")
        ),
        protocol_sha256=str(row.get("protocol_sha256", "")),
        input_asset_fingerprint=str(row.get("input_asset_fingerprint", "")),
    )


def find_route_qualification(
    project_root: Path,
    *,
    task_id: str,
    scout_artifact_ids: tuple[str, ...],
    expert_artifact_id: str,
    routing_policy: str,
    requested_budget: float,
    request_scope: str = "cached_prediction_replay",
) -> RouteQualificationDecision | None:
    path = project_root / OUTPUT_RELATIVE_DIR / "route_qualification_matrix.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    rows = _find_registered_route_rows(
        frame,
        task_id=task_id,
        pairing_id="",
        scout_artifact_ids=scout_artifact_ids,
        expert_artifact_id=expert_artifact_id,
        routing_policy=routing_policy,
        requested_budget=requested_budget,
    )
    if len(rows) != 1:
        return None
    contract, contract_sha = load_route_qualification_contract(project_root)
    request = route_qualification_request_from_row(
        rows.iloc[0],
        request_scope=request_scope,
    )
    return evaluate_route_qualification(
        request,
        contract=contract,
        contract_sha256=contract_sha,
    )


def _find_registered_route_rows(
    frame: pd.DataFrame,
    *,
    task_id: str,
    pairing_id: str,
    scout_artifact_ids: tuple[str, ...],
    expert_artifact_id: str,
    routing_policy: str,
    requested_budget: float,
) -> pd.DataFrame:
    base_mask = (
        frame["task_id"].astype(str).eq(task_id)
        & frame["routing_policy"].astype(str).eq(routing_policy)
        & np.isclose(
            pd.to_numeric(frame["requested_budget"], errors="coerce"),
            requested_budget,
        )
    )
    if pairing_id:
        exact_rows = frame.loc[
            base_mask & frame["pairing_id"].astype(str).eq(pairing_id)
        ]
        if len(exact_rows) == 1:
            return exact_rows
    if not scout_artifact_ids or not expert_artifact_id:
        return frame.iloc[0:0]
    scout_key = "|".join(sorted(scout_artifact_ids))
    return frame.loc[
        base_mask
        & frame["scout_artifact_ids"]
        .astype(str)
        .map(lambda value: "|".join(sorted(value.split("|"))))
        .eq(scout_key)
        & frame["expert_artifact_id"].astype(str).eq(expert_artifact_id)
    ]


def find_route_qualification_record(
    project_root: Path,
    *,
    task_id: str,
    pairing_id: str,
    routing_policy: str,
    requested_budget: float,
    scout_artifact_ids: tuple[str, ...] = (),
    expert_artifact_id: str = "",
    request_scope: str = "cached_prediction_replay",
) -> tuple[RouteQualificationDecision, dict[str, Any]] | None:
    """Return the registered decision and its non-sensitive evidence record."""

    path = project_root / OUTPUT_RELATIVE_DIR / "route_qualification_matrix.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    rows = _find_registered_route_rows(
        frame,
        task_id=task_id,
        pairing_id=pairing_id,
        scout_artifact_ids=scout_artifact_ids,
        expert_artifact_id=expert_artifact_id,
        routing_policy=routing_policy,
        requested_budget=requested_budget,
    )
    if len(rows) != 1:
        return None
    row = rows.iloc[0]
    contract, contract_sha = load_route_qualification_contract(project_root)
    request = route_qualification_request_from_row(
        row,
        request_scope=request_scope,
    )
    decision = evaluate_route_qualification(
        request,
        contract=contract,
        contract_sha256=contract_sha,
    )
    evidence_columns = [
        "task_id",
        "pairing_id",
        "scout_artifact_ids",
        "expert_artifact_id",
        "routing_policy",
        "requested_budget",
        "primary_metric",
        "metric_used",
        "validation_delta_vs_scout",
        "validation_delta_vs_best_single",
        "validation_corrected",
        "validation_introduced",
        "validation_net",
        "stability_ci_lower",
        "stability_ci_upper",
        "frozen_delta_vs_scout",
        "frozen_corrected",
        "frozen_introduced",
        "frozen_net",
        "expected_cost_ms_per_image",
        "protocol_sha256",
        "input_asset_fingerprint",
        "qualification_contract_sha256",
    ]
    evidence = {
        column: (
            None
            if pd.isna(row.get(column))
            else row.get(column)
        )
        for column in evidence_columns
        if column in row.index
    }
    return decision, evidence


def build_gate_comparison(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for task_id, frame in [
        *matrix.groupby("task_id"),
        ("all_tasks", matrix),
    ]:
        total = len(frame)
        favorable = frame["frozen_favorable"].fillna(False).astype(bool)
        granted = frame["allow_case_simulation"].fillna(False).astype(bool)
        ineffective = ~favorable
        reversals = frame["error_codes"].astype(str).str.contains(
            "RQ_FROZEN_REVERSAL"
        )
        introduced = pd.to_numeric(
            frame["frozen_introduced"], errors="coerce"
        ).fillna(0).gt(0)

        def rate(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        rows.append(
            {
                "task_id": task_id,
                "formal_frozen_routes": total,
                "frozen_favorable_routes": int(favorable.sum()),
                "ungated_research_actions": total,
                "gated_research_actions": int(granted.sum()),
                "ineffective_route_interception_rate": rate(
                    int((ineffective & ~granted).sum()),
                    int(ineffective.sum()),
                ),
                "beneficial_route_retention_rate": rate(
                    int((favorable & granted).sum()),
                    int(favorable.sum()),
                ),
                "ungated_false_grant_rate": rate(
                    int(ineffective.sum()),
                    total,
                ),
                "gated_false_grant_rate": rate(
                    int((ineffective & granted).sum()),
                    int(granted.sum()),
                ),
                "gated_false_rejection_rate": rate(
                    int((favorable & ~granted).sum()),
                    int(favorable.sum()),
                ),
                "test_reversal_interception_rate": rate(
                    int((reversals & ~granted).sum()),
                    int(reversals.sum()),
                ),
                "introduced_risk_deployment_limited_rate": rate(
                    int(
                        (
                            introduced
                            & ~frame["allow_new_case_route"]
                            .fillna(False)
                            .astype(bool)
                        ).sum()
                    ),
                    int(introduced.sum()),
                ),
                "research_action_coverage": rate(int(granted.sum()), total),
                "mean_expected_cost_ms_per_image": pd.to_numeric(
                    frame.loc[granted, "expected_cost_ms_per_image"],
                    errors="coerce",
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def write_route_qualification_artifacts(
    project_root: Path,
) -> dict[str, Path]:
    output_dir = project_root / OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_route_qualification_matrix(project_root)
    comparison = build_gate_comparison(matrix)
    matrix_path = output_dir / "route_qualification_matrix.csv"
    comparison_path = output_dir / "gate_retrospective_comparison.csv"
    matrix.to_csv(matrix_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    contract_path = project_root / CONTRACT_RELATIVE_PATH
    manifest = {
        "schema_version": "ophagent.route_qualification_artifacts.v1",
        "route_count": len(matrix),
        "task_count": int(matrix["task_id"].nunique()) if not matrix.empty else 0,
        "contract_path": CONTRACT_RELATIVE_PATH,
        "contract_sha256": file_sha256(contract_path),
        "matrix_sha256": file_sha256(matrix_path),
        "comparison_sha256": file_sha256(comparison_path),
        "source_commit_sha": _git_commit(project_root),
        "frozen_inputs_modified": False,
        "clinical_route_eligible": False,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "matrix": matrix_path,
        "comparison": comparison_path,
        "manifest": manifest_path,
    }
