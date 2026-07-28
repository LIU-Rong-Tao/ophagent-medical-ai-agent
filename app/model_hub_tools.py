"""Model Hub V1 的统一工具契约、资格门禁与本地调用轨迹。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image

from app.audit_core import compute_confidence, compute_entropy, compute_margin
from app.checkpoints import (
    ModelArtifact,
    compute_file_sha256,
    discover_model_artifacts,
    select_preferred_artifacts,
)
from app.inference import run_single_image_inference
from app.model_hub_index import build_task_asset_index
from app.route_qualification import (
    RouteQualificationRequest,
    evaluate_route_qualification,
    find_route_qualification_record,
    load_route_qualification_contract,
)


TOOL_NAMES = (
    "case_input.validate",
    "model_registry.inspect",
    "model_inference.run",
    "prediction_asset.validate",
    "result_risk_audit.run",
    "routing_protocol.evaluate",
)

QUALIFICATION_LEVELS = (
    "analytical_asset_only",
    "offline_batch_inference_ready",
    "online_case_inference_ready",
)

ERROR_MESSAGES = {
    "OK": "调用完成",
    "INVALID_REQUEST": "请求字段不完整或不合法",
    "INPUT_INVALID": "病例输入未通过完整性检查",
    "TASK_MISMATCH": "输入与任务契约不匹配",
    "ASSET_NOT_FOUND": "未找到已登记资产",
    "ASSET_INVALID": "预测资产未通过完整性检查",
    "QUALIFICATION_BLOCKED": "当前资格不允许执行该调用",
    "TEST_LOCKED": "冻结 Test 不允许在审阅工作台中读取",
    "ROUTE_RESEARCH_ONLY": "当前路由仅允许研究模拟",
    "OFFLINE_POLICY_VIOLATION": "离线模式禁止访问外部资源",
    "UPSTREAM_FAILED": "上游工具失败，后续调用已停止",
    "TOOL_EXECUTION_FAILED": "工具执行失败",
}

TASK_CONTRACTS = {
    "aptos_dr_5class": {
        "task_name": "APTOS DR 五级分级",
        "class_order": [0, 1, 2, 3, 4],
        "class_labels": ["0级", "1级", "2级", "3级", "4级"],
        "modality": "CFP",
        "risk_proxy": "DR 等级概率分布代理",
        "asset_registry": (
            "experiments/opening_risk_routing_closure/configs/protocols/"
            "aptos_h100_prediction_assets.csv"
        ),
        "route_trace": (
            "experiments/opening_risk_routing_closure/outputs/"
            "model_hub_validation_expanded_pool/case_routing_trace.csv"
        ),
        "route_protocol": (
            "experiments/opening_risk_routing_closure/configs/protocols/"
            "aptos_h100_ten_model_frozen_test_protocol.json"
        ),
        "route_pairing_id": (
            "aptos_dr_5class__flair__ret_clip__to__retfound_cfp"
        ),
        "route_policy": "disagreement_then_uncertainty",
        "route_budget": 0.20,
        "manifest": (
            "experiments/opening_risk_routing_closure/replays/preti/"
            "seed42_20260722/preflight/val_manifest.csv"
        ),
        "data_root": "/training_data/lizekun/data/RETFound/Data_split/APTOS2019",
    },
    "glaucoma_3class": {
        "task_name": "青光眼三分类",
        "class_order": [0, 1, 2],
        "class_labels": ["正常对照", "早期青光眼", "进展/晚期青光眼"],
        "modality": "CFP",
        "risk_proxy": "青光眼等级概率分布代理",
        "asset_registry": (
            "experiments/model_hub/tasks/glaucoma_3class/configs/"
            "glaucoma_h100_prediction_assets.csv"
        ),
        "route_trace": (
            "experiments/model_hub/tasks/glaucoma_3class/outputs/"
            "validation_pool/case_routing_trace.csv"
        ),
        "route_protocol": (
            "experiments/model_hub/tasks/glaucoma_3class/outputs/"
            "validation_pool/run_config.yaml"
        ),
        "route_pairing_id": (
            "glaucoma_3class__glaucoma_retfound_dinov2__"
            "glaucoma_vit_b__to__glaucoma_swin_tiny"
        ),
        "route_policy": "disagreement_then_uncertainty",
        "route_budget": 0.20,
        "manifest": "",
        "data_root": (
            "/training_data/lizekun/data/RETFound/Data_split/Glaucoma_fundus"
        ),
    },
}


class ToolError(RuntimeError):
    """带稳定错误码的工具错误。"""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class ToolRequest:
    tool_name: str
    task_id: str
    case_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class ToolResponse:
    ok: bool
    tool_name: str
    request_id: str
    code: str
    message: str
    data: dict[str, Any]
    qualification: str | None
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    request_id: str
    tool_name: str
    task_id: str
    case_alias: str
    status: str
    code: str
    started_at: str
    duration_ms: float
    request_fingerprint: str
    qualification: str | None
    message: str


@dataclass
class ToolContext:
    project_root: Path
    asset_registry: pd.DataFrame
    online_artifacts: dict[str, ModelArtifact]
    offline_mode: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _fingerprint_request(request: ToolRequest) -> str:
    safe_payload = {
        key: value
        for key, value in request.payload.items()
        if key not in {
            "image_paths",
            "raw_case_id",
            "patient_id",
            "source_case_key",
        }
    }
    payload = {
        "tool_name": request.tool_name,
        "task_id": request.task_id,
        "case_id": request.case_id,
        "payload": safe_payload,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value or "").strip()).expanduser()
    return path if path.is_absolute() else project_root / path


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


def _qualification_for_asset(row: pd.Series, project_root: Path) -> tuple[str, str]:
    checkpoint = _resolve(project_root, row.get("checkpoint_path"))
    adapter = str(row.get("adapter_type", ""))
    reproducible = bool(row.get("current_run_reproducible", False))
    batch_adapters = {
        "timm_classifier",
        "retfound_classifier",
        "preti_classifier",
    }
    if checkpoint.is_file() and reproducible and adapter in batch_adapters:
        return (
            "offline_batch_inference_ready",
            "已有当前任务 checkpoint 与批量重放证据；尚未登记单病例原图入口",
        )
    return (
        "analytical_asset_only",
        "已有冻结 validation prediction asset；仅允许只读审计和研究模拟",
    )


def build_review_asset_registry(project_root: Path) -> tuple[pd.DataFrame, dict[str, ModelArtifact]]:
    """Reuse the Model Hub projection for task assets and case endpoints."""

    rows: list[dict[str, Any]] = []
    registry = build_task_asset_index(project_root)
    for _, source in registry.iterrows():
        qualification, reason = _qualification_for_asset(source, project_root)
        rows.append(
            {
                **source.to_dict(),
                "display_name": str(source.get("artifact_id", "")),
                "qualification": qualification,
                "qualification_reason": reason,
                "online_artifact_key": "",
                "route_eligible": bool(source.get("route_eligible", False)),
            }
        )

    online_artifacts = select_preferred_artifacts(discover_model_artifacts(project_root))
    for key, artifact in online_artifacts.items():
        if not artifact.can_attempt_load:
            continue
        checkpoint_sha256 = (
            compute_file_sha256(artifact.checkpoint_path)
            if artifact.checkpoint_path is not None
            else ""
        )
        rows.append(
            {
                "task_id": "aptos_dr_5class",
                "artifact_id": f"online::{key}",
                "display_name": f"{artifact.display_name} · 单病例链",
                "model_family": key,
                "architecture": artifact.loader_model_name,
                "adapter_type": "legacy_verified_single_case",
                "pretraining_source": "existing_ophagent_artifact",
                "validation_prediction_path": "",
                "test_prediction_path": "",
                "checkpoint_path": str(artifact.checkpoint_path or ""),
                "checkpoint_sha256": checkpoint_sha256,
                "preprocessing_id": "registered_legacy_single_case_transform",
                "forward_cost_ms_per_image": np.nan,
                "cost_scope": "not_measured_for_v1_case_mode",
                "cost_status": "unmeasured",
                "current_run_reproducible": True,
                "validation_selection_eligible": False,
                "qualification_status": "single_case_runtime_available",
                "qualification": "online_case_inference_ready",
                "qualification_reason": "已有严格 checkpoint Loader、类别映射和单病例原图推理入口",
                "online_artifact_key": key,
                "route_eligible": False,
                "role_candidates": "",
                "source_version": artifact.commit_or_unknown,
                "notes": "仅单模型原图推理；不授予路由资格",
            }
        )
    return pd.DataFrame(rows), online_artifacts


def build_default_tool_context(project_root: Path) -> ToolContext:
    registry, online_artifacts = build_review_asset_registry(project_root)
    return ToolContext(
        project_root=project_root,
        asset_registry=registry,
        online_artifacts=online_artifacts,
        offline_mode=True,
    )


def capability_matrix(context: ToolContext) -> pd.DataFrame:
    """返回六项工具的真实实现与限制。"""

    online_count = int(
        context.asset_registry.get("qualification", pd.Series(dtype=str))
        .astype(str)
        .eq("online_case_inference_ready")
        .sum()
    )
    return pd.DataFrame(
        [
            ("case_input.validate", "implemented", "本地图像、任务、数量与格式检查"),
            ("model_registry.inspect", "implemented", "统一展示任务资产与三层资格"),
            (
                "model_inference.run",
                "implemented_limited",
                f"仅 {online_count} 个已登记旧单病例链可调用；离线资产被门禁阻止",
            ),
            (
                "prediction_asset.validate",
                "implemented",
                "复用正式 prediction validator；工作台只允许 validation",
            ),
            (
                "result_risk_audit.run",
                "implemented",
                "复用 audit_core 计算 entropy、margin 与模型分歧",
            ),
            (
                "routing_protocol.evaluate",
                "implemented_research_only",
                "只读复用冻结 validation route trace；route_eligible=false",
            ),
        ],
        columns=["tool_name", "status", "boundary"],
    )


class ToolRuntime:
    """按顺序执行工具并保存不含患者标识的结构化轨迹。"""

    def __init__(self, context: ToolContext, *, trace_id: str | None = None):
        self.context = context
        self.trace_id = trace_id or f"trace-{uuid4().hex[:12]}"
        self.events: list[TraceEvent] = []
        self.halted = False

    def run(self, request: ToolRequest) -> ToolResponse:
        started = time.perf_counter()
        started_at = _utc_now()
        fingerprint = _fingerprint_request(request)
        if request.tool_name not in TOOL_NAMES:
            return self._record(
                request,
                started,
                started_at,
                fingerprint,
                ok=False,
                code="INVALID_REQUEST",
                message=f"未登记工具：{request.tool_name}",
                data={},
                qualification=None,
            )
        if self.halted:
            return self._record(
                request,
                started,
                started_at,
                fingerprint,
                ok=False,
                code="UPSTREAM_FAILED",
                message=ERROR_MESSAGES["UPSTREAM_FAILED"],
                data={},
                qualification=None,
                status="skipped",
                halt=False,
            )
        handler = TOOL_HANDLERS[request.tool_name]
        try:
            data, qualification = handler(self.context, request)
        except ToolError as exc:
            return self._record(
                request,
                started,
                started_at,
                fingerprint,
                ok=False,
                code=exc.code,
                message=str(exc),
                data=exc.details,
                qualification=exc.details.get("qualification"),
            )
        except Exception as exc:  # pragma: no cover - exact errors are covered by handlers
            return self._record(
                request,
                started,
                started_at,
                fingerprint,
                ok=False,
                code="TOOL_EXECUTION_FAILED",
                message=f"{type(exc).__name__}: {str(exc)[:300]}",
                data={},
                qualification=None,
            )
        return self._record(
            request,
            started,
            started_at,
            fingerprint,
            ok=True,
            code="OK",
            message=ERROR_MESSAGES["OK"],
            data=data,
            qualification=qualification,
            halt=False,
        )

    def _record(
        self,
        request: ToolRequest,
        started: float,
        started_at: str,
        fingerprint: str,
        *,
        ok: bool,
        code: str,
        message: str,
        data: dict[str, Any],
        qualification: str | None,
        status: str | None = None,
        halt: bool = True,
    ) -> ToolResponse:
        event = TraceEvent(
            sequence=len(self.events) + 1,
            request_id=request.request_id,
            tool_name=request.tool_name,
            task_id=request.task_id,
            case_alias=request.case_id,
            status=status or ("succeeded" if ok else "failed"),
            code=code,
            started_at=started_at,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            request_fingerprint=fingerprint,
            qualification=qualification,
            message=message,
        )
        self.events.append(event)
        if not ok and halt and status != "skipped":
            self.halted = True
        return ToolResponse(
            ok=ok,
            tool_name=request.tool_name,
            request_id=request.request_id,
            code=code,
            message=message,
            data=data,
            qualification=qualification,
            trace_id=self.trace_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ophagent.model_hub_trace.v1",
            "trace_id": self.trace_id,
            "offline_mode": self.context.offline_mode,
            "git_commit": _git_commit(self.context.project_root),
            "events": [asdict(event) for event in self.events],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path


def _task_contract(request: ToolRequest) -> dict[str, Any]:
    try:
        return TASK_CONTRACTS[request.task_id]
    except KeyError as exc:
        raise ToolError(
            "TASK_MISMATCH",
            f"未登记任务：{request.task_id}",
        ) from exc


def _case_input_validate(
    context: ToolContext,
    request: ToolRequest,
) -> tuple[dict[str, Any], str | None]:
    contract = _task_contract(request)
    split = str(request.payload.get("split", "validation")).lower()
    if split in {"test", "locked_test"}:
        raise ToolError("TEST_LOCKED", ERROR_MESSAGES["TEST_LOCKED"])
    image_paths = [Path(str(value)) for value in request.payload.get("image_paths", [])]
    if not image_paths or len(image_paths) > 8:
        raise ToolError("INPUT_INVALID", "病例必须包含 1 至 8 张本地图像")
    images = []
    for path in image_paths:
        if str(path).startswith(("http://", "https://")):
            raise ToolError(
                "OFFLINE_POLICY_VIOLATION",
                "离线工作台不接受公网图像地址",
            )
        if not path.is_file():
            raise ToolError("INPUT_INVALID", "病例图像不存在或当前账号不可读")
        try:
            with Image.open(path) as image:
                images.append(
                    {
                        "display_name": f"图像 {len(images) + 1}",
                        "width": int(image.width),
                        "height": int(image.height),
                        "mode": str(image.mode),
                    }
                )
        except OSError as exc:
            raise ToolError("INPUT_INVALID", "病例图像无法解码") from exc
    return (
        {
            "case_alias": request.case_id,
            "task_name": contract["task_name"],
            "modality": contract["modality"],
            "split": split,
            "image_count": len(images),
            "images": images,
            "structured_fields": sorted(request.payload.get("structured_info", {}).keys()),
            "input_complete": True,
        },
        None,
    )


def _model_registry_inspect(
    context: ToolContext,
    request: ToolRequest,
) -> tuple[dict[str, Any], str | None]:
    _task_contract(request)
    rows = context.asset_registry.loc[
        context.asset_registry["task_id"].astype(str).eq(request.task_id)
    ].copy()
    if rows.empty:
        raise ToolError("ASSET_NOT_FOUND", "当前任务没有已登记模型资产")
    records = []
    for _, row in rows.iterrows():
        cost = pd.to_numeric(
            pd.Series([row.get("forward_cost_ms_per_image")]), errors="coerce"
        ).iloc[0]
        records.append(
            {
                "artifact_id": str(row.get("artifact_id", "")),
                "display_name": str(row.get("display_name", row.get("artifact_id", ""))),
                "architecture": str(row.get("architecture", "")),
                "qualification": str(row.get("qualification", "")),
                "qualification_reason": str(row.get("qualification_reason", "")),
                "cost_ms_per_image": float(cost) if pd.notna(cost) else None,
                "cost_scope": str(row.get("cost_scope", "")),
                "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
                "preprocessing_id": str(row.get("preprocessing_id", "")),
                "route_eligible": bool(row.get("route_eligible", False)),
            }
        )
    return (
        {
            "task_id": request.task_id,
            "models": records,
            "online_case_inference_count": sum(
                item["qualification"] == "online_case_inference_ready"
                for item in records
            ),
        },
        None,
    )


def _asset_row(context: ToolContext, task_id: str, artifact_id: str) -> pd.Series:
    rows = context.asset_registry.loc[
        context.asset_registry["task_id"].astype(str).eq(task_id)
        & context.asset_registry["artifact_id"].astype(str).eq(artifact_id)
    ]
    if rows.empty:
        raise ToolError("ASSET_NOT_FOUND", f"未登记模型资产：{artifact_id}")
    return rows.iloc[0]


def _model_inference_run(
    context: ToolContext,
    request: ToolRequest,
) -> tuple[dict[str, Any], str | None]:
    _task_contract(request)
    artifact_id = str(request.payload.get("artifact_id", ""))
    row = _asset_row(context, request.task_id, artifact_id)
    qualification = str(row.get("qualification", ""))
    if qualification != "online_case_inference_ready":
        raise ToolError(
            "QUALIFICATION_BLOCKED",
            "该资产只能用于离线批量推理或只读审计，不能调用新病例原图",
            details={
                "artifact_id": artifact_id,
                "qualification": qualification,
                "required": "online_case_inference_ready",
            },
        )
    image_paths = request.payload.get("image_paths", [])
    if len(image_paths) != 1:
        raise ToolError("INPUT_INVALID", "当前单病例模型入口一次只接受一张图像")
    online_key = str(row.get("online_artifact_key", ""))
    artifact = context.online_artifacts.get(online_key)
    if artifact is None:
        raise ToolError("ASSET_NOT_FOUND", "在线单病例 Loader 未找到")
    with Image.open(Path(str(image_paths[0]))) as source:
        result = run_single_image_inference(source.convert("RGB"), artifact)
    if not result.ok:
        raise ToolError(
            "TOOL_EXECUTION_FAILED",
            f"模型推理失败（{result.stage}）：{result.error_message}",
            details={"stage": result.stage, "error_type": result.error_type},
        )
    return (
        {
            "artifact_id": artifact_id,
            "pred_label": int(result.pred_grade),
            "probabilities": list(result.probabilities or []),
            "confidence": result.confidence,
            "margin": result.margin,
            "entropy": result.entropy_norm,
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "preprocessing_id": str(row.get("preprocessing_id", "")),
        },
        qualification,
    )


def _prediction_asset_validate(
    context: ToolContext,
    request: ToolRequest,
) -> tuple[dict[str, Any], str | None]:
    contract = _task_contract(request)
    split = str(request.payload.get("split", "validation")).lower()
    if split != "validation":
        raise ToolError("TEST_LOCKED", ERROR_MESSAGES["TEST_LOCKED"])
    artifact_id = str(request.payload.get("artifact_id", ""))
    row = _asset_row(context, request.task_id, artifact_id)
    qualification = str(row.get("qualification", ""))
    path = _resolve(context.project_root, row.get("validation_prediction_path"))
    if not path.is_file():
        raise ToolError("ASSET_NOT_FOUND", "validation prediction asset 不存在")
    try:
        from scripts.routing.run_interactive_model_hub import (
            normalize_and_validate_prediction,
        )

        frame = normalize_and_validate_prediction(
            path,
            n_classes=len(contract["class_order"]),
        )
    except Exception as exc:
        raise ToolError("ASSET_INVALID", f"prediction asset 校验失败：{exc}") from exc
    source_case_key = str(request.payload.get("source_case_key", request.case_id))
    case_rows = frame.loc[frame["image_key"].astype(str).eq(source_case_key)]
    if case_rows.empty:
        raise ToolError("ASSET_NOT_FOUND", "该 validation asset 中没有当前病例")
    case = case_rows.iloc[0]
    probability_columns = [
        f"prob_{index}" for index in range(len(contract["class_order"]))
    ]
    probabilities = [float(case[column]) for column in probability_columns]
    return (
        {
            "artifact_id": artifact_id,
            "pred_label": int(case["pred_label"]),
            "probabilities": probabilities,
            "confidence": float(case["confidence"]),
            "margin": float(case["margin"]),
            "entropy": float(case["entropy"]),
            "prediction_asset_sha256": compute_file_sha256(path),
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "preprocessing_id": str(row.get("preprocessing_id", "")),
            "split": "validation",
            "read_only": True,
        },
        qualification,
    )


def _result_risk_audit_run(
    context: ToolContext,
    request: ToolRequest,
) -> tuple[dict[str, Any], str | None]:
    contract = _task_contract(request)
    predictions = list(request.payload.get("predictions", []))
    if not predictions:
        raise ToolError("INVALID_REQUEST", "风险审计至少需要一组概率")
    rows = []
    for prediction in predictions:
        probabilities = [float(value) for value in prediction.get("probabilities", [])]
        if len(probabilities) != len(contract["class_order"]):
            raise ToolError("ASSET_INVALID", "模型概率数量与任务类别顺序不一致")
        row = {
            "artifact_id": str(prediction.get("artifact_id", "")),
            **{
                f"prob_{index}": probability
                for index, probability in enumerate(probabilities)
            },
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    probability_columns = [
        f"prob_{index}" for index in range(len(contract["class_order"]))
    ]
    frame["confidence"] = compute_confidence(frame, probability_columns)
    frame["margin"] = compute_margin(frame, probability_columns)
    frame["entropy"] = compute_entropy(frame, probability_columns)
    predictions_array = frame[probability_columns].to_numpy(dtype=float)
    labels = predictions_array.argmax(axis=1)
    disagreement = len(set(labels.tolist())) > 1
    mean_probabilities = predictions_array.mean(axis=0)
    proxy: dict[str, Any]
    if request.task_id == "aptos_dr_5class":
        proxy = {
            "name": "DR 重症等级概率质量代理",
            "value": float(mean_probabilities[3] + mean_probabilities[4]),
            "definition": "各模型 P(3级)+P(4级) 的平均值",
        }
    else:
        proxy = {
            "name": "青光眼非正常等级概率质量代理",
            "value": float(mean_probabilities[1:].sum()),
            "definition": "各模型非正常类别概率质量的平均值",
        }
    return (
        {
            "models": frame[
                ["artifact_id", "confidence", "margin", "entropy"]
            ].to_dict("records"),
            "model_disagreement": disagreement,
            "predicted_labels": labels.astype(int).tolist(),
            "mean_probabilities": mean_probabilities.tolist(),
            "task_proxy": proxy,
            "semantics": "model_output_error_risk_not_clinical_consequence",
        },
        "analytical_asset_only",
    )


@lru_cache(maxsize=8)
def _read_route_trace(
    path_text: str,
    modified_ns: int,
    pairing_id: str,
    policy: str,
    budget: float,
) -> pd.DataFrame:
    del modified_ns
    columns = [
        "pairing_id",
        "task_id",
        "image_key",
        "primary_scout_artifact_id",
        "expert_artifact_id",
        "routing_policy",
        "requested_budget",
        "realized_budget",
        "scout_pred_labels",
        "scout_confidences",
        "scout_entropies",
        "scout_margins",
        "scout_disagreement",
        "routing_score",
        "is_reviewed_by_expert",
        "expert_pred_label",
        "final_pred_label",
        "final_source",
    ]
    frame = pd.read_csv(path_text, usecols=columns, low_memory=False)
    return frame.loc[
        frame["pairing_id"].astype(str).eq(pairing_id)
        & frame["routing_policy"].astype(str).eq(policy)
        & np.isclose(frame["requested_budget"].astype(float), budget)
    ].copy()


def _routing_protocol_evaluate(
    context: ToolContext,
    request: ToolRequest,
) -> tuple[dict[str, Any], str | None]:
    contract = _task_contract(request)
    split = str(request.payload.get("split", "validation")).strip().lower()
    if split not in {"validation", "val"}:
        raise ToolError(
            "TEST_LOCKED",
            "病例工作台只能读取 Validation 路由轨迹",
        )
    trace_path = context.project_root / str(contract["route_trace"])
    protocol_path = context.project_root / str(contract["route_protocol"])
    if not trace_path.is_file() or not protocol_path.is_file():
        raise ToolError("ASSET_NOT_FOUND", "冻结路由 trace 或 protocol 不存在")
    frame = _read_route_trace(
        str(trace_path),
        trace_path.stat().st_mtime_ns,
        str(contract["route_pairing_id"]),
        str(contract["route_policy"]),
        float(contract["route_budget"]),
    )
    source_case_key = str(request.payload.get("source_case_key", request.case_id))
    rows = frame.loc[frame["image_key"].astype(str).eq(source_case_key)]
    if rows.empty:
        raise ToolError("ASSET_NOT_FOUND", "冻结 validation trace 中没有当前病例")
    row = rows.iloc[0]
    try:
        scout_artifact_ids = tuple(
            str(value)
            for value in json.loads(str(row["scout_pred_labels"])).keys()
        )
    except (AttributeError, json.JSONDecodeError, TypeError):
        scout_artifact_ids = ()
    if not scout_artifact_ids:
        scout_artifact_ids = (str(row["primary_scout_artifact_id"]),)

    qualification_record = find_route_qualification_record(
        context.project_root,
        task_id=request.task_id,
        pairing_id=str(row["pairing_id"]),
        routing_policy=str(row["routing_policy"]),
        requested_budget=float(row["requested_budget"]),
        scout_artifact_ids=scout_artifact_ids,
        expert_artifact_id=str(row["expert_artifact_id"]),
        request_scope="cached_prediction_replay",
    )
    if qualification_record is None:
        qualification_contract, qualification_contract_sha = (
            load_route_qualification_contract(context.project_root)
        )
        qualification_request = RouteQualificationRequest(
            task_id=request.task_id,
            pairing_id=str(row["pairing_id"]),
            scout_artifact_ids=scout_artifact_ids,
            expert_artifact_id=str(row["expert_artifact_id"]),
            request_scope="cached_prediction_replay",
            prediction_assets_valid=True,
            protocol_frozen=True,
            selection_split="",
            protocol_sha256=compute_file_sha256(protocol_path),
        )
        route_qualification = evaluate_route_qualification(
            qualification_request,
            contract=qualification_contract,
            contract_sha256=qualification_contract_sha,
        )
        gate_evidence: dict[str, Any] = {
            "matrix_record_found": False,
            "reason": "尚未生成正式资格矩阵，仅允许保守只读回放",
        }
    else:
        route_qualification, gate_evidence = qualification_record
        gate_evidence["matrix_record_found"] = True

    if not route_qualification.allow_cached_replay:
        raise ToolError(
            "QUALIFICATION_BLOCKED",
            "冻结路由未通过资产与任务门禁",
            details={
                "qualification": route_qualification.execution_level,
                "route_qualification": route_qualification.to_dict(),
                "gate_evidence": gate_evidence,
            },
        )
    return (
        {
            "execution_mode": "cached_prediction_replay",
            "evaluation_design": "research_simulation",
            "route_eligible": False,
            "pairing_id": str(row["pairing_id"]),
            "scout_artifact_ids": list(scout_artifact_ids),
            "routing_policy": str(row["routing_policy"]),
            "requested_budget": float(row["requested_budget"]),
            "realized_budget": float(row["realized_budget"]),
            "primary_scout_artifact_id": str(row["primary_scout_artifact_id"]),
            "expert_artifact_id": str(row["expert_artifact_id"]),
            "scout_disagreement": bool(row["scout_disagreement"]),
            "routing_score": float(row["routing_score"]),
            "expert_invoked": bool(row["is_reviewed_by_expert"]),
            "expert_pred_label": (
                int(row["expert_pred_label"])
                if pd.notna(row["expert_pred_label"])
                else None
            ),
            "final_pred_label": int(row["final_pred_label"]),
            "final_source": str(row["final_source"]),
            "protocol_sha256": compute_file_sha256(protocol_path),
            "trace_asset_sha256": compute_file_sha256(trace_path),
            "git_commit": _git_commit(context.project_root),
            "test_content_used": False,
            "route_qualification": route_qualification.to_dict(),
            "gate_evidence": gate_evidence,
        },
        route_qualification.execution_level,
    )


TOOL_HANDLERS: dict[
    str,
    Callable[[ToolContext, ToolRequest], tuple[dict[str, Any], str | None]],
] = {
    "case_input.validate": _case_input_validate,
    "model_registry.inspect": _model_registry_inspect,
    "model_inference.run": _model_inference_run,
    "prediction_asset.validate": _prediction_asset_validate,
    "result_risk_audit.run": _result_risk_audit_run,
    "routing_protocol.evaluate": _routing_protocol_evaluate,
}


def trace_frame(runtime: ToolRuntime) -> pd.DataFrame:
    return pd.DataFrame([asdict(event) for event in runtime.events])
