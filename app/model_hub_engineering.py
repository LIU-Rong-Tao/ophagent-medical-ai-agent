"""模型工程工作区：模型接入、研究评测与任务运行记录。"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import asdict, replace
from datetime import datetime
import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from app.model_asset_runtime import RUNTIME_ROOT, build_asset_readiness, latest_smoke_run
from app.model_asset_smoke_jobs import (
    ASSET_SMOKE_JOBS_ROOT,
    archive_asset_smoke_job,
    cancel_asset_smoke_job,
    checkpoint_smoke_evidence,
    delete_asset_smoke_job,
    import_legacy_smoke_run,
    list_asset_smoke_jobs,
    read_asset_smoke_log_tail,
    resume_asset_smoke_job,
    retry_asset_smoke_subset,
    submit_asset_smoke_job,
)
from app.model_hub_data import build_unified_model_catalog
from app.model_hub_research import render_research_workspace
from app.model_hub_scan_jobs import (
    list_global_scan_jobs,
    load_global_scan_results,
    read_global_scan_log_tail,
)
from app.training_config import (
    TrainingConfigError,
    build_official_trial_recipe,
    build_training_draft,
    discover_official_recipe_profiles,
    discover_training_recipes,
    expand_official_recipe_trials,
    match_official_recipe_profiles,
    parse_yaml_text,
    save_training_recipe,
)
from app.model_hub_ui import human_family, human_model, task_label
from app.ophbench_task_adapter import STANDARD_ARTIFACT_ID
from app.training_jobs import (
    TrainingRequest,
    adaptation_artifact_id,
    archive_training_job,
    build_adaptation_request,
    build_retry_request,
    build_training_context,
    cancel_training_job,
    delete_training_job_and_outputs,
    list_training_jobs,
    load_training_progress,
    read_job_log_tail,
    registered_dataset_options,
    submit_training_job,
    validate_training_request,
)
from models.datasets.imagefolder_classification import DatasetInspection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_RECIPE_ROOT = PROJECT_ROOT / "experiments/model_hub/registry/training_recipes"
OFFICIAL_RECIPE_PROFILE_ROOT = PROJECT_ROOT / "experiments/model_hub/registry/official_recipe_profiles"
LEGACY_TRAINING_RECIPES = PROJECT_ROOT / "experiments/v0_8_6_interactive_model_hub/configs/training_recipes.csv"
TASK_REGISTRY = PROJECT_ROOT / "experiments/v0_8_5_model_registry_scout_expert_protocol/configs/task_registry.csv"
MODEL_HUB_ROOT = PROJECT_ROOT / "experiments/model_hub"
TRAINING_JOBS_ROOT = MODEL_HUB_ROOT / "runtime/training_jobs"
TRAINING_RUNS_ROOT = MODEL_HUB_ROOT / "runs/training"

GLOBAL_SCAN_PREVIEW_RENAME = {
    "global_rank_primary": "全局排名",
    "scout_ids": "路由模型",
    "active_expert_ids": "专家模型",
    "routing_policy": "路由机制",
    "realized_budget": "专家调用比例",
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "qwk": "QWK",
    "estimated_total_compute_ms_per_image": "估算前向成本（ms/图）",
}

GLOBAL_SCAN_POLICY_LABELS = {
    "low_confidence": "低置信度优先",
    "low_margin": "Top1/Top2 间隔小优先",
    "high_entropy": "高熵优先",
    "disagreement_then_uncertainty": "路由模型分歧优先，其次不确定性",
    "mean_uncertainty": "路由模型平均不确定性",
}

KNOWN_TRAINING_LOG_WARNINGS = {
    "scheduler.step()": "PyTorch 学习率调度器弃用提示",
    "EPOCH_DEPRECATION_WARNING": "PyTorch 学习率调度器弃用提示",
}

ASSET_PROBE_PROFILE_LABELS = {
    "deretfound_mae_runtime": "RETFound-DE MAE 原生前向",
    "deretfound_sd_runtime": "RETFound-DE 生成 UNet 前向",
    "eyeclip_runtime": "EyeCLIP 图文前向",
    "flair_runtime": "FLAIR 图文前向",
    "fmue_runtime": "FMUE 16 类 OCT 前向",
    "keepfit_runtime": "KeepFIT 图文前向",
    "mirage_runtime": "MIRAGE 配对模态前向",
    "preti_runtime": "PRETI 编码前向",
    "ret_clip_runtime": "RET-CLIP 图文前向",
    "retfound_runtime": "RETFound 编码前向",
    "retfound_green_runtime": "RETFound-Green 编码前向",
    "retizero_runtime": "RetiZero 图文前向",
    "urfound_runtime": "UrFound 原生前向",
    "vilref_runtime": "ViLReF 图文前向",
    "visionunite_resource_audit": "VisionUnite 资源契约审计",
    "torch_checkpoint_structure": "PyTorch checkpoint 结构",
    "safetensors_structure": "safetensors 结构",
    "zip_structure": "ZIP 资产结构",
    "metadata_only": "仅元数据",
}

ASSET_SMOKE_STATUS_LABELS = {
    "queued": "等待启动",
    "running": "运行中",
    "succeeded": "已完成",
    "completed_with_blockers": "已完成，有未通过项",
    "failed": "任务框架失败",
    "cancelled": "已取消",
    "pending": "等待运行",
    "runtime_smoke_passed": "完整前向通过",
    "asset_probe_passed": "资产探测通过",
    "resource_blocked": "资源阻塞",
    "timeout": "运行超时",
    "not_applicable": "不适用",
}


def _asset_smoke_parent_status_label(
    status: str, counts: dict[str, object]
) -> str:
    if status != "completed_with_blockers":
        return ASSET_SMOKE_STATUS_LABELS.get(status, status)
    failed_count = int(counts.get("failed", 0) or 0) + int(
        counts.get("timeout", 0) or 0
    )
    blocked_count = int(counts.get("resource_blocked", 0) or 0)
    if failed_count and blocked_count:
        return "已完成，有失败和资源阻塞"
    if failed_count:
        return "已完成，有失败项"
    if blocked_count:
        return "已完成，有资源阻塞"
    return "已完成，有未通过项"


def _ensure_historical_asset_smoke_job() -> str | None:
    source = RUNTIME_ROOT / "20260716T021348Z-all"
    if not source.is_dir():
        return None
    return import_legacy_smoke_run(source, jobs_root=ASSET_SMOKE_JOBS_ROOT)


def _render_asset_runtime_readiness() -> None:
    readiness = build_asset_readiness()
    local_count = int(readiness["local_asset_exists"].sum())
    try:
        _ensure_historical_asset_smoke_job()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"历史 Smoke 证据暂时无法登记：{exc}")
    jobs = list_asset_smoke_jobs(include_archived=False)
    latest_job = jobs[0] if jobs else None
    summary = dict(latest_job.get("summary", {})) if latest_job else {}
    _, legacy_results = latest_smoke_run()
    evidence = checkpoint_smoke_evidence()
    results = pd.DataFrame(
        [entry["latest"] for entry in evidence.values() if entry.get("latest")]
    )
    if results.empty:
        results = legacy_results
    asset_probe_passed = (
        int(results.get("asset_probe_passed", pd.Series(dtype=bool)).fillna(False).sum())
        if not results.empty
        else 0
    )
    runtime_smoke_passed = (
        int(results.get("runtime_smoke_passed", pd.Series(dtype=bool)).fillna(False).sum())
        if not results.empty
        else 0
    )
    resource_blocked = int(
        results.get("status", pd.Series(dtype=str))
        .isin(["resource_blocked", "skipped"])
        .sum()
    )
    failed_count = int(
        results.get("status", pd.Series(dtype=str)).isin(["failed", "timeout"]).sum()
    )
    waiting_count = max(0, local_count - runtime_smoke_passed - resource_blocked - failed_count)
    first_row = st.columns(4, gap="small")
    first_row[0].metric("Registry 登记", f"{len(readiness):,}")
    first_row[1].metric("本机资产", f"{local_count:,}")
    first_row[2].metric("资产探测通过", f"{asset_probe_passed:,}")
    first_row[3].metric("完整 runtime smoke", f"{runtime_smoke_passed:,}")
    second_row = st.columns(3, gap="small")
    second_row[0].metric("等待官方契约或 Adapter", f"{waiting_count:,}")
    second_row[1].metric("资源阻塞", f"{resource_blocked:,}")
    second_row[2].metric("运行失败", f"{failed_count:,}")
    st.caption(
        f"OphBench Registry 全量登记 {len(readiness)} 个 checkpoint，其中包含8个当前范围排除的 VisionFM；"
        "结构探测只确认文件容器可读取；完整前向还要求明确 Adapter、预处理和输出契约。"
        "两者都不会自动授予任务推理或路由资格。"
    )
    if summary:
        batch_completed = int(summary.get("completed_count", 0) or 0)
        batch_target = int(summary.get("target_count", 0) or 0)
        status_text = (
            f"最近批次 {summary.get('job_id', latest_job.get('job_id') if latest_job else '—')}："
            f"进度 {batch_completed}/{batch_target}，"
            f"资产探测通过 {summary.get('asset_probe_passed_count', 0)}，"
            f"完整 runtime smoke {summary.get('runtime_smoke_passed_count', 0)}，"
            f"失败 {summary.get('failed_count', 0)}，"
            f"资源阻塞 {summary.get('resource_blocked_count', summary.get('skipped_count', 0))}，"
            f"超时 {summary.get('timeout_count', 0)}。"
        )
        (st.success if not summary.get("failed_count") and not summary.get("timeout_count") else st.warning)(
            status_text
        )
    control_columns = st.columns([1.1, 1.5, 1, 0.8], gap="small")
    local_checkpoint_ids = readiness.loc[
        readiness["local_asset_exists"].astype(bool), "checkpoint_id"
    ].astype(str).tolist()
    selected_checkpoint = control_columns[1].selectbox(
        "选中 checkpoint",
        local_checkpoint_ids,
        key="asset_smoke_selected_checkpoint",
    )
    selected_device = control_columns[2].selectbox(
        "运行设备",
        ["auto", "cuda:0", "cpu"],
        format_func=lambda value: {
            "auto": "自动选择空闲 GPU",
            "cuda:0": "GPU 0",
            "cpu": "CPU",
        }[value],
        key="asset_smoke_device",
    )
    force = control_columns[3].toggle(
        "强制重跑", value=False, key="asset_smoke_force"
    )
    if control_columns[0].button(
        "运行全部本机资产",
        icon=":material/play_arrow:",
        width="stretch",
        key="submit_all_asset_smoke",
    ):
        try:
            job_id = submit_asset_smoke_job(
                selection_mode="all_local",
                device=selected_device,
                force=force,
            )
        except Exception as exc:
            st.error(f"Smoke 任务提交失败：{exc}")
        else:
            st.session_state["asset_smoke_job_flash"] = f"已提交模型资产 Smoke：{job_id}"
            st.session_state["pending_engineering_layer"] = "任务运行记录"
            st.session_state["pending_hub_workspace"] = "任务运行记录"
            st.rerun()
    if st.button(
        "仅运行选中模型",
        icon=":material/play_arrow:",
        key="submit_selected_asset_smoke",
    ):
        try:
            job_id = submit_asset_smoke_job(
                checkpoint_ids=[selected_checkpoint],
                selection_mode="selected",
                device=selected_device,
                force=force,
            )
        except Exception as exc:
            st.error(f"Smoke 任务提交失败：{exc}")
        else:
            st.session_state["asset_smoke_job_flash"] = f"已提交模型资产 Smoke：{job_id}"
            st.session_state["pending_engineering_layer"] = "任务运行记录"
            st.session_state["pending_hub_workspace"] = "任务运行记录"
            st.rerun()
    st.caption(
        "一次批量提交只生成一条父任务；checkpoint 在父任务内独立运行。"
        "Smoke 通过不会自动授予任务推理或路由资格。"
    )
    with st.expander("查看 checkpoint 运行准备矩阵"):
        display = readiness[
            [
                "model_id",
                "checkpoint_id",
                "modalities",
                "artifact_type",
                "local_asset_exists",
                "size_matches_manifest",
                "registry_sha256_evidence",
                "adapter_implemented",
                "preprocessing_status",
                "expected_output",
                "probe_profile",
                "probe_eligible",
                "runtime_smoke_eligible",
                "blocked_reason",
            ]
        ].copy()
        display["最近状态"] = display["checkpoint_id"].map(
            lambda checkpoint_id: ASSET_SMOKE_STATUS_LABELS.get(
                str((evidence.get(str(checkpoint_id), {}).get("latest") or {}).get("status", "")),
                "尚未测试",
            )
        )
        display["最近成功证据"] = display["checkpoint_id"].map(
            lambda checkpoint_id: (
                (evidence.get(str(checkpoint_id), {}).get("latest_success") or {}).get(
                    "job_completed_at_utc", "—"
                )
            )
        )
        display["probe_profile"] = display["probe_profile"].replace(
            ASSET_PROBE_PROFILE_LABELS
        )
        display = display.rename(
            columns={
                "model_id": "模型",
                "checkpoint_id": "Checkpoint",
                "modalities": "模态",
                "artifact_type": "资产类型",
                "local_asset_exists": "本机文件",
                "size_matches_manifest": "大小一致",
                "registry_sha256_evidence": "Registry 哈希证据",
                "adapter_implemented": "Adapter",
                "preprocessing_status": "预处理核验",
                "expected_output": "预期输出",
                "probe_profile": "探测方式",
                "probe_eligible": "可结构探测",
                "runtime_smoke_eligible": "可完整前向",
                "blocked_reason": "尚缺条件",
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
        if not results.empty:
            result_columns = [
                "checkpoint_id",
                "probe_profile",
                "status",
                "achieved_stage",
                "runtime_smoke_eligible",
                "asset_probe_passed",
                "runtime_smoke_passed",
            ]
            result_display = results.reindex(columns=result_columns).copy()
            result_display["probe_profile"] = result_display["probe_profile"].replace(
                ASSET_PROBE_PROFILE_LABELS
            )
            st.markdown("##### 最近一次运行结果")
            st.dataframe(result_display, hide_index=True, width="stretch")


def _role_text(value: object) -> str:
    roles = str(value or "").split("|")
    labels = []
    if any(role in {"scout", "adapter_scout"} for role in roles):
        labels.append("路由候选")
    if any(role in {"expert", "legacy_expert"} for role in roles):
        labels.append("专家候选")
    return " / ".join(labels) or "角色待确认"


def _task_selector(models: pd.DataFrame, key: str) -> str:
    task_ids = sorted(models["task_id"].dropna().astype(str).unique())
    return st.selectbox("任务与数据集", task_ids, format_func=task_label, key=key)


def _load_training_recipes() -> pd.DataFrame:
    rows = []
    for payload in discover_training_recipes(TRAINING_RECIPE_ROOT):
        metadata = payload["recipe"]
        rows.append(
            {
                "recipe_id": metadata["recipe_id"],
                "display_name": metadata.get("display_name", metadata["recipe_id"]),
                "recipe_kind": metadata.get("recipe_kind", "engineering"),
                "evaluation_role": metadata.get("evaluation_role", "engineering_screening"),
                "trainer_adapter": metadata["trainer_adapter"],
                "supported_model_families": metadata["supported_model_families"],
                "description": metadata.get("description", ""),
                "enabled": 1,
                "payload": payload,
            }
        )
    return pd.DataFrame(rows)


def official_profile_for_model(model_family: str, architecture: str) -> dict | None:
    profiles = discover_official_recipe_profiles(OFFICIAL_RECIPE_PROFILE_ROOT)
    matches = match_official_recipe_profiles(profiles, model_family, architecture)
    if not matches:
        return None
    if len(matches) > 1:
        profile_ids = [item["profile"]["profile_id"] for item in matches]
        raise TrainingConfigError(f"同一 architecture 匹配到多个官方协议档案：{profile_ids}")
    payload = dict(matches[0])
    payload["trials"] = expand_official_recipe_trials(matches[0])
    return payload


def _load_task_registry() -> pd.DataFrame:
    if not TASK_REGISTRY.is_file():
        return pd.DataFrame()
    return pd.read_csv(TASK_REGISTRY)


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_parameter_count(value: object) -> str:
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        return "未登记"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f} M"
    if count >= 1_000:
        return f"{count / 1_000:.1f} K"
    return str(count)


@st.cache_data(show_spinner=False)
def _registered_architecture_parameter_count(
    model_family: str,
    architecture: str,
    num_classes: int,
) -> int | None:
    if str(model_family).lower() not in {"convnext", "swin", "vit"} or not architecture:
        return None
    try:
        import timm

        model = timm.create_model(str(architecture), pretrained=False, num_classes=int(num_classes))
    except Exception:
        return None
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _model_parameter_count(row: pd.Series) -> int | None:
    recorded = pd.to_numeric(pd.Series([row.get("parameter_count")]), errors="coerce").iloc[0]
    if pd.notna(recorded):
        return int(recorded)
    return _registered_architecture_parameter_count(
        str(row.get("model_family", "")),
        str(row.get("architecture", "")),
        int(pd.to_numeric(pd.Series([row.get("n_classes")]), errors="coerce").fillna(0).iloc[0]),
    )


def _checkpoint_size_text(row: pd.Series) -> str:
    recorded = pd.to_numeric(pd.Series([row.get("checkpoint_mb")]), errors="coerce").iloc[0]
    if pd.notna(recorded):
        return _format_file_size(float(recorded) * 1024 * 1024)
    path = Path(str(row.get("checkpoint_path", "") or ""))
    if path.is_file():
        return _format_file_size(path.stat().st_size)
    return "未登记"


def _format_file_size(size_bytes: float | int) -> str:
    size = float(size_bytes)
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def _compact_number(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "尚未登记"
    return str(int(number)) if float(number).is_integer() else str(number)


def _runtime_architecture(row: pd.Series) -> str:
    if str(row.get("model_family", "")).lower() == "retfound":
        return "ViT-Large/16 图像编码器"
    return architecture_display(row.get("architecture"))


def _input_size_display(value: object) -> str:
    return ui_value(value, empty="").replace("x", " × ", 1)


def training_capability(row: pd.Series, recipes: pd.DataFrame) -> tuple[bool, str]:
    if recipes.empty:
        return False, "训练 recipe registry 缺失"
    enabled = recipes.loc[recipes["enabled"].map(_enabled)].copy()
    if "supported_model_families" in enabled.columns:
        family = str(row.get("model_family", ""))
        matches = enabled.loc[
            enabled["supported_model_families"].map(
                lambda value: family in value if isinstance(value, list) else family in str(value).split("|")
            )
        ]
    else:
        matches = enabled.loc[
            enabled["model_family"].astype(str).eq(str(row.get("model_family", "")))
            & enabled["architecture"].astype(str).eq(str(row.get("architecture", "")))
        ]
    if matches.empty:
        return False, "缺少已验证的训练 recipe / Loader"
    return True, "可提交受控微调任务"


def filter_global_model_catalog(
    catalog: pd.DataFrame,
    *,
    status: str | None = None,
    family: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    filtered = catalog.copy()
    if status:
        filtered = filtered.loc[filtered["target_task_status"].astype(str).eq(str(status))]
    if family:
        filtered = filtered.loc[filtered["model_family"].astype(str).eq(str(family))]
    if provider:
        filtered = filtered.loc[filtered["provider_id"].astype(str).eq(str(provider))]
    return filtered.reset_index(drop=True)


def partition_model_catalog(catalog: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按实际可执行层级拆分目录；候选基础模型永不进入任务模型视图。"""

    task_ready = catalog["task_checkpoint"].astype(bool) & catalog[
        "task_inference_ready"
    ].astype(bool)
    replay_ready = (
        catalog["task_checkpoint"].astype(bool)
        & catalog.get("replay_eligible", pd.Series(False, index=catalog.index)).astype(bool)
        & ~task_ready
    )
    adaptable = (
        ~catalog["task_checkpoint"].astype(bool)
        & catalog["base_adapter_ready"].astype(bool)
        & ~catalog["task_inference_ready"].astype(bool)
    )
    return {
        "在线可用任务模型": catalog.loc[task_ready].reset_index(drop=True),
        "离线预测回放资产": catalog.loc[replay_ready].reset_index(drop=True),
        "可适配基础模型": catalog.loc[adaptable].reset_index(drop=True),
        "候选基础模型库": catalog.loc[
            ~catalog["task_checkpoint"].astype(bool) & ~adaptable
        ].reset_index(drop=True),
    }


TASK_MODALITY_BY_DATASET = {
    "APTOS2019": "CFP",
    "Glaucoma_fundus": "CFP",
}


def current_task_candidate_catalog(
    catalog: pd.DataFrame,
    *,
    dataset_id: str,
) -> tuple[pd.DataFrame, str | None]:
    """Hide upstream scope exclusions and assets incompatible with the task input."""

    target_modality = TASK_MODALITY_BY_DATASET.get(str(dataset_id))
    external = catalog["provider_id"].astype(str).eq("ophbench")
    eligible_external = external & catalog["download_status"].astype(str).eq(
        "downloaded"
    )
    eligible_external &= ~catalog["artifact_type"].astype(str).eq("generative_model")
    if target_modality:
        eligible_external &= catalog["modalities"].fillna("").astype(str).map(
            lambda value: target_modality in value.split("|")
        )
    visible = catalog.loc[~external | eligible_external].reset_index(drop=True)
    return visible, target_modality


def ui_value(value: object, *, empty: str = "尚未登记") -> str:
    """禁止把 pandas/内部状态的缺失标记直接暴露给用户。"""

    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return empty
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "missing", "unknown"}:
        return empty
    return text


def source_access_display(row: pd.Series) -> str:
    status = str(row.get("source_access_status", ""))
    return {
        "open": "开放获取",
        "authentication_required": "需要认证或授权",
        "weights_missing": "尚未登记可用权重",
        "unavailable": "暂不可用",
    }.get(status, "尚未核验")


def artifact_type_display(row: pd.Series) -> str:
    artifact_type = str(row.get("artifact_type", ""))
    labels = {
        "foundation_encoder": "基础编码器",
        "vision_language_model": "视觉语言模型",
        "task_checkpoint": "任务 checkpoint",
        "generative_model": "生成模型",
        "ablation_checkpoint": "消融 checkpoint",
        "multimodal_full_model": "完整多模态模型",
    }
    label = labels.get(artifact_type, "资产类型待核验")
    if str(row.get("source_model_id", "")) == "fmue":
        return f"{label}（16 类 OCT）"
    return label


def verification_evidence_display(value: object) -> str:
    return {
        "verified": "已核验",
        "pending": "待核验",
        "blocked": "存在限制",
        "not_applicable": "不适用",
    }.get(str(value), "未报告")


def upstream_download_display(value: object) -> str:
    return {
        "downloaded": "OphBench 建库时已下载",
        "excluded_by_project_scope": "项目范围排除重复下载",
    }.get(str(value), "上游未报告")


def integrity_display(value: object) -> str:
    return {
        "local_size_sha256_and_non_html_verified": "OphBench 大小与 SHA256 已核验",
        "not_applicable": "不适用",
        "provider_sha256_matched": "官方 SHA256 已匹配",
        "provider_revision_and_checksum_unavailable": "官方未提供版本校验值",
        "provider_checksum_not_recorded": "官方校验值未登记",
        "not_checked": "未核验",
    }.get(str(value), ui_value(value, empty="未报告"))


def readiness_value(passed: bool, passed_text: str, pending_text: str) -> str:
    return passed_text if passed else pending_text


def _detail_section(title: str, cells: list[tuple[str, str]]) -> str:
    content = "".join(
        '<div class="model-detail-cell">'
        f'<span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>'
        for label, value in cells
    )
    return (
        f'<section class="model-readiness-section"><h4>{html.escape(title)}</h4>'
        f'<div class="model-detail-grid">{content}</div></section>'
    )


def adapter_status_display(value: object) -> str:
    return {
        "not_implemented": "Adapter 尚未实现",
        "implemented_not_tested": "Adapter 已实现，尚未通过 smoke",
        "smoke_test_passed": "基础 Adapter 已通过 smoke",
        "failed": "Adapter 验证失败",
    }.get(str(value), "尚未核验")


def architecture_display(value: object) -> str:
    text = ui_value(value, empty="架构尚未登记")
    return text.replace("ResNet-50图像", "ResNet-50 图像").replace(
        "BioClinicalBERT文本", "BioClinicalBERT 文本"
    )


TARGET_STATUS_LABELS = {
    "direct_inference": ("当前任务可直接推理", "badge-live"),
    "offline_replay": ("当前任务仅离线回放", "badge-replay"),
    "adaptable": ("可适配当前任务", "badge-wait"),
    "blocked": ("当前任务不可接入", "badge-wait"),
}


def apply_pending_engineering_navigation(state: MutableMapping[str, object]) -> None:
    pending = state.pop("pending_engineering_layer", None)
    if pending:
        state["engineering_layer"] = pending


def format_global_scan_preview(preview: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in GLOBAL_SCAN_PREVIEW_RENAME if column in preview.columns]
    table = preview[columns].rename(columns=GLOBAL_SCAN_PREVIEW_RENAME).copy()
    if "路由模型" in table:
        table["路由模型"] = table["路由模型"].map(
            lambda value: " + ".join(human_model(item) for item in str(value).split("|") if item)
        )
    if "专家模型" in table:
        table["专家模型"] = table["专家模型"].map(
            lambda value: " + ".join(human_model(item) for item in str(value).split("|") if item)
        )
    if "路由机制" in table:
        table["路由机制"] = table["路由机制"].map(lambda value: GLOBAL_SCAN_POLICY_LABELS.get(str(value), str(value)))
    return table


def split_training_log_messages(log_text: str) -> tuple[str, list[str]]:
    visible_lines: list[str] = []
    hidden_warning_labels: list[str] = []
    for line in log_text.splitlines():
        matched_label = next(
            (label for marker, label in KNOWN_TRAINING_LOG_WARNINGS.items() if marker in line),
            None,
        )
        if matched_label:
            hidden_warning_labels.append(matched_label)
            continue
        visible_lines.append(line)
    return "\n".join(visible_lines).strip(), sorted(set(hidden_warning_labels))


def _catalog_summary_html(catalog: pd.DataFrame) -> str:
    status_counts = catalog["target_task_status"].astype(str).value_counts().to_dict()
    cards = [
        ("可直接推理", int(status_counts.get("direct_inference", 0)), "当前任务在线链"),
        ("仅离线回放", int(status_counts.get("offline_replay", 0)), "已有冻结输出"),
        ("可适配", int(status_counts.get("adaptable", 0)), "需受控训练"),
        ("不可接入", int(status_counts.get("blocked", 0)), "缺少协议或产物"),
    ]
    return (
        '<div class="hub-mini-strip">'
        + "".join(
            '<div class="hub-mini-stat">'
            f'<span>{html.escape(label)}</span><b>{value}</b><small>{html.escape(note)}</small>'
            "</div>"
            for label, value, note in cards
        )
        + "</div>"
    )


def _render_model_entry(row: pd.Series) -> None:
    artifact_id = str(row["artifact_id"])
    is_candidate = not bool(row.get("task_checkpoint", False))
    display_name = ui_value(row.get("model_name"), empty=human_model(artifact_id))
    model_id = str(row.get("model_id", f"{row.get('task_id')}::{artifact_id}"))
    status, css_class = TARGET_STATUS_LABELS.get(
        str(row.get("target_task_status", "blocked")),
        ("当前任务状态待核验", "badge-wait"),
    )
    selected = st.session_state.get("selected_model_id") == model_id
    if bool(row.get("route_eligible", False)):
        role_text = _role_text(row.get("role_candidates"))
    elif bool(row.get("replay_eligible", False)):
        role_text = "仅离线回放"
    else:
        role_text = "不可路由"
    with st.container(border=True):
        columns = st.columns([1.35, 1.05, 1.35, 0.7])
        with columns[0]:
            st.markdown(
                f'<span class="model-entry-label">模型资产</span>'
                f'<span class="model-entry-title">{html.escape(display_name)}</span>'
                f'<span class="model-entry-copy">{html.escape(task_label(row.get("task_id")))}</span>',
                unsafe_allow_html=True,
            )
        with columns[1]:
            parameter_count = _format_parameter_count(_model_parameter_count(row))
            st.markdown(
                f'<span class="model-entry-label">架构</span>'
                f'<span class="model-entry-title">{html.escape(architecture_display(row.get("architecture")))}</span>'
                f'<span class="model-entry-copy">{html.escape(parameter_count)}</span>',
                unsafe_allow_html=True,
            )
        with columns[2]:
            st.markdown(
                f'<span class="model-entry-label">当前任务状态</span>'
                f'<span class="model-entry-title">{html.escape(status)}</span>'
                f'<span class="badge {css_class}">{html.escape(role_text)}</span>'
                f'<span class="model-entry-copy">{html.escape(str(row.get("target_task_reason", "")))}</span>',
                unsafe_allow_html=True,
            )
        with columns[3]:
            button_label = "已打开" if selected else ("查看模型卡" if is_candidate else "选择")
            if st.button(button_label, key=f"select_model_{model_id}", disabled=selected, width="stretch"):
                st.session_state["selected_model_id"] = model_id
                st.rerun()


def _default_training_output(task_id: str, artifact_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(MODEL_HUB_ROOT / "runs" / "training" / task_id / artifact_id / stamp)


def _job_request_payload(job: dict[str, object]) -> dict[str, object]:
    request = job.get("request")
    if isinstance(request, dict):
        return request
    job_dir = str(job.get("job_dir", "")).strip()
    if not job_dir:
        return {}
    request_path = Path(job_dir) / "request.json"
    if not request_path.is_file():
        return {}
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _job_display_artifact_id(job: dict[str, object]) -> str:
    artifact_id = str(job.get("artifact_id", "")).strip()
    request = _job_request_payload(job)
    if str(request.get("initialization_source", "")).strip() == "timm_pretrained":
        try:
            return adaptation_artifact_id(
                str(request.get("source_artifact_id") or artifact_id),
                str(request.get("task_id") or request.get("target_task_id") or job.get("task_id", "")),
                initialization_source="timm_pretrained",
                architecture=str(request.get("architecture", "")),
            )
        except (KeyError, ValueError):
            return artifact_id
    return artifact_id


def _job_record_title(job: dict[str, object], label: str) -> str:
    display_id = _job_display_artifact_id(job)
    return f"{job['job_id']} · {human_model(display_id)} · {label}"


def _request_fingerprint(request: TrainingRequest) -> str:
    return json.dumps(asdict(request), ensure_ascii=False, sort_keys=True)


def _config_diff_paths(base: object, submitted: object, prefix: str = "") -> list[str]:
    if isinstance(base, dict) and isinstance(submitted, dict):
        paths = []
        for key in sorted(set(base) | set(submitted)):
            child = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_config_diff_paths(base.get(key), submitted.get(key), child))
        return paths
    return [] if base == submitted else [prefix]


def _render_training_wizard(
    row: pd.Series,
    recipes: pd.DataFrame,
    *,
    target_task_id: str,
) -> None:
    source_artifact_id = str(row["artifact_id"])
    supported, reason = training_capability(row, recipes)
    with st.expander("模型微调任务提交器", expanded=True):
        st.caption("面向研发人员的配置化实验发射器：编辑 YAML、严格预检，再提交独立后台任务。")
        st.markdown("**1. 任务与数据集**")
        task_registry = _load_task_registry()
        target_rows = task_registry.loc[
            task_registry["task_id"].astype(str).eq(str(target_task_id))
        ] if not task_registry.empty else pd.DataFrame()
        if target_rows.empty:
            st.error("目标任务未登记，无法创建适配任务。")
            return
        target_task = target_rows.iloc[0]
        st.write(
            f"目标任务：{task_label(target_task_id)} · "
            f"源模型：{human_model(source_artifact_id)}（{task_label(row.get('task_id'))}）"
        )
        dataset_options = registered_dataset_options(TASK_REGISTRY, target_task_id)
        ready_datasets = dataset_options.loc[
            dataset_options["availability_status"].astype(str).eq("ready")
        ] if not dataset_options.empty else pd.DataFrame()
        if ready_datasets.empty:
            st.error("当前任务没有已登记且通过预检的 ImageFolder 数据目录。请先更新任务注册表并运行受控扫描。")
            if not dataset_options.empty:
                for _, option in dataset_options.iterrows():
                    st.caption(f"{option.get('dataset_id', '未知数据集')}：{option.get('availability_reason', '不可用')}")
            return
        dataset_ids = ready_datasets["dataset_id"].astype(str).tolist()
        selected_dataset_id = st.selectbox(
            "已登记数据集",
            dataset_ids,
            key=f"training_dataset_{target_task_id}_{source_artifact_id}",
        )
        selected_dataset = ready_datasets.loc[
            ready_datasets["dataset_id"].astype(str).eq(selected_dataset_id)
        ].iloc[0]
        data_root = str(selected_dataset["data_root"])
        st.caption(f"受控目录：{data_root}")
        st.json(
            {
                "类别映射": selected_dataset["class_to_idx"],
                "数据划分": selected_dataset["split_sizes"],
            },
            expanded=False,
        )

        st.markdown("**2. 模型与训练配置**")
        cross_task = str(row.get("task_id", "")) != str(target_task_id)
        is_retfound_probe = (
            str(row.get("provider_id", "")) == "ophbench"
            and str(row.get("source_model_id", "")) == "retfound"
            and str(row.get("source_checkpoint_id", "")) == "retfound-cfp"
        )
        checkpoint_path = str(row.get("checkpoint_path", "") or "").strip()
        if is_retfound_probe:
            checkpoint_path = str(
                PROJECT_ROOT / "models/checkpoints/retfound/RETFound_mae_natureCFP.pth"
            )
        checkpoint_available = str(row.get("checkpoint_status", "")) == "found" and bool(checkpoint_path)
        checkpoint_available = checkpoint_available or (
            is_retfound_probe and Path(checkpoint_path).is_file()
        )
        initialization_options = ["registered_checkpoint"] if is_retfound_probe else ["timm_pretrained"]
        if checkpoint_available:
            initialization_options.append("registered_checkpoint")
        initialization_labels = {
            "timm_pretrained": "原始预训练权重（推荐）",
            "registered_checkpoint": (
                "跨疾病 checkpoint 迁移（研究实验）"
                if cross_task
                else "继续训练现有 checkpoint"
            ),
        }
        initialization_source = st.selectbox(
            "训练初始化",
            initialization_options,
            format_func=lambda value: initialization_labels[value],
            help=(
                "默认从 timm 登记的原始自然图像预训练权重开始新的目标任务微调。"
                "只有明确选择现有 checkpoint 时，才继续训练已有模型或进行跨疾病研究迁移。"
            ),
            key=f"training_initialization_{target_task_id}_{source_artifact_id}",
        )
        if initialization_source == "timm_pretrained":
            st.info("本次训练从原始预训练骨干开始，不继承现有眼病分类 checkpoint。")
            st.caption(f"初始化模型：timm.create_model('{row.get('architecture')}', pretrained=True)")
            if checkpoint_available:
                st.caption(f"现有模型 checkpoint（本次不加载）：{checkpoint_path}")
        else:
            if cross_task:
                st.warning("研究性跨疾病迁移：将加载源眼病 checkpoint，再替换为目标任务分类头。")
            else:
                st.warning("继续训练模式：将加载现有同任务 checkpoint，不属于独立的全新微调基线。")
            st.caption(f"本次加载 checkpoint：{checkpoint_path}")
        family = str(row.get("model_family", ""))
        architecture = str(row.get("architecture", ""))
        matching = recipes.loc[
            recipes["enabled"].map(_enabled)
            & recipes["supported_model_families"].map(
                lambda value: family in value if isinstance(value, list) else family in str(value).split("|")
            )
        ] if not recipes.empty else pd.DataFrame()
        official_profile = official_profile_for_model(family, architecture)
        mode_options = ["工程链路验证"]
        if official_profile is not None:
            mode_options.append("科研候选实验")
        experiment_mode = st.segmented_control(
            "训练用途",
            mode_options,
            default=mode_options[0],
            key=f"training_mode_{target_task_id}_{source_artifact_id}",
        )

        recipe_id = ""
        base_recipe: dict = {}
        if experiment_mode == "科研候选实验" and official_profile is not None:
            profile = official_profile["profile"]
            search_plan = official_profile["search_plan"]
            translation = profile["translation"]
            source = profile["source"]
            st.markdown(f"**{profile['display_name']}**")
            st.caption(
                f"来源范围：{source.get('scope', '未记录')} · "
                f"转译状态：{translation.get('status', '未记录')}"
            )
            st.markdown(
                f"[官方仓库]({source.get('url')}) · "
                f"[原始配置/说明]({source.get('config_url')})"
            )
            st.info(str(translation.get("note", "未记录协议转译边界")))
            unsupported_fields = translation.get("unsupported_fields", [])
            if unsupported_fields:
                st.caption("当前 trainer 未逐项对齐：" + "、".join(map(str, unsupported_fields)))

            trials = official_profile["trials"]
            trial_frame = pd.DataFrame(trials).rename(
                columns={
                    "trial_id": "试验",
                    "optimizer.learning_rate": "Learning Rate",
                    "optimizer.weight_decay": "Weight Decay",
                    "selection_split": "选择数据划分",
                    "final_seed_count": "入围后种子数",
                }
            )
            st.dataframe(
                trial_frame[["试验", "Learning Rate", "Weight Decay", "选择数据划分"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"{search_plan.get('display_name')}：{len(trials)} 次验证集 trial；"
                f"入围配置再跑 {len(search_plan.get('final_seeds', []))} 个种子，"
                f"合计 {len(trials) + len(search_plan.get('final_seeds', []))} 个训练任务。"
            )
            st.caption("当前固定预算仅搜索 Learning Rate 与 Weight Decay；其余可执行字段保持该模型协议锚点配置不变。")
            trial_ids = [item["trial_id"] for item in trials]
            trial_id = st.selectbox(
                "当前提交的 trial",
                trial_ids,
                key=f"official_trial_{target_task_id}_{source_artifact_id}",
            )
            base_recipe_id = str(translation.get("base_recipe_id", ""))
            base_match = matching.loc[matching["recipe_id"].astype(str).eq(base_recipe_id)]
            if str(translation.get("status")) != "task_adapted_subset":
                supported = False
                reason = "官方协议与当前 trainer 尚未对齐，本页只展示档案和候选网格，不允许冒充官方复现提交。"
            elif base_match.empty:
                supported = False
                reason = f"缺少官方协议转译所需的基础 recipe：{base_recipe_id}"
            else:
                try:
                    base_recipe = build_official_trial_recipe(
                        base_match.iloc[0]["payload"], official_profile, trial_id
                    )
                except TrainingConfigError as exc:
                    supported = False
                    reason = str(exc)
                else:
                    recipe_id = str(base_recipe["recipe"]["recipe_id"])
                    with st.expander("查看当前 trial 的可执行转译配置"):
                        st.code(
                            yaml.safe_dump(base_recipe, allow_unicode=True, sort_keys=False),
                            language="yaml",
                        )
                    supported = False
                    reason = (
                        "当前 v0.8.6 训练器在训练后会读取 test split。"
                        "为避免用测试集选参，科研 trial 本版只登记和预览；"
                        "待 val-only search runner 与 frozen-test evaluator 分离后再开放提交。"
                    )
            st.caption("六次 trial 属于同一个模型×数据集实验系列，不会注册为六个永久模板。")
        else:
            engineering = matching.loc[matching["recipe_kind"].astype(str).eq("engineering")].copy()
            recipe_ids = engineering["recipe_id"].astype(str).tolist() if not engineering.empty else []
            recipe_display_names = (
                engineering.set_index("recipe_id")["display_name"].astype(str).to_dict()
                if not engineering.empty
                else {}
            )
            recipe_id = st.selectbox(
                "工程配置模板",
                recipe_ids or ["无可用 recipe"],
                format_func=lambda value: (
                    f"{recipe_display_names.get(value, value)}（{value}）"
                    if value in recipe_display_names
                    else value
                ),
                key=f"training_recipe_{target_task_id}_{source_artifact_id}",
                disabled=not supported,
            )
            if recipe_id in recipe_ids:
                selected_recipe = engineering.loc[
                    engineering["recipe_id"].astype(str).eq(str(recipe_id))
                ].iloc[0]
                base_recipe = selected_recipe["payload"]
                st.caption(
                    f"当前模板：{selected_recipe.get('display_name', recipe_id)}（{recipe_id}） · "
                    f"{selected_recipe.get('description', '未填写用途说明')}"
                )
            st.warning("工程模板用于链路验证、迁移探针和统一初筛，不代表模型的官方协议或性能上限。")
            with st.popover("工程模板说明"):
                for _, recipe_row in engineering.sort_values("recipe_id").iterrows():
                    st.markdown(
                        f"**{recipe_row.get('display_name', recipe_row['recipe_id'])}** "
                        f"（`{recipe_row['recipe_id']}`）：{recipe_row.get('description', '未填写用途说明')}"
                    )
        if not supported:
            st.warning(reason)
            return
        target_artifact_id = (
            STANDARD_ARTIFACT_ID
            if is_retfound_probe
            else adaptation_artifact_id(
                source_artifact_id,
                target_task_id,
                initialization_source=initialization_source,
                architecture=str(row.get("architecture", "")),
            )
        )
        if initialization_source == "timm_pretrained" and target_artifact_id != f"{source_artifact_id}__{target_task_id}__adapter":
            st.caption(
                f"新模型标识：{target_artifact_id}。当前从 timm 原始预训练权重初始化，"
                "不会把旧眼病 adapter 名称写入新模型标识。"
            )
        output_key = f"training_output_path_{target_task_id}_{source_artifact_id}"
        if output_key not in st.session_state:
            st.session_state[output_key] = _default_training_output(target_task_id, target_artifact_id)
        output_dir = str(st.session_state[output_key])
        request_row = row.copy()
        if is_retfound_probe:
            request_row["checkpoint_path"] = checkpoint_path
            request_row["checkpoint_status"] = "found"
            request_row["n_classes"] = 0
        base_request = build_adaptation_request(
            request_row,
            target_task,
            data_root=data_root,
            recipe_id=recipe_id,
            output_dir=output_dir,
            initialization_source=initialization_source,
        )
        trainer_adapter = str(base_recipe.get("recipe", {}).get("trainer_adapter", ""))
        if is_retfound_probe:
            base_request = replace(
                base_request,
                artifact_id=STANDARD_ARTIFACT_ID,
                output_dir=output_dir,
                trainer_adapter=trainer_adapter,
                source_checkpoint_path=checkpoint_path,
                source_num_classes=0,
                base_model_provider="ophbench",
                base_model_id="retfound",
                base_checkpoint_id="retfound-cfp",
                encoder_checkpoint_sha256=str(row.get("sha256", "")),
            )
        inspection = DatasetInspection(
            class_to_idx=dict(selected_dataset["class_to_idx"]),
            split_sizes=dict(selected_dataset["split_sizes"]),
        )
        context = build_training_context(base_request, inspection)
        draft = build_training_draft(base_recipe, context) if base_recipe else {}
        editor_key = f"training_yaml_{target_task_id}_{source_artifact_id}"
        editor_recipe_key = f"training_yaml_recipe_{target_task_id}_{source_artifact_id}"
        if st.session_state.get(editor_recipe_key) != recipe_id:
            st.session_state[editor_recipe_key] = recipe_id
            st.session_state[editor_key] = yaml.safe_dump(draft, allow_unicode=True, sort_keys=False)
        yaml_text = st.text_area(
            "完整训练配置（YAML）",
            height=620,
            key=editor_key,
            help="任务、数据、类别映射、输出目录和跨任务初始化为锁定字段；训练行为可自由编辑。",
        )
        try:
            submitted_config = parse_yaml_text(yaml_text)
            yaml_error = ""
        except TrainingConfigError as exc:
            submitted_config = {}
            yaml_error = str(exc)
            st.error(yaml_error)
        request = replace(
            base_request,
            base_recipe=base_recipe,
            submitted_config=submitted_config,
            trainer_adapter=trainer_adapter,
        )
        changed_paths = _config_diff_paths(draft, submitted_config) if submitted_config else []
        with st.expander(f"配置差异 · {len(changed_paths)} 项"):
            st.write(changed_paths or ["当前配置与模板一致"])

        action_cols = st.columns([1.2, 1.2, 1.6])
        new_recipe_id = action_cols[0].text_input(
            "新 recipe ID",
            placeholder="例如 glaucoma_convnext_v1",
            key=f"new_recipe_{target_task_id}_{source_artifact_id}",
        )
        if action_cols[1].button(
            "保存为新 recipe",
            disabled=bool(yaml_error or not new_recipe_id.strip()),
            key=f"save_recipe_{target_task_id}_{source_artifact_id}",
        ):
            try:
                validate_training_request(request, LEGACY_TRAINING_RECIPES)
                saved_recipe = save_training_recipe(TRAINING_RECIPE_ROOT, submitted_config, new_recipe_id)
            except Exception as exc:
                st.error(f"保存失败：{exc}")
            else:
                st.success(f"已保存：{saved_recipe}")
        effective_path = Path(output_dir) / "configs" / "effective_config.yaml"
        trainer_script = (
            "scripts/training/train_ophbench_retfound_linear_probe.py"
            if trainer_adapter == "ophbench_retfound_linear_probe_v1"
            else "scripts/training/train_timm_classifier.py"
        )
        export_command = (
            "/data/conda_envs/ophagent/bin/python "
            f"{trainer_script} "
            f"--config {effective_path}"
        )
        with action_cols[2].expander("导出命令"):
            st.code(export_command, language="bash")

        st.markdown("**3. 配置预检**")
        st.caption(f"统一运行目录：{output_dir}")
        fingerprint = _request_fingerprint(request)
        preflight_key = f"training_preflight_{target_task_id}_{source_artifact_id}"
        if st.button(
            "执行配置预检",
            key=f"preflight_{target_task_id}_{source_artifact_id}",
            disabled=not supported or bool(yaml_error),
        ):
            try:
                preflight = validate_training_request(request, LEGACY_TRAINING_RECIPES)
            except Exception as exc:
                st.session_state[preflight_key] = {"ok": False, "fingerprint": fingerprint, "message": str(exc)}
            else:
                st.session_state[preflight_key] = {
                    "ok": True,
                    "fingerprint": fingerprint,
                    "generated_config": preflight.generated_config,
                    "validation_report": preflight.validation_report,
                    "message": (
                        f"预检通过：{len(preflight.inspection.class_to_idx)} 类，"
                        f"train/val/test = {preflight.inspection.split_sizes}"
                    ),
                }
        preflight_state = st.session_state.get(preflight_key, {})
        current_preflight_ok = bool(
            preflight_state.get("ok") and preflight_state.get("fingerprint") == fingerprint
        )
        if preflight_state.get("message"):
            (st.success if current_preflight_ok else st.error)(preflight_state["message"])
        if current_preflight_ok:
            st.json(preflight_state["validation_report"], expanded=False)
        st.caption(f"提交时统一保存：{output_dir}/configs/base_recipe.yaml、submitted_config.yaml、effective_config.yaml 和 validation_report.json")

        st.markdown("**4. 人工确认与提交**")
        confirmed = st.checkbox(
            "我已确认数据目录、类别映射、初始化来源、YAML 配置和输出目录",
            key=f"training_confirm_{target_task_id}_{source_artifact_id}",
        )
        if st.button(
            "提交后台任务",
            icon=":material/play_arrow:",
            key=f"submit_training_{target_task_id}_{source_artifact_id}",
            disabled=not (supported and current_preflight_ok and confirmed),
        ):
            try:
                job_id = submit_training_job(request, TRAINING_JOBS_ROOT, LEGACY_TRAINING_RECIPES)
            except Exception as exc:
                st.error(f"任务提交失败：{exc}")
            else:
                st.session_state["training_job_flash"] = f"已提交后台任务：{job_id}"
                st.session_state["pending_engineering_layer"] = "任务运行记录"
                st.session_state["pending_hub_workspace"] = "任务运行记录"
                st.rerun()


def _family_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _checkpoint_task_evidence(
    checkpoints: pd.DataFrame,
    task_assets: pd.DataFrame,
) -> pd.DataFrame:
    evidence = checkpoints.copy()
    if evidence.empty:
        return evidence
    if task_assets.empty:
        evidence["task_asset_count"] = 0
        evidence["task_count"] = 0
        evidence["offline_batch_count"] = 0
        return evidence
    task_assets = task_assets.copy()
    task_assets["family_key"] = task_assets["model_family"].map(_family_key)
    family_summary = (
        task_assets.groupby("family_key", as_index=False)
        .agg(
            task_asset_count=("artifact_id", "size"),
            task_count=("task_id", "nunique"),
            offline_batch_count=("offline_batch_inference_ready", "sum"),
        )
        .set_index("family_key")
    )
    evidence["family_key"] = evidence["model_id"].map(_family_key)
    for column in ("task_asset_count", "task_count", "offline_batch_count"):
        evidence[column] = (
            evidence["family_key"].map(family_summary[column]).fillna(0).astype(int)
        )
    return evidence


def _render_unified_asset_catalog(hub_index: dict[str, pd.DataFrame]) -> None:
    checkpoints = _checkpoint_task_evidence(
        hub_index["checkpoints"],
        hub_index["task_assets"],
    )
    if checkpoints.empty:
        st.error("OphBench checkpoint 注册表当前不可读取。")
        return
    runtime_passed = checkpoints["runtime_smoke_passed"].astype(bool)
    blocked = checkpoints["smoke_status"].astype(str).eq("resource_blocked")
    summary = [
        ("基础 Checkpoint", len(checkpoints), "OphBench 完整登记"),
        ("完整前向通过", int(runtime_passed.sum()), "Checkpoint 级 Runtime Smoke"),
        ("任务预测关联", int((checkpoints["task_asset_count"] > 0).sum()), "至少一个离线任务资产"),
        ("资源阻塞", int(blocked.sum()), "不等同于运行代码失败"),
    ]
    st.markdown(
        '<div class="hub-mini-strip">'
        + "".join(
            '<div class="hub-mini-stat">'
            f"<span>{html.escape(label)}</span><b>{value}</b>"
            f"<small>{html.escape(note)}</small></div>"
            for label, value, note in summary
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "这里分别展示基础资产、Runtime Smoke、任务 prediction、离线批量链和正式路由资格；"
        "任一状态通过都不会自动推出下一层。"
    )
    filter_columns = st.columns([1, 1, 1.2])
    modality_options = sorted(
        {
            modality
            for value in checkpoints["modalities_text"].astype(str)
            for modality in value.split(" / ")
            if modality
        }
    )
    modality = filter_columns[0].selectbox(
        "输入模态",
        [""] + modality_options,
        format_func=lambda value: "全部模态" if not value else value,
        key="unified_asset_modality",
    )
    smoke_status = filter_columns[1].selectbox(
        "Runtime 状态",
        ["", "runtime_smoke_passed", "resource_blocked", "not_smoked"],
        format_func=lambda value: {
            "": "全部 Runtime 状态",
            "runtime_smoke_passed": "完整前向通过",
            "resource_blocked": "资源阻塞",
            "not_smoked": "尚未 Smoke",
        }[value],
        key="unified_asset_smoke",
    )
    scope = filter_columns[2].segmented_control(
        "资产范围",
        ["当前 CFP 研究相关", "完整 27 Checkpoint"],
        default="当前 CFP 研究相关",
        key="unified_asset_scope",
    )
    visible = checkpoints.copy()
    if modality:
        visible = visible.loc[
            visible["modalities_text"].astype(str).str.split(" / ").map(
                lambda values: modality in values
            )
        ]
    if smoke_status:
        visible = visible.loc[visible["smoke_status"].astype(str).eq(smoke_status)]
    if scope == "当前 CFP 研究相关":
        visible = visible.loc[
            visible["modalities_text"].astype(str).str.split(" / ").map(
                lambda values: "CFP" in values
            )
            & ~visible["model_id"].astype(str).eq("visionfm")
            & ~visible["artifact_type"].astype(str).eq("generative_model")
        ]
    table = visible[
        [
            "model_name",
            "checkpoint_name",
            "modalities_text",
            "artifact_type",
            "smoke_status",
            "task_asset_count",
            "task_count",
            "offline_batch_count",
            "route_eligible",
        ]
    ].rename(
        columns={
            "model_name": "模型",
            "checkpoint_name": "Checkpoint",
            "modalities_text": "模态",
            "artifact_type": "资产类型",
            "smoke_status": "Runtime Smoke",
            "task_asset_count": "任务预测资产",
            "task_count": "任务数",
            "offline_batch_count": "离线批量链",
            "route_eligible": "正式路由资格",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(620, 38 * (len(table) + 1)),
        column_config={
            "正式路由资格": st.column_config.CheckboxColumn(),
            "任务预测资产": st.column_config.NumberColumn(format="%d"),
            "任务数": st.column_config.NumberColumn(format="%d"),
            "离线批量链": st.column_config.NumberColumn(format="%d"),
        },
    )
    if visible.empty:
        return
    options = visible["checkpoint_id"].astype(str).tolist()
    selected = st.selectbox(
        "查看 Checkpoint 证据",
        options,
        format_func=lambda checkpoint_id: (
            f"{visible.loc[visible['checkpoint_id'].astype(str).eq(checkpoint_id), 'model_name'].iloc[0]}"
            f" · {checkpoint_id}"
        ),
        key="unified_asset_selected",
    )
    row = visible.loc[visible["checkpoint_id"].astype(str).eq(selected)].iloc[0]
    task_rows = hub_index["task_assets"].loc[
        hub_index["task_assets"]["model_family"].map(_family_key).eq(
            _family_key(row["model_id"])
        )
    ]
    evidence_columns = st.columns(4, gap="small")
    evidence_columns[0].markdown(
        f"**资产来源**\n\n{row['checkpoint_provider'] or row['provider_id']}  \n"
        f"{row['access_type'] or '访问方式未登记'} · {row['license'] or '许可证待核验'}"
    )
    evidence_columns[1].markdown(
        "**运行证据**\n\n"
        + (
            "完整模型加载与前向通过"
            if bool(row["runtime_smoke_passed"])
            else (
                f"资源阻塞：{row['smoke_blocker'] or '依赖资源不完整'}"
                if row["smoke_status"] == "resource_blocked"
                else "尚无 Runtime Smoke 证据"
            )
        )
    )
    evidence_columns[2].markdown(
        f"**任务证据**\n\n{int(row['task_asset_count'])} 条 prediction asset  \n"
        f"{int(row['task_count'])} 个任务 · {int(row['offline_batch_count'])} 条离线批量链"
    )
    evidence_columns[3].markdown(
        "**中转台资格**\n\n"
        f"单病例原图：{'已登记' if bool(row.get('task_inference_ready', False)) else '未登记'}  \n"
        f"正式路由：{'可用' if bool(row['route_eligible']) else '未授予'}"
    )
    if not task_rows.empty:
        linked = task_rows[
            [
                "dataset_label",
                "task_label",
                "artifact_id",
                "qualification_status",
                "cost_status",
                "route_eligible",
            ]
        ].rename(
            columns={
                "dataset_label": "数据集 / 实验",
                "task_label": "任务",
                "artifact_id": "任务模型",
                "qualification_status": "评测资格",
                "cost_status": "成本证据",
                "route_eligible": "正式路由资格",
            }
        )
        st.dataframe(linked, hide_index=True, width="stretch")


def _render_unified_task_catalog(hub_index: dict[str, pd.DataFrame]) -> None:
    datasets = hub_index["datasets"]
    task_assets = hub_index["task_assets"]
    route_runs = hub_index["route_runs"]
    if datasets.empty:
        st.info("尚无正式任务与数据集记录。")
        return
    labels = datasets["dataset_label"].astype(str).tolist()
    selected_label = st.selectbox(
        "任务与数据集",
        labels,
        key="unified_task_dataset",
    )
    dataset = datasets.loc[datasets["dataset_label"].astype(str).eq(selected_label)].iloc[0]
    task_id = str(dataset["task_id"])
    assets = task_assets.loc[task_assets["task_id"].astype(str).eq(task_id)].copy()
    runs = route_runs.loc[route_runs["task_id"].astype(str).eq(task_id)].copy()
    if int(dataset["prediction_assets"]) == 0:
        assets = task_assets.iloc[0:0].copy()
    if int(dataset["route_runs"]) == 0:
        runs = route_runs.iloc[0:0].copy()
    summary = [
        ("任务预测资产", int(dataset["prediction_assets"]), "validation / test 独立登记"),
        ("Validation 资产", int(dataset["validation_assets"]), "只用于结构与阈值选择"),
        ("路由结果包", int(dataset["route_runs"]), "历史扫描与冻结评估"),
        ("正式路由资格", int(bool(dataset["route_eligible"])), "当前仍为研究评测"),
    ]
    st.markdown(
        '<div class="hub-mini-strip">'
        + "".join(
            '<div class="hub-mini-stat">'
            f"<span>{html.escape(label)}</span><b>{value}</b>"
            f"<small>{html.escape(note)}</small></div>"
            for label, value, note in summary
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hub-band">'
        f"<strong>任务：</strong>{html.escape(str(dataset['task_label']))}　"
        f"<strong>数据集 / 口径：</strong>{html.escape(selected_label)}　"
        f"<strong>阶段：</strong>{html.escape(str(dataset['admission_status']))}<br>"
        f"{html.escape(str(dataset['qualification_note']))}</div>",
        unsafe_allow_html=True,
    )
    if assets.empty:
        st.info("该数据集已完成准入登记，但尚未生成正式任务 prediction asset。")
        return
    asset_table = assets[
        [
            "artifact_id",
            "adapter_type",
            "validation_asset_exists",
            "test_asset_exists",
            "current_run_reproducible",
            "validation_selection_eligible",
            "qualification_status",
            "forward_cost_ms_per_image",
            "cost_status",
            "route_eligible",
        ]
    ].rename(
        columns={
            "artifact_id": "任务模型",
            "adapter_type": "适配类型",
            "validation_asset_exists": "Validation",
            "test_asset_exists": "冻结结果集",
            "current_run_reproducible": "当前运行可复现",
            "validation_selection_eligible": "可参与 Validation 选择",
            "qualification_status": "资格限制",
            "forward_cost_ms_per_image": "H100 成本（ms/图）",
            "cost_status": "成本状态",
            "route_eligible": "正式路由资格",
        }
    )
    st.dataframe(
        asset_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Validation": st.column_config.CheckboxColumn(),
            "冻结结果集": st.column_config.CheckboxColumn(),
            "当前运行可复现": st.column_config.CheckboxColumn(),
            "可参与 Validation 选择": st.column_config.CheckboxColumn(),
            "正式路由资格": st.column_config.CheckboxColumn(),
            "H100 成本（ms/图）": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    if not runs.empty:
        st.markdown("#### 关联路由评测")
        run_table = runs[
            [
                "stage",
                "protocol_version",
                "model_count",
                "pairing_count",
                "result_rows",
                "status",
                "route_eligible",
            ]
        ].rename(
            columns={
                "stage": "评测阶段",
                "protocol_version": "协议",
                "model_count": "模型池",
                "pairing_count": "组合",
                "result_rows": "结果行",
                "status": "产物状态",
                "route_eligible": "正式路由资格",
            }
        )
        st.dataframe(run_table, hide_index=True, width="stretch")


def _render_unified_job_records(hub_index: dict[str, pd.DataFrame]) -> None:
    jobs = hub_index["jobs"]
    if jobs.empty:
        return
    st.markdown("#### 全部模型中转台任务")
    st.caption(
        "统一显示 Smoke、训练、冻结编码器适配、批量推理和历史扫描；"
        "下方保留可操作任务的原生详情。"
    )
    type_options = sorted(jobs["job_type"].astype(str).unique())
    selected_types = st.multiselect(
        "任务类型",
        type_options,
        default=type_options,
        key="unified_job_types",
    )
    visible = jobs.loc[jobs["job_type"].isin(selected_types)].copy()
    table = visible[
        ["job_type", "job_id", "status", "progress", "created_at", "updated_at"]
    ].rename(
        columns={
            "job_type": "任务类型",
            "job_id": "任务 ID",
            "status": "状态",
            "progress": "进度",
            "created_at": "创建时间",
            "updated_at": "结束 / 更新时间",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch", height=520)


def _render_model_access(models: pd.DataFrame, *, page_mode: str = "assets") -> None:
    if page_mode == "tasks":
        st.markdown("#### 模型接入 · 当前任务模型")
        st.caption("只显示已有当前任务预测产物或任务推理链的模型；离线回放与在线资格保持分离。")
    else:
        st.markdown("#### 模型接入 · 资产目录")
        st.caption("显示当前输入模态下可交接的模型资产，并逐项核对来源、Adapter 和任务适配状态。")
        _render_asset_runtime_readiness()
    task_id = _task_selector(models, "engineering_task")
    recipes = _load_training_recipes()
    catalog = build_unified_model_catalog(models, target_task_id=task_id, recipes=recipes)
    target_models = catalog.loc[catalog["task_id"].astype(str).eq(task_id)]
    first = target_models.iloc[0]
    for health in catalog.attrs.get("provider_health", ()):
        if health.provider_id == "ophbench" and not health.available:
            detail = health.detail or "未提供详细错误。"
            st.error(
                f"OphBench 注册表暂不可用：{health.message}\n\n"
                f"详细错误：`{detail}`\n\n"
                "本地模型与 timm 仍可继续使用。"
            )
    foundation = catalog.loc[catalog["provider_id"].astype(str).eq("ophbench")]
    visible_catalog, target_modality = current_task_candidate_catalog(
        catalog,
        dataset_id=str(first.get("dataset_id", "")),
    )
    layers = partition_model_catalog(visible_catalog)
    visible_foundation = visible_catalog.loc[
        visible_catalog["provider_id"].astype(str).eq("ophbench")
    ]
    if page_mode == "tasks":
        selected_layer = "当前任务模型"
        layer_catalog = pd.concat(
            [layers["在线可用任务模型"], layers["离线预测回放资产"]],
            ignore_index=True,
        )
        summary_cards = [
            ("任务模型", len(layer_catalog), "当前任务预测资产"),
            ("仅离线回放", len(layers["离线预测回放资产"]), "不能推出在线推理资格"),
            ("任务推理可用", len(layers["在线可用任务模型"]), "可输出标准类别概率"),
            (
                "可进入路由池",
                int(layer_catalog.get("route_eligible", pd.Series(dtype=bool)).astype(bool).sum()),
                "统一评测与成本门控后",
            ),
        ]
        status_options = ["direct_inference", "offline_replay"]
    else:
        selected_layer = "当前任务候选模型资产"
        layer_catalog = pd.concat(
            [layers["可适配基础模型"], layers["候选基础模型库"]],
            ignore_index=True,
        )
        summary_cards = [
            ("登记模型", int(foundation["source_model_id"].nunique()), "OphBench 模型资产"),
            (
                "可交接 Checkpoint",
                int(foundation["download_status"].astype(str).eq("downloaded").sum()),
                "不等于本机权重已核验",
            ),
            (
                f"当前 {target_modality or '任务'} 候选",
                len(visible_foundation),
                "已按模态与资产类型筛选",
            ),
            (
                "基础加载通过",
                int(foundation["encoder_smoke_passed"].astype(bool).sum()),
                "Checkpoint 级 smoke",
            ),
        ]
        status_options = ["adaptable", "blocked"]
    st.markdown(
        '<div class="hub-mini-strip">'
        + "".join(
            '<div class="hub-mini-stat">'
            f'<span>{html.escape(label)}</span><b>{value}</b><small>{html.escape(note)}</small></div>'
            for label, value, note in summary_cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    record_label = (
        "当前任务模型"
        if page_mode == "tasks"
        else "当前目录候选记录（OphBench + 本地）"
    )
    st.markdown(
        f'<div class="hub-band"><strong>数据集：</strong>{html.escape(str(first.get("dataset_display_name", first.get("dataset_id", "—"))))}　'
        f'<strong>来源：</strong>{html.escape(str(first.get("dataset_source", "待核实")))}　'
        f'<strong>{html.escape(record_label)}：</strong>{len(layer_catalog)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "候选视图默认隐藏 OphBench 范围排除的 VisionFM legacy 资产、"
        "与当前输入模态不匹配的 checkpoint 及生成模型；完整登记仍保留在上游 Registry。"
    )
    if st.button("重新扫描受控目录", icon=":material/refresh:"):
        st.info("扫描由受控 inventory runner 执行；当前页面不会扫描服务器全盘。")
    if layer_catalog.empty:
        st.info(f"当前没有{selected_layer}。")
        return
    filter_cols = st.columns(3)
    selected_status = filter_cols[0].selectbox(
        "当前任务状态",
        [""] + status_options,
        format_func=lambda value: "全部状态" if not value else TARGET_STATUS_LABELS[value][0],
    )
    provider_options = sorted(layer_catalog["provider_id"].dropna().astype(str).unique())
    selected_provider = filter_cols[1].selectbox(
        "模型来源",
        [""] + provider_options,
        format_func=lambda value: "全部来源" if not value else value,
    )
    family_options = sorted(layer_catalog["model_family"].dropna().astype(str).unique())
    selected_family = filter_cols[2].selectbox(
        "模型家族",
        [""] + family_options,
        format_func=lambda value: "全部家族" if not value else human_family(value),
    )
    visible_models = filter_global_model_catalog(
        layer_catalog,
        status=selected_status or None,
        family=selected_family or None,
        provider=selected_provider or None,
    )
    if visible_models.empty:
        st.info("当前筛选条件下没有模型。")
        return

    visible_ids = set(visible_models["model_id"].astype(str))
    if st.session_state.get("selected_model_id") not in visible_ids:
        st.session_state["selected_model_id"] = str(visible_models.iloc[0]["model_id"])

    library_col, detail_col = st.columns([1.15, 0.85], gap="large")
    with library_col:
        st.markdown("#### " + str(selected_layer))
        group_column = "model_family" if page_mode == "tasks" else "source_model_id"
        for family, family_models in visible_models.groupby(group_column, sort=True):
            usable_n = int(family_models["target_task_status"].astype(str).isin({"direct_inference", "offline_replay"}).sum())
            model_name = ui_value(family_models.iloc[0].get("model_name"), empty=human_family(family))
            checkpoint_note = f"{len(family_models)} 个 checkpoint" if group_column == "source_model_id" else f"{len(family_models)} 个任务模型"
            with st.expander(
                f"{model_name} · {checkpoint_note} · {usable_n} 个当前任务可用",
                expanded=False,
            ):
                for _, row in family_models.sort_values("artifact_id").iterrows():
                    _render_model_entry(row)
    with detail_col:
        selected_model_id = st.session_state["selected_model_id"]
        row = visible_models.loc[visible_models["model_id"].astype(str).eq(selected_model_id)].iloc[0]
        selected_id = str(row["artifact_id"])
        status, css_class = TARGET_STATUS_LABELS[str(row["target_task_status"])]
        if bool(row.get("route_eligible", False)):
            role_text = _role_text(row.get("role_candidates"))
        elif bool(row.get("replay_eligible", False)):
            role_text = "仅离线回放"
        else:
            role_text = "不可路由"
        st.markdown("#### 模型操作")
        is_task = bool(row.get("task_checkpoint", False))
        if is_task:
            base_size = pd.to_numeric(
                pd.Series([row.get("base_checkpoint_mb", 1157.1)]), errors="coerce"
            ).iloc[0]
            head_bytes = pd.to_numeric(
                pd.Series([row.get("task_head_bytes")]), errors="coerce"
            ).iloc[0]
            detail_content = _detail_section("当前任务模型", [
                ("基础模型", human_family(row.get("base_model_id", row.get("model_family")))),
                ("基础 Checkpoint", f"{ui_value(row.get('base_checkpoint_id'), empty='retfound-cfp')} / CFP"),
                ("基础权重", f"{_format_file_size(float(base_size) * 1024 * 1024)} · 需要认证"),
                ("任务头", ui_value(row.get("classifier_type"), empty="Logistic Regression")),
                ("任务头文件", f"{ui_value(row.get('task_head_file'), empty='linear_probe.joblib')} · {_format_file_size(head_bytes) if pd.notna(head_bytes) else '尚未登记'}"),
                ("任务 / 标签空间", f"{task_label(row.get('task_id'))} / {ui_value(row.get('label_space'))}"),
                ("适配方法", "冻结编码器 + Logistic Regression"),
                ("Recipe / Run", f"{ui_value(row.get('recipe_id'))} / {ui_value(row.get('run_id'))}"),
                ("Selected C", _compact_number(row.get("selected_C"))),
                ("主要指标", f"QWK {float(row.get('quadratic_kappa')):.4f} · Macro-F1 {float(row.get('macro_f1')):.4f}" if pd.notna(row.get("quadratic_kappa")) and pd.notna(row.get("macro_f1")) else "尚未登记"),
                ("在线推理", "已验证" if bool(row.get("task_inference_ready", False)) else "不可用（仅离线回放）"),
                ("Scout / Expert", "可用" if bool(row.get("route_eligible", False)) else "仅回放可用"),
                ("Forward-only 成本", f"{float(row.get('forward_cost_ms_per_image')):.3f} ms/图" if str(row.get("cost_status")) == "measured" else "尚未完成统一测量"),
                ("科研声明", "集成验证，不用于模型优劣比较"),
            ])
        else:
            asset_cells = [
                ("正式名称", ui_value(row.get("model_name"), empty=human_model(selected_id))),
                ("Checkpoint", ui_value(row.get("checkpoint_name"))),
                ("模型覆盖模态", ui_value(row.get("model_modalities"))),
                ("Checkpoint 输入", ui_value(row.get("modalities"))),
                ("架构", _runtime_architecture(row)),
                ("资产类型", artifact_type_display(row)),
                ("权重提供方", ui_value(row.get("checkpoint_provider"))),
                ("权重访问", source_access_display(row)),
                (
                    "目录登记",
                    readiness_value(
                        bool(row.get("catalog_registered", False)),
                        "已登记",
                        "未登记",
                    ),
                ),
                (
                    "官方来源",
                    readiness_value(
                        bool(row.get("official_source_verified", False)),
                        "来源已核验",
                        "来源待核验",
                    ),
                ),
                ("上游下载记录", upstream_download_display(row.get("download_status"))),
                ("上游文件核验", integrity_display(row.get("local_integrity_status"))),
                ("Provider 核验", integrity_display(row.get("provider_verification_status"))),
                ("许可证", ui_value(row.get("license"), empty="尚未明确")),
                (
                    "许可证状态",
                    verification_evidence_display(row.get("upstream_license_status")),
                ),
            ]
            readiness_cells = [
                (
                    "OphAgent 本地权重",
                    readiness_value(
                        bool(row.get("local_asset_verified", False)),
                        "本机已核验",
                        "本机未核验",
                    ),
                ),
                ("Adapter", adapter_status_display(row.get("base_adapter_status"))),
                (
                    "基础加载",
                    readiness_value(
                        bool(row.get("encoder_smoke_passed", False)),
                        "基础加载已通过",
                        "基础加载待验证",
                    ),
                ),
                (
                    "特征输出",
                    readiness_value(
                        bool(row.get("feature_output_verified", False)),
                        "特征输出已验证",
                        "特征输出待验证",
                    ),
                ),
                (
                    "原生预处理",
                    verification_evidence_display(
                        row.get("preprocessing_verification_status")
                    ),
                ),
                (
                    "当前任务适配",
                    readiness_value(
                        bool(row.get("task_adapted", False)),
                        "任务适配已完成",
                        "任务适配待完成",
                    ),
                ),
                ("Embedding 维度", _compact_number(row.get("embedding_dim"))),
            ]
            routing_cells = [
                (
                    "任务推理",
                    readiness_value(
                        bool(row.get("task_inference_ready", False)),
                        "可输出标准类别概率",
                        "任务推理暂不可用",
                    ),
                ),
                (
                    "统一任务评测",
                    readiness_value(
                        bool(row.get("unified_evaluation_complete", False)),
                        "统一评测已完成",
                        "统一评测待完成",
                    ),
                ),
                (
                    "推理成本",
                    readiness_value(
                        bool(row.get("cost_verified", False)),
                        "推理成本已核验",
                        "推理成本待核验",
                    ),
                ),
                (
                    "路由资格",
                    readiness_value(
                        bool(row.get("route_eligible", False)),
                        "可进入路由池",
                        "暂不可路由",
                    ),
                ),
            ]
            detail_content = (
                _detail_section("A. 模型资产目录", asset_cells)
                + _detail_section("B. 接入准备度", readiness_cells)
                + _detail_section("C. 中转台资格", routing_cells)
            )
        st.markdown(
            f'<div class="detail-panel"><h3>{html.escape(ui_value(row.get("model_name"), empty=human_model(selected_id)))}</h3>'
            f'<span class="badge {css_class}">{html.escape(status)}</span>'
            f'<span class="badge badge-live">{html.escape(role_text)}</span>'
            f'{detail_content}'
            f'<div class="case-list-note"><strong>目标任务判断：</strong>{html.escape(str(row.get("target_task_reason", "待核验")))}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        if not bool(row.get("task_checkpoint", False)):
            links = []
            if ui_value(row.get("paper_url"), empty=""):
                links.append(f"[论文]({row.get('paper_url')})")
            if ui_value(row.get("code_url"), empty=""):
                links.append(f"[官方代码]({row.get('code_url')})")
            if ui_value(row.get("weight_url"), empty=""):
                links.append(f"[官方权重入口]({row.get('weight_url')})")
            if links:
                st.markdown("　".join(links))
        is_candidate = not bool(row.get("task_checkpoint", False)) and not bool(row.get("base_adapter_ready", False))
        if is_candidate:
            st.caption("该记录用于模型调研与 Adapter 规划，不是可运行任务模型。")
            st.button("规划 Adapter 接入", width="stretch", disabled=True)
        elif st.button("检查兼容性", width="stretch"):
            if str(row.get("target_task_status")) in {"direct_inference", "offline_replay"}:
                st.success(str(row.get("target_task_reason")))
            else:
                st.warning(str(row.get("target_task_reason")))
        can_prepare, prepare_reason = training_capability(row, recipes)
        if not is_candidate and st.button("准备微调", width="stretch", disabled=not can_prepare):
            st.session_state["training_wizard_model"] = selected_model_id
            st.rerun()
        if not is_candidate and not can_prepare:
            st.caption(prepare_reason)
    if st.session_state.get("training_wizard_model") == str(row["model_id"]):
        _render_training_wizard(row, recipes, target_task_id=task_id)


def _render_asset_smoke_job_records() -> None:
    st.markdown("#### 模型资产 Smoke 任务")
    flash = st.session_state.pop("asset_smoke_job_flash", None)
    if flash:
        st.success(str(flash))
    try:
        _ensure_historical_asset_smoke_job()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"历史 Smoke 证据暂时无法登记：{exc}")
    all_jobs = list_asset_smoke_jobs(include_archived=True)
    if not all_jobs:
        st.info("尚无模型资产 Smoke 任务。可在“模型资产”页面提交。")
        return
    visible_jobs = [job for job in all_jobs if not job.get("archived")]
    active_jobs = [
        job for job in visible_jobs if str(job.get("status")) in {"queued", "running"}
    ]
    recent_jobs = [job for job in visible_jobs if job not in active_jobs][:10]
    jobs = active_jobs + recent_jobs
    if st.button(
        "刷新 Smoke 状态",
        icon=":material/refresh:",
        key="refresh_asset_smoke_jobs",
    ):
        st.rerun()
    st.caption(
        f"当前展示 {len(jobs)} 个批次；共登记 {len(all_jobs)} 个。"
        "每个批次仅占一条顶层任务记录。"
    )
    for job in jobs:
        status = str(job.get("status", "unknown"))
        progress = dict(job.get("progress", {}))
        summary = dict(job.get("summary", {}))
        counts = dict(progress.get("counts", {}))
        label = _asset_smoke_parent_status_label(status, counts)
        completed = int(progress.get("completed_count", 0) or 0)
        total = int(progress.get("total_count", job.get("target_count", 0)) or 0)
        title = f"{job.get('job_id')} · {completed}/{total} · {label}"
        with st.expander(title, expanded=status in {"queued", "running"}):
            metrics = st.columns(6, gap="small")
            metrics[0].metric("状态", label)
            metrics[1].metric("进度", f"{completed}/{total}")
            metrics[2].metric(
                "完整前向", int(counts.get("runtime_smoke_passed", 0) or 0)
            )
            metrics[3].metric(
                "资源阻塞", int(counts.get("resource_blocked", 0) or 0)
            )
            metrics[4].metric(
                "失败/超时",
                int(counts.get("failed", 0) or 0)
                + int(counts.get("timeout", 0) or 0),
            )
            metrics[5].metric("设备", job.get("resolved_device", "—"))
            active_checkpoint = progress.get("active_checkpoint_id")
            if active_checkpoint:
                st.info(f"当前运行：{active_checkpoint}")
            if job.get("parent_job_id"):
                st.caption(f"来源批次：{job.get('parent_job_id')}")
            if job.get("source_kind") == "legacy_reference":
                st.caption("该记录引用既有 CLI Smoke 证据，没有复制权重或大体积产物。")
                if not job.get("source_available", True):
                    st.warning("历史证据引用当前不可访问；任务摘要仍保留，但不能展开原始结果。")
            if job.get("error_message"):
                st.error(f"{job.get('error_type', '错误')}：{job.get('error_message')}")
            child_frame = pd.DataFrame(progress.get("children", []))
            if not child_frame.empty:
                child_columns = [
                    "model_id",
                    "checkpoint_id",
                    "probe_profile",
                    "status",
                    "elapsed_seconds",
                    "asset_probe_passed",
                    "runtime_smoke_passed",
                    "error_detail",
                ]
                child_display = child_frame.reindex(columns=child_columns).copy()
                child_display["probe_profile"] = child_display[
                    "probe_profile"
                ].replace(ASSET_PROBE_PROFILE_LABELS)
                child_display["status"] = child_display["status"].replace(
                    ASSET_SMOKE_STATUS_LABELS
                )
                child_display = child_display.rename(
                    columns={
                        "model_id": "模型",
                        "checkpoint_id": "Checkpoint",
                        "probe_profile": "运行契约",
                        "status": "状态",
                        "elapsed_seconds": "耗时（秒）",
                        "asset_probe_passed": "资产探测",
                        "runtime_smoke_passed": "完整前向",
                        "error_detail": "失败或阻塞摘要",
                    }
                )
                st.dataframe(child_display, hide_index=True, width="stretch")
            actions = st.columns(6)
            if status in {"queued", "running"} and actions[0].button(
                "取消",
                key=f"cancel_asset_smoke_{job['job_id']}",
            ):
                try:
                    cancel_asset_smoke_job(job["job_dir"])
                except Exception as exc:
                    st.error(f"取消失败：{exc}")
                else:
                    st.rerun()
            if status == "cancelled" and actions[1].button(
                "恢复",
                key=f"resume_asset_smoke_{job['job_id']}",
            ):
                try:
                    resume_asset_smoke_job(job["job_dir"])
                except Exception as exc:
                    st.error(f"恢复失败：{exc}")
                else:
                    st.rerun()
            if actions[2].button(
                "重跑失败项",
                key=f"retry_failed_asset_smoke_{job['job_id']}",
                disabled=not bool(
                    int(counts.get("failed", 0) or 0)
                    + int(counts.get("timeout", 0) or 0)
                ),
            ):
                try:
                    new_job_id = retry_asset_smoke_subset(
                        job["job_dir"], statuses={"failed", "timeout"}
                    )
                except Exception as exc:
                    st.error(f"重跑失败：{exc}")
                else:
                    st.session_state["asset_smoke_job_flash"] = (
                        f"已从 {job['job_id']} 创建重跑批次：{new_job_id}"
                    )
                    st.rerun()
            if actions[3].button(
                "重查阻塞项",
                key=f"retry_blocked_asset_smoke_{job['job_id']}",
                disabled=not bool(int(counts.get("resource_blocked", 0) or 0)),
            ):
                try:
                    new_job_id = retry_asset_smoke_subset(
                        job["job_dir"], statuses={"resource_blocked"}
                    )
                except Exception as exc:
                    st.error(f"重查失败：{exc}")
                else:
                    st.session_state["asset_smoke_job_flash"] = (
                        f"已从 {job['job_id']} 创建阻塞项批次：{new_job_id}"
                    )
                    st.rerun()
            if status in {
                "succeeded",
                "completed_with_blockers",
                "failed",
                "cancelled",
            } and actions[4].button(
                "归档",
                key=f"archive_asset_smoke_{job['job_id']}",
            ):
                archive_asset_smoke_job(job["job_dir"], archived=True)
                st.rerun()
            terminal = status in {
                "succeeded",
                "completed_with_blockers",
                "failed",
                "cancelled",
            }
            with actions[5].popover("删除", disabled=not terminal):
                st.warning("永久删除该批次的任务记录、日志和结果摘要，无法恢复。模型权重不会被删除。")
                delete_confirmed = st.checkbox(
                    "我确认永久删除该 Smoke 批次",
                    key=f"delete_asset_smoke_confirm_{job['job_id']}",
                )
                if st.button(
                    "确认永久删除",
                    key=f"delete_asset_smoke_{job['job_id']}",
                    disabled=not delete_confirmed,
                    width="stretch",
                ):
                    try:
                        delete_asset_smoke_job(
                            job["job_dir"], jobs_root=ASSET_SMOKE_JOBS_ROOT
                        )
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")
                    else:
                        st.session_state["asset_smoke_job_flash"] = (
                            "Smoke 批次及其实验记录已永久删除。"
                        )
                        st.rerun()
            log_tail = read_asset_smoke_log_tail(job["job_dir"])
            if log_tail:
                with st.expander("批次日志"):
                    st.code(log_tail, language="text")
            if summary:
                st.caption(
                    "Smoke 结果仅证明模型资产运行契约；"
                    f"任务推理可用 {summary.get('task_inference_ready_count', 0)}，"
                    f"可进入路由池 {summary.get('route_eligible_count', 0)}。"
                )


def _render_global_scan_job_records() -> None:
    st.markdown("#### 全局候选扫描任务")
    scan_jobs = list_global_scan_jobs(include_archived=True)
    if not scan_jobs:
        st.info("尚无全局候选扫描任务。可在“研究评测”的全局候选扫描区域提交后台任务。")
        return
    status_labels = {
        "queued": "等待启动",
        "running": "运行中",
        "succeeded": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
        "unknown": "未知",
    }
    for job in scan_jobs[:20]:
        status = str(job.get("status", "unknown"))
        label = status_labels.get(status, status)
        title = (
            f"{job.get('job_id')} · {task_label(job.get('task_id'))} · "
            f"{int(job.get('estimated_points', 0) or 0):,} 个候选点 · {label}"
        )
        with st.expander(title, expanded=status == "running"):
            cols = st.columns(4)
            cols[0].metric("状态", label)
            cols[1].metric("任务", task_label(job.get("task_id")))
            cols[2].metric("候选点", f"{int(job.get('estimated_points', 0) or 0):,}")
            cols[3].metric("进程 PID", job.get("pid", "尚未记录"))
            st.caption(f"输出目录：{job.get('output_dir', '—')}")
            if job.get("error_message"):
                st.error(f"{job.get('error_type', '错误')}：{job['error_message']}")
            if status == "succeeded" and job.get("output_exists"):
                results = load_global_scan_results(job)
                if results.empty:
                    st.warning("结果目录存在，但未找到 global_scan_results.csv。")
                else:
                    completed = results.loc[results["scan_status"].astype(str).eq("completed")].copy()
                    st.caption(f"已生成 global_scan_results.csv；可用候选 {len(completed)} / 总行 {len(results)}。")
                    preview = completed.sort_values("global_rank_primary", na_position="last").head(5)
                    if not preview.empty:
                        preview_table = format_global_scan_preview(preview)
                        st.dataframe(preview_table, hide_index=True, width="stretch")
                    if st.button("载入研究评测区", key=f"load_scan_job_{job['job_id']}"):
                        st.session_state[f"model_hub_global_scan_{job.get('task_id')}"] = results
                        st.session_state["pending_engineering_layer"] = "研究评测"
                        st.session_state["pending_hub_workspace"] = "研究评测"
                        st.rerun()
            log_tail = read_global_scan_log_tail(job["job_dir"])
            if log_tail:
                with st.expander("扫描任务日志"):
                    st.code(log_tail, language="text")


def _render_job_records() -> None:
    st.subheader("任务运行记录")
    st.caption("任务在独立后台进程中运行；本页读取状态、日志和本地运行产物，不依赖外部实验跟踪服务。")
    task_type = st.segmented_control(
        "任务类型筛选",
        ["全部", "训练适配任务", "全局候选扫描任务", "模型资产 Smoke 任务"],
        default="全部",
        key="job_record_type_filter",
    )
    if task_type in {"全部", "模型资产 Smoke 任务"}:
        _render_asset_smoke_job_records()
    if task_type == "模型资产 Smoke 任务":
        return
    if task_type in {"全部", "全局候选扫描任务"}:
        _render_global_scan_job_records()
    if task_type == "全局候选扫描任务":
        return
    st.markdown("#### 训练适配任务")
    st.caption("训练产物与任务记录分别保存；手动删除输出目录不会删除任务记录。归档只隐藏记录，不删除 checkpoint、日志或评测结果。")
    flash = st.session_state.pop("training_job_flash", None)
    if flash:
        st.success(str(flash))
    all_jobs = list_training_jobs(TRAINING_JOBS_ROOT, include_archived=True)
    controls = st.columns([1, 2, 1, 1])
    if controls[0].button("刷新任务状态", icon=":material/refresh:"):
        st.rerun()
    status_labels = {
        "queued": "等待启动",
        "running": "运行中",
        "succeeded": "已完成",
        "failed": "失败",
        "cancelled": "已取消",
    }
    available_statuses = [
        status
        for status in ("running", "queued", "succeeded", "failed", "cancelled")
        if any(str(job.get("status")) == status for job in all_jobs)
    ]
    selected_statuses = controls[1].multiselect(
        "状态筛选",
        available_statuses,
        default=available_statuses,
        format_func=lambda value: status_labels.get(value, value),
    )
    display_limit = controls[2].selectbox("显示数量", [10, 20, 50, 0], format_func=lambda value: "全部" if value == 0 else f"最近 {value} 条")
    show_archived = controls[3].toggle("显示已归档", value=False)
    jobs = [
        job
        for job in all_jobs
        if (show_archived or not job.get("archived"))
        and str(job.get("status")) in selected_statuses
    ]
    if display_limit:
        jobs = jobs[: int(display_limit)]
    if not all_jobs:
        st.info("尚无训练任务。请先在“模型接入”中选择可用模型并完成预检。")
        return
    if not jobs:
        st.info("当前筛选条件下没有任务记录。")
        return
    st.caption(f"当前显示 {len(jobs)} 条；任务记录总数 {len(all_jobs)} 条。")
    for job in jobs:
        status = str(job.get("status", "unknown"))
        label = status_labels.get(status, status)
        if job.get("archived"):
            label += " · 已归档"
        with st.expander(_job_record_title(job, label), expanded=status == "running"):
            cols = st.columns(3)
            cols[0].metric("状态", label)
            cols[1].metric("任务", task_label(job.get("task_id")))
            cols[2].metric("进程 PID", job.get("pid", "尚未记录"))
            st.caption(f"输出目录：{job.get('output_dir', '—')}")
            if job.get("output_dir") and not job.get("output_exists"):
                st.warning("该任务对应的训练产物目录已不存在，但任务状态记录仍保留。可归档此记录，或在服务器删除对应 training_jobs/<job_id> 目录。")
            for warning in job.get("startup_warnings", []):
                st.warning(str(warning))
            if job.get("error_message"):
                st.error(f"{job.get('error_type', '错误')}：{job['error_message']}")
            progress = load_training_progress(job.get("output_dir", ""))
            history = progress["history"]
            summary = progress["summary"]
            test_metrics = progress["test_metrics"]
            if not history.empty and "epoch" in history:
                summary_columns = st.columns(4)
                summary_columns[0].metric("已记录 Epoch", int(history["epoch"].max()))
                summary_columns[1].metric("最佳 Epoch", summary.get("best_epoch", "训练中"))
                best_score = summary.get("best_validation_score")
                summary_columns[2].metric(
                    "最佳验证分数",
                    f"{float(best_score):.4f}" if best_score is not None else "训练中",
                )
                train_time = summary.get("total_train_time_sec")
                summary_columns[3].metric(
                    "训练耗时",
                    f"{float(train_time):.1f} 秒" if train_time is not None else "训练中",
                )

                loss_columns = [column for column in ("train_loss", "val_loss") if column in history]
                metric_columns = [
                    column
                    for column in ("val_accuracy", "val_macro_f1", "val_qwk")
                    if column in history and pd.to_numeric(history[column], errors="coerce").notna().any()
                ]
                charts = st.columns(2)
                if loss_columns:
                    loss_data = history[["epoch", *loss_columns]].melt(
                        "epoch", var_name="曲线", value_name="数值"
                    )
                    loss_labels = {"train_loss": "训练损失", "val_loss": "验证损失"}
                    loss_data["曲线"] = loss_data["曲线"].map(loss_labels)
                    loss_chart = (
                        alt.Chart(loss_data)
                        .mark_line(point=True, strokeWidth=2.5)
                        .encode(
                            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
                            y=alt.Y("数值:Q", title="Loss", scale=alt.Scale(zero=False)),
                            color=alt.Color(
                                "曲线:N",
                                title=None,
                                scale=alt.Scale(domain=list(loss_labels.values()), range=["#168579", "#D95D4F"]),
                            ),
                            tooltip=["epoch:Q", "曲线:N", alt.Tooltip("数值:Q", format=".4f")],
                        )
                        .properties(title="训练与验证损失", height=260)
                    )
                    charts[0].altair_chart(loss_chart, width="stretch")
                if metric_columns:
                    metric_data = history[["epoch", *metric_columns]].melt(
                        "epoch", var_name="指标", value_name="数值"
                    )
                    metric_labels = {
                        "val_accuracy": "Accuracy",
                        "val_macro_f1": "Macro-F1",
                        "val_qwk": "QWK",
                    }
                    metric_data["指标"] = metric_data["指标"].map(metric_labels)
                    metric_chart = (
                        alt.Chart(metric_data)
                        .mark_line(point=True, strokeWidth=2.5)
                        .encode(
                            x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
                            y=alt.Y("数值:Q", title="指标值", scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color(
                                "指标:N",
                                title=None,
                                scale=alt.Scale(
                                    domain=["Accuracy", "Macro-F1", "QWK"],
                                    range=["#2F6FB0", "#168579", "#D28B28"],
                                ),
                            ),
                            tooltip=["epoch:Q", "指标:N", alt.Tooltip("数值:Q", format=".4f")],
                        )
                        .properties(title="验证集性能", height=260)
                    )
                    charts[1].altair_chart(metric_chart, width="stretch")
                if len(history) == 1:
                    st.caption("当前为 1 Epoch 快速链路验证，因此图中只有一个观测点；正式训练会形成完整曲线。")
            else:
                st.caption("尚未生成 epoch 级训练记录。任务启动后可点击“刷新任务状态”查看。")

            if test_metrics:
                metric_columns = st.columns(3)
                for column, (name, label) in zip(
                    metric_columns,
                    (("accuracy", "测试 Accuracy"), ("macro_f1", "测试 Macro-F1"), ("qwk", "测试 QWK")),
                ):
                    value = test_metrics.get(name)
                    column.metric(label, f"{float(value):.4f}" if value is not None else "不适用")
            log_tail = read_job_log_tail(job["job_dir"])
            if log_tail:
                visible_log, hidden_warnings = split_training_log_messages(log_tail)
                if hidden_warnings:
                    st.info("已收起 " + "、".join(hidden_warnings) + "；这类提示不影响训练记录曲线展示。")
                with st.expander("原始训练日志"):
                    st.code(visible_log or "日志中仅包含已收起的已知提示。", language="text")
            action_cols = st.columns(4)
            can_cancel = status in {"queued", "running"} and bool(job.get("pid"))
            if action_cols[0].button("取消任务", key=f"cancel_{job['job_id']}", disabled=not can_cancel):
                try:
                    cancel_training_job(job["job_dir"])
                except Exception as exc:
                    st.error(f"取消失败：{exc}")
                else:
                    st.rerun()
            if action_cols[1].button(
                "安全重试",
                key=f"retry_{job['job_id']}",
                disabled=status not in {"failed", "cancelled"},
                help="生成一个使用 cuda:2 的新任务；原失败记录和原配置保持不变。",
            ):
                request_path = Path(job["job_dir"]) / "request.json"
                request = TrainingRequest(**json.loads(request_path.read_text(encoding="utf-8")))
                retry_output = request.output_dir + "-retry-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                retry_request = build_retry_request(
                    request,
                    retry_output,
                    device_override="cuda:2",
                )
                try:
                    new_job_id = submit_training_job(retry_request, TRAINING_JOBS_ROOT, LEGACY_TRAINING_RECIPES)
                except Exception as exc:
                    st.error(f"重试提交失败：{exc}")
                else:
                    st.success(f"已生成 cuda:2 重试任务：{new_job_id}")
            archive_label = "恢复记录" if job.get("archived") else "归档记录"
            if action_cols[2].button(
                archive_label,
                key=f"archive_{job['job_id']}",
                disabled=status not in {"succeeded", "failed", "cancelled"},
                help="只改变任务记录的显示状态，不删除训练产物。",
            ):
                try:
                    archive_training_job(job["job_dir"], archived=not bool(job.get("archived")))
                except Exception as exc:
                    st.error(f"记录操作失败：{exc}")
                else:
                    st.rerun()
            with action_cols[3].popover("删除任务及产物", disabled=status not in {"succeeded", "failed", "cancelled"}):
                st.warning("此操作将永久删除该任务的 checkpoint、训练日志、评测结果和任务记录，无法恢复。")
                delete_confirmed = st.checkbox(
                    "我确认删除该任务及其全部实验产物",
                    key=f"delete_confirm_{job['job_id']}",
                )
                if st.button(
                    "确认永久删除",
                    key=f"delete_job_{job['job_id']}",
                    disabled=not delete_confirmed,
                    width="stretch",
                ):
                    try:
                        delete_training_job_and_outputs(
                            job["job_dir"],
                            jobs_root=TRAINING_JOBS_ROOT,
                            runs_root=TRAINING_RUNS_ROOT,
                        )
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")
                    else:
                        st.session_state["training_job_flash"] = "任务记录及对应实验产物已删除。"
                        st.rerun()


def render_engineering_workspace(
    data: dict[str, object],
    *,
    view: str | None = None,
) -> None:
    models = data["models"]
    hub_index = data.get("hub_index")
    apply_pending_engineering_navigation(st.session_state)
    if view is None:
        legacy_view = st.segmented_control(
            "模型工程",
            ["模型接入", "研究评测", "任务运行记录"],
            default="模型接入",
            key="engineering_layer",
        )
        view = {
            "模型接入": "模型资产",
            "研究评测": "研究评测",
            "任务运行记录": "任务运行记录",
        }.get(str(legacy_view), "模型资产")
    if view == "模型资产":
        if isinstance(hub_index, dict):
            _render_unified_asset_catalog(hub_index)
            with st.expander("Adapter 与训练接入操作", expanded=False):
                _render_model_access(models, page_mode="assets")
        else:
            _render_model_access(models, page_mode="assets")
    elif view == "任务模型":
        if isinstance(hub_index, dict):
            _render_unified_task_catalog(hub_index)
        else:
            _render_model_access(models, page_mode="tasks")
    elif view == "研究评测":
        render_research_workspace(models, hub_index=hub_index)
    else:
        if isinstance(hub_index, dict):
            _render_unified_job_records(hub_index)
            with st.expander("可操作任务详情", expanded=False):
                _render_job_records()
        else:
            _render_job_records()
