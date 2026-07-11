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
from app.model_hub_ui import human_family, human_model, human_pretraining_source, task_label
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
        return f"{float(recorded):.1f} MB"
    path = Path(str(row.get("checkpoint_path", "") or ""))
    if path.is_file():
        return f"{path.stat().st_size / 1024 / 1024:.1f} MB"
    return "未登记"


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
    model_id = str(row.get("model_id", f"{row.get('task_id')}::{artifact_id}"))
    status, css_class = TARGET_STATUS_LABELS.get(
        str(row.get("target_task_status", "blocked")),
        ("当前任务状态待核验", "badge-wait"),
    )
    selected = st.session_state.get("selected_model_id") == model_id
    with st.container(border=True):
        columns = st.columns([1.35, 1.05, 1.35, 0.7])
        with columns[0]:
            st.markdown(
                f'<span class="model-entry-label">模型资产</span>'
                f'<span class="model-entry-title">{html.escape(human_model(artifact_id))}</span>'
                f'<span class="model-entry-copy">{html.escape(human_pretraining_source(row.get("pretraining_source")))} · '
                f'{html.escape(task_label(row.get("task_id")))}</span>',
                unsafe_allow_html=True,
            )
        with columns[1]:
            parameter_count = _format_parameter_count(_model_parameter_count(row))
            st.markdown(
                f'<span class="model-entry-label">架构</span>'
                f'<span class="model-entry-title">{html.escape(str(row.get("architecture", "未识别架构")))}</span>'
                f'<span class="model-entry-copy">{html.escape(parameter_count)}</span>',
                unsafe_allow_html=True,
            )
        with columns[2]:
            st.markdown(
                f'<span class="model-entry-label">当前任务状态</span>'
                f'<span class="model-entry-title">{html.escape(status)}</span>'
                f'<span class="badge {css_class}">{html.escape(_role_text(row.get("role_candidates")))}</span>'
                f'<span class="model-entry-copy">{html.escape(str(row.get("target_task_reason", "")))}</span>',
                unsafe_allow_html=True,
            )
        with columns[3]:
            if st.button("已选择" if selected else "选择", key=f"select_model_{model_id}", disabled=selected, width="stretch"):
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
        checkpoint_path = str(row.get("checkpoint_path", "") or "").strip()
        checkpoint_available = str(row.get("checkpoint_status", "")) == "found" and bool(checkpoint_path)
        initialization_options = ["timm_pretrained"]
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
        target_artifact_id = adaptation_artifact_id(
            source_artifact_id,
            target_task_id,
            initialization_source=initialization_source,
            architecture=str(row.get("architecture", "")),
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
        base_request = build_adaptation_request(
            row,
            target_task,
            data_root=data_root,
            recipe_id=recipe_id,
            output_dir=output_dir,
            initialization_source=initialization_source,
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
            trainer_adapter="timm_imagefolder_v1",
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
        export_command = (
            "/data/conda_envs/ophagent/bin/python "
            "scripts/training/train_timm_classifier.py "
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
                st.rerun()


def _render_model_access(models: pd.DataFrame) -> None:
    st.subheader("模型接入")
    st.caption("显示受控目录中发现的全部模型；目标任务只决定兼容状态，不会隐藏跨任务模型。")
    task_id = _task_selector(models, "engineering_task")
    recipes = _load_training_recipes()
    catalog = build_unified_model_catalog(models, target_task_id=task_id, recipes=recipes)
    target_models = catalog.loc[catalog["task_id"].astype(str).eq(task_id)]
    first = target_models.iloc[0]
    for health in catalog.attrs.get("provider_health", ()):
        if health.provider_id == "ophbench" and not health.available:
            st.info(f"OphBench 注册表暂不可用：{health.message} 本地模型与 timm 仍可继续使用。")
    st.markdown(_catalog_summary_html(catalog), unsafe_allow_html=True)
    st.markdown(
        f'<div class="hub-band"><strong>数据集：</strong>{html.escape(str(first.get("dataset_display_name", first.get("dataset_id", "—"))))}　'
        f'<strong>来源：</strong>{html.escape(str(first.get("dataset_source", "待核实")))}　'
        f'<strong>全局已发现模型：</strong>{len(catalog)}</div>',
        unsafe_allow_html=True,
    )
    if st.button("重新扫描受控目录", icon=":material/refresh:"):
        st.info("扫描由受控 inventory runner 执行；当前页面不会扫描服务器全盘。")

    status_options = ["direct_inference", "offline_replay", "adaptable", "blocked"]
    filter_cols = st.columns(3)
    selected_status = filter_cols[0].selectbox(
        "当前任务状态",
        [""] + status_options,
        format_func=lambda value: "全部状态" if not value else TARGET_STATUS_LABELS[value][0],
    )
    provider_options = sorted(catalog["provider_id"].dropna().astype(str).unique())
    selected_provider = filter_cols[1].selectbox(
        "模型来源",
        [""] + provider_options,
        format_func=lambda value: "全部来源" if not value else value,
    )
    family_options = sorted(catalog["model_family"].dropna().astype(str).unique())
    selected_family = filter_cols[2].selectbox(
        "模型家族",
        [""] + family_options,
        format_func=lambda value: "全部家族" if not value else human_family(value),
    )
    visible_models = filter_global_model_catalog(
        catalog,
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
        st.markdown("#### 模型家族")
        for family, family_models in visible_models.groupby("model_family", sort=True):
            usable_n = int(family_models["target_task_status"].astype(str).isin({"direct_inference", "offline_replay"}).sum())
            with st.expander(f"{human_family(family)} · {len(family_models)} 个 · {usable_n} 个当前任务可用", expanded=len(family_models) == 1):
                for _, row in family_models.sort_values("artifact_id").iterrows():
                    _render_model_entry(row)
    with detail_col:
        selected_model_id = st.session_state["selected_model_id"]
        row = visible_models.loc[visible_models["model_id"].astype(str).eq(selected_model_id)].iloc[0]
        selected_id = str(row["artifact_id"])
        status, css_class = TARGET_STATUS_LABELS[str(row["target_task_status"])]
        st.markdown("#### 模型操作")
        parameter_count = _model_parameter_count(row)
        detail_cells = [
            ("Provider", str(row.get("provider_id", "—"))),
            ("模型家族", human_family(row.get("model_family"))),
            ("来源任务", task_label(row.get("task_id"))),
            ("架构", str(row.get("architecture", "—"))),
            ("参数量", _format_parameter_count(parameter_count)),
            ("Checkpoint 大小", _checkpoint_size_text(row)),
            ("预训练来源", human_pretraining_source(row.get("pretraining_source"))),
            ("Checkpoint", str(row.get("checkpoint_status", "未登记"))),
            ("现有结果", str(row.get("prediction_source", "missing"))),
            ("来源访问", str(row.get("source_access_status", "unknown"))),
            ("基础 Adapter", str(row.get("base_adapter_status", "unknown"))),
            ("任务推理", "ready" if bool(row.get("task_inference_ready", False)) else "not ready"),
            ("路由资格", "eligible" if bool(row.get("route_eligible", False)) else "ineligible"),
        ]
        detail_grid = "".join(
            '<div class="model-detail-cell">'
            f'<span>{html.escape(label)}</span><b>{html.escape(str(value))}</b>'
            "</div>"
            for label, value in detail_cells
        )
        st.markdown(
            f'<div class="detail-panel"><h3>{html.escape(human_model(selected_id))}</h3>'
            f'<span class="badge {css_class}">{html.escape(status)}</span>'
            f'<span class="badge badge-live">{html.escape(_role_text(row.get("role_candidates")))}</span>'
            f'<div class="model-detail-grid">{detail_grid}</div>'
            f'<div class="case-list-note"><strong>目标任务判断：</strong>{html.escape(str(row.get("target_task_reason", "待核验")))}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("检查兼容性", width="stretch"):
            if str(row.get("target_task_status")) in {"direct_inference", "offline_replay"}:
                st.success(str(row.get("target_task_reason")))
            else:
                st.warning(str(row.get("target_task_reason")))
        can_prepare, prepare_reason = training_capability(row, recipes)
        if st.button("准备微调", width="stretch", disabled=not can_prepare):
            st.session_state["training_wizard_model"] = selected_model_id
            st.rerun()
        if not can_prepare:
            st.caption(prepare_reason)
    if st.session_state.get("training_wizard_model") == str(row["model_id"]):
        _render_training_wizard(row, recipes, target_task_id=task_id)


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
        ["全部", "训练适配任务", "全局候选扫描任务"],
        default="全部",
        key="job_record_type_filter",
    )
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


def render_engineering_workspace(data: dict[str, object]) -> None:
    models = data["models"]
    apply_pending_engineering_navigation(st.session_state)
    view = st.segmented_control(
        "模型工程",
        ["模型接入", "研究评测", "任务运行记录"],
        default="模型接入",
        key="engineering_layer",
    )
    if view == "模型接入":
        _render_model_access(models)
    elif view == "研究评测":
        render_research_workspace(models)
    else:
        _render_job_records()
