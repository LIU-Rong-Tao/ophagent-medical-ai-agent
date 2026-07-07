"""OphAgent 已知模型实验目录的静态发现与能力解析。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DR_PROTOCOL_ID = "dr_icdr_5class_proxy_v1"
GENERIC_PROTOCOL_ID = "generic_multiclass_v1"


@dataclass(frozen=True)
class ModelSpec:
    model_key: str
    display_name: str
    experiment_root: str
    loader_model_name: str
    protocol_id: str = DR_PROTOCOL_ID


KNOWN_MODEL_SPECS = (
    ModelSpec(
        "convnext_tiny",
        "ConvNeXt-Tiny",
        "aptos_convnext_tiny",
        "convnext_tiny",
    ),
    ModelSpec(
        "swin_tiny",
        "Swin-Tiny",
        "aptos_swin_tiny",
        "swin_tiny_patch4_window7_224.ms_in1k",
    ),
    ModelSpec(
        "vit_b_imagenet",
        "ViT-B ImageNet",
        "aptos_vit_base_patch16_imagenet",
        "vit_base_patch16_224",
    ),
    ModelSpec(
        "vit_l_official_like",
        "ViT-L official-like",
        "aptos_vit_large_patch16_official_like",
        "vit_large_patch16_224",
    ),
)


@dataclass(frozen=True)
class ModelArtifact:
    model_key: str
    display_name: str
    protocol_id: str
    loader_model_name: str
    experiment_dir: Path
    checkpoint_path: Path | None
    config_path: Path | None
    class_to_idx_path: Path | None
    env_info_path: Path | None
    metrics_path: Path | None
    test_predictions_path: Path | None
    summary_path: Path | None
    checkpoint_meta_path: Path | None
    checkpoint_size: int | None
    checkpoint_mtime_ns: int | None
    prediction_csv_sha256: str | None
    prediction_columns: tuple[str, ...]
    generated_time_or_unknown: str
    commit_or_unknown: str
    num_best_checkpoints: int
    artifact_status: str
    static_complete: bool
    can_attempt_load: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = value.as_posix()
        return payload


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _prediction_columns(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    try:
        return tuple(pd.read_csv(path, nrows=0).columns.astype(str))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return ()


@lru_cache(maxsize=128)
def _sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    """按路径、大小与修改时间缓存文件 SHA256。"""

    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_cached(str(resolved), stat.st_size, stat.st_mtime_ns)


def _artifact_for_run(spec: ModelSpec, run_dir: Path) -> ModelArtifact:
    checkpoints = sorted((run_dir / "checkpoints").glob("*best*.pth"))
    checkpoint_path = checkpoints[0] if len(checkpoints) == 1 else None
    config_path = _first_file(run_dir / "configs" / "config.json")
    mapping_path = _first_file(run_dir / "configs" / "class_to_idx.json")
    env_path = _first_file(run_dir / "configs" / "env_info.json")
    metrics_path = _first_file(run_dir / "evaluation" / "test" / "metrics.json")
    predictions_path = _first_file(
        run_dir / "evaluation" / "test" / "test_predictions.csv"
    )
    summary_path = _first_file(run_dir / "logs" / "summary.json")
    checkpoint_meta_path = _first_file(
        run_dir / "checkpoints" / "checkpoint_meta.json"
    )
    metadata = _read_json(checkpoint_meta_path)
    summary = _read_json(summary_path)
    generated = next(
        (
            str(value)
            for value in (
                metadata.get("generated_at"),
                metadata.get("created_at"),
                summary.get("generated_at"),
            )
            if value
        ),
        "unknown",
    )
    commit = next(
        (
            str(value)
            for value in (
                metadata.get("git_commit"),
                metadata.get("commit"),
                summary.get("git_commit"),
            )
            if value
        ),
        "unknown",
    )

    checkpoint_stat = checkpoint_path.stat() if checkpoint_path else None
    core_complete = all([checkpoint_path, config_path, mapping_path])
    static_complete = bool(core_complete and predictions_path and metrics_path)
    can_attempt_load = bool(core_complete and spec.loader_model_name)
    if len(checkpoints) > 1:
        artifact_status = "checkpoint_ambiguous"
        can_attempt_load = False
        static_complete = False
    elif static_complete:
        artifact_status = "static_complete"
    elif can_attempt_load:
        artifact_status = "inference_only"
    elif predictions_path:
        artifact_status = "offline_only"
    else:
        artifact_status = "artifact_missing"

    return ModelArtifact(
        model_key=spec.model_key,
        display_name=spec.display_name,
        protocol_id=spec.protocol_id,
        loader_model_name=spec.loader_model_name,
        experiment_dir=run_dir,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        class_to_idx_path=mapping_path,
        env_info_path=env_path,
        metrics_path=metrics_path,
        test_predictions_path=predictions_path,
        summary_path=summary_path,
        checkpoint_meta_path=checkpoint_meta_path,
        checkpoint_size=checkpoint_stat.st_size if checkpoint_stat else None,
        checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns if checkpoint_stat else None,
        prediction_csv_sha256=(
            compute_file_sha256(predictions_path) if predictions_path else None
        ),
        prediction_columns=_prediction_columns(predictions_path),
        generated_time_or_unknown=generated,
        commit_or_unknown=commit,
        num_best_checkpoints=len(checkpoints),
        artifact_status=artifact_status,
        static_complete=static_complete,
        can_attempt_load=can_attempt_load,
    )


def discover_model_artifacts(repo_root: str | Path) -> list[ModelArtifact]:
    """仅扫描六个已知 APTOS 实验根目录下一层 run，不加载模型。"""

    root = Path(repo_root)
    artifacts: list[ModelArtifact] = []
    for spec in KNOWN_MODEL_SPECS:
        experiment_root = root / "experiments" / spec.experiment_root
        if not experiment_root.is_dir():
            continue
        for run_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
            artifacts.append(_artifact_for_run(spec, run_dir))
    return artifacts


def select_preferred_artifacts(
    artifacts: Iterable[ModelArtifact],
) -> dict[str, ModelArtifact]:
    """每个模型选择最完整的一个 run；不按文件时间猜测“最新”。"""

    priority = {
        "static_complete": 0,
        "inference_only": 1,
        "offline_only": 2,
        "artifact_missing": 3,
        "checkpoint_ambiguous": 4,
    }
    grouped: dict[str, list[ModelArtifact]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.model_key, []).append(artifact)
    return {
        model_key: sorted(
            candidates,
            key=lambda item: (
                priority.get(item.artifact_status, 99),
                item.experiment_dir.as_posix(),
            ),
        )[0]
        for model_key, candidates in grouped.items()
    }


def resolve_capabilities(
    protocol_id: str,
    prediction_columns: Iterable[str],
) -> dict[str, bool]:
    """根据显式任务协议和预测字段解析可计算能力。"""

    columns = {str(column) for column in prediction_columns}
    probability_columns = {column for column in columns if column.startswith("prob_")}
    has_prediction = bool({"pred_idx", "pred_grade", "pred_class"} & columns)
    has_truth = bool({"true_idx", "true_grade", "true_class"} & columns)
    probability_audit = len(probability_columns) >= 2 and has_prediction
    is_dr = protocol_id == DR_PROTOCOL_ID
    has_five_probabilities = (
        {f"prob_{grade}" for grade in range(5)}.issubset(columns)
        or {
            "prob_No DR",
            "prob_Mild DR",
            "prob_Moderate DR",
            "prob_Severe DR",
            "prob_Proliferative DR",
        }.issubset(columns)
    )
    ordinal_dr = bool(is_dr and has_five_probabilities and has_prediction)
    retrospective = bool(probability_audit and has_truth)
    return {
        "supports_online_inference": False,
        "supports_probability_audit": probability_audit,
        "supports_pre_review_ranking": probability_audit,
        "supports_retrospective_validation": retrospective,
        "supports_target_class_miss": probability_audit,
        "supports_ordinal_dr_audit": ordinal_dr,
        "supports_expected_gap": ordinal_dr,
        "supports_large_undergrading": bool(ordinal_dr and has_truth),
        "supports_vtdr_miss_proxy": bool(ordinal_dr and has_truth),
    }


def _empty_finding() -> dict[str, Any]:
    return {
        "rank_at_10": None,
        "status_at_10": "未评估",
        "rank_at_20": None,
        "status_at_20": "未评估",
        "rank_at_30": None,
        "status_at_30": "未评估",
        "top20_recall": None,
        "delta_to_best_comparator_at_20": None,
    }


def summarize_frozen_model_finding(
    path: str | Path,
    *,
    backbone: str,
    event: str,
    method: str,
) -> dict[str, Any]:
    """从冻结 v0.6.7c 表提取排名事实，不解释为科研发现成立。"""

    source = Path(path)
    if not source.is_file():
        return _empty_finding()
    frame = pd.read_csv(source)
    required = {
        "backbone",
        "ranking_method",
        "clinical_event",
        "review_budget",
        "dangerous_error_recall_at_k",
    }
    if not required.issubset(frame.columns):
        return _empty_finding()
    subset = frame[
        (frame["backbone"].astype(str) == backbone)
        & (frame["clinical_event"].astype(str) == event)
    ].copy()
    if subset.empty or method not in set(subset["ranking_method"].astype(str)):
        return _empty_finding()

    result = _empty_finding()
    for budget, suffix in ((0.1, "10"), (0.2, "20"), (0.3, "30")):
        point = subset[np.isclose(subset["review_budget"].astype(float), budget)]
        target = point[point["ranking_method"].astype(str) == method]
        if target.empty:
            continue
        recalls = point.groupby("ranking_method")[
            "dangerous_error_recall_at_k"
        ].mean()
        target_recall = float(target["dangerous_error_recall_at_k"].mean())
        descending = recalls.rank(method="min", ascending=False)
        rank = int(descending.loc[method])
        best = float(recalls.max())
        tie_count = int(np.isclose(recalls.to_numpy(dtype=float), best).sum())
        status = "非第一"
        if rank == 1:
            status = "并列第一" if tie_count > 1 else "第一"
        result[f"rank_at_{suffix}"] = rank
        result[f"status_at_{suffix}"] = status
        if suffix == "20":
            comparators = recalls.drop(labels=[method], errors="ignore")
            best_comparator = (
                float(comparators.max()) if not comparators.empty else np.nan
            )
            result["top20_recall"] = target_recall
            result["delta_to_best_comparator_at_20"] = (
                target_recall - best_comparator
                if np.isfinite(best_comparator)
                else None
            )
    return result
