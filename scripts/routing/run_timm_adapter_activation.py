#!/usr/bin/env python3
"""启用服务器已有 timm 分类 checkpoint 的真实 adapter 推理闭环。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository-root bootstrap above must run before these imports.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.routing.timm_adapter_runtime import (  # noqa: E402
    AdapterStageError,
    Backend,
    classification_metrics,
    clean_text,
    execute_timm_backend,
    normalize_prediction_frame,
    resolve_path,
    validate_job,
    write_adapter_outputs,
)


VALID_STAGES = {"audit", "onboarding", "replay", "all"}
JOB_REQUIRED_COLUMNS = {
    "job_id",
    "task_id",
    "dataset_id",
    "disease_family",
    "artifact_id",
    "role_candidates",
    "arch",
    "checkpoint_path",
    "config_path",
    "legacy_prediction_path",
    "data_root",
    "class_to_idx_path",
    "num_classes",
    "input_size",
    "norm",
    "batch_size",
    "device",
    "precision",
    "warmup_runs",
    "timed_runs",
    "label_structure",
    "enabled",
}
REPLAY_REQUIRED_COLUMNS = {
    "replay_id",
    "task_id",
    "scout_job_id",
    "expert_artifact_id",
    "expert_legacy_prediction_path",
    "policies",
    "budgets",
    "prediction_source_mode",
    "enabled",
}


class ActivationProtocolError(RuntimeError):
    pass


@dataclass
class ActivationResult:
    output_dir: Path
    stage: str
    files: list[Path]


def truthy(value: Any) -> bool:
    return clean_text(value).lower() not in {"", "0", "false", "no", "off"}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ActivationProtocolError(f"协议文件不存在：{path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ActivationProtocolError("读取 YAML 需要 PyYAML") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ActivationProtocolError("协议根节点必须是对象")
    return data


def _protocol_path(value: Any) -> Path:
    path = Path(clean_text(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_tables(config_path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = load_yaml(config_path)
    missing = sorted(
        {"protocol_id", "timm_adapter_jobs", "routing_replay_protocols", "output_dir"}
        - set(config)
    )
    if missing:
        raise ActivationProtocolError(f"协议缺少字段：{missing}")
    jobs_path = _protocol_path(config["timm_adapter_jobs"])
    replays_path = _protocol_path(config["routing_replay_protocols"])
    if not jobs_path.exists() or not replays_path.exists():
        raise ActivationProtocolError("jobs 或 routing replay 配置不存在")
    jobs = pd.read_csv(jobs_path, dtype=str, keep_default_na=False)
    replays = pd.read_csv(replays_path, dtype=str, keep_default_na=False)
    job_missing = sorted(JOB_REQUIRED_COLUMNS - set(jobs.columns))
    replay_missing = sorted(REPLAY_REQUIRED_COLUMNS - set(replays.columns))
    if job_missing:
        raise ActivationProtocolError(f"timm_adapter_jobs 缺少字段：{job_missing}")
    if replay_missing:
        raise ActivationProtocolError(f"routing_replay_protocols 缺少字段：{replay_missing}")
    for column in ("job_id", "artifact_id"):
        if jobs[column].duplicated().any():
            raise ActivationProtocolError(f"jobs 的 {column} 必须唯一")
    if replays["replay_id"].duplicated().any():
        raise ActivationProtocolError("replay_id 必须唯一")
    return config, jobs, replays


def write_frame(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def audit_jobs(jobs: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, Path]]]:
    rows: list[dict[str, Any]] = []
    manifests: dict[str, tuple[pd.DataFrame, Path]] = {}
    for _, job in jobs.iterrows():
        if not truthy(job["enabled"]):
            rows.append(
                {
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "status": "skipped_disabled",
                    "n_images": "",
                    "input_manifest_path": "",
                    "notes": "job disabled",
                }
            )
            continue
        row, manifest, manifest_path = validate_job(job, output_dir)
        rows.append(row)
        if manifest is not None and manifest_path is not None:
            manifests[str(job["job_id"])] = (manifest, manifest_path)
    inventory = pd.DataFrame(rows)
    write_frame(
        output_dir / "model_inventory.csv",
        inventory,
        ["job_id", "task_id", "artifact_id", "status", "n_images", "input_manifest_path", "notes"],
    )
    return inventory, manifests


def load_or_create_audit_outputs(
    jobs: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, Path]]]:
    inventory_path = output_dir / "model_inventory.csv"
    if not inventory_path.exists():
        return audit_jobs(jobs, output_dir)
    inventory = pd.read_csv(inventory_path)
    manifests: dict[str, tuple[pd.DataFrame, Path]] = {}
    for _, row in inventory.iterrows():
        if str(row.get("status", "")) != "ready_for_adapter":
            continue
        manifest_path = Path(str(row.get("input_manifest_path", "")))
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.exists():
            return audit_jobs(jobs, output_dir)
        manifests[str(row["job_id"])] = (pd.read_csv(manifest_path), manifest_path)
    return inventory, manifests


def _legacy_metrics(job: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    legacy_path = resolve_path(job["legacy_prediction_path"])
    frame = normalize_prediction_frame(legacy_path, num_classes=int(job["num_classes"]))
    metrics = classification_metrics(frame, label_structure=job["label_structure"])
    return frame, metrics


def compare_predictions(job: pd.Series, adapter_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy, legacy_metrics = _legacy_metrics(job)
    adapter = pd.read_csv(adapter_path)
    prob_cols = [f"prob_{index}" for index in range(int(job["num_classes"]))]
    merged = legacy.merge(
        adapter,
        on="image_key",
        how="inner",
        suffixes=("_legacy", "_adapter"),
        validate="one_to_one",
    )
    if merged.empty:
        prediction_status = "not_comparable"
        agreement = np.nan
        mean_diff = np.nan
        max_diff = np.nan
    else:
        agreement = float(
            (merged["pred_label_legacy"].astype(int) == merged["pred_label_adapter"].astype(int)).mean()
        )
        differences = np.concatenate(
            [
                np.abs(
                    merged[f"{column}_legacy"].astype(float).to_numpy()
                    - merged[f"{column}_adapter"].astype(float).to_numpy()
                )
                for column in prob_cols
            ]
        )
        mean_diff = float(differences.mean())
        max_diff = float(differences.max())
        if agreement == 1.0 and max_diff <= 1e-6:
            prediction_status = "matched"
        elif agreement >= 0.999 and max_diff <= 1e-3:
            prediction_status = "close_but_not_identical"
        else:
            prediction_status = "different_but_explained"
    adapter_metrics = classification_metrics(adapter, label_structure=job["label_structure"])
    prediction_row = {
        "job_id": job["job_id"],
        "task_id": job["task_id"],
        "artifact_id": job["artifact_id"],
        "legacy_prediction_path": str(resolve_path(job["legacy_prediction_path"])),
        "adapter_prediction_path": str(adapter_path),
        "n_legacy": len(legacy),
        "n_adapter": len(adapter),
        "n_overlap": len(merged),
        "pred_label_agreement": agreement,
        "mean_abs_prob_diff": mean_diff,
        "max_abs_prob_diff": max_diff,
        "accuracy_diff": adapter_metrics["accuracy"] - legacy_metrics["accuracy"],
        "macro_f1_diff": adapter_metrics["macro_f1"] - legacy_metrics["macro_f1"],
        "status": prediction_status,
        "notes": "同 checkpoint/transform/split 的 sanity check；不是 strict reproduction 声明",
    }
    baseline_row = {
        "job_id": job["job_id"],
        "task_id": job["task_id"],
        "artifact_id": job["artifact_id"],
        "n_images_legacy": legacy_metrics["n_images"],
        "n_images_adapter": adapter_metrics["n_images"],
        "accuracy_legacy": legacy_metrics["accuracy"],
        "accuracy_adapter": adapter_metrics["accuracy"],
        "accuracy_diff": adapter_metrics["accuracy"] - legacy_metrics["accuracy"],
        "macro_f1_legacy": legacy_metrics["macro_f1"],
        "macro_f1_adapter": adapter_metrics["macro_f1"],
        "macro_f1_diff": adapter_metrics["macro_f1"] - legacy_metrics["macro_f1"],
        "qwk_legacy": legacy_metrics["qwk"],
        "qwk_adapter": adapter_metrics["qwk"],
        "status": prediction_status,
        "notes": "sanity check；差异不被强行修正",
    }
    return prediction_row, baseline_row


def run_onboarding(
    jobs: pd.DataFrame,
    output_dir: Path,
    *,
    backend: Backend,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory, manifests = load_or_create_audit_outputs(jobs, output_dir)
    inventory_status = dict(zip(inventory["job_id"].astype(str), inventory["status"].astype(str)))
    summary_rows: list[dict[str, Any]] = []
    prediction_checks: list[dict[str, Any]] = []
    baseline_checks: list[dict[str, Any]] = []
    for _, job in jobs.iterrows():
        job_id = str(job["job_id"])
        status = inventory_status.get(job_id, "skipped_disabled")
        if status != "ready_for_adapter":
            summary_rows.append(
                {
                    "job_id": job_id,
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "adapter_id": "timm_classifier_v1",
                    "status": status,
                    "n_images": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "qwk": "",
                    "estimated_forward_ms_per_image": "",
                    "cost_scope": "",
                    "predictions_path": "",
                    "outputs_dir": "",
                    "notes": inventory.loc[inventory["job_id"] == job_id, "notes"].iloc[0],
                }
            )
            continue
        manifest, manifest_path = manifests[job_id]
        try:
            result = backend(job, manifest)
            summary = write_adapter_outputs(job, manifest, manifest_path, result, output_dir)
            summary_rows.append(summary)
            prediction_row, baseline_row = compare_predictions(
                job,
                Path(summary["predictions_path"]),
            )
            prediction_checks.append(prediction_row)
            baseline_checks.append(baseline_row)
        except AdapterStageError as exc:
            summary_rows.append(
                {
                    "job_id": job_id,
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "adapter_id": "timm_classifier_v1",
                    "status": exc.status,
                    "n_images": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "qwk": "",
                    "estimated_forward_ms_per_image": "",
                    "cost_scope": "",
                    "predictions_path": "",
                    "outputs_dir": "",
                    "notes": str(exc)[:1000],
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "job_id": job_id,
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "adapter_id": "timm_classifier_v1",
                    "status": "failed_inference",
                    "n_images": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "qwk": "",
                    "estimated_forward_ms_per_image": "",
                    "cost_scope": "",
                    "predictions_path": "",
                    "outputs_dir": "",
                    "notes": str(exc)[:1000],
                }
            )
    summary = pd.DataFrame(summary_rows)
    prediction_check = pd.DataFrame(prediction_checks)
    baseline_check = pd.DataFrame(baseline_checks)
    write_frame(output_dir / "adapter_job_summary.csv", summary)
    write_frame(
        output_dir / "adapter_vs_legacy_prediction_check.csv",
        prediction_check,
        [
            "job_id", "task_id", "artifact_id", "legacy_prediction_path",
            "adapter_prediction_path", "n_legacy", "n_adapter", "n_overlap",
            "pred_label_agreement", "mean_abs_prob_diff", "max_abs_prob_diff",
            "accuracy_diff", "macro_f1_diff", "status", "notes",
        ],
    )
    write_frame(
        output_dir / "adapter_vs_legacy_baseline_check.csv",
        baseline_check,
        [
            "job_id", "task_id", "artifact_id", "n_images_legacy", "n_images_adapter",
            "accuracy_legacy", "accuracy_adapter", "accuracy_diff", "macro_f1_legacy",
            "macro_f1_adapter", "macro_f1_diff", "qwk_legacy", "qwk_adapter", "status", "notes",
        ],
    )
    _publish_adapter_aggregates(output_dir)
    return summary, prediction_check, baseline_check


def _publish_adapter_aggregates(output_dir: Path) -> None:
    job_dirs = sorted((output_dir / "onboarded_models").glob("*"))
    specifications = {
        "model_baselines_from_adapters.csv": "model_baseline.csv",
        "forward_cost_summary_from_adapters.csv": "forward_cost_summary.csv",
        "adapter_manifest.csv": "adapter_manifest.csv",
    }
    for target, source_name in specifications.items():
        frames = [pd.read_csv(path / source_name) for path in job_dirs if (path / source_name).exists()]
        write_frame(output_dir / target, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    onboarded_rows: list[dict[str, Any]] = []
    for job_dir in job_dirs:
        manifest_path = job_dir / "adapter_manifest.csv"
        if not manifest_path.exists():
            continue
        row = pd.read_csv(manifest_path).iloc[0]
        onboarded_rows.append(
            {
                "job_id": row["job_id"],
                "task_id": row["task_id"],
                "artifact_id": row["artifact_id"],
                "predictions_path": row["predictions_path"],
                "baseline_path": row["model_baseline_path"],
                "cost_path": row["forward_cost_summary_path"],
                "manifest_path": str(manifest_path),
            }
        )
    write_frame(output_dir / "onboarded_models.csv", pd.DataFrame(onboarded_rows))


def _uncertainty(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "low_confidence":
        return 1.0 - frame["confidence"].astype(float)
    if policy == "low_margin":
        return 1.0 - frame["margin"].astype(float)
    if policy == "high_entropy":
        return frame["entropy"].astype(float)
    raise ActivationProtocolError(f"不支持 replay policy={policy}")


def run_replays(
    jobs: pd.DataFrame,
    replays: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    job_map = {str(row["job_id"]): row for _, row in jobs.iterrows()}
    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for _, replay in replays.iterrows():
        if not truthy(replay["enabled"]):
            continue
        replay_id = str(replay["replay_id"])
        scout_job_id = str(replay["scout_job_id"])
        base = {
            "replay_id": replay_id,
            "task_id": replay["task_id"],
            "routing_type": "single_scout",
            "prediction_source_mode": replay["prediction_source_mode"],
            "scout_artifact_ids": job_map.get(scout_job_id, {}).get("artifact_id", scout_job_id),
            "expert_artifact_id": replay["expert_artifact_id"],
        }
        scout_path = output_dir / "onboarded_models" / scout_job_id / "predictions.csv"
        expert_path = resolve_path(replay["expert_legacy_prediction_path"])
        if not scout_path.exists():
            summary_rows.append({**base, "status": "skipped_missing_scout_predictions", "n_rows": 0, "notes": str(scout_path)})
            continue
        if not expert_path.exists():
            summary_rows.append({**base, "status": "skipped_missing_expert_predictions", "n_rows": 0, "notes": str(expert_path)})
            continue
        job = job_map[scout_job_id]
        scout = pd.read_csv(scout_path)
        expert = normalize_prediction_frame(expert_path, num_classes=int(job["num_classes"]))
        merged = scout.merge(
            expert[["image_key", "true_label", "pred_label"]].rename(
                columns={"true_label": "expert_true_label", "pred_label": "expert_pred_label"}
            ),
            on="image_key",
            how="inner",
            validate="one_to_one",
        )
        if merged.empty or not np.array_equal(
            merged["true_label"].astype(int), merged["expert_true_label"].astype(int)
        ):
            summary_rows.append({**base, "status": "skipped_incompatible_predictions", "n_rows": len(merged), "notes": "标签或 image_key 不一致"})
            continue
        for policy in clean_text(replay["policies"]).split("|"):
            score = _uncertainty(merged, policy)
            for budget_text in clean_text(replay["budgets"]).split("|"):
                budget = float(budget_text)
                selected_n = min(len(merged), max(0, int(round(len(merged) * budget))))
                ranked = pd.DataFrame({"image_key": merged["image_key"], "score": score}).sort_values(
                    ["score", "image_key"], ascending=[False, True], kind="mergesort"
                )
                selected = set(ranked.head(selected_n)["image_key"].astype(str))
                routed_pred = np.where(
                    merged["image_key"].astype(str).isin(selected),
                    merged["expert_pred_label"].astype(int),
                    merged["pred_label"].astype(int),
                )
                routed = merged.copy()
                routed["pred_label"] = routed_pred
                metrics = classification_metrics(routed, label_structure=job["label_structure"])
                result_rows.append(
                    {
                        **base,
                        "policy": policy,
                        "budget": budget,
                        "n_rows": len(merged),
                        "selected_n": selected_n,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "qwk": metrics["qwk"],
                    }
                )
        subset = [row for row in result_rows if row["replay_id"] == replay_id]
        best = max(subset, key=lambda row: (row["macro_f1"], row["accuracy"]))
        summary_rows.append(
            {
                **base,
                "status": "completed",
                "n_rows": best["n_rows"],
                "best_policy": best["policy"],
                "best_budget": best["budget"],
                "best_accuracy": best["accuracy"],
                "best_macro_f1": best["macro_f1"],
                "notes": "工程链路 sanity replay；mixed adapter/legacy 不是新科研实验",
            }
        )
    results = pd.DataFrame(result_rows)
    summary = pd.DataFrame(summary_rows)
    write_frame(output_dir / "single_scout_routing_results_from_adapters.csv", results)
    write_frame(output_dir / "routing_replay_summary.csv", summary)
    return results, summary


def render_report(output_dir: Path) -> None:
    def count_status(name: str, status: str) -> int:
        path = output_dir / name
        if not path.exists():
            return 0
        frame = pd.read_csv(path)
        return int((frame.get("status", pd.Series(dtype=str)).astype(str) == status).sum())

    inventory_path = output_dir / "model_inventory.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    body = f"""<!doctype html>
<html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>v0.8.5c timm adapter</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1080px;margin:40px auto;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px;text-align:left}}.note{{background:#f4f7fa;padding:16px;border-left:4px solid #16877c}}</style>
<body><h1>v0.8.5c：真实 timm 分类模型 adapter 启用</h1>
<p>已审计模型：{len(inventory)}；ready：{count_status('model_inventory.csv','ready_for_adapter')}；completed：{count_status('adapter_job_summary.csv','completed')}；replay completed：{count_status('routing_replay_summary.csv','completed')}。</p>
<div class=\"note\">forward-only cost 只计模型前向传播，不包含图像读取、解码、预处理、CPU-GPU 传输、模型加载、后处理、服务排队，因此不是真实部署端到端延迟。</div>
<p>adapter 与历史 prediction 的比较是 sanity check，不宣称 strict reproduction。mixed_adapter_legacy replay 只验证工程链路，不作为新的科研结论。</p>
</body></html>"""
    (output_dir / "summary.html").write_text(body, encoding="utf-8")


def run_protocol(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    stage: str = "all",
    dry_run: bool = False,
    backend: Backend | None = None,
) -> ActivationResult:
    if stage not in VALID_STAGES:
        raise ActivationProtocolError(f"不支持 stage={stage}")
    config, jobs, replays = load_tables(config_path)
    target = output_dir or _protocol_path(config["output_dir"])
    if dry_run:
        print("[DRY-RUN] v0.8.5c protocol valid")
        return ActivationResult(target, stage, [])
    target.mkdir(parents=True, exist_ok=True)
    selected_backend = backend or execute_timm_backend
    if stage in {"audit", "all"}:
        audit_jobs(jobs, target)
    if stage in {"onboarding", "all"}:
        run_onboarding(jobs, target, backend=selected_backend)
    if stage in {"replay", "all"}:
        run_replays(jobs, replays, target)
    render_report(target)
    files = [path for path in target.rglob("*") if path.is_file()]
    print(f"[DONE] v0.8.5c stage={stage} outputs: {target}")
    return ActivationResult(target, stage, files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=sorted(VALID_STAGES), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_protocol(
        args.config,
        output_dir=args.output_dir,
        stage=args.stage,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
