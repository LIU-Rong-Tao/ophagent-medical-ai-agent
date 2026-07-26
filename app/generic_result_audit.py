"""Generic prediction-table import, validation, and model-error risk audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


FIELD_CANDIDATES = {
    "case_id": ("case_id", "sample_id", "image_id", "record_id", "id"),
    "true_label": ("y_true", "true_label", "label", "ground_truth", "target"),
    "pred_label": ("y_pred", "pred_label", "prediction", "predicted_class"),
    "confidence": ("confidence", "max_probability", "score"),
    "split": ("split", "dataset_split", "partition"),
    "latency": ("latency_ms", "latency", "inference_ms"),
}
PROBABILITY_PREFIXES = ("prob_", "probability_", "score_")
KNOWN_VARIANT_SUFFIXES = {
    "_no_tta": "base",
    "_base": "base",
    "_tta": "tta",
    "_ensemble": "ensemble",
}
REVIEW_BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
LEAKAGE_STATUS_CLEAR = "未发现明显问题"
LEAKAGE_STATUS_SUSPICIOUS = "发现可疑风险"
LEAKAGE_STATUS_UNKNOWN = "当前无法评估"


@dataclass(frozen=True)
class VariantMapping:
    name: str
    prediction_column: str | None = None
    probability_columns: Mapping[str, str] = field(default_factory=dict)
    confidence_column: str | None = None
    latency_column: str | None = None


@dataclass(frozen=True)
class ResultTableMapping:
    case_id_column: str
    true_label_column: str
    variants: tuple[VariantMapping, ...]
    split_column: str | None = None
    metadata_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationIssue:
    check_id: str
    severity: str
    message: str
    count: int = 0


@dataclass
class ValidationReport:
    summary: dict[str, Any]
    issues: list[ValidationIssue]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def offline_evaluation_eligible(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_imported": True,
            "validation_passed": self.passed,
            "offline_evaluation_eligible": self.offline_evaluation_eligible,
            "adapter_implemented": False,
            "task_inference_ready": False,
            "route_eligible": False,
            "summary": self.summary,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass
class AuditResult:
    primary_variant: str
    summary: pd.DataFrame
    class_metrics: pd.DataFrame
    confusion_pairs: pd.DataFrame
    case_risk_scores: pd.DataFrame
    review_budget_results: pd.DataFrame
    risk_coverage: pd.DataFrame
    variant_stability: pd.DataFrame
    leakage_checks: pd.DataFrame


@dataclass
class ObservedPositiveAudit:
    """Positive-only audit that never treats an unobserved class as a negative."""

    summary: dict[str, Any]
    case_scores: pd.DataFrame


def _parse_observed_label_ids(value: Any, *, n_classes: int) -> tuple[int, ...]:
    labels = tuple(
        sorted(
            {
                int(item.strip())
                for item in re.split(r"[;|]", str(value))
                if item.strip()
            }
        )
    )
    if not labels or labels[0] < 0 or labels[-1] >= n_classes:
        raise ValueError("观测阳性标签为空或超出概率列范围。")
    return labels


def run_observed_positive_audit(
    predictions: pd.DataFrame,
    *,
    probability_columns: Iterable[str],
    high_confidence_threshold: float = 0.8,
) -> ObservedPositiveAudit:
    """Audit ranks of observed positives without inferring complete multilabel truth."""

    if not 0 <= high_confidence_threshold <= 1:
        raise ValueError("高置信阈值必须位于 [0,1]。")
    probability_columns = list(probability_columns)
    required = {"case_id", "observed_label_ids"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"观测阳性审计缺少字段：{missing}")
    if len(probability_columns) < 2:
        raise ValueError("观测阳性审计至少需要两个类别概率列。")
    probabilities = predictions[probability_columns].astype(float).to_numpy()
    if (
        probabilities.shape != (len(predictions), len(probability_columns))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("观测阳性审计收到非法概率矩阵。")

    ranking = np.argsort(-probabilities, axis=1, kind="stable")
    inverse_rank = np.empty_like(ranking)
    inverse_rank[np.arange(len(predictions))[:, None], ranking] = np.arange(
        1, len(probability_columns) + 1
    )
    rows: list[dict[str, Any]] = []
    for row_index, (_, source) in enumerate(predictions.reset_index(drop=True).iterrows()):
        observed = _parse_observed_label_ids(
            source["observed_label_ids"],
            n_classes=len(probability_columns),
        )
        observed_array = np.asarray(observed, dtype=int)
        ranks = inverse_rank[row_index, observed_array]
        top_class = int(ranking[row_index, 0])
        confidence = float(probabilities[row_index, top_class])
        observed_mass = float(probabilities[row_index, observed_array].sum())
        top1_consistent = top_class in observed
        rows.append(
            {
                "case_id": str(source["case_id"]),
                "observed_label_ids": ";".join(str(item) for item in observed),
                "observed_label_count": len(observed),
                "predicted_class": top_class,
                "confidence": confidence,
                "best_observed_rank": int(ranks.min()),
                "mean_observed_reciprocal_rank": float(np.mean(1.0 / ranks)),
                "observed_positive_probability_mass": observed_mass,
                "top1_observed_consistent": top1_consistent,
                "high_confidence_observed_label_inconsistency": bool(
                    not top1_consistent and confidence >= high_confidence_threshold
                ),
                "observed_label_review_score": float(
                    0.5 * (1.0 - observed_mass)
                    + 0.5 * ((ranks.min() - 1) / (len(probability_columns) - 1))
                ),
            }
        )
    case_scores = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_cases": int(len(case_scores)),
        "n_classes": int(len(probability_columns)),
        "label_semantics": "observed_positive_only",
        "unobserved_classes_treated_as_negative": False,
        "mean_observed_positive_probability_mass": float(
            case_scores["observed_positive_probability_mass"].mean()
        ),
        "mean_observed_reciprocal_rank": float(
            case_scores["mean_observed_reciprocal_rank"].mean()
        ),
        "median_best_observed_rank": float(
            case_scores["best_observed_rank"].median()
        ),
        "top1_observed_consistency": float(
            case_scores["top1_observed_consistent"].mean()
        ),
        "high_confidence_observed_label_inconsistency_count": int(
            case_scores["high_confidence_observed_label_inconsistency"].sum()
        ),
    }
    for cutoff in (1, 3, 5, 10):
        if cutoff <= len(probability_columns):
            summary[f"observed_positive_hit_at_{cutoff}"] = float(
                (case_scores["best_observed_rank"] <= cutoff).mean()
            )
    return ObservedPositiveAudit(summary=summary, case_scores=case_scores)


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def detect_field_candidates(columns: Iterable[Any]) -> dict[str, list[str]]:
    """Return ordered field candidates without assuming a task or class count."""

    originals = [str(column) for column in columns]
    normalized = {_normalized_name(column): column for column in originals}
    result: dict[str, list[str]] = {}
    for field_name, aliases in FIELD_CANDIDATES.items():
        exact = [normalized[alias] for alias in aliases if alias in normalized]
        fuzzy = [
            column
            for column in originals
            if column not in exact
            and any(alias in _normalized_name(column) for alias in aliases)
        ]
        result[field_name] = [*exact, *fuzzy]
    return result


def _split_probability_column(column: str) -> tuple[str, str] | None:
    text = str(column).strip()
    lowered = text.casefold()
    prefix = next((item for item in PROBABILITY_PREFIXES if lowered.startswith(item)), None)
    if prefix is None:
        return None
    remainder = text[len(prefix) :]
    lowered_remainder = lowered[len(prefix) :]
    variant = "base"
    for suffix, variant_name in sorted(
        KNOWN_VARIANT_SUFFIXES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if lowered_remainder.endswith(suffix):
            remainder = remainder[: -len(suffix)]
            variant = variant_name
            break
    return variant, remainder


def detect_probability_groups(columns: Iterable[Any]) -> dict[str, dict[str, str]]:
    """Detect probability groups, including but not requiring common TTA suffixes."""

    groups: dict[str, dict[str, str]] = {}
    for raw_column in columns:
        column = str(raw_column)
        parsed = _split_probability_column(column)
        if parsed is None:
            continue
        variant, class_name = parsed
        if class_name:
            groups.setdefault(variant, {})[class_name] = column
    return groups


def detect_variant_predictions(columns: Iterable[Any]) -> dict[str, str]:
    originals = [str(column) for column in columns]
    result: dict[str, str] = {}
    for column in originals:
        normalized = _normalized_name(column)
        if normalized in FIELD_CANDIDATES["pred_label"]:
            result.setdefault("base", column)
            continue
        for suffix, variant in KNOWN_VARIANT_SUFFIXES.items():
            if normalized.endswith(suffix) and any(
                normalized.startswith(alias) for alias in FIELD_CANDIDATES["pred_label"]
            ):
                result[variant] = column
    return result


def detect_variant_field(columns: Iterable[Any], field_name: str) -> dict[str, str]:
    originals = [str(column) for column in columns]
    aliases = FIELD_CANDIDATES[field_name]
    result: dict[str, str] = {}
    for column in originals:
        normalized = _normalized_name(column)
        if normalized in aliases:
            result.setdefault("base", column)
            continue
        for suffix, variant in KNOWN_VARIANT_SUFFIXES.items():
            if normalized.endswith(suffix) and any(normalized.startswith(alias) for alias in aliases):
                result[variant] = column
    return result


def suggest_mapping(frame: pd.DataFrame) -> ResultTableMapping | None:
    candidates = detect_field_candidates(frame.columns)
    if not candidates["case_id"] or not candidates["true_label"]:
        return None
    probability_groups = detect_probability_groups(frame.columns)
    predictions = detect_variant_predictions(frame.columns)
    confidences = detect_variant_field(frame.columns, "confidence")
    latencies = detect_variant_field(frame.columns, "latency")
    variant_names = list(dict.fromkeys([*probability_groups, *predictions])) or ["base"]
    variants = tuple(
        VariantMapping(
            name=name,
            prediction_column=predictions.get(name),
            probability_columns=probability_groups.get(name, {}),
            confidence_column=confidences.get(name),
            latency_column=latencies.get(name),
        )
        for name in variant_names
    )
    return ResultTableMapping(
        case_id_column=candidates["case_id"][0],
        true_label_column=candidates["true_label"][0],
        split_column=candidates["split"][0] if candidates["split"] else None,
        variants=variants,
    )


def read_result_table(
    source: bytes | BinaryIO | str | Path,
    *,
    filename: str,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    """Read CSV/XLS/XLSX bytes with explicit, user-facing format failures."""

    suffix = Path(filename).suffix.casefold()
    if isinstance(source, bytes):
        payload: Any = BytesIO(source)
    else:
        payload = source
    if suffix == ".csv":
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return pd.read_csv(BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 编码无法识别，请另存为 UTF-8 CSV 后重试。")
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError("仅支持 .csv、.xlsx 和 .xls 结果表。")
    try:
        return pd.read_excel(payload, sheet_name=sheet_name if sheet_name is not None else 0)
    except ImportError as exc:
        if suffix == ".xls":
            raise ValueError("读取 .xls 需要安装 xlrd；也可先另存为 .xlsx。") from exc
        raise


def list_excel_sheets(source: bytes, *, filename: str) -> list[str]:
    suffix = Path(filename).suffix.casefold()
    if suffix == ".csv":
        return []
    try:
        return list(pd.ExcelFile(BytesIO(source)).sheet_names)
    except ImportError as exc:
        if suffix == ".xls":
            raise ValueError("读取 .xls 需要安装 xlrd；也可先另存为 .xlsx。") from exc
        raise


def _clean_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else None


def normalize_result_table(frame: pd.DataFrame, mapping: ResultTableMapping) -> pd.DataFrame:
    required = {mapping.case_id_column, mapping.true_label_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"字段映射指向不存在的列：{', '.join(missing)}")
    if not mapping.variants:
        raise ValueError("至少需要配置一个预测版本。")

    normalized_frames: list[pd.DataFrame] = []
    for variant in mapping.variants:
        if not variant.prediction_column and not variant.probability_columns:
            raise ValueError(f"预测版本 {variant.name} 缺少预测标签和概率列。")
        selected_columns = [
            column
            for column in (
                variant.prediction_column,
                variant.confidence_column,
                variant.latency_column,
            )
            if column
        ]
        selected_columns.extend(variant.probability_columns.values())
        absent = sorted(set(selected_columns) - set(frame.columns))
        if absent:
            raise ValueError(f"预测版本 {variant.name} 映射列不存在：{', '.join(absent)}")

        result = pd.DataFrame(
            {
                "case_id": frame[mapping.case_id_column].map(_clean_label),
                "true_label": frame[mapping.true_label_column].map(_clean_label),
                "variant": str(variant.name).strip() or "base",
                "source_row": np.arange(len(frame), dtype=int),
            }
        )
        if mapping.split_column:
            result["split"] = frame[mapping.split_column].map(_clean_label)
        if variant.prediction_column:
            result["y_pred"] = frame[variant.prediction_column].map(_clean_label)
        probability_names: list[str] = []
        for class_name, column in variant.probability_columns.items():
            canonical = str(class_name).strip()
            output_column = f"prob::{canonical}"
            probability_names.append(output_column)
            result[output_column] = pd.to_numeric(frame[column], errors="coerce")
        if probability_names:
            values = result[probability_names].to_numpy(dtype=float)
            finite_for_argmax = np.where(np.isfinite(values), values, -np.inf)
            argmax = finite_for_argmax.argmax(axis=1)
            inferred = [probability_names[index].removeprefix("prob::") for index in argmax]
            if "y_pred" not in result:
                result["y_pred"] = inferred
            result["argmax_label"] = inferred
            result["probability_complete"] = np.isfinite(values).all(axis=1)
        if variant.confidence_column:
            result["reported_confidence"] = pd.to_numeric(
                frame[variant.confidence_column], errors="coerce"
            )
        if variant.latency_column:
            result["latency_ms"] = pd.to_numeric(frame[variant.latency_column], errors="coerce")
        normalized_frames.append(result)
    return pd.concat(normalized_frames, ignore_index=True, sort=False)


def validate_normalized_predictions(
    normalized: pd.DataFrame,
    *,
    probability_tolerance: float = 0.01,
    numeric_tolerance: float = 1e-6,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    required = {"case_id", "true_label", "variant", "y_pred"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        issues.append(ValidationIssue("required_columns", "error", f"缺少必要字段：{', '.join(missing)}"))
        return ValidationReport({"total_rows": len(normalized)}, issues)

    variants = normalized["variant"].dropna().astype(str).unique().tolist()
    per_variant_counts = normalized.groupby("variant")["case_id"].nunique(dropna=True).to_dict()
    duplicate_pairs = int(normalized.duplicated(["case_id", "variant"], keep=False).sum())
    missing_case = int(normalized["case_id"].isna().sum())
    missing_truth = int(normalized["true_label"].isna().sum())
    missing_pred = int(normalized["y_pred"].isna().sum())
    for check_id, count, message in (
        ("missing_case_id", missing_case, "存在缺失病例标识"),
        ("missing_true_label", missing_truth, "存在缺失真实标签"),
        ("missing_prediction", missing_pred, "存在缺失预测标签"),
        ("duplicate_case_variant", duplicate_pairs, "同一预测版本内存在重复病例标识"),
    ):
        if count:
            issues.append(ValidationIssue(check_id, "error", message, count))
    if len(set(per_variant_counts.values())) > 1:
        issues.append(
            ValidationIssue(
                "variant_coverage",
                "warning",
                "不同预测版本覆盖的唯一病例数不一致",
                max(per_variant_counts.values()) - min(per_variant_counts.values()),
            )
        )

    probability_columns = [column for column in normalized if column.startswith("prob::")]
    rows_with_probabilities = 0
    if probability_columns:
        for variant_name, group in normalized.groupby("variant", sort=False):
            active_columns = [column for column in probability_columns if group[column].notna().any()]
            if not active_columns:
                continue
            values = group[active_columns].to_numpy(dtype=float)
            rows_with_probabilities += len(group)
            non_finite = ~np.isfinite(values)
            if non_finite.any():
                issues.append(
                    ValidationIssue(
                        f"probability_missing::{variant_name}",
                        "error",
                        f"预测版本 {variant_name} 的类别概率不完整",
                        int(non_finite.any(axis=1).sum()),
                    )
                )
                continue
            out_of_range = (values < 0) | (values > 1)
            if out_of_range.any():
                issues.append(
                    ValidationIssue(
                        f"probability_range::{variant_name}",
                        "error",
                        f"预测版本 {variant_name} 存在超出 [0,1] 的概率",
                        int(out_of_range.any(axis=1).sum()),
                    )
                )
            invalid_sum = np.abs(values.sum(axis=1) - 1.0) > probability_tolerance
            if invalid_sum.any():
                issues.append(
                    ValidationIssue(
                        f"probability_sum::{variant_name}",
                        "error",
                        f"预测版本 {variant_name} 存在概率和不接近 1 的记录",
                        int(invalid_sum.sum()),
                    )
                )
            if "argmax_label" in group:
                mismatch = group["y_pred"].astype(str) != group["argmax_label"].astype(str)
                if mismatch.any():
                    issues.append(
                        ValidationIssue(
                            f"prediction_argmax::{variant_name}",
                            "error",
                            f"预测版本 {variant_name} 的 y_pred 与概率最大类别不一致",
                            int(mismatch.sum()),
                        )
                    )
            if "reported_confidence" in group:
                computed = values.max(axis=1)
                reported = group["reported_confidence"].to_numpy(dtype=float)
                mismatch = ~np.isfinite(reported) | (np.abs(reported - computed) > numeric_tolerance)
                if mismatch.any():
                    issues.append(
                        ValidationIssue(
                            f"confidence_mismatch::{variant_name}",
                            "warning",
                            f"预测版本 {variant_name} 的 confidence 与最大概率不一致，将采用计算值",
                            int(mismatch.sum()),
                        )
                    )

    class_values = pd.concat(
        [normalized["true_label"], normalized["y_pred"]], ignore_index=True
    ).dropna()
    summary = {
        "total_rows": int(len(normalized)),
        "unique_cases": int(normalized["case_id"].nunique(dropna=True)),
        "class_count": int(class_values.astype(str).nunique()),
        "variant_count": len(variants),
        "variants": variants,
        "per_variant_case_count": {str(key): int(value) for key, value in per_variant_counts.items()},
        "duplicate_case_variant_rows": duplicate_pairs,
        "missing_true_labels": missing_truth,
        "missing_predictions": missing_pred,
        "has_complete_probabilities": bool(probability_columns and rows_with_probabilities == len(normalized)),
    }
    return ValidationReport(summary, issues)


def _probability_columns_for_group(group: pd.DataFrame) -> list[str]:
    return [column for column in group if column.startswith("prob::") and group[column].notna().any()]


def _with_case_risk_scores(group: pd.DataFrame, high_confidence_threshold: float) -> pd.DataFrame:
    result = group.copy().reset_index(drop=True)
    result["is_error"] = result["true_label"].astype(str) != result["y_pred"].astype(str)
    probability_columns = _probability_columns_for_group(result)
    if probability_columns:
        values = result[probability_columns].to_numpy(dtype=float)
        result["confidence"] = values.max(axis=1)
        safe = np.clip(values, np.finfo(float).tiny, 1.0)
        result["entropy_normalized"] = -(values * np.log(safe)).sum(axis=1) / math.log(
            len(probability_columns)
        )
        ordered = np.sort(values, axis=1)
        result["top1_top2_margin"] = ordered[:, -1] - ordered[:, -2]
        index_by_label = {
            column.removeprefix("prob::"): index for index, column in enumerate(probability_columns)
        }
        result["true_class_probability"] = [
            values[row_index, index_by_label.get(str(label), 0)]
            if str(label) in index_by_label
            else np.nan
            for row_index, label in enumerate(result["true_label"])
        ]
        result["high_confidence_error"] = result["is_error"] & (
            result["confidence"] >= high_confidence_threshold
        )
        result["priority_review_candidate"] = (
            (result["confidence"] <= result["confidence"].quantile(0.20))
            | (result["entropy_normalized"] >= result["entropy_normalized"].quantile(0.80))
            | (result["top1_top2_margin"] <= result["top1_top2_margin"].quantile(0.20))
        )
    else:
        result["confidence"] = np.nan
        result["entropy_normalized"] = np.nan
        result["top1_top2_margin"] = np.nan
        result["true_class_probability"] = np.nan
        result["high_confidence_error"] = False
        result["priority_review_candidate"] = False
    result["prediction_variant_instability"] = False

    def labels(row: pd.Series) -> str:
        values: list[str] = []
        if row["is_error"]:
            values.append("预测错误")
        if row["high_confidence_error"]:
            values.append("高置信错误")
        if pd.notna(row["entropy_normalized"]) and row["entropy_normalized"] >= 0.75:
            values.append("高不确定性")
        if pd.notna(row["top1_top2_margin"]) and row["top1_top2_margin"] <= 0.10:
            values.append("类别边界不稳定")
        if row["prediction_variant_instability"]:
            values.append("多版本预测不一致")
        if row["priority_review_candidate"]:
            values.append("优先复核候选")
        return "；".join(values) if values else "常规观察"

    result["risk_labels"] = result.apply(labels, axis=1)
    result["display_case"] = [f"病例 {index + 1:04d}" for index in range(len(result))]
    return result


def _metric_frames(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = group["true_label"].astype(str)
    pred = group["y_pred"].astype(str)
    labels = sorted(set(truth) | set(pred))
    accuracy = accuracy_score(truth, pred)
    macro = precision_recall_fscore_support(truth, pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, pred, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    summary = pd.DataFrame(
        [
            {
                "variant": str(group["variant"].iloc[0]),
                "sample_count": len(group),
                "class_count": len(labels),
                "accuracy": accuracy,
                "macro_precision": macro[0],
                "macro_recall": macro[1],
                "macro_f1": macro[2],
                "weighted_f1": weighted[2],
                "cohen_kappa": cohen_kappa_score(truth, pred, labels=labels),
                "error_count": int((truth != pred).sum()),
            }
        ]
    )
    class_metrics = pd.DataFrame(
        {
            "class_label": labels,
            "precision": per_class[0],
            "recall": per_class[1],
            "f1": per_class[2],
            "support": per_class[3].astype(int),
        }
    )
    matrix = confusion_matrix(truth, pred, labels=labels)
    pairs = pd.DataFrame(
        [
            {"true_label": true_label, "predicted_label": predicted_label, "count": int(matrix[i, j])}
            for i, true_label in enumerate(labels)
            for j, predicted_label in enumerate(labels)
            if matrix[i, j] > 0
        ]
    )
    return summary, class_metrics, pairs


def _review_tables(case_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_specs = {
        "低 confidence": ("confidence", True),
        "高 entropy": ("entropy_normalized", False),
        "低 margin": ("top1_top2_margin", True),
    }
    review_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    total_errors = int(case_scores["is_error"].sum())
    n = len(case_scores)
    for signal_name, (column, ascending) in signal_specs.items():
        if column not in case_scores or case_scores[column].isna().any():
            continue
        ranked = case_scores.sort_values(column, ascending=ascending, kind="mergesort")
        for budget in REVIEW_BUDGETS:
            reviewed_n = min(n, max(1, math.ceil(n * budget)))
            captured = int(ranked.head(reviewed_n)["is_error"].sum())
            remaining = total_errors - captured
            review_error_rate = captured / reviewed_n
            random_error_rate = total_errors / n if n else np.nan
            review_rows.append(
                {
                    "signal": signal_name,
                    "review_budget": budget,
                    "reviewed_cases": reviewed_n,
                    "captured_errors": captured,
                    "error_recall": captured / total_errors if total_errors else np.nan,
                    "review_error_rate": review_error_rate,
                    "remaining_errors": remaining,
                    "enrichment_vs_random": review_error_rate / random_error_rate
                    if random_error_rate
                    else np.nan,
                }
            )
        for review_fraction in np.linspace(0, 0.95, 20):
            reviewed_n = int(math.floor(n * review_fraction))
            retained = ranked.iloc[reviewed_n:]
            coverage_rows.append(
                {
                    "signal": signal_name,
                    "coverage": len(retained) / n if n else 0.0,
                    "selective_error_rate": float(retained["is_error"].mean())
                    if len(retained)
                    else 0.0,
                    "reviewed_cases": reviewed_n,
                }
            )
    return pd.DataFrame(review_rows), pd.DataFrame(coverage_rows)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    tiny = np.finfo(float).tiny
    left_safe = np.clip(left, tiny, 1.0)
    right_safe = np.clip(right, tiny, 1.0)
    midpoint_safe = np.clip(midpoint, tiny, 1.0)
    return 0.5 * (
        (left * np.log(left_safe / midpoint_safe)).sum(axis=1)
        + (right * np.log(right_safe / midpoint_safe)).sum(axis=1)
    )


def compute_variant_stability(normalized: pd.DataFrame, primary_variant: str) -> pd.DataFrame:
    variants = normalized["variant"].astype(str).unique().tolist()
    if len(variants) < 2:
        return pd.DataFrame()
    primary = normalized.loc[normalized["variant"].astype(str).eq(primary_variant)].copy()
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant == primary_variant:
            continue
        comparison = normalized.loc[normalized["variant"].astype(str).eq(variant)].copy()
        merged = primary.merge(comparison, on="case_id", suffixes=("_primary", "_comparison"))
        if merged.empty:
            continue
        primary_correct = merged["true_label_primary"].astype(str).eq(merged["y_pred_primary"].astype(str))
        comparison_correct = merged["true_label_comparison"].astype(str).eq(
            merged["y_pred_comparison"].astype(str)
        )
        row: dict[str, Any] = {
            "primary_variant": primary_variant,
            "comparison_variant": variant,
            "aligned_cases": len(merged),
            "prediction_changed": int(
                merged["y_pred_primary"].astype(str).ne(merged["y_pred_comparison"].astype(str)).sum()
            ),
            "error_to_correct": int((~primary_correct & comparison_correct).sum()),
            "correct_to_error": int((primary_correct & ~comparison_correct).sum()),
            "always_correct": int((primary_correct & comparison_correct).sum()),
            "always_error": int((~primary_correct & ~comparison_correct).sum()),
            "primary_accuracy": float(primary_correct.mean()),
            "comparison_accuracy": float(comparison_correct.mean()),
        }
        primary_metric_summary, _, _ = _metric_frames(primary)
        comparison_metric_summary, _, _ = _metric_frames(comparison)
        row["primary_macro_f1"] = float(primary_metric_summary.iloc[0]["macro_f1"])
        row["comparison_macro_f1"] = float(comparison_metric_summary.iloc[0]["macro_f1"])
        primary_probs = sorted(
            column.removesuffix("_primary")
            for column in merged
            if column.startswith("prob::") and column.endswith("_primary")
        )
        comparison_probs = {
            column.removesuffix("_comparison")
            for column in merged
            if column.startswith("prob::") and column.endswith("_comparison")
        }
        shared = [column for column in primary_probs if column in comparison_probs]
        if shared:
            left = merged[[f"{column}_primary" for column in shared]].to_numpy(dtype=float)
            right = merged[[f"{column}_comparison" for column in shared]].to_numpy(dtype=float)
            row["mean_probability_l1"] = float(np.abs(left - right).sum(axis=1).mean())
            row["mean_js_divergence"] = float(_js_divergence(left, right).mean())
        else:
            row["mean_probability_l1"] = np.nan
            row["mean_js_divergence"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _token_contains_label(value: Any, labels: set[str]) -> bool:
    if pd.isna(value):
        return False
    tokens = {_normalized_name(token) for token in re.split(r"[^\w]+", str(value)) if token}
    return bool(tokens & labels)


def run_lightweight_leakage_checks(
    source: pd.DataFrame,
    mapping: ResultTableMapping,
) -> pd.DataFrame:
    """Check only risks observable from the supplied result table."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, status: str, explanation: str, evidence_count: int = 0) -> None:
        rows.append(
            {
                "check_id": check_id,
                "status": status,
                "evidence_count": int(evidence_count),
                "explanation": explanation,
            }
        )

    case_ids = source[mapping.case_id_column].map(_clean_label)
    duplicate_count = int(case_ids.duplicated(keep=False).sum())
    add(
        "duplicate_case_id",
        LEAKAGE_STATUS_SUSPICIOUS if duplicate_count else LEAKAGE_STATUS_CLEAR,
        "结果表中存在重复病例标识。" if duplicate_count else "当前结果表未发现重复病例标识。",
        duplicate_count,
    )
    if mapping.split_column:
        split_counts = (
            pd.DataFrame({"case_id": case_ids, "split": source[mapping.split_column].map(_clean_label)})
            .dropna()
            .groupby("case_id")["split"]
            .nunique()
        )
        overlap = int((split_counts > 1).sum())
        add(
            "case_id_cross_split",
            LEAKAGE_STATUS_SUSPICIOUS if overlap else LEAKAGE_STATUS_CLEAR,
            "同一病例标识出现在多个 split。" if overlap else "未发现同一病例标识跨 split。",
            overlap,
        )
    else:
        add("case_id_cross_split", LEAKAGE_STATUS_UNKNOWN, "缺少 split 字段，无法检查病例跨划分重叠。")

    labels = {
        _normalized_name(value)
        for value in source[mapping.true_label_column].dropna().astype(str).unique()
        if not str(value).strip().isdigit()
    }
    excluded = {
        mapping.true_label_column,
        mapping.case_id_column,
        *(column for variant in mapping.variants for column in variant.probability_columns.values()),
        *(variant.prediction_column for variant in mapping.variants if variant.prediction_column),
        *(variant.confidence_column for variant in mapping.variants if variant.confidence_column),
    }
    candidate_columns = [column for column in source.columns if column not in excluded]
    direct_value_columns = list(
        dict.fromkeys(
            [
                mapping.case_id_column,
                *(column for column in mapping.metadata_columns if column in source.columns),
            ]
        )
    )
    shortcut_hits = 0
    shortcut_columns: list[str] = []
    if labels:
        for column in direct_value_columns:
            hits = int(source[column].map(lambda value: _token_contains_label(value, labels)).sum())
            if hits:
                shortcut_hits += hits
                shortcut_columns.append(str(column))
    add(
        "label_name_in_metadata",
        LEAKAGE_STATUS_SUSPICIOUS if shortcut_hits else LEAKAGE_STATUS_CLEAR,
        (
            f"发现元数据值直接包含类别名称；涉及列：{', '.join(shortcut_columns[:5])}。"
            if shortcut_hits
            else "未在可检查的标识或元数据值中发现直接类别名称。"
        ),
        shortcut_hits,
    )

    answer_name_pattern = re.compile(r"diagnosis|disease|ground[_ ]?truth|true[_ ]?label|target", re.I)
    answer_columns = [str(column) for column in candidate_columns if answer_name_pattern.search(str(column))]
    add(
        "obvious_answer_columns",
        LEAKAGE_STATUS_SUSPICIOUS if answer_columns else LEAKAGE_STATUS_CLEAR,
        (
            f"发现可能直接承载答案的额外字段：{', '.join(answer_columns[:5])}。"
            if answer_columns
            else "未发现额外的 diagnosis、disease、ground_truth 等明显答案字段。"
        ),
        len(answer_columns),
    )

    deterministic_columns: list[str] = []
    truth = source[mapping.true_label_column].astype(str)
    for column in mapping.metadata_columns:
        if column not in source.columns:
            continue
        values = source[column]
        unique_count = values.nunique(dropna=True)
        if unique_count < 2 or unique_count > max(20, int(len(source) * 0.5)):
            continue
        table = pd.DataFrame({"value": values.astype(str), "truth": truth}).dropna()
        if table.empty:
            continue
        purity = table.groupby("value")["truth"].apply(lambda item: item.value_counts(normalize=True).max())
        weighted_purity = float((purity * table["value"].value_counts(normalize=True)).sum())
        if weighted_purity >= 0.98:
            deterministic_columns.append(str(column))
    if not mapping.metadata_columns:
        add("metadata_label_mapping", LEAKAGE_STATUS_UNKNOWN, "未选择可检查的元数据列。")
    else:
        add(
            "metadata_label_mapping",
            LEAKAGE_STATUS_SUSPICIOUS if deterministic_columns else LEAKAGE_STATUS_CLEAR,
            (
                f"发现与真实标签近乎确定映射的低基数元数据：{', '.join(deterministic_columns[:5])}。"
                if deterministic_columns
                else "未发现与真实标签近乎确定映射的低基数元数据。"
            ),
            len(deterministic_columns),
        )
    add("patient_overlap", LEAKAGE_STATUS_UNKNOWN, "当前结果表未提供稳定患者标识和完整划分清单。")
    add("image_overlap", LEAKAGE_STATUS_UNKNOWN, "当前结果表未提供图像指纹和完整划分清单。")
    add("training_test_isolation", LEAKAGE_STATUS_UNKNOWN, "仅凭预测结果表无法确认训练、调参与测试流程隔离。")
    return pd.DataFrame(rows)


def run_generic_risk_audit(
    normalized: pd.DataFrame,
    *,
    source: pd.DataFrame,
    mapping: ResultTableMapping,
    primary_variant: str,
    high_confidence_threshold: float = 0.8,
) -> AuditResult:
    if not 0 <= high_confidence_threshold <= 1:
        raise ValueError("高置信错误阈值必须位于 [0,1]。")
    variants = normalized["variant"].astype(str).unique().tolist()
    if primary_variant not in variants:
        raise ValueError(f"找不到主预测版本：{primary_variant}")
    primary = normalized.loc[normalized["variant"].astype(str).eq(primary_variant)].copy()
    summary, class_metrics, confusion_pairs = _metric_frames(primary)
    other_summaries = []
    for variant in variants:
        if variant == primary_variant:
            continue
        variant_group = normalized.loc[normalized["variant"].astype(str).eq(variant)]
        variant_summary, _, _ = _metric_frames(variant_group)
        other_summaries.append(variant_summary)
    case_scores = _with_case_risk_scores(primary, high_confidence_threshold)
    other_predictions = normalized.loc[
        ~normalized["variant"].astype(str).eq(primary_variant), ["case_id", "y_pred"]
    ]
    if not other_predictions.empty:
        primary_predictions = case_scores.set_index("case_id")["y_pred"].astype(str)
        changed_cases: set[str] = set()
        for case_id, values in other_predictions.groupby("case_id")["y_pred"]:
            if str(case_id) in primary_predictions and any(
                str(value) != primary_predictions.loc[str(case_id)] for value in values
            ):
                changed_cases.add(str(case_id))
        changed = case_scores["case_id"].astype(str).isin(changed_cases)
        case_scores.loc[changed, "prediction_variant_instability"] = True
        case_scores.loc[changed, "risk_labels"] = case_scores.loc[changed, "risk_labels"].map(
            lambda value: (
                f"{value}；多版本预测不一致"
                if value and value != "常规观察"
                else "多版本预测不一致"
            )
        )
    summary["high_confidence_error_count"] = int(case_scores["high_confidence_error"].sum())
    summary["has_probabilities"] = bool(_probability_columns_for_group(primary))
    if other_summaries:
        summary = pd.concat([summary, *other_summaries], ignore_index=True, sort=False)
    review, coverage = _review_tables(case_scores)
    stability = compute_variant_stability(normalized, primary_variant)
    leakage = run_lightweight_leakage_checks(source, mapping)
    return AuditResult(
        primary_variant=primary_variant,
        summary=summary,
        class_metrics=class_metrics,
        confusion_pairs=confusion_pairs,
        case_risk_scores=case_scores,
        review_budget_results=review,
        risk_coverage=coverage,
        variant_stability=stability,
        leakage_checks=leakage,
    )


def export_audit_result(
    output_dir: str | Path,
    *,
    normalized: pd.DataFrame,
    validation: ValidationReport,
    audit: AuditResult,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(directory / "normalized_predictions.csv", index=False, encoding="utf-8-sig")
    (directory / "validation.json").write_text(
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs = {
        "audit_summary.csv": audit.summary,
        "class_metrics.csv": audit.class_metrics,
        "confusion_pairs.csv": audit.confusion_pairs,
        "case_risk_scores.csv": audit.case_risk_scores,
        "review_budget_results.csv": audit.review_budget_results,
        "risk_coverage.csv": audit.risk_coverage,
        "variant_stability.csv": audit.variant_stability,
        "leakage_checks.csv": audit.leakage_checks,
    }
    for filename, frame in outputs.items():
        frame.to_csv(directory / filename, index=False, encoding="utf-8-sig")
    return directory
