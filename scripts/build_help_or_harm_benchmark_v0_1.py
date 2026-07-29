#!/usr/bin/env python3
"""Build the read-only OphAgent Help-or-Harm feasibility benchmark v0.1.

The script consumes existing frozen probability assets.  It does not run a
model, select a route, modify a frozen split, or use retrospective outcomes to
fit a feature/threshold.  All file-system paths written to evidence tables are
logical asset URIs rather than host-private absolute paths.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.help_or_harm_benchmark import (  # noqa: E402
    ConsultationPolicyBaselineV1_1,
    LEGAL_FEATURE_COLUMNS,
    apply_expert_history_profile,
    build_cross_fitted_expert_history,
    build_cross_fitted_reference_js,
    build_scout_feature_frame,
    compute_case_outcomes,
    deterministic_random_scores,
    fit_expert_history_profile,
    jensen_shannon_divergence,
    rank_top_budget,
)


PROTOCOL_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "help_or_harm_benchmark_v0_1.json"
)
OUTPUT_RELATIVE_DIR = (
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
APTOS_REGISTRY_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "aptos_h100_prediction_assets.csv"
)
APTOS_LEGACY_IDENTITY_RELATIVE_PATH = (
    "experiments/summary/v0_7_0/external_dr_dataset_inventory.csv"
)
DEEPDRID_NATIVE_SUMMARY_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/external_validation/"
    "deepdrid/native_adaptation.json"
)
V1_1_EVIDENCE_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/outputs/"
    "route_qualification_benchmark_v1_1/"
    "route_qualification_evidence_matrix.csv"
)
DEFAULT_ASSET_ROOT = Path("/training_data/lizekun/ophagent_assets")
DEFAULT_APTOS_ROOT = Path(
    "/training_data/lizekun/data/RETFound/Data_split/APTOS2019"
)
DEFAULT_APTOS_MANIFEST = DEFAULT_ASSET_ROOT / (
    "experiments/model_hub/runs/training/aptos_dr_5class/"
    "aptos2019-flair-resnet50-official-lp-project-v1/"
    "flair-aptos-20260721T000026Z/dataset_manifest.json"
)
DEFAULT_DEEPDRID_MANIFEST = Path(
    "/training_data/lizekun/data/deepdrid/admission_v1/"
    "deepdrid_admission_manifest.csv"
)
DEFAULT_DEEPDRID_NATIVE_ROOT = DEFAULT_ASSET_ROOT / (
    "experiments/model_hub/runs/native_adaptation/deepdrid_v1.1"
)
DEFAULT_DEEPDRID_TRANSFER_ROOT = DEFAULT_ASSET_ROOT / (
    "experiments/model_hub/runs/external_transfer/deepdrid_v1.1/"
    "frozen_official_validation"
)
PROBABILITY_COLUMNS = tuple(f"prob_{index}" for index in range(5))
SCOUT_PROBABILITY_COLUMNS = tuple(f"scout_prob_{index}" for index in range(5))
BENCHMARK_BUDGETS = (0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class ModelAsset:
    model_id: str
    split_paths: Mapping[str, Path]
    checkpoint_sha256: str
    preprocessing_id: str
    forward_cost_ms_per_image: float | None
    cost_scope: str
    cost_status: str
    source_task_id: str


@dataclass(frozen=True)
class DesignSpec:
    task_id: str
    dataset_id: str
    design: str
    models: Mapping[str, ModelAsset]
    split_roles: tuple[str, ...]
    profile_source_task_id: str
    profile_source_split: str = "development"
    cost_protocol_id: str = "h100_gpu_forward_only_batch16_component_v0_1"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--aptos-root", type=Path, default=DEFAULT_APTOS_ROOT)
    parser.add_argument(
        "--aptos-manifest",
        type=Path,
        default=DEFAULT_APTOS_MANIFEST,
    )
    parser.add_argument(
        "--deepdrid-manifest",
        type=Path,
        default=DEFAULT_DEEPDRID_MANIFEST,
    )
    parser.add_argument(
        "--deepdrid-native-root",
        type=Path,
        default=DEFAULT_DEEPDRID_NATIVE_ROOT,
    )
    parser.add_argument(
        "--deepdrid-transfer-root",
        type=Path,
        default=DEFAULT_DEEPDRID_TRANSFER_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / OUTPUT_RELATIVE_DIR,
    )
    parser.add_argument(
        "--force-image-rehash",
        action="store_true",
        help="Recompute image identity inventory even when a prior formal file exists.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def current_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def logical_asset_uri(
    path: Path,
    *,
    project_root: Path,
    asset_root: Path,
) -> str:
    resolved = path.resolve()
    for prefix, root in (
        ("repo://", project_root.resolve()),
        ("ophagent_asset://", asset_root.resolve()),
    ):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return prefix + relative.as_posix()
    return f"dataset_asset://{path.name}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def perceptual_hashes(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        phash_array = np.asarray(
            grayscale.resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        ahash_array = np.asarray(
            grayscale.resize((8, 8), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        dhash_array = np.asarray(
            grayscale.resize((9, 8), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    transformed = dct(
        dct(phash_array, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )
    low = transformed[:8, :8].flatten()
    phash_bits = low > float(np.median(low[1:]))
    ahash_bits = ahash_array.flatten() > float(ahash_array.mean())
    dhash_bits = (dhash_array[:, 1:] > dhash_array[:, :-1]).flatten()
    values: list[int] = []
    for bits in (phash_bits, ahash_bits, dhash_bits):
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        values.append(value)
    return values[0], values[1], values[2]


def _load_existing_identity_inventory(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {
        "dataset_id",
        "case_id",
        "source_split",
        "y_true",
        "image_sha256",
        "phash64",
        "ahash64",
        "dhash64",
    }
    if required.issubset(frame.columns):
        return frame
    return None


def build_image_identity_inventory(
    *,
    aptos_root: Path,
    aptos_manifest_path: Path,
    deepdrid_manifest_path: Path,
    reusable_path: Path,
    force_rehash: bool,
) -> pd.DataFrame:
    if not force_rehash:
        existing = _load_existing_identity_inventory(reusable_path)
        if existing is not None:
            print(f"reused_image_identity_inventory={reusable_path}", flush=True)
            return existing

    payload = json.loads(aptos_manifest_path.read_text(encoding="utf-8"))
    aptos_records: list[dict[str, Any]] = []
    for index, entry in enumerate(payload["entries"], start=1):
        relative_path = Path(str(entry["relative_path"]))
        image_path = aptos_root / relative_path
        phash_value, ahash_value, dhash_value = perceptual_hashes(image_path)
        aptos_records.append(
            {
                "dataset_id": "APTOS2019",
                "case_id": image_path.stem,
                "patient_group_id": "",
                "eye": "",
                "exam_id": "",
                "image_id": image_path.stem,
                "view": "",
                "source_split": str(entry["split"]),
                "y_true": int(entry["label"]),
                "image_sha256": file_sha256(image_path),
                "phash64": f"{phash_value:016x}",
                "ahash64": f"{ahash_value:016x}",
                "dhash64": f"{dhash_value:016x}",
                "identity_source": "aptos_frozen_dataset_manifest_and_image_bytes",
            }
        )
        if index % 500 == 0:
            print(f"aptos_images_hashed={index}", flush=True)

    deepdrid = pd.read_csv(deepdrid_manifest_path)
    deepdrid = deepdrid.loc[deepdrid["y_true"].notna()].copy()
    deepdrid_records = pd.DataFrame(
        {
            "dataset_id": "DeepDRiD_v1.1",
            "case_id": deepdrid["case_id"].astype(str),
            "patient_group_id": deepdrid["patient_id"].fillna("").astype(str),
            "eye": deepdrid["eye"].fillna("").astype(str),
            "exam_id": deepdrid["patient_id"].fillna("").astype(str),
            "image_id": deepdrid["image_id"].fillna("").astype(str),
            "view": deepdrid["view"].fillna("").astype(str),
            "source_split": deepdrid["source_split"].astype(str),
            "y_true": deepdrid["y_true"].astype(int),
            "image_sha256": deepdrid["image_sha256"].astype(str),
            "phash64": deepdrid["phash64"].astype(str).str.zfill(16),
            "ahash64": deepdrid["ahash64"].astype(str).str.zfill(16),
            "dhash64": deepdrid["dhash64"].astype(str).str.zfill(16),
            "identity_source": "deepdrid_admission_manifest",
        }
    )
    return pd.concat(
        [pd.DataFrame(aptos_records), deepdrid_records],
        ignore_index=True,
    )


def annotate_exact_identity(inventory: pd.DataFrame) -> pd.DataFrame:
    frame = inventory.copy()
    frame["exact_group_size"] = 1
    frame["exact_group_cross_split"] = False
    frame["exact_group_label_conflict"] = False
    frame["exact_group_representative"] = True
    frame["analysis_unit_id"] = (
        frame["dataset_id"].astype(str)
        + "::sha256::"
        + frame["image_sha256"].astype(str)
    )
    for (_, _), group in frame.groupby(
        ["dataset_id", "image_sha256"],
        sort=False,
    ):
        indices = group.index
        size = len(group)
        cross_split = group["source_split"].astype(str).nunique() > 1
        label_conflict = group["y_true"].astype(int).nunique() > 1
        frame.loc[indices, "exact_group_size"] = size
        frame.loc[indices, "exact_group_cross_split"] = cross_split
        frame.loc[indices, "exact_group_label_conflict"] = label_conflict
        if size > 1 and not cross_split and not label_conflict:
            representative = (
                group.sort_values(["case_id"], kind="mergesort").index[0]
            )
            frame.loc[indices, "exact_group_representative"] = False
            frame.loc[representative, "exact_group_representative"] = True
    frame["primary_identity_eligible"] = (
        ~frame["exact_group_cross_split"].astype(bool)
        & ~frame["exact_group_label_conflict"].astype(bool)
        & frame["exact_group_representative"].astype(bool)
    )
    frame["resampling_group_id"] = np.where(
        frame["patient_group_id"].fillna("").astype(str).str.strip().ne(""),
        frame["patient_group_id"].astype(str),
        frame["analysis_unit_id"].astype(str),
    )
    return frame


def build_near_duplicate_candidates(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset in inventory.groupby("dataset_id", sort=True):
        by_split = {
            str(split): group.to_dict("records")
            for split, group in dataset.groupby("source_split", sort=True)
        }
        for left_split, right_split in combinations(sorted(by_split), 2):
            for left in by_split[left_split]:
                left_phash = int(str(left["phash64"]), 16)
                left_ahash = int(str(left["ahash64"]), 16)
                left_dhash = int(str(left["dhash64"]), 16)
                for right in by_split[right_split]:
                    if left["image_sha256"] == right["image_sha256"]:
                        continue
                    phash_distance = (
                        left_phash ^ int(str(right["phash64"]), 16)
                    ).bit_count()
                    if phash_distance > 4:
                        continue
                    ahash_distance = (
                        left_ahash ^ int(str(right["ahash64"]), 16)
                    ).bit_count()
                    dhash_distance = (
                        left_dhash ^ int(str(right["dhash64"]), 16)
                    ).bit_count()
                    if ahash_distance > 6 or dhash_distance > 6:
                        continue
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "left_case_id": left["case_id"],
                            "left_split": left_split,
                            "left_label": int(left["y_true"]),
                            "right_case_id": right["case_id"],
                            "right_split": right_split,
                            "right_label": int(right["y_true"]),
                            "phash_hamming": phash_distance,
                            "ahash_hamming": ahash_distance,
                            "dhash_hamming": dhash_distance,
                            "label_conflict": int(left["y_true"])
                            != int(right["y_true"]),
                            "status": "unconfirmed_candidate",
                            "primary_exclusion": False,
                            "use": "near_duplicate_sensitivity_only",
                        }
                    )
    columns = [
        "dataset_id",
        "left_case_id",
        "left_split",
        "left_label",
        "right_case_id",
        "right_split",
        "right_label",
        "phash_hamming",
        "ahash_hamming",
        "dhash_hamming",
        "label_conflict",
        "status",
        "primary_exclusion",
        "use",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_exact_duplicate_members(inventory: pd.DataFrame) -> pd.DataFrame:
    duplicates = inventory.loc[inventory["exact_group_size"].astype(int) > 1].copy()
    columns = [
        "dataset_id",
        "image_sha256",
        "case_id",
        "source_split",
        "y_true",
        "exact_group_size",
        "exact_group_cross_split",
        "exact_group_label_conflict",
        "exact_group_representative",
        "primary_identity_eligible",
    ]
    return duplicates.loc[:, columns].sort_values(
        ["dataset_id", "image_sha256", "source_split", "case_id"],
        kind="mergesort",
    )


def build_image_leakage_summary(
    inventory: pd.DataFrame,
    near_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, frame in inventory.groupby("dataset_id", sort=True):
        groups = list(frame.groupby("image_sha256", sort=False))
        duplicate_groups = [group for _, group in groups if len(group) > 1]
        cross_groups = [
            group
            for group in duplicate_groups
            if group["source_split"].astype(str).nunique() > 1
        ]
        conflict_groups = [
            group
            for group in duplicate_groups
            if group["y_true"].astype(int).nunique() > 1
        ]
        cross_conflict_groups = [
            group
            for group in cross_groups
            if group["y_true"].astype(int).nunique() > 1
        ]
        patient = frame["patient_group_id"].fillna("").astype(str).str.strip()
        patient_frame = frame.loc[patient.ne("")]
        patient_cross_split = (
            int(
                (
                    patient_frame.groupby("patient_group_id")["source_split"].nunique()
                    > 1
                ).sum()
            )
            if not patient_frame.empty
            else 0
        )
        candidates = near_candidates.loc[
            near_candidates["dataset_id"].astype(str).eq(str(dataset_id))
        ]
        rows.append(
            {
                "dataset_id": dataset_id,
                "images": len(frame),
                "unique_case_ids": frame["case_id"].nunique(),
                "duplicate_case_ids": int(frame["case_id"].duplicated().sum()),
                "exact_duplicate_groups": len(duplicate_groups),
                "exact_duplicate_rows": sum(len(group) for group in duplicate_groups),
                "cross_split_exact_groups": len(cross_groups),
                "cross_split_exact_rows": sum(len(group) for group in cross_groups),
                "label_conflict_exact_groups": len(conflict_groups),
                "cross_split_label_conflict_exact_groups": len(
                    cross_conflict_groups
                ),
                "patient_id_coverage": float(patient.ne("").mean()),
                "eye_id_coverage": float(
                    frame["eye"].fillna("").astype(str).str.strip().ne("").mean()
                ),
                "exam_id_coverage": float(
                    frame["exam_id"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .mean()
                ),
                "patient_cross_split_groups": patient_cross_split,
                "near_duplicate_candidate_pairs": len(candidates),
                "near_duplicate_status": "unconfirmed_candidate",
                "near_duplicate_primary_exclusion": False,
                "primary_identity_eligible_images": int(
                    frame["primary_identity_eligible"].astype(bool).sum()
                ),
                "analysis_unit": "single_CFP_image",
                "resampling_group": (
                    "patient_group_id"
                    if patient.ne("").all()
                    else "exact_image_sha256_then_case_id"
                ),
            }
        )
    return pd.DataFrame(rows)


def _first_prediction_metadata(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, nrows=1)
    frame.columns = [str(value).lstrip("\ufeff") for value in frame.columns]
    if frame.empty:
        raise ValueError(f"Prediction asset is empty: {path}")
    return {
        "checkpoint_sha256": str(frame.iloc[0].get("checkpoint_sha256", "")),
        "preprocessing_id": str(frame.iloc[0].get("preprocessing_id", "")),
    }


def load_design_specs(
    *,
    project_root: Path,
    asset_root: Path,
    deepdrid_native_root: Path,
    deepdrid_transfer_root: Path,
) -> list[DesignSpec]:
    aptos_registry = pd.read_csv(project_root / APTOS_REGISTRY_RELATIVE_PATH)
    aptos_models: dict[str, ModelAsset] = {}
    aptos_costs: dict[str, dict[str, Any]] = {}
    for _, row in aptos_registry.iterrows():
        model_id = str(row["artifact_id"])
        cost = pd.to_numeric(pd.Series([row["forward_cost_ms_per_image"]]), errors="coerce").iloc[
            0
        ]
        numeric_cost = float(cost) if pd.notna(cost) else None
        aptos_costs[model_id] = {
            "cost": numeric_cost,
            "scope": str(row["cost_scope"]),
            "status": str(row["cost_status"]),
        }
        aptos_models[model_id] = ModelAsset(
            model_id=model_id,
            split_paths={
                "development": project_root / str(row["validation_prediction_path"]),
                "retrospective_frozen": project_root
                / str(row["test_prediction_path"]),
            },
            checkpoint_sha256=str(row["checkpoint_sha256"]),
            preprocessing_id=str(row["preprocessing_id"]),
            forward_cost_ms_per_image=numeric_cost,
            cost_scope=str(row["cost_scope"]),
            cost_status=str(row["cost_status"]),
            source_task_id="aptos_dr_5class",
        )

    native_summary = json.loads(
        (project_root / DEEPDRID_NATIVE_SUMMARY_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    native_models: dict[str, ModelAsset] = {}
    for row in native_summary["models"]:
        model_id = str(row["model_id"])
        run_id = str(row["run_id"])
        base = deepdrid_native_root / model_id / run_id / "predictions"
        paths = {
            "development": base / "validation_predictions.csv",
            "retrospective_frozen": base / "test_predictions.csv",
        }
        metadata = _first_prediction_metadata(paths["development"])
        native_models[model_id] = ModelAsset(
            model_id=model_id,
            split_paths=paths,
            checkpoint_sha256=metadata["checkpoint_sha256"],
            preprocessing_id=metadata["preprocessing_id"],
            forward_cost_ms_per_image=float(row["batch16_ms_per_image"]),
            cost_scope="H100 GPU forward-only batch16",
            cost_status=(
                "partial"
                if aptos_costs.get(model_id, {}).get("status") == "partial"
                else "measured"
            ),
            source_task_id="deepdrid_dr_5class_native",
        )

    transfer_models: dict[str, ModelAsset] = {}
    for prediction_path in sorted(deepdrid_transfer_root.glob("*/predictions.csv")):
        model_id = prediction_path.parent.name
        metadata = _first_prediction_metadata(prediction_path)
        cost = aptos_costs.get(model_id, {})
        source_model = aptos_models.get(model_id)
        if source_model is None:
            raise ValueError(
                f"DeepDRiD transfer model has no APTOS profile source: {model_id}"
            )
        if metadata["checkpoint_sha256"] != source_model.checkpoint_sha256:
            raise ValueError(
                "DeepDRiD transfer checkpoint does not match its APTOS "
                f"development profile source: {model_id}"
            )
        if metadata["preprocessing_id"] != source_model.preprocessing_id:
            raise ValueError(
                "DeepDRiD transfer preprocessing does not match its APTOS "
                f"development profile source: {model_id}"
            )
        transfer_models[model_id] = ModelAsset(
            model_id=model_id,
            split_paths={"retrospective_external": prediction_path},
            checkpoint_sha256=metadata["checkpoint_sha256"],
            preprocessing_id=metadata["preprocessing_id"],
            forward_cost_ms_per_image=cost.get("cost"),
            cost_scope=str(cost.get("scope", "")),
            cost_status=str(cost.get("status", "missing")),
            source_task_id="aptos_dr_5class",
        )

    return [
        DesignSpec(
            task_id="aptos_dr_5class",
            dataset_id="APTOS2019",
            design="aptos_task_adaptation",
            models=aptos_models,
            split_roles=("development", "retrospective_frozen"),
            profile_source_task_id="aptos_dr_5class",
        ),
        DesignSpec(
            task_id="deepdrid_dr_5class_external",
            dataset_id="DeepDRiD_v1.1",
            design="cross_dataset_frozen_transfer",
            models=transfer_models,
            split_roles=("retrospective_external",),
            profile_source_task_id="aptos_dr_5class",
        ),
        DesignSpec(
            task_id="deepdrid_dr_5class_native",
            dataset_id="DeepDRiD_v1.1",
            design="deepdrid_native_adaptation",
            models=native_models,
            split_roles=("development", "retrospective_frozen"),
            profile_source_task_id="deepdrid_dr_5class_native",
        ),
    ]


def standardize_prediction(
    path: Path,
    *,
    benchmark_split: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(value).lstrip("\ufeff") for value in frame.columns]
    aliases = {
        "true_label": "y_true",
        "pred_label": "y_pred",
        "image_id": "case_id",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    if "case_id" not in frame and "image_key" in frame:
        frame["case_id"] = frame["image_key"]
    if "image_key" not in frame:
        frame["image_key"] = frame["case_id"]
    if "patient_id" not in frame:
        frame["patient_id"] = ""
    required = {
        "case_id",
        "patient_id",
        "image_key",
        "y_true",
        "y_pred",
        *PROBABILITY_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    frame["case_id"] = frame["case_id"].astype(str)
    frame["patient_id"] = frame["patient_id"].fillna("").astype(str)
    frame["image_key"] = frame["image_key"].astype(str)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="raise").astype(int)
    frame[list(PROBABILITY_COLUMNS)] = frame[list(PROBABILITY_COLUMNS)].apply(
        pd.to_numeric,
        errors="raise",
    )
    frame["benchmark_split"] = benchmark_split
    frame["prediction_split"] = (
        frame["split"].astype(str) if "split" in frame else benchmark_split
    )
    return frame


def audit_prediction_asset(
    frame: pd.DataFrame,
    *,
    task_id: str,
    dataset_id: str,
    design: str,
    model: ModelAsset,
    split: str,
    path: Path,
    project_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    sums = probabilities.sum(axis=1)
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "evaluation_design": design,
        "model_id": model.model_id,
        "benchmark_split": split,
        "prediction_asset_uri": logical_asset_uri(
            path,
            project_root=project_root,
            asset_root=asset_root,
        ),
        "prediction_asset_sha256": file_sha256(path),
        "rows": len(frame),
        "unique_case_ids": frame["case_id"].nunique(),
        "duplicate_case_ids": int(frame["case_id"].duplicated().sum()),
        "missing_true_labels": int(frame["y_true"].isna().sum()),
        "missing_predictions": int(frame["y_pred"].isna().sum()),
        "missing_probability_rows": int(np.isnan(probabilities).any(axis=1).sum()),
        "probability_out_of_range_rows": int(
            ((probabilities < -1e-8) | (probabilities > 1 + 1e-8)).any(axis=1).sum()
        ),
        "probability_sum_conflicts": int((np.abs(sums - 1.0) > 1e-5).sum()),
        "prediction_argmax_conflicts": int(
            (probabilities.argmax(axis=1) != frame["y_pred"].to_numpy()).sum()
        ),
        "patient_id_coverage": float(
            frame["patient_id"].astype(str).str.strip().ne("").mean()
        ),
        "image_key_coverage": float(
            frame["image_key"].astype(str).str.strip().ne("").mean()
        ),
        "prediction_split_values": "|".join(
            sorted(frame["prediction_split"].astype(str).unique())
        ),
        "checkpoint_sha256": model.checkpoint_sha256,
        "preprocessing_id": model.preprocessing_id,
        "asset_complete": bool(
            len(frame)
            and not frame["case_id"].duplicated().any()
            and np.isfinite(probabilities).all()
            and np.allclose(sums, 1.0, atol=1e-5)
            and np.array_equal(
                probabilities.argmax(axis=1),
                frame["y_pred"].to_numpy(),
            )
        ),
    }


def load_frozen_v1_1_routes(project_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    frame = pd.read_csv(project_root / V1_1_EVIDENCE_RELATIVE_PATH)
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        scouts = [
            value
            for value in str(row["scout_artifact_ids"]).split("|")
            if value
        ]
        if len(scouts) != 1:
            continue
        key = (
            str(row["task_id"]),
            scouts[0],
            str(row["expert_artifact_id"]),
        )
        result[key] = {
            "route_id": str(row["pairing_id"]),
            "routing_policy": str(row["routing_policy"]),
            "budget": float(row["requested_budget"]),
            "qualification_level": str(row["execution_level"]),
            "qualification_error_codes": str(row["error_codes"]),
        }
    return result


def model_cost_complete(model: ModelAsset) -> bool:
    return bool(
        model.forward_cost_ms_per_image is not None
        and np.isfinite(model.forward_cost_ms_per_image)
        and model.cost_status == "measured"
    )


def build_route_inventory(
    specs: Sequence[DesignSpec],
    *,
    v1_1_routes: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        model_ids = sorted(spec.models)
        for scout_id in model_ids:
            for expert_id in model_ids:
                if scout_id == expert_id:
                    continue
                scout = spec.models[scout_id]
                expert = spec.models[expert_id]
                frozen = v1_1_routes.get((spec.task_id, scout_id, expert_id))
                rows.append(
                    {
                        "task_id": spec.task_id,
                        "dataset_id": spec.dataset_id,
                        "evaluation_design": spec.design,
                        "route_id": (
                            f"{spec.task_id}::{scout_id}__to__{expert_id}"
                        ),
                        "scout_id": scout_id,
                        "expert_id": expert_id,
                        "split_roles": "|".join(spec.split_roles),
                        "profile_source_task_id": spec.profile_source_task_id,
                        "profile_source_split": spec.profile_source_split,
                        "scout_checkpoint_sha256": scout.checkpoint_sha256,
                        "expert_checkpoint_sha256": expert.checkpoint_sha256,
                        "scout_preprocessing_id": scout.preprocessing_id,
                        "expert_preprocessing_id": expert.preprocessing_id,
                        "cost_protocol_id": spec.cost_protocol_id,
                        "scout_cost_ms_per_image": scout.forward_cost_ms_per_image,
                        "expert_cost_ms_per_image": expert.forward_cost_ms_per_image,
                        "scout_cost_status": scout.cost_status,
                        "expert_cost_status": expert.cost_status,
                        "cost_comparable": model_cost_complete(scout)
                        and model_cost_complete(expert),
                        "formal_v1_1_route_id": (
                            str(frozen["route_id"]) if frozen else ""
                        ),
                        "formal_v1_1_policy": (
                            str(frozen["routing_policy"]) if frozen else ""
                        ),
                        "formal_v1_1_budget": (
                            float(frozen["budget"]) if frozen else np.nan
                        ),
                        "formal_v1_1_qualification_level": (
                            str(frozen["qualification_level"]) if frozen else ""
                        ),
                        "formal_v1_1_error_codes": (
                            str(frozen["qualification_error_codes"])
                            if frozen
                            else ""
                        ),
                        "v1_1_case_baseline_available": frozen is not None,
                        "v1_1_unavailable_reason": (
                            ""
                            if frozen
                            else "not_a_frozen_single_scout_v1_1_route"
                        ),
                        "route_enumeration_basis": (
                            "all_ordered_distinct_model_pairs_no_test_selection"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_semantic_compatibility_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task_id": "aptos_dr_5class",
                "dataset_id": "APTOS2019",
                "label_space": "ordered_DR_grade_0_1_2_3_4",
                "disease_semantics": "diabetic_retinopathy_grade",
                "grade_mapping": "identity_0_to_4",
                "label_source": "frozen_directory_grade_manifest",
                "preprocessing": "model_specific_registered_adapter",
                "shared_case_outcome_contract": True,
                "direct_metric_pooling_allowed": False,
                "dataset_id_allowed_as_feature": False,
                "patient_level_claim_allowed": False,
                "domain_shift_status": "in_domain_task_adaptation",
                "boundary": (
                    "grade-match error proxy; not a clinical harm or outcome label"
                ),
            },
            {
                "task_id": "deepdrid_dr_5class_external",
                "dataset_id": "DeepDRiD_v1.1",
                "label_space": "ordered_DR_grade_0_1_2_3_4",
                "disease_semantics": "diabetic_retinopathy_grade",
                "grade_mapping": "identity_0_to_4",
                "label_source": "DeepDRiD_v1.1 official grade manifest",
                "preprocessing": "same_frozen_model_adapter_cross_domain",
                "shared_case_outcome_contract": True,
                "direct_metric_pooling_allowed": False,
                "dataset_id_allowed_as_feature": False,
                "patient_level_claim_allowed": True,
                "domain_shift_status": "external_shift_without_in_domain_selection",
                "boundary": (
                    "external retrospective grade proxy; not route selection evidence"
                ),
            },
            {
                "task_id": "deepdrid_dr_5class_native",
                "dataset_id": "DeepDRiD_v1.1",
                "label_space": "ordered_DR_grade_0_1_2_3_4",
                "disease_semantics": "diabetic_retinopathy_grade",
                "grade_mapping": "identity_0_to_4",
                "label_source": "DeepDRiD_v1.1 official grade manifest",
                "preprocessing": "registered_native_adapter",
                "shared_case_outcome_contract": True,
                "direct_metric_pooling_allowed": False,
                "dataset_id_allowed_as_feature": False,
                "patient_level_claim_allowed": True,
                "domain_shift_status": "native_patient_grouped_adaptation",
                "boundary": (
                    "grade-match error proxy; not a clinical harm or outcome label"
                ),
            },
        ]
    )


def build_legal_feature_audit() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "scout_prob_0": "Scout probability for grade 0",
        "scout_prob_1": "Scout probability for grade 1",
        "scout_prob_2": "Scout probability for grade 2",
        "scout_prob_3": "Scout probability for grade 3",
        "scout_prob_4": "Scout probability for grade 4",
        "scout_confidence": "maximum Scout probability",
        "scout_entropy": "normalized Shannon entropy of Scout probabilities",
        "scout_margin": "top-1 minus top-2 Scout probability",
        "scout_severe_probability_mass": "Scout probability mass for grades 3 and 4",
        "expert_history_corrected_rate": (
            "smoothed corrected rate from other development folds"
        ),
        "expert_history_introduced_rate": (
            "smoothed introduced rate from other development folds"
        ),
        "expert_history_net": (
            "development-only history corrected rate minus introduced rate"
        ),
        "expert_history_support": "matching development-history stratum support",
        "scout_reference_js_divergence": (
            "unlabeled JS divergence from other-fold/full-development Scout reference"
        ),
    }
    for feature in LEGAL_FEATURE_COLUMNS:
        rows.append(
            {
                "feature": feature,
                "status": "allowed_v0_1",
                "definition": definitions[feature],
                "current_case_expert_output_used": False,
                "test_label_or_threshold_used": False,
                "additional_scout_inference_used": False,
                "identity_or_private_path_used": False,
            }
        )
    rows.extend(
        [
            {
                "feature": "registered_image_quality_signal",
                "status": "not_available_v0_1",
                "definition": (
                    "Allowed only after a registered algorithm and measured cost exist"
                ),
                "current_case_expert_output_used": False,
                "test_label_or_threshold_used": False,
                "additional_scout_inference_used": False,
                "identity_or_private_path_used": False,
            },
            {
                "feature": "current_case_expert_output_or_embedding",
                "status": "forbidden",
                "definition": "Post-consultation leakage",
                "current_case_expert_output_used": True,
                "test_label_or_threshold_used": False,
                "additional_scout_inference_used": False,
                "identity_or_private_path_used": False,
            },
            {
                "feature": "raw_dataset_id",
                "status": "forbidden_as_default_predictor",
                "definition": "Metadata retained for stratification only",
                "current_case_expert_output_used": False,
                "test_label_or_threshold_used": False,
                "additional_scout_inference_used": False,
                "identity_or_private_path_used": False,
            },
            {
                "feature": "test_derived_feature_or_threshold",
                "status": "forbidden",
                "definition": "Frozen retrospective outcomes cannot define a method",
                "current_case_expert_output_used": False,
                "test_label_or_threshold_used": True,
                "additional_scout_inference_used": False,
                "identity_or_private_path_used": False,
            },
            {
                "feature": "uncosted_additional_scout",
                "status": "forbidden",
                "definition": "No hidden multi-Scout inference in a pair benchmark",
                "current_case_expert_output_used": False,
                "test_label_or_threshold_used": False,
                "additional_scout_inference_used": True,
                "identity_or_private_path_used": False,
            },
        ]
    )
    return pd.DataFrame(rows)


class BenchmarkBuilder:
    def __init__(
        self,
        *,
        project_root: Path,
        asset_root: Path,
        specs: Sequence[DesignSpec],
        identity: pd.DataFrame,
        near_candidates: pd.DataFrame,
        route_inventory: pd.DataFrame,
    ) -> None:
        self.project_root = project_root
        self.asset_root = asset_root
        self.specs = {spec.task_id: spec for spec in specs}
        self.identity = {
            str(dataset_id): frame.set_index("case_id", drop=False)
            for dataset_id, frame in identity.groupby("dataset_id", sort=False)
        }
        self.near_case_ids: dict[str, set[str]] = defaultdict(set)
        for _, row in near_candidates.iterrows():
            dataset_id = str(row["dataset_id"])
            self.near_case_ids[dataset_id].update(
                [str(row["left_case_id"]), str(row["right_case_id"])]
            )
        self.route_inventory = route_inventory.set_index("route_id", drop=False)
        self.prediction_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        self.asset_audits: list[dict[str, Any]] = []
        self.base_case_cache: dict[
            tuple[str, str, str, str], tuple[pd.DataFrame, dict[str, Any]]
        ] = {}
        self.full_profile_cache: dict[
            tuple[str, str, str], tuple[dict[str, Any], np.ndarray, int]
        ] = {}

    def load_prediction(
        self,
        task_id: str,
        model_id: str,
        split: str,
    ) -> pd.DataFrame:
        key = (task_id, model_id, split)
        if key in self.prediction_cache:
            return self.prediction_cache[key]
        spec = self.specs[task_id]
        model = spec.models[model_id]
        path = model.split_paths[split]
        frame = standardize_prediction(path, benchmark_split=split)
        self.prediction_cache[key] = frame
        self.asset_audits.append(
            audit_prediction_asset(
                frame,
                task_id=task_id,
                dataset_id=spec.dataset_id,
                design=spec.design,
                model=model,
                split=split,
                path=path,
                project_root=self.project_root,
                asset_root=self.asset_root,
            )
        )
        return frame

    def build_base_cases(
        self,
        task_id: str,
        split: str,
        scout_id: str,
        expert_id: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        key = (task_id, split, scout_id, expert_id)
        if key in self.base_case_cache:
            frame, audit = self.base_case_cache[key]
            return frame.copy(), dict(audit)
        spec = self.specs[task_id]
        scout = self.load_prediction(task_id, scout_id, split)
        expert = self.load_prediction(task_id, expert_id, split)
        columns = [
            "case_id",
            "patient_id",
            "image_key",
            "y_true",
            "y_pred",
            *PROBABILITY_COLUMNS,
        ]
        merged = scout[columns].merge(
            expert[columns],
            on="case_id",
            how="outer",
            suffixes=("_scout", "_expert"),
            indicator=True,
            validate="one_to_one",
        )
        aligned = merged.loc[merged["_merge"].eq("both")].copy()
        true_conflict = (
            aligned["y_true_scout"].astype(str)
            != aligned["y_true_expert"].astype(str)
        )
        image_key_conflict = (
            aligned["image_key_scout"].astype(str)
            != aligned["image_key_expert"].astype(str)
        )
        patient_conflict = (
            aligned["patient_id_scout"].fillna("").astype(str)
            != aligned["patient_id_expert"].fillna("").astype(str)
        )
        aligned = aligned.loc[~true_conflict].copy()
        identity = self.identity[spec.dataset_id]
        missing_identity = ~aligned["case_id"].astype(str).isin(identity.index)
        if missing_identity.any():
            missing_examples = aligned.loc[missing_identity, "case_id"].head().tolist()
            raise ValueError(
                f"{task_id}/{split}: missing identity rows {missing_examples}"
            )
        identity_rows = identity.loc[aligned["case_id"].astype(str)].reset_index(
            drop=True
        )
        aligned = aligned.reset_index(drop=True)
        identity_label_conflict = (
            aligned["y_true_scout"].astype(int).to_numpy()
            != identity_rows["y_true"].astype(int).to_numpy()
        )
        identity_patient = identity_rows["patient_group_id"].fillna("").astype(str)
        prediction_patient = (
            aligned["patient_id_scout"].fillna("").astype(str).reset_index(drop=True)
        )
        identity_patient_conflict = (
            identity_patient.ne("")
            & prediction_patient.ne("")
            & identity_patient.ne(prediction_patient)
        )
        probabilities = aligned[
            [f"{column}_scout" for column in PROBABILITY_COLUMNS]
        ].to_numpy(dtype=float)
        scout_features = build_scout_feature_frame(probabilities)
        outcomes = compute_case_outcomes(
            y_true=aligned["y_true_scout"].astype(int),
            scout_pred=aligned["y_pred_scout"].astype(int),
            expert_pred=aligned["y_pred_expert"].astype(int),
        )
        route_id = f"{task_id}::{scout_id}__to__{expert_id}"
        route = self.route_inventory.loc[route_id]
        cases = pd.DataFrame(
            {
                "schema_version": "ophagent.help_or_harm_case_table.v0_1",
                "task_id": task_id,
                "dataset_id": spec.dataset_id,
                "evaluation_design": spec.design,
                "route_id": route_id,
                "scout_id": scout_id,
                "expert_id": expert_id,
                "benchmark_split": split,
                "case_id": aligned["case_id"].astype(str),
                "patient_group_id": identity_rows["patient_group_id"]
                .fillna("")
                .astype(str),
                "eye": identity_rows["eye"].fillna("").astype(str),
                "exam_id": identity_rows["exam_id"].fillna("").astype(str),
                "image_id": identity_rows["image_id"].fillna("").astype(str),
                "view": identity_rows["view"].fillna("").astype(str),
                "source_split": identity_rows["source_split"].astype(str),
                "analysis_unit_id": identity_rows["analysis_unit_id"].astype(str),
                "resampling_group_id": identity_rows[
                    "resampling_group_id"
                ].astype(str),
                "image_sha256": identity_rows["image_sha256"].astype(str),
                "y_true": aligned["y_true_scout"].astype(int),
                "scout_pred": aligned["y_pred_scout"].astype(int),
                "expert_pred": aligned["y_pred_expert"].astype(int),
                "identity_label_conflict": identity_label_conflict,
                "identity_patient_conflict": identity_patient_conflict.to_numpy(),
                "exact_group_size": identity_rows["exact_group_size"].astype(int),
                "cross_split_exact_duplicate": identity_rows[
                    "exact_group_cross_split"
                ].astype(bool),
                "exact_duplicate_label_conflict": identity_rows[
                    "exact_group_label_conflict"
                ].astype(bool),
                "exact_group_representative": identity_rows[
                    "exact_group_representative"
                ].astype(bool),
                "near_duplicate_candidate": aligned["case_id"]
                .astype(str)
                .isin(self.near_case_ids[spec.dataset_id]),
                "primary_cohort_eligible": identity_rows[
                    "primary_identity_eligible"
                ].astype(bool)
                & ~identity_label_conflict
                & ~identity_patient_conflict.to_numpy(),
                "scout_cost_ms_per_image": route["scout_cost_ms_per_image"],
                "expert_cost_ms_per_image": route["expert_cost_ms_per_image"],
                "cost_protocol_id": route["cost_protocol_id"],
                "cost_comparable": route["cost_comparable"],
                "scout_prediction_asset_sha256": next(
                    row["prediction_asset_sha256"]
                    for row in self.asset_audits
                    if row["task_id"] == task_id
                    and row["model_id"] == scout_id
                    and row["benchmark_split"] == split
                ),
                "expert_prediction_asset_sha256": next(
                    row["prediction_asset_sha256"]
                    for row in self.asset_audits
                    if row["task_id"] == task_id
                    and row["model_id"] == expert_id
                    and row["benchmark_split"] == split
                ),
            }
        )
        for index, column in enumerate(SCOUT_PROBABILITY_COLUMNS):
            cases[column] = scout_features[column].to_numpy()
            cases[f"expert_prob_{index}"] = aligned[
                f"prob_{index}_expert"
            ].to_numpy(dtype=float)
        for column in (
            "scout_confidence",
            "scout_entropy",
            "scout_margin",
            "scout_severe_probability_mass",
        ):
            cases[column] = scout_features[column].to_numpy()
        for column in outcomes:
            cases[column] = outcomes[column].to_numpy(dtype=bool)
        cases["near_duplicate_sensitivity_eligible"] = (
            cases["primary_cohort_eligible"].astype(bool)
            & ~cases["near_duplicate_candidate"].astype(bool)
        )
        audit = {
            "task_id": task_id,
            "dataset_id": spec.dataset_id,
            "evaluation_design": spec.design,
            "route_id": route_id,
            "scout_id": scout_id,
            "expert_id": expert_id,
            "benchmark_split": split,
            "scout_rows": len(scout),
            "expert_rows": len(expert),
            "aligned_rows": len(aligned),
            "scout_only_rows": int(merged["_merge"].eq("left_only").sum()),
            "expert_only_rows": int(merged["_merge"].eq("right_only").sum()),
            "true_label_conflicts_between_assets": int(true_conflict.sum()),
            "image_key_conflicts": int(image_key_conflict.sum()),
            "patient_id_conflicts_between_assets": int(patient_conflict.sum()),
            "identity_missing_rows": int(missing_identity.sum()),
            "identity_label_conflicts": int(identity_label_conflict.sum()),
            "identity_patient_conflicts": int(identity_patient_conflict.sum()),
            "alignment_complete": bool(
                not merged["_merge"].ne("both").any()
                and not true_conflict.any()
                and not image_key_conflict.any()
                and not patient_conflict.any()
                and not missing_identity.any()
                and not identity_label_conflict.any()
                and not identity_patient_conflict.any()
            ),
        }
        self.base_case_cache[key] = (cases.copy(), dict(audit))
        return cases, audit

    def full_profile(
        self,
        *,
        source_task_id: str,
        scout_id: str,
        expert_id: str,
    ) -> tuple[dict[str, Any], np.ndarray, int]:
        key = (source_task_id, scout_id, expert_id)
        if key in self.full_profile_cache:
            return self.full_profile_cache[key]
        source, _ = self.build_base_cases(
            source_task_id,
            "development",
            scout_id,
            expert_id,
        )
        eligible = source.loc[source["primary_cohort_eligible"].astype(bool)].copy()
        if eligible.empty:
            raise ValueError(f"No eligible development profile cases for {key}")
        profile = fit_expert_history_profile(eligible, alpha=0.5)
        reference = eligible[list(SCOUT_PROBABILITY_COLUMNS)].to_numpy(
            dtype=float
        ).mean(axis=0)
        value = (profile, reference, len(eligible))
        self.full_profile_cache[key] = value
        return value

    def add_profile_features(self, cases: pd.DataFrame) -> pd.DataFrame:
        result = cases.copy()
        task_id = str(result["task_id"].iloc[0])
        scout_id = str(result["scout_id"].iloc[0])
        expert_id = str(result["expert_id"].iloc[0])
        split = str(result["benchmark_split"].iloc[0])
        spec = self.specs[task_id]
        source_task = spec.profile_source_task_id
        profile, reference, support = self.full_profile(
            source_task_id=source_task,
            scout_id=scout_id,
            expert_id=expert_id,
        )
        if split == "development" and task_id == source_task:
            eligible = result["primary_cohort_eligible"].astype(bool)
            cross_history = build_cross_fitted_expert_history(
                result.loc[eligible],
                group_column="resampling_group_id",
                n_folds=5,
                alpha=0.5,
                salt=f"{task_id}:{scout_id}:{expert_id}:history",
            )
            cross_reference = build_cross_fitted_reference_js(
                result.loc[eligible],
                group_column="resampling_group_id",
                probability_columns=SCOUT_PROBABILITY_COLUMNS,
                n_folds=5,
                salt=f"{task_id}:{scout_id}:reference",
            )
            for column in (
                "expert_history_corrected_rate",
                "expert_history_introduced_rate",
                "expert_history_net",
                "expert_history_support",
                "profile_fold",
                "profile_training_case_count",
            ):
                result.loc[eligible, column] = cross_history[column]
            result.loc[
                eligible, "scout_reference_js_divergence"
            ] = cross_reference["scout_reference_js_divergence"]
            result.loc[
                eligible, "reference_training_case_count"
            ] = cross_reference["reference_training_case_count"]
            result.loc[eligible, "expert_profile_source"] = (
                "other_development_folds_only"
            )
            result.loc[eligible, "expert_profile_excludes_current_group"] = True
            noneligible = ~eligible
            if noneligible.any():
                applied = apply_expert_history_profile(
                    result.loc[noneligible],
                    profile,
                )
                for column in applied:
                    result.loc[noneligible, column] = applied[column]
                result.loc[
                    noneligible, "scout_reference_js_divergence"
                ] = jensen_shannon_divergence(
                    result.loc[
                        noneligible, list(SCOUT_PROBABILITY_COLUMNS)
                    ].to_numpy(dtype=float),
                    reference,
                )
                result.loc[noneligible, "profile_fold"] = -1
                result.loc[noneligible, "profile_training_case_count"] = support
                result.loc[noneligible, "reference_training_case_count"] = support
                result.loc[noneligible, "expert_profile_source"] = (
                    "full_primary_development_for_nonprimary_audit_row"
                )
                result.loc[
                    noneligible, "expert_profile_excludes_current_group"
                ] = False
        else:
            applied = apply_expert_history_profile(result, profile)
            for column in applied:
                result[column] = applied[column]
            result["scout_reference_js_divergence"] = jensen_shannon_divergence(
                result[list(SCOUT_PROBABILITY_COLUMNS)].to_numpy(dtype=float),
                reference,
            )
            result["profile_fold"] = -1
            result["profile_training_case_count"] = support
            result["reference_training_case_count"] = support
            result["expert_profile_source"] = (
                f"{source_task}:full_development_only"
            )
            result["expert_profile_excludes_current_group"] = True
        result["expert_profile_source_task_id"] = source_task
        result["expert_profile_source_split"] = "development"
        result["expert_profile_domain_shift"] = source_task != task_id
        result["current_case_expert_output_used_for_feature"] = False
        result["test_outcome_used_for_feature_or_threshold"] = False
        return result

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        case_frames: list[pd.DataFrame] = []
        alignment_rows: list[dict[str, Any]] = []
        for route_id, route in self.route_inventory.iterrows():
            task_id = str(route["task_id"])
            spec = self.specs[task_id]
            for split in spec.split_roles:
                cases, alignment = self.build_base_cases(
                    task_id,
                    split,
                    str(route["scout_id"]),
                    str(route["expert_id"]),
                )
                cases = self.add_profile_features(cases)
                case_frames.append(cases)
                alignment_rows.append(alignment)
            print(f"built_route={route_id}", flush=True)
        cases = pd.concat(case_frames, ignore_index=True)
        assets = pd.DataFrame(self.asset_audits).drop_duplicates(
            ["task_id", "model_id", "benchmark_split"]
        )
        alignments = pd.DataFrame(alignment_rows)
        return cases, assets, alignments


def event_summary_row(
    frame: pd.DataFrame,
    *,
    cohort: str,
) -> dict[str, Any]:
    corrected = int(frame["corrected"].astype(bool).sum())
    introduced = int(frame["introduced"].astype(bool).sum())
    return {
        "cohort": cohort,
        "n_cases": len(frame),
        "corrected": corrected,
        "introduced": introduced,
        "both_correct": int(frame["both_correct"].astype(bool).sum()),
        "both_wrong": int(frame["both_wrong"].astype(bool).sum()),
        "dangerous_introduced": int(
            frame["dangerous_introduced"].astype(bool).sum()
        ),
        "net": corrected - introduced,
        "corrected_prevalence": corrected / len(frame) if len(frame) else np.nan,
        "introduced_prevalence": introduced / len(frame) if len(frame) else np.nan,
    }


def build_route_event_audit(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = [
        "task_id",
        "dataset_id",
        "evaluation_design",
        "route_id",
        "scout_id",
        "expert_id",
        "benchmark_split",
    ]
    for keys, frame in cases.groupby(grouping, sort=True):
        base = dict(zip(grouping, keys, strict=True))
        cohorts = {
            "all_aligned": frame,
            "leakage_controlled_primary": frame.loc[
                frame["primary_cohort_eligible"].astype(bool)
            ],
            "near_duplicate_excluded_sensitivity": frame.loc[
                frame["near_duplicate_sensitivity_eligible"].astype(bool)
            ],
        }
        for cohort, cohort_frame in cohorts.items():
            rows.append({**base, **event_summary_row(cohort_frame, cohort=cohort)})
    return pd.DataFrame(rows)


def _safe_binary_metric(
    target: np.ndarray,
    score: np.ndarray,
    metric: str,
) -> float | None:
    if np.unique(target).size != 2:
        return None
    if metric == "auroc":
        return float(roc_auc_score(target, score))
    return float(average_precision_score(target, score))


def build_development_signal_results(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    development = cases.loc[
        cases["benchmark_split"].eq("development")
        & cases["primary_cohort_eligible"].astype(bool)
    ].copy()
    feature_specs = {
        "scout_entropy": ("scout_entropy", 1.0),
        "scout_inverse_margin": ("scout_margin", -1.0),
        "scout_severe_probability_mass": (
            "scout_severe_probability_mass",
            1.0,
        ),
        "expert_history_net": ("expert_history_net", 1.0),
        "scout_reference_js_divergence": (
            "scout_reference_js_divergence",
            1.0,
        ),
    }
    grouping = [
        "task_id",
        "dataset_id",
        "evaluation_design",
        "route_id",
        "scout_id",
        "expert_id",
    ]
    for keys, frame in development.groupby(grouping, sort=True):
        base = dict(zip(grouping, keys, strict=True))
        scout_correct = frame["scout_pred"].astype(int).eq(
            frame["y_true"].astype(int)
        )
        cohort_specs = (
            ("all_cases", "corrected", frame),
            ("all_cases", "introduced", frame),
            ("scout_wrong_only", "corrected", frame.loc[~scout_correct]),
            ("scout_correct_only", "introduced", frame.loc[scout_correct]),
            (
                "corrected_or_introduced_only",
                "corrected",
                frame.loc[
                    frame["corrected"].astype(bool)
                    | frame["introduced"].astype(bool)
                ],
            ),
        )
        for feature_name, (column, multiplier) in feature_specs.items():
            for evaluation_cohort, outcome, cohort_frame in cohort_specs:
                score = (
                    cohort_frame[column].to_numpy(dtype=float) * multiplier
                )
                target = cohort_frame[outcome].astype(int).to_numpy()
                rows.append(
                    {
                        **base,
                        "split": "development",
                        "cohort": "leakage_controlled_primary",
                        "evaluation_cohort": evaluation_cohort,
                        "feature": feature_name,
                        "outcome": outcome,
                        "n_cases": len(cohort_frame),
                        "events": int(target.sum()),
                        "prevalence": (
                            float(target.mean()) if len(target) else np.nan
                        ),
                        "auroc": _safe_binary_metric(target, score, "auroc"),
                        "auprc": _safe_binary_metric(target, score, "auprc"),
                        "direction_predeclared": True,
                        "test_used": False,
                        "fit_performed": False,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_selected_cases(
    frame: pd.DataFrame,
    *,
    selected: np.ndarray,
    policy: str,
    requested_budget: float,
    budget_source: str,
    route: pd.Series,
    cohort: str,
    qualification_level: str = "",
) -> dict[str, Any]:
    selected_n = int(selected.sum())
    n_cases = len(frame)
    truth = frame["y_true"].to_numpy(dtype=int)
    scout = frame["scout_pred"].to_numpy(dtype=int)
    expert = frame["expert_pred"].to_numpy(dtype=int)
    final = np.where(selected, expert, scout)
    corrected_total = int(frame["corrected"].astype(bool).sum())
    introduced_total = int(frame["introduced"].astype(bool).sum())
    corrected_selected = int(
        (selected & frame["corrected"].astype(bool).to_numpy()).sum()
    )
    introduced_selected = int(
        (selected & frame["introduced"].astype(bool).to_numpy()).sum()
    )
    both_correct_selected = int(
        (selected & frame["both_correct"].astype(bool).to_numpy()).sum()
    )
    both_wrong_selected = int(
        (selected & frame["both_wrong"].astype(bool).to_numpy()).sum()
    )
    dangerous_selected = int(
        (selected & frame["dangerous_introduced"].astype(bool).to_numpy()).sum()
    )
    realized_budget = selected_n / n_cases if n_cases else 0.0
    scout_cost = pd.to_numeric(
        pd.Series([route["scout_cost_ms_per_image"]]),
        errors="coerce",
    ).iloc[0]
    expert_cost = pd.to_numeric(
        pd.Series([route["expert_cost_ms_per_image"]]),
        errors="coerce",
    ).iloc[0]
    comparable = bool(route["cost_comparable"])
    total_cost = (
        float(scout_cost + realized_budget * expert_cost)
        if comparable and pd.notna(scout_cost) and pd.notna(expert_cost)
        else np.nan
    )
    return {
        "task_id": str(frame["task_id"].iloc[0]),
        "dataset_id": str(frame["dataset_id"].iloc[0]),
        "evaluation_design": str(frame["evaluation_design"].iloc[0]),
        "route_id": str(frame["route_id"].iloc[0]),
        "scout_id": str(frame["scout_id"].iloc[0]),
        "expert_id": str(frame["expert_id"].iloc[0]),
        "benchmark_split": str(frame["benchmark_split"].iloc[0]),
        "cohort": cohort,
        "policy": policy,
        "requested_budget": requested_budget,
        "budget_source": budget_source,
        "n_cases": n_cases,
        "selected_n": selected_n,
        "realized_budget": realized_budget,
        "corrected_total": corrected_total,
        "introduced_total": introduced_total,
        "corrected_selected": corrected_selected,
        "introduced_selected": introduced_selected,
        "dangerous_introduced_selected": dangerous_selected,
        "neutral_selected": both_correct_selected + both_wrong_selected,
        "net_selected": corrected_selected - introduced_selected,
        "corrected_capture_rate": (
            corrected_selected / corrected_total if corrected_total else np.nan
        ),
        "introduced_capture_rate": (
            introduced_selected / introduced_total if introduced_total else np.nan
        ),
        "help_precision_among_consulted": (
            corrected_selected / selected_n if selected_n else np.nan
        ),
        "harm_rate_among_consulted": (
            introduced_selected / selected_n if selected_n else np.nan
        ),
        "scout_accuracy": float(accuracy_score(truth, scout)),
        "final_accuracy": float(accuracy_score(truth, final)),
        "final_macro_f1": float(
            f1_score(truth, final, average="macro", zero_division=0)
        ),
        "final_qwk": float(cohen_kappa_score(truth, final, weights="quadratic")),
        "cost_protocol_id": str(route["cost_protocol_id"]),
        "cost_comparable": comparable,
        "estimated_component_cost_ms_per_case": total_cost,
        "qualification_level": qualification_level,
        "current_case_expert_output_used_for_ranking": False,
        "test_used_for_feature_threshold_or_route_selection": False,
        "retrospective_only": str(frame["benchmark_split"].iloc[0])
        != "development",
    }


def build_baseline_results(
    cases: pd.DataFrame,
    route_inventory: pd.DataFrame,
) -> pd.DataFrame:
    route_lookup = route_inventory.set_index("route_id", drop=False)
    rows: list[dict[str, Any]] = []
    grouping = ["route_id", "benchmark_split"]
    for (route_id, split), full_frame in cases.groupby(grouping, sort=True):
        route = route_lookup.loc[route_id]
        cohorts = {
            "leakage_controlled_primary": full_frame.loc[
                full_frame["primary_cohort_eligible"].astype(bool)
            ].copy(),
            "near_duplicate_excluded_sensitivity": full_frame.loc[
                full_frame["near_duplicate_sensitivity_eligible"].astype(bool)
            ].copy(),
        }
        for cohort, frame in cohorts.items():
            if frame.empty:
                continue
            for budget in BENCHMARK_BUDGETS:
                policy_scores = {
                    "random": deterministic_random_scores(
                        frame["case_id"].astype(str),
                        salt=f"{route_id}:{split}:{cohort}:{budget}:random-v0.1",
                    ),
                    "entropy": frame["scout_entropy"].to_numpy(dtype=float),
                    "margin": 1.0
                    - frame["scout_margin"].to_numpy(dtype=float),
                }
                for policy, scores in policy_scores.items():
                    selected = rank_top_budget(
                        case_ids=frame["case_id"].astype(str),
                        scores=scores,
                        budget=budget,
                    )
                    rows.append(
                        evaluate_selected_cases(
                            frame,
                            selected=selected,
                            policy=policy,
                            requested_budget=budget,
                            budget_source="fixed_benchmark_grid",
                            route=route,
                            cohort=cohort,
                        )
                    )
            if bool(route["v1_1_case_baseline_available"]):
                baseline = ConsultationPolicyBaselineV1_1(
                    route_id=str(route["formal_v1_1_route_id"]),
                    routing_policy=str(route["formal_v1_1_policy"]),
                    budget=float(route["formal_v1_1_budget"]),
                    qualification_level=str(
                        route["formal_v1_1_qualification_level"]
                    ),
                )
                selected = rank_top_budget(
                    case_ids=frame["case_id"].astype(str),
                    scores=baseline.scores(frame),
                    budget=baseline.budget,
                )
                rows.append(
                    evaluate_selected_cases(
                        frame,
                        selected=selected,
                        policy="consultation_policy_baseline_v1_1",
                        requested_budget=baseline.budget,
                        budget_source="frozen_v1_1_route_budget",
                        route=route,
                        cohort=cohort,
                        qualification_level=baseline.qualification_level,
                    )
                )
    return pd.DataFrame(rows)


def build_expert_profile_audit(cases: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "task_id",
        "route_id",
        "benchmark_split",
        "expert_profile_source_task_id",
        "expert_profile_source_split",
        "expert_profile_domain_shift",
        "expert_profile_source",
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in cases.groupby(grouping, sort=True):
        base = dict(zip(grouping, keys, strict=True))
        rows.append(
            {
                **base,
                "n_cases": len(frame),
                "profile_training_case_count_min": int(
                    frame["profile_training_case_count"].min()
                ),
                "profile_training_case_count_max": int(
                    frame["profile_training_case_count"].max()
                ),
                "profile_excludes_current_group_all": bool(
                    frame["expert_profile_excludes_current_group"].astype(bool).all()
                ),
                "test_outcome_used": bool(
                    frame["test_outcome_used_for_feature_or_threshold"]
                    .astype(bool)
                    .any()
                ),
                "current_case_expert_output_used": bool(
                    frame["current_case_expert_output_used_for_feature"]
                    .astype(bool)
                    .any()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_case_table_schema(cases: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": "ophagent.help_or_harm_case_table_schema.v0_1",
        "analysis_unit": "single_CFP_image",
        "resampling_group": {
            "DeepDRiD_v1.1": "patient_group_id",
            "APTOS2019": "exact_image_sha256_then_case_id",
        },
        "class_order": [0, 1, 2, 3, 4],
        "outcomes": {
            "corrected": "scout_pred != y_true and expert_pred == y_true",
            "introduced": "scout_pred == y_true and expert_pred != y_true",
            "both_correct": "scout_pred == y_true and expert_pred == y_true",
            "both_wrong": "scout_pred != y_true and expert_pred != y_true",
            "dangerous_introduced": (
                "scout is not grade>=3 undergrading and expert is grade>=3 undergrading"
            ),
        },
        "legal_feature_columns": list(LEGAL_FEATURE_COLUMNS),
        "metadata_not_default_features": [
            "dataset_id",
            "task_id",
            "patient_group_id",
            "eye",
            "exam_id",
            "image_id",
        ],
        "post_consultation_audit_only_columns": [
            "expert_pred",
            *[f"expert_prob_{index}" for index in range(5)],
            "corrected",
            "introduced",
            "both_correct",
            "both_wrong",
            "dangerous_introduced",
        ],
        "private_paths_included": False,
        "rows": len(cases),
        "columns": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in cases.dtypes.items()
        ],
    }


def build_summary(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    route_inventory: pd.DataFrame,
    assets: pd.DataFrame,
    alignments: pd.DataFrame,
    cases: pd.DataFrame,
    leakage: pd.DataFrame,
    signals: pd.DataFrame,
    baselines: pd.DataFrame,
) -> dict[str, Any]:
    alignment_errors = int(
        (
            alignments[
                [
                    "scout_only_rows",
                    "expert_only_rows",
                    "true_label_conflicts_between_assets",
                    "image_key_conflicts",
                    "patient_id_conflicts_between_assets",
                    "identity_missing_rows",
                    "identity_label_conflicts",
                    "identity_patient_conflicts",
                ]
            ].sum(axis=1)
            > 0
        ).sum()
    )
    aptos_leakage = leakage.loc[leakage["dataset_id"].eq("APTOS2019")].iloc[0]
    deepdrid_leakage = leakage.loc[
        leakage["dataset_id"].eq("DeepDRiD_v1.1")
    ].iloc[0]
    development_signals = signals.loc[
        (
            signals["outcome"].eq("corrected")
            & signals["evaluation_cohort"].eq("scout_wrong_only")
        )
        | (
            signals["outcome"].eq("introduced")
            & signals["evaluation_cohort"].eq("scout_correct_only")
        )
    ]
    signal_summary: dict[str, Any] = {}
    for (task_id, feature, outcome, evaluation_cohort), frame in (
        development_signals.groupby(
            ["task_id", "feature", "outcome", "evaluation_cohort"],
            sort=True,
        )
    ):
        valid = pd.to_numeric(frame["auroc"], errors="coerce").dropna()
        signal_summary[
            f"{task_id}::{feature}::{outcome}::{evaluation_cohort}"
        ] = {
            "route_count": len(frame),
            "informative_route_count": len(valid),
            "median_auroc": float(valid.median()) if len(valid) else None,
            "q25_auroc": float(valid.quantile(0.25)) if len(valid) else None,
            "q75_auroc": float(valid.quantile(0.75)) if len(valid) else None,
        }
    decision = "CONDITIONAL_GO"
    if alignment_errors or not bool(assets["asset_complete"].all()):
        decision = "NO_GO"
    elif (
        int(deepdrid_leakage["patient_cross_split_groups"]) == 0
        and int(deepdrid_leakage["cross_split_exact_groups"]) == 0
        and int(aptos_leakage["cross_split_exact_groups"]) == 0
        and float(aptos_leakage["patient_id_coverage"]) == 1.0
    ):
        decision = "GO"
    split_counts = (
        cases.groupby(["task_id", "benchmark_split"])
        .size()
        .rename("rows")
        .reset_index()
    )
    primary_counts = (
        cases.loc[cases["primary_cohort_eligible"].astype(bool)]
        .groupby(["task_id", "benchmark_split"])
        .size()
        .rename("rows")
        .reset_index()
    )
    return {
        "schema_version": "ophagent.help_or_harm_benchmark_summary.v0_1",
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": protocol_sha256,
        "source_commit_sha": current_commit(project_root),
        "decision": decision,
        "decision_reason": (
            "病例概率资产可一一对齐，DeepDRiD 具备患者分组开发证据；"
            "但 APTOS 冻结模型使用的 256×256 派生输入存在跨 split "
            "字节级重复且缺少患者/眼别标识，全部冻结 Test 也仅能回顾性使用。"
        ),
        "candidate_routes": len(route_inventory),
        "candidate_routes_by_task": {
            str(key): int(value)
            for key, value in route_inventory["task_id"].value_counts().items()
        },
        "route_split_rows": int(
            cases[["route_id", "benchmark_split"]].drop_duplicates().shape[0]
        ),
        "case_route_rows": len(cases),
        "prediction_assets": len(assets),
        "prediction_assets_complete": bool(assets["asset_complete"].all()),
        "route_alignment_error_rows": alignment_errors,
        "all_aligned_case_counts": split_counts.to_dict("records"),
        "primary_case_counts": primary_counts.to_dict("records"),
        "image_leakage": leakage.to_dict("records"),
        "development_signal_summary": signal_summary,
        "baseline_rows": len(baselines),
        "v1_1_baseline_rows": int(
            baselines["policy"]
            .eq("consultation_policy_baseline_v1_1")
            .sum()
        ),
        "test_policy": {
            "test_used_for_feature_definition": False,
            "test_used_for_threshold": False,
            "test_used_for_route_selection": False,
            "retrospective_only": True,
            "independent_confirmation_missing": True,
        },
        "clinical_claim_boundary": (
            "corrected/introduced are label-defined model-error proxies, "
            "not patient harm, clinical benefit, diagnosis, or treatment outcomes"
        ),
    }


def build_report(
    *,
    summary: Mapping[str, Any],
    route_inventory: pd.DataFrame,
    alignments: pd.DataFrame,
    leakage: pd.DataFrame,
    signals: pd.DataFrame,
    baselines: pd.DataFrame,
) -> str:
    aptos = leakage.loc[leakage["dataset_id"].eq("APTOS2019")].iloc[0]
    deepdrid = leakage.loc[
        leakage["dataset_id"].eq("DeepDRiD_v1.1")
    ].iloc[0]
    signal_lines: list[str] = []
    conditional = signals.loc[
        (
            signals["outcome"].eq("corrected")
            & signals["evaluation_cohort"].eq("scout_wrong_only")
        )
        | (
            signals["outcome"].eq("introduced")
            & signals["evaluation_cohort"].eq("scout_correct_only")
        )
    ]
    for (task_id, feature, outcome, evaluation_cohort), frame in conditional.groupby(
        ["task_id", "feature", "outcome", "evaluation_cohort"],
        sort=True,
    ):
        auc = pd.to_numeric(frame["auroc"], errors="coerce").dropna()
        if len(auc):
            signal_lines.append(
                f"- `{task_id}` / `{feature}`：{len(auc)} 条可计算路线，"
                f"`{outcome}`@`{evaluation_cohort}` AUROC 中位数 "
                f"{auc.median():.3f}"
                f"（IQR {auc.quantile(0.25):.3f}–{auc.quantile(0.75):.3f}）。"
            )
    if not signal_lines:
        signal_lines.append("- 开发集没有同时包含阳性与阴性的可计算路线。")
    v1_rows = baselines.loc[
        baselines["policy"].eq("consultation_policy_baseline_v1_1")
        & baselines["cohort"].eq("leakage_controlled_primary")
    ]
    v1_lines: list[str] = []
    for _, row in v1_rows.sort_values(
        ["task_id", "route_id", "benchmark_split"]
    ).iterrows():
        v1_lines.append(
            f"- `{row['route_id']}` / `{row['benchmark_split']}`："
            f"预算 {float(row['requested_budget']):.0%}，"
            f"捕获 corrected {int(row['corrected_selected'])}/"
            f"{int(row['corrected_total'])}，"
            f"引入 introduced {int(row['introduced_selected'])}/"
            f"{int(row['introduced_total'])}，"
            f"net={int(row['net_selected'])}；"
            f"资格层级 `{row['qualification_level']}`。"
        )
    if not v1_lines:
        v1_lines.append("- 没有可重建的冻结单 Scout v1.1 路线。")
    return "\n".join(
        [
            "# OphAgent Help-or-Harm 病例级可行性审计与 Benchmark v0.1",
            "",
            f"结论：**{summary['decision']}**。",
            "",
            str(summary["decision_reason"]),
            "",
            "## 1. 审计范围",
            "",
            f"- 共枚举 {len(route_inventory)} 条有向单 Scout→单 Expert 路线："
            "APTOS 90、DeepDRiD 冻结迁移 30、DeepDRiD 原生适配 90。",
            f"- 共核对 {len(alignments)} 个路线×split 对齐单元；"
            f"异常单元 {summary['route_alignment_error_rows']}。",
            "- 所有路线均由冻结概率资产做只读笛卡尔枚举；未根据 Test "
            "筛选路线、特征、阈值或预算。",
            "",
            "## 2. 图像、患者与划分泄漏",
            "",
            "- 本报告中的 APTOS 图像是冻结模型实际使用的 256×256 派生输入，"
            "不是对原始高分辨率 APTOS 文件的溯源结论。",
            f"- APTOS：{int(aptos['exact_duplicate_groups'])} 个 SHA256 "
            f"完全重复组（{int(aptos['exact_duplicate_rows'])} 个文件），"
            f"其中 {int(aptos['cross_split_exact_groups'])} 个跨 split；"
            f"{int(aptos['label_conflict_exact_groups'])} 个重复组标签冲突，"
            f"其中 {int(aptos['cross_split_label_conflict_exact_groups'])} "
            "个同时跨 split。患者与眼别标识不可用。",
            f"- DeepDRiD：{int(deepdrid['exact_duplicate_groups'])} 个完全"
            f"重复组，患者跨 split 数 {int(deepdrid['patient_cross_split_groups'])}；"
            "患者、眼别和图像标识覆盖完整。",
            f"- 感知哈希只产生“待确认候选”：APTOS "
            f"{int(aptos['near_duplicate_candidate_pairs'])} 对，DeepDRiD "
            f"{int(deepdrid['near_duplicate_candidate_pairs'])} 对。它们不被"
            "当作已证实重复，只进入敏感性分析。",
            "- 主队列不修改原始 split，而是在 Benchmark 视图中排除跨 split "
            "完全重复、标签冲突，并将同 split 完全重复固定为一个分析单位。",
            "",
            "## 3. 标签与任务兼容边界",
            "",
            "- APTOS 与 DeepDRiD 均可映射到固定的 DR 0–4 有序标签，"
            "因此 corrected/introduced 的定义可复用。",
            "- 两者标签来源、采集域、适配设计和预处理不同，禁止直接混合"
            "模型排名、成本排名或把 dataset_id 当成预测特征。",
            "- `dangerous_introduced` 只是 grade≥3 被降为 <3 的错误代理，"
            "不是临床伤害终点。",
            "",
            "## 4. 合法预咨询特征与基本可识别信号",
            "",
            "- 特征仅来自 Scout 概率、熵、margin、重症概率质量、其他开发折"
            "形成的 Expert 历史画像，以及无标签 Scout 分布偏移信号。",
            "- 开发病例使用确定性分组五折画像；DeepDRiD 按患者分组，APTOS "
            "按完全图像组分组。回顾性病例只使用完整开发折画像。",
            "- 当前病例 Expert 输出、Expert embedding、Test 派生特征/阈值、"
            "dataset_id、身份字段、私有路径和未计成本的额外 Scout 均不进入"
            "特征合同。",
            "- 为避免把“识别 Scout 犯错”误称为 Help-or-Harm，主信号只看："
            "Scout 已错病例中的 corrected，以及 Scout 正确病例中的 introduced；"
            "全病例 AUROC 仅保留为次要描述。",
            *signal_lines,
            "- AUROC 仅是开发集描述性信号，不构成正式方法训练或独立验证。",
            "",
            "## 5. 固定基线",
            "",
            "- 已生成随机、entropy、margin 的 5%/10%/20%/30% 固定预算"
            "基线。v1.1 仅对具有唯一冻结单 Scout 协议身份的路线按原政策和"
            "原预算重建；未向其余候选补造 v1.1 身份。",
            *v1_lines,
            "- 成本仅在同一 `h100_gpu_forward_only_batch16_component_v0_1` "
            "协议内报告；部分成本模型不参与可比成本结论。",
            "",
            "## 6. 架构边界",
            "",
            "- `SafetyEligibilityGate` 继续委托唯一 "
            "`evaluate_route_qualification` 服务，负责不可绕过的任务、模态、"
            "资产、隐私、权限与运行边界。",
            "- `ConsultationPolicyBaselineV1_1` 只对病例排序，不能授予资格、"
            "改预算或直接调用 Expert，后续 Help-or-Harm 方法只能替换这一层。",
            "",
            "## 7. Test 与确认性缺口",
            "",
            "- APTOS frozen Test、DeepDRiD official validation 以及既有路由"
            "结果均已被观察，只能作为回顾性比较，不能承担最终独立确认。",
            "- 进入确认性阶段前需要一套未参与模型、路线、特征、阈值和提示"
            "选择的患者级数据；须有眼别/检查关联、预声明标签映射、冻结适配器"
            "生成的 Scout/Expert 概率、实测成本，以及足够的 corrected 与 "
            "introduced 事件数。",
            "",
            "因此本轮允许建立研究 Benchmark，但不允许据此声称已能预测真实"
            "临床获益/伤害，也不授予 deployment 或 clinical route 资格。",
            "",
        ]
    )


def ensure_no_private_paths(frame: pd.DataFrame) -> None:
    for column in frame.select_dtypes(include=["object"]).columns:
        values = frame[column].astype("string").fillna("")
        if values.str.contains(
            r"/training_data/|/data/team/|/data/LRT/|[A-Za-z]:\\",
            regex=True,
        ).any():
            raise ValueError(f"Private absolute path leaked into column {column}")


def assert_contract_invariants(
    *,
    route_inventory: pd.DataFrame,
    assets: pd.DataFrame,
    alignments: pd.DataFrame,
    cases: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    expected_routes = {
        "aptos_dr_5class": 90,
        "deepdrid_dr_5class_external": 30,
        "deepdrid_dr_5class_native": 90,
    }
    observed = route_inventory["task_id"].value_counts().to_dict()
    if observed != expected_routes:
        raise ValueError(f"Candidate route inventory mismatch: {observed}")
    if not assets["asset_complete"].all():
        raise ValueError("At least one frozen prediction asset is incomplete.")
    if not alignments["alignment_complete"].all():
        raise ValueError("At least one Scout/Expert case alignment failed.")
    if cases["current_case_expert_output_used_for_feature"].astype(bool).any():
        raise ValueError("Current-case Expert output leaked into legal features.")
    if cases["test_outcome_used_for_feature_or_threshold"].astype(bool).any():
        raise ValueError("Retrospective outcome leaked into features or thresholds.")
    if profiles["test_outcome_used"].astype(bool).any():
        raise ValueError("Expert history profile used a retrospective outcome.")
    if profiles["current_case_expert_output_used"].astype(bool).any():
        raise ValueError("Expert history profile used current-case Expert output.")
    missing_features = sorted(set(LEGAL_FEATURE_COLUMNS) - set(cases.columns))
    if missing_features:
        raise ValueError(f"Case table is missing legal features: {missing_features}")
    if cases.duplicated(["route_id", "benchmark_split", "case_id"]).any():
        raise ValueError("Case table has a duplicate route/split/case key.")
    legal_values = cases[list(LEGAL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(legal_values).all():
        raise ValueError("A legal pre-consultation feature is non-finite.")
    outcome_sum = cases[
        ["corrected", "introduced", "both_correct", "both_wrong"]
    ].astype(int).sum(axis=1)
    if not outcome_sum.eq(1).all():
        raise ValueError("Case error outcomes are not mutually exclusive.")
    primary = cases.loc[cases["primary_cohort_eligible"].astype(bool)]
    if primary.duplicated(
        ["route_id", "benchmark_split", "analysis_unit_id"]
    ).any():
        raise ValueError("Primary cohort repeats an exact-image analysis unit.")
    for frame in (route_inventory, assets, alignments, cases, profiles):
        ensure_no_private_paths(frame)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = project_root / PROTOCOL_RELATIVE_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = file_sha256(protocol_path)

    identity_path = output_dir / "image_identity_inventory.csv.gz"
    identity = build_image_identity_inventory(
        aptos_root=args.aptos_root,
        aptos_manifest_path=args.aptos_manifest,
        deepdrid_manifest_path=args.deepdrid_manifest,
        reusable_path=identity_path,
        force_rehash=bool(args.force_image_rehash),
    )
    identity = annotate_exact_identity(identity)
    near_candidates = build_near_duplicate_candidates(identity)
    near_case_ids: dict[str, set[str]] = defaultdict(set)
    for _, row in near_candidates.iterrows():
        near_case_ids[str(row["dataset_id"])].update(
            [str(row["left_case_id"]), str(row["right_case_id"])]
        )
    identity["near_duplicate_candidate"] = [
        str(case_id) in near_case_ids[str(dataset_id)]
        for dataset_id, case_id in zip(
            identity["dataset_id"],
            identity["case_id"],
            strict=True,
        )
    ]
    exact_members = build_exact_duplicate_members(identity)
    leakage_summary = build_image_leakage_summary(identity, near_candidates)

    specs = load_design_specs(
        project_root=project_root,
        asset_root=args.asset_root,
        deepdrid_native_root=args.deepdrid_native_root,
        deepdrid_transfer_root=args.deepdrid_transfer_root,
    )
    v1_1_routes = load_frozen_v1_1_routes(project_root)
    route_inventory = build_route_inventory(specs, v1_1_routes=v1_1_routes)
    builder = BenchmarkBuilder(
        project_root=project_root,
        asset_root=args.asset_root,
        specs=specs,
        identity=identity,
        near_candidates=near_candidates,
        route_inventory=route_inventory,
    )
    cases, assets, alignments = builder.build()
    route_events = build_route_event_audit(cases)
    signals = build_development_signal_results(cases)
    baselines = build_baseline_results(cases, route_inventory)
    profiles = build_expert_profile_audit(cases)
    semantics = build_semantic_compatibility_audit()
    feature_audit = build_legal_feature_audit()
    case_schema = build_case_table_schema(cases)
    summary = build_summary(
        project_root=project_root,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        route_inventory=route_inventory,
        assets=assets,
        alignments=alignments,
        cases=cases,
        leakage=leakage_summary,
        signals=signals,
        baselines=baselines,
    )
    report = build_report(
        summary=summary,
        route_inventory=route_inventory,
        alignments=alignments,
        leakage=leakage_summary,
        signals=signals,
        baselines=baselines,
    )
    assert_contract_invariants(
        route_inventory=route_inventory,
        assets=assets,
        alignments=alignments,
        cases=cases,
        profiles=profiles,
    )

    outputs: list[tuple[str, pd.DataFrame, bool]] = [
        ("candidate_route_inventory.csv", route_inventory, False),
        ("prediction_asset_audit.csv", assets, False),
        ("case_alignment_audit.csv", alignments, False),
        ("image_identity_inventory.csv.gz", identity, True),
        ("exact_duplicate_members.csv", exact_members, False),
        ("near_duplicate_candidates.csv", near_candidates, False),
        ("image_leakage_audit.csv", leakage_summary, False),
        ("semantic_compatibility_audit.csv", semantics, False),
        ("legal_feature_contract.csv", feature_audit, False),
        ("case_level_benchmark.csv.gz", cases, True),
        ("route_event_audit.csv", route_events, False),
        ("development_signal_results.csv", signals, False),
        ("baseline_results.csv", baselines, False),
        ("expert_history_profile_audit.csv", profiles, False),
    ]
    for name, frame, compressed in outputs:
        path = output_dir / name
        if compressed:
            write_csv_gzip(path, frame)
        else:
            write_csv(path, frame)
    write_json(output_dir / "case_table_schema.json", case_schema)
    write_json(output_dir / "benchmark_summary.json", summary)
    (output_dir / "feasibility_audit.md").write_text(
        report,
        encoding="utf-8",
    )

    output_paths = [
        output_dir / name for name, _, _ in outputs
    ] + [
        output_dir / "case_table_schema.json",
        output_dir / "benchmark_summary.json",
        output_dir / "feasibility_audit.md",
    ]
    input_assets = [
        {
            "uri": str(row["prediction_asset_uri"]),
            "sha256": str(row["prediction_asset_sha256"]),
        }
        for _, row in assets.sort_values(
            ["task_id", "model_id", "benchmark_split"]
        ).iterrows()
    ]
    input_assets.extend(
        [
            {
                "uri": "repo://" + APTOS_LEGACY_IDENTITY_RELATIVE_PATH,
                "sha256": file_sha256(
                    project_root / APTOS_LEGACY_IDENTITY_RELATIVE_PATH
                ),
            },
            {
                "uri": "dataset_asset://aptos_dataset_manifest.json",
                "sha256": file_sha256(args.aptos_manifest),
            },
            {
                "uri": "dataset_asset://deepdrid_admission_manifest.csv",
                "sha256": file_sha256(args.deepdrid_manifest),
            },
        ]
    )
    manifest = {
        "schema_version": "ophagent.help_or_harm_artifact_manifest.v0_1",
        "protocol_id": protocol["protocol_id"],
        "protocol_uri": "repo://" + PROTOCOL_RELATIVE_PATH,
        "protocol_sha256": protocol_sha256,
        "source_commit_sha": current_commit(project_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": summary["decision"],
        "frozen_source_assets_modified": False,
        "model_inference_performed": False,
        "route_selection_performed": False,
        "test_used_for_feature_threshold_or_route_selection": False,
        "input_assets": input_assets,
        "outputs": [
            {
                "uri": "repo://"
                + path.relative_to(project_root).as_posix(),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    print(
        stable_json(
            {
                "decision": summary["decision"],
                "candidate_routes": len(route_inventory),
                "case_route_rows": len(cases),
                "output_dir": output_dir,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
