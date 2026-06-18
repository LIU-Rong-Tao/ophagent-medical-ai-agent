#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0.7.0 外部 DR 数据集预检查脚本。

本脚本只审计外部数据集的结构、图像质量、重复风险、跨数据集重叠情况，
以及是否具备基于 DR 分级构造 grade-based risk proxy 的基本条件。

本脚本不运行模型推理，不评估分类性能，也不判断 direct external validation 是否成功。

设计原则：
- 图像路径只保存相对路径，不写入服务器绝对路径；
- direct external validation 的数据结构要求以 test split 可用为核心，不强制要求 train / val 存在；
- hash 重复检查必须显式记录：未运行 hash 时，duplicate_check_passed_for_direct_external_test 不允许为 True；
- 如果 APTOS 数据存在，必须检查 APTOS 与外部数据之间的跨数据集重复；
- 若未发现明确的患者 / 双眼分组元数据，则后续只能声明 image-level analysis；
- 本阶段只输出结构性预检查结果，不输出“外部验证成功”结论。
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]

DATASET_NAMES = ["APTOS2019", "IDRiD_data", "MESSIDOR2"]
EXTERNAL_DATASETS = ["IDRiD_data", "MESSIDOR2"]
REFERENCE_DATASETS = ["APTOS2019"]

GRADE_ALIASES = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,

    "anodr": 0,
    "anoDR": 0,
    "noDR": 0,
    "no_dr": 0,
    "normal": 0,

    "bmilddr": 1,
    "bmildDR": 1,
    "milddr": 1,
    "mildDR": 1,
    "mild": 1,

    "cmoderatedr": 2,
    "cmoderateDR": 2,
    "moderatedr": 2,
    "moderateDR": 2,
    "moderate": 2,

    "dseveredr": 3,
    "dsevereDR": 3,
    "severedr": 3,
    "severeDR": 3,
    "severe": 3,

    "eproliferativedr": 4,
    "eproDR": 4,
    "proliferativedr": 4,
    "proliferativeDR": 4,
    "proliferative": 4,
    "pdr": 4,
}


def normalize_name(name: str) -> str:
    return name.strip()


def grade_from_class_dir(name: str) -> Optional[int]:
    raw = normalize_name(name)
    if raw in GRADE_ALIASES:
        return GRADE_ALIASES[raw]
    lower = raw.lower()
    if lower in GRADE_ALIASES:
        return GRADE_ALIASES[lower]
    return None


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def inspect_image(path: Path) -> dict:
    row = {
        "image_readable": False,
        "image_width": None,
        "image_height": None,
        "image_mode": "",
        "image_channels": None,
        "image_error": "",
    }
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            row["image_readable"] = True
            row["image_width"] = int(img.width)
            row["image_height"] = int(img.height)
            row["image_mode"] = str(img.mode)
            if img.mode in ["1", "L", "P"]:
                row["image_channels"] = 1
            elif img.mode in ["RGB", "YCbCr"]:
                row["image_channels"] = 3
            elif img.mode in ["RGBA", "CMYK"]:
                row["image_channels"] = 4
            else:
                row["image_channels"] = None
    except (UnidentifiedImageError, OSError, ValueError) as e:
        row["image_error"] = repr(e)
    return row




def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """将 DataFrame 转成简单 Markdown 表格，避免依赖 pandas 的 tabulate 可选包。"""
    if df is None or df.empty:
        return "无记录。"

    out = df.reset_index() if index else df.copy()
    out = out.fillna("")

    columns = [str(c) for c in out.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in out.iterrows():
        values = []
        for c in out.columns:
            v = str(row[c])
            v = v.replace("|", "\\|").replace("\n", " ")
            values.append(v)
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def metadata_grouping_status(dataset_root: Path) -> dict:
    metadata_files = []
    for pattern in ["*.csv", "*.xlsx", "*.xls", "*.json", "*.txt"]:
        metadata_files.extend(dataset_root.glob(pattern))

    names = " ".join(p.name.lower() for p in metadata_files)
    has_patient_hint = any(k in names for k in ["patient", "subject", "id", "eye", "laterality", "left", "right", "od", "os"])

    return {
        "metadata_files_found": ";".join(sorted(p.name for p in metadata_files)),
        "patient_or_eye_grouping_metadata_observed": bool(has_patient_hint),
        "analysis_unit": "patient_or_eye_possible_if_metadata_verified" if has_patient_hint else "image_level_only",
    }


def collect_dataset(data_root: Path, dataset_name: str, compute_hash: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_root = data_root / dataset_name

    inventory_rows = []
    class_rows = []
    quality_rows = []

    grouping = metadata_grouping_status(dataset_root) if dataset_root.exists() else {
        "metadata_files_found": "",
        "patient_or_eye_grouping_metadata_observed": False,
        "analysis_unit": "image_level_only",
    }

    for split in SPLITS:
        split_dir = dataset_root / split
        split_exists = split_dir.exists()

        observed_dirs = [p for p in split_dir.iterdir() if p.is_dir()] if split_exists else []
        expected_grade_counts = defaultdict(int)
        unexpected_dirs = []

        for class_dir in sorted(observed_dirs):
            grade = grade_from_class_dir(class_dir.name)
            if grade is None:
                unexpected_dirs.append(class_dir.name)
                continue

            image_paths = sorted(p for p in class_dir.rglob("*") if is_image(p))
            expected_grade_counts[grade] += len(image_paths)

            for p in image_paths:
                inspect = inspect_image(p)
                file_hash = md5_file(p) if compute_hash else ""

                inventory_rows.append({
                    "dataset": dataset_name,
                    "split": split,
                    "class_dir": class_dir.name,
                    "grade": grade,
                    "relative_image_path": relative_to_root(p, data_root),
                    "image_name": p.name,
                    "suffix": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                    "hash_check_performed": bool(compute_hash),
                    "md5": file_hash,
                    **grouping,
                    **inspect,
                })

        for grade in range(5):
            class_rows.append({
                "dataset": dataset_name,
                "split": split,
                "split_exists": bool(split_exists),
                "grade": grade,
                "n_images": int(expected_grade_counts.get(grade, 0)),
                "unexpected_class_dirs": ";".join(sorted(unexpected_dirs)),
                **grouping,
            })

        quality_rows.append({
            "dataset": dataset_name,
            "split": split,
            "split_exists": bool(split_exists),
            "unexpected_class_dirs": ";".join(sorted(unexpected_dirs)),
            "n_unexpected_class_dirs": len(unexpected_dirs),
            **grouping,
        })

    return pd.DataFrame(inventory_rows), pd.DataFrame(class_rows), pd.DataFrame(quality_rows)


def build_quality_summary(inventory: pd.DataFrame, quality_base: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        q = quality_base.copy()
        q["n_images"] = 0
        q["n_unreadable_images"] = 0
        q["min_width"] = None
        q["min_height"] = None
        q["median_width"] = None
        q["median_height"] = None
        q["observed_modes"] = ""
        q["observed_channels"] = ""
        return q

    grouped_rows = []
    for (dataset, split), g in inventory.groupby(["dataset", "split"], dropna=False):
        grouped_rows.append({
            "dataset": dataset,
            "split": split,
            "n_images": int(len(g)),
            "n_unreadable_images": int((~g["image_readable"]).sum()),
            "min_width": int(g["image_width"].dropna().min()) if g["image_width"].notna().any() else None,
            "min_height": int(g["image_height"].dropna().min()) if g["image_height"].notna().any() else None,
            "median_width": float(g["image_width"].dropna().median()) if g["image_width"].notna().any() else None,
            "median_height": float(g["image_height"].dropna().median()) if g["image_height"].notna().any() else None,
            "observed_modes": ";".join(sorted(str(x) for x in g["image_mode"].dropna().unique() if str(x))),
            "observed_channels": ";".join(sorted(str(int(x)) for x in g["image_channels"].dropna().unique())),
        })

    agg = pd.DataFrame(grouped_rows)
    q = quality_base.merge(agg, on=["dataset", "split"], how="left")
    fill_cols = ["n_images", "n_unreadable_images"]
    for c in fill_cols:
        q[c] = q[c].fillna(0).astype(int)
    return q


def build_overlap_audit(inventory: pd.DataFrame, compute_hash: bool) -> pd.DataFrame:
    rows = []

    if inventory.empty:
        return pd.DataFrame(rows)

    # filename-level overlap is only a weak warning.
    for key, label in [("image_name", "filename"), ("md5", "md5")]:
        if key == "md5" and not compute_hash:
            continue

        valid = inventory[inventory[key].astype(str).ne("")]
        if valid.empty:
            continue

        for value, g in valid.groupby(key):
            datasets = sorted(g["dataset"].unique())
            splits = sorted(g["split"].unique())

            if len(g) <= 1:
                continue

            overlap_type = []
            if g["dataset"].nunique() > 1:
                overlap_type.append("cross_dataset")
            if g.groupby("dataset")["split"].nunique().max() > 1:
                overlap_type.append("cross_split_within_dataset")
            if not overlap_type:
                overlap_type.append("within_split_or_duplicate_name")

            rows.append({
                "overlap_key_type": label,
                "overlap_key": value,
                "n_rows": int(len(g)),
                "datasets": ";".join(datasets),
                "splits": ";".join(splits),
                "overlap_type": ";".join(overlap_type),
                "relative_image_paths": ";".join(sorted(g["relative_image_path"].astype(str).tolist())),
                "hash_check_performed": bool(compute_hash),
            })

    return pd.DataFrame(rows)


def build_precheck_summary_table(
    distribution: pd.DataFrame,
    quality: pd.DataFrame,
    overlap: pd.DataFrame,
    compute_hash: bool,
) -> pd.DataFrame:
    rows = []

    for dataset in EXTERNAL_DATASETS:
        d = distribution[distribution["dataset"] == dataset]
        q = quality[quality["dataset"] == dataset]

        test = d[d["split"] == "test"]
        test_exists = bool(test["split_exists"].any()) if not test.empty else False
        test_total = int(test["n_images"].sum()) if not test.empty else 0
        event_sample_size = int(test.loc[test["grade"].isin([3, 4]), "n_images"].sum()) if not test.empty else 0
        all_5_classes_present_test = bool((test["n_images"] > 0).all()) if len(test) == 5 else False

        n_unexpected_dirs = int(q["n_unexpected_class_dirs"].sum()) if not q.empty else 0
        n_unreadable = int(q["n_unreadable_images"].sum()) if not q.empty and "n_unreadable_images" in q else 0

        overlap_ds = overlap[
            overlap["overlap_type"].astype(str).str.contains("cross_dataset", na=False)
        ] if not overlap.empty else pd.DataFrame()

        overlap_cross_split = overlap[
            overlap["overlap_type"].astype(str).str.contains("cross_split_within_dataset", na=False)
        ] if not overlap.empty else pd.DataFrame()

        overlap_involving_dataset = 0
        if not overlap_ds.empty:
            overlap_involving_dataset = int(overlap_ds["datasets"].astype(str).str.contains(dataset).sum())

        if compute_hash:
            md5_cross_dataset = overlap_ds[
                overlap_ds["overlap_key_type"].eq("md5")
                & overlap_ds["datasets"].astype(str).str.contains(dataset)
            ]

            md5_internal_cross_split = overlap_cross_split[
                overlap_cross_split["overlap_key_type"].eq("md5")
                & overlap_cross_split["datasets"].astype(str).str.contains(dataset)
            ]

            cross_dataset_md5_overlap_rows = int(len(md5_cross_dataset))
            external_internal_cross_split_md5_overlap_rows = int(len(md5_internal_cross_split))
            duplicate_check_passed_for_direct_external_test = bool(
                cross_dataset_md5_overlap_rows == 0
            )
        else:
            cross_dataset_md5_overlap_rows = None
            external_internal_cross_split_md5_overlap_rows = None
            duplicate_check_passed_for_direct_external_test = False

        rows.append({
            "dataset": dataset,
            "test_split_exists": test_exists,
            "test_total_images": test_total,
            "event_sample_size_grade_3_or_4": event_sample_size,
            "structurally_eligible": bool(test_exists and test_total > 0 and event_sample_size > 0 and n_unreadable == 0),
            "hash_check_performed": bool(compute_hash),
            "duplicate_check_passed_for_direct_external_test": duplicate_check_passed_for_direct_external_test,
            "cross_dataset_md5_overlap_rows": cross_dataset_md5_overlap_rows,
            "external_internal_cross_split_md5_overlap_rows": external_internal_cross_split_md5_overlap_rows,
            "cross_dataset_overlap_rows_involving_dataset": overlap_involving_dataset,
            "n_unexpected_class_dirs": n_unexpected_dirs,
            "n_unreadable_images": n_unreadable,
            "all_5_classes_present_test": all_5_classes_present_test,
            "all_5_classes_present_warning": not all_5_classes_present_test,
            "patient_or_eye_grouping_metadata_observed": bool(q["patient_or_eye_grouping_metadata_observed"].any()) if not q.empty else False,
            "analysis_unit": "patient_or_eye_possible_if_metadata_verified" if (not q.empty and q["patient_or_eye_grouping_metadata_observed"].any()) else "image_level_only",
            "statistical_adequacy_pending": True,
        })

    return pd.DataFrame(rows)


def write_markdown_summary(
    out_path: Path,
    precheck: pd.DataFrame,
    distribution: pd.DataFrame,
    quality: pd.DataFrame,
    overlap: pd.DataFrame,
    compute_hash: bool,
) -> None:
    lines = []
    lines.append("# v0.7.0 外部 DR 数据预检查结果")
    lines.append("")
    lines.append("本文件由 `scripts/precheck_v070_external_dr_datasets.py` 自动生成。")
    lines.append("")
    lines.append("本阶段只检查数据结构、质量、重复风险和 grade-based proxy 承接条件；不运行模型推理，不给出外部泛化结论。")
    lines.append("")
    lines.append("## 核心摘要")
    lines.append("")
    lines.append(df_to_markdown(precheck, index=False))
    lines.append("")
    lines.append("## 类别分布")
    lines.append("")

    for dataset in sorted(distribution["dataset"].unique()):
        lines.append(f"### {dataset}")
        lines.append("")
        d = distribution[distribution["dataset"] == dataset]
        pivot = d.pivot_table(index="split", columns="grade", values="n_images", aggfunc="sum", fill_value=0).reset_index()
        lines.append(df_to_markdown(pivot, index=False))
        lines.append("")

    lines.append("## 数据质量摘要")
    lines.append("")
    q_cols = [
        "dataset", "split", "split_exists", "n_images", "n_unreadable_images",
        "min_width", "min_height", "median_width", "median_height",
        "observed_modes", "observed_channels", "n_unexpected_class_dirs",
        "unexpected_class_dirs", "analysis_unit",
    ]
    q_cols = [c for c in q_cols if c in quality.columns]
    lines.append(df_to_markdown(quality[q_cols], index=False))
    lines.append("")

    lines.append("## 重复与重叠审计")
    lines.append("")
    lines.append(f"- hash_check_performed: `{bool(compute_hash)}`")
    lines.append(f"- overlap rows: `{len(overlap)}`")
    lines.append("")
    if len(overlap) > 0:
        show_cols = ["overlap_key_type", "overlap_key", "n_rows", "datasets", "splits", "overlap_type", "hash_check_performed"]
        lines.append(df_to_markdown(overlap[show_cols].head(30), index=False))
        lines.append("")

    lines.append("## 解释边界")
    lines.append("")
    lines.append("- `structurally_eligible=True` 只表示 test split、图像读取和 grade 3/4 proxy 存在初步承接条件。")
    lines.append("- `duplicate_check_passed_for_direct_external_test=True` 只表示 hash 检查已运行，且未发现 APTOS 与外部数据之间的跨数据集 md5 重叠。")
    lines.append("- `cross_dataset_md5_overlap_rows` 用于判断 APTOS 与外部数据是否存在图像级重复。")
    lines.append("- `external_internal_cross_split_md5_overlap_rows` 用于记录外部数据集内部 train/val/test 是否存在图像级重复；该字段主要影响后续目标数据重训实验。")
    lines.append("- `event_sample_size_grade_3_or_4` 只是外部 test 中 grade 3/4 样本数，不等于最终 dangerous event 数。")
    lines.append("- `statistical_adequacy_pending=True` 表示统计充分性必须等 v0.7.1 推理结果和危险事件数量出来后再判断。")
    lines.append("- `all_5_classes_present_test=False` 是警告条件，不是自动阻断条件。")
    lines.append("- 若无患者或双眼元数据，后续只能声明 image-level analysis。")
    lines.append("- 无 DME、病灶级或临床终点标签时，只能称为 grade-only proxy，不能称为真实 VTDR 临床终点。")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/LRT/RETFound/Data_split"), help="包含 APTOS2019、IDRiD_data、MESSIDOR2 的数据根目录")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/summary/v0_7_0"), help="v0.7.0 预检查输出目录")
    parser.add_argument("--compute-hash", action="store_true", help="计算图像 MD5，用于更严格的重复和跨数据集重叠检查")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventories = []
    distributions = []
    qualities_base = []

    for dataset_name in DATASET_NAMES:
        print(f"正在预检查数据集：{dataset_name}")
        inv, dist, qual = collect_dataset(args.data_root, dataset_name, args.compute_hash)
        inventories.append(inv)
        distributions.append(dist)
        qualities_base.append(qual)

    inventory = pd.concat(inventories, ignore_index=True) if inventories else pd.DataFrame()
    distribution = pd.concat(distributions, ignore_index=True) if distributions else pd.DataFrame()
    quality_base = pd.concat(qualities_base, ignore_index=True) if qualities_base else pd.DataFrame()

    quality = build_quality_summary(inventory, quality_base)
    overlap = build_overlap_audit(inventory, args.compute_hash)
    precheck = build_precheck_summary_table(distribution, quality, overlap, args.compute_hash)

    inventory.to_csv(args.out_dir / "external_dr_dataset_inventory.csv", index=False)
    distribution.to_csv(args.out_dir / "external_dr_class_distribution.csv", index=False)
    quality.to_csv(args.out_dir / "external_dr_data_quality.csv", index=False)
    overlap.to_csv(args.out_dir / "external_dr_overlap_audit.csv", index=False)
    precheck.to_csv(args.out_dir / "external_dr_precheck_table.csv", index=False)

    write_markdown_summary(
        out_path=args.out_dir / "external_dr_precheck_summary.md",
        precheck=precheck,
        distribution=distribution,
        quality=quality,
        overlap=overlap,
        compute_hash=args.compute_hash,
    )

    print("已保存 v0.7.0 外部 DR 数据预检查输出到：", args.out_dir)
    print(precheck.to_string(index=False))


if __name__ == "__main__":
    main()
