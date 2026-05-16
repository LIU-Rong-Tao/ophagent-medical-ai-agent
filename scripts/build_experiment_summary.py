import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def build_summary(run_dir: Path) -> pd.DataFrame:
    config_path = run_dir / "configs" / "config.json"
    train_summary_path = run_dir / "logs" / "summary.json"
    metrics_path = run_dir / "evaluation" / "test" / "metrics.json"

    config = read_optional_json(config_path)
    train_summary = read_optional_json(train_summary_path)
    metrics = read_optional_json(metrics_path)

    if not config:
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if not train_summary:
        raise FileNotFoundError(f"Missing training summary file: {train_summary_path}")

    if not metrics:
        raise FileNotFoundError(f"Missing evaluation metrics file: {metrics_path}")

    row = {
        "project": "OphAgent",
        "stage": "Vision Baseline",
        "dataset": train_summary.get("dataset"),
        "task": "5-class diabetic retinopathy classification",
        "backbone": metrics.get("backbone", config.get("backbone")),
        "input_size": config.get("image_size"),
        "seed": train_summary.get("seed", config.get("seed")),
        "checkpoint": metrics.get("checkpoint"),
        "test_accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("precision_macro"),
        "macro_recall": metrics.get("recall_macro"),
        "macro_f1": metrics.get("f1_macro"),
        "weighted_f1": metrics.get("f1_weighted"),
        "num_samples": metrics.get("num_samples"),
        "best_val_acc": train_summary.get("best_acc"),
        "best_epoch": train_summary.get("best_epoch"),
        "batch_size": config.get("batch_size"),
        "num_epochs": config.get("num_epochs"),
        "learning_rate": config.get("learning_rate"),
        "pretrained": config.get("pretrained"),
        "run_dir": str(run_dir),
        "intended_use": (
            "Research and engineering demo only. "
            "Not for clinical diagnosis."
        ),
    }

    return pd.DataFrame([row])


def build_training_curve_summary(run_dir: Path) -> pd.DataFrame:
    log_path = run_dir / "logs" / "train_log.csv"

    if not log_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(log_path)

    required_cols = {"epoch", "train_loss", "val_loss", "val_acc", "best_acc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in train_log.csv: {sorted(missing)}")

    best_idx = df["val_acc"].idxmax()
    best_row = df.loc[best_idx]

    summary = {
        "num_epochs_logged": len(df),
        "best_epoch": int(best_row["epoch"]),
        "best_val_acc": float(best_row["val_acc"]),
        "best_recorded_acc_before_epoch": float(best_row["best_acc"]),
        "final_epoch": int(df.iloc[-1]["epoch"]),
        "final_train_loss": float(df.iloc[-1]["train_loss"]),
        "final_val_loss": float(df.iloc[-1]["val_loss"]),
        "final_val_acc": float(df.iloc[-1]["val_acc"]),
        "min_train_loss": float(df["train_loss"].min()),
        "min_val_loss": float(df["val_loss"].min()),
    }

    return pd.DataFrame([summary])


def build_class_mapping(run_dir: Path) -> pd.DataFrame:
    class_path = run_dir / "configs" / "class_to_idx.json"

    if not class_path.exists():
        return pd.DataFrame()

    mapping = load_json(class_path)

    rows = []
    for class_name, class_idx in mapping.items():
        rows.append(
            {
                "class_name": class_name,
                "class_idx": class_idx,
            }
        )

    return pd.DataFrame(rows).sort_values("class_idx")


def write_summary_md(
    output_path: Path,
    summary_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    class_df: pd.DataFrame,
):
    lines = []

    lines.append("# Experiment Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")

    if not summary_df.empty:
        row = summary_df.iloc[0].to_dict()
        lines.append(f"- Project: `{row.get('project')}`")
        lines.append(f"- Version: `{row.get('version')}`")
        lines.append(f"- Stage: `{row.get('stage')}`")
        lines.append(f"- Dataset: `{row.get('dataset')}`")
        lines.append(f"- Task: `{row.get('task')}`")
        lines.append(f"- Backbone: `{row.get('backbone')}`")
        lines.append(f"- Input size: `{row.get('input_size')}`")
        lines.append(f"- Seed: `{row.get('seed')}`")
        lines.append(f"- Checkpoint: `{row.get('checkpoint')}`")
        lines.append("")

        lines.append("## Test Metrics")
        lines.append("")
        lines.append(f"- Test accuracy: `{row.get('test_accuracy')}`")
        lines.append(f"- Macro precision: `{row.get('macro_precision')}`")
        lines.append(f"- Macro recall: `{row.get('macro_recall')}`")
        lines.append(f"- Macro F1: `{row.get('macro_f1')}`")
        lines.append(f"- Weighted F1: `{row.get('weighted_f1')}`")
        lines.append("")

        lines.append("## Training Config")
        lines.append("")
        lines.append(f"- Batch size: `{row.get('batch_size')}`")
        lines.append(f"- Number of epochs: `{row.get('num_epochs')}`")
        lines.append(f"- Learning rate: `{row.get('learning_rate')}`")
        lines.append(f"- Pretrained: `{row.get('pretrained')}`")
        lines.append("")

    if not curve_df.empty:
        curve = curve_df.iloc[0].to_dict()
        lines.append("## Training Curve Summary")
        lines.append("")
        lines.append(
            f"- Logged epochs: `{int(curve.get('num_epochs_logged'))}`"
        )
        lines.append(
            f"- Best epoch: `{int(curve.get('best_epoch'))}`"
        )
        lines.append(f"- Best validation accuracy: `{curve.get('best_val_acc')}`")
        lines.append(f"- Final validation accuracy: `{curve.get('final_val_acc')}`")
        lines.append(f"- Final train loss: `{curve.get('final_train_loss')}`")
        lines.append(f"- Final validation loss: `{curve.get('final_val_loss')}`")
        lines.append("")

    if not summary_df.empty:
        row = summary_df.iloc[0].to_dict()
        intended_use = row.get("intended_use")
        if intended_use:
            lines.append("## Intended Use")
            lines.append("")
            lines.append(f"{intended_use}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build summary artifacts for an existing OphAgent training run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to an experiment run directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for summary artifacts.",
    )

    args = parser.parse_args()

    run_dir = args.run_dir
    output_dir = args.output

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary(run_dir)
    curve_df = build_training_curve_summary(run_dir)
    class_df = build_class_mapping(run_dir)

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    curve_df.to_csv(output_dir / "training_curve_summary.csv", index=False)
    class_df.to_csv(output_dir / "class_mapping.csv", index=False)

    write_summary_md(
        output_path=output_dir / "summary.md",
        summary_df=summary_df,
        curve_df=curve_df,
        class_df=class_df,
    )

    print(f"Saved experiment summary artifacts to: {output_dir}")


if __name__ == "__main__":
    main()