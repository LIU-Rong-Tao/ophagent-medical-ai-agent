import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

PROB_COLS = [
    "prob_No DR",
    "prob_Mild DR",
    "prob_Moderate DR",
    "prob_Severe DR",
    "prob_Proliferative DR",
]


EXPERIMENTS = {
    "ConvNeXt-Tiny": {
        "path": "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
        "setting": "lightweight baseline",
    },
    "Swin-Tiny": {
        "path": "experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
        "setting": "lightweight baseline",
    },
    "ViT-B/16": {
        "path": "experiments/aptos_vit_base_patch16_imagenet/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
        "setting": "lightweight baseline",
    },
    "ViT-B/16 official-like": {
        "path": "experiments/aptos_vit_base_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
        "display_name": "ViT-B/16",
        "setting": "official-like",
    },
    "RETFound-MAE-CFP official-like": {
        "path": "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
        "display_name": "RETFound-MAE-CFP",
        "setting": "official-like",
    },
}


def prediction_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log(probs + eps), axis=1)


def top1_top2_margin(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def compute_metrics(y_true, y_pred, probs):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mean_prediction_entropy": float(np.mean(prediction_entropy(probs))),
        "mean_top1_top2_margin": float(np.mean(top1_top2_margin(probs))),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    return metrics, report, cm


def per_class_f1_dataframe(report: dict, backbone: str) -> pd.DataFrame:
    rows = []

    for label, values in report.items():
        if not isinstance(values, dict):
            continue

        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue

        rows.append(
            {
                "backbone": backbone,
                "class": label,
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1": values.get("f1-score"),
                "support": values.get("support"),
            }
        )

    return pd.DataFrame(rows)


def load_prediction_file(path: Path):
    df = pd.read_csv(path)

    required_cols = ["true_idx", "pred_idx"] + PROB_COLS
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    y_true = df["true_idx"].to_numpy()
    y_pred = df["pred_idx"].to_numpy()
    probs = df[PROB_COLS].to_numpy(dtype=np.float64)

    return df, y_true, y_pred, probs


def save_confusion_matrix(cm, class_names, output_path, title):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/summary/v0_5_2",
        help="Output directory for v0.5.2 benchmark summaries.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cm_dir = output_dir / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = []
    per_class_rows = []

    for backbone, info in EXPERIMENTS.items():
        pred_path = Path(info["path"])
        setting = info["setting"]

        if not pred_path.exists():
            print(f"[WARN] Missing prediction file: {pred_path}")
            continue

        print(f"[INFO] Processing {backbone}: {pred_path}")

        df, y_true, y_pred, probs = load_prediction_file(pred_path)

        metrics, report, cm = compute_metrics(y_true, y_pred, probs)

        row = {
            "backbone": info.get("display_name", backbone),
            "setting": setting,
            "n_samples": len(df),
        }
        row.update(metrics)
        benchmark_rows.append(row)

        per_class_df = per_class_f1_dataframe(report, backbone)
        per_class_rows.append(per_class_df)

        safe_name = backbone.replace("/", "_").replace(" ", "_")
        cm_output = cm_dir / f"{safe_name}_confusion_matrix.png"

        save_confusion_matrix(
            cm,
            CLASS_NAMES,
            cm_output,
            title=f"{backbone} Confusion Matrix",
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(output_dir / "benchmark_metrics.csv", index=False)

    if per_class_rows:
        all_per_class = pd.concat(per_class_rows, ignore_index=True)
        all_per_class.to_csv(output_dir / "per_class_f1.csv", index=False)

    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# v0.5.2 Benchmark Consistency Repair\n\n")
        f.write(
            "This folder contains the unified multi-metric benchmark results "
            "after v0.5.2 consistency repair.\n\n"
        )
        f.write("Generated files:\n\n")
        f.write("- benchmark_metrics.csv\n")
        f.write("- per_class_f1.csv\n")
        f.write("- confusion_matrices/\n")

    print(f"[DONE] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()