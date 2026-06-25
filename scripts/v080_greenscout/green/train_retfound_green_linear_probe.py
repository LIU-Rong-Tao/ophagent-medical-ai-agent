#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


EMB_DIR = Path("experiments/v0_8_0_greenscout_feasibility/green_embeddings")
OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/green_probe")
PRED_OUT = Path("experiments/v0_8_0_greenscout_feasibility/predictions/retfound_green_test_predictions.csv")
METRIC_OUT = OUT_DIR / "retfound_green_linear_probe_metrics.csv"
MODEL_OUT = OUT_DIR / "retfound_green_linear_probe.joblib"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def load_split(split: str):
    p = EMB_DIR / f"retfound_green_{split}_embeddings.csv"
    df = pd.read_csv(p)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) != 384:
        raise ValueError(f"{split}: expected 384 embedding cols, got {len(emb_cols)}")

    X = df[emb_cols].to_numpy(dtype=np.float32)
    y = df["true_label"].to_numpy(dtype=np.int64)
    meta = df[["image_key", "image_path", "true_label", "true_label_name"]].copy()
    return X, y, meta


def metric_row(split: str, y_true, y_pred):
    return {
        "model_name": "retfound_green_linear_probe",
        "split": split,
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)

    X_train, y_train, train_meta = load_split("train")
    X_val, y_val, val_meta = load_split("val")
    X_test, y_test, test_meta = load_split("test")

    print("[INFO] loaded embeddings")
    print("train:", X_train.shape, np.bincount(y_train, minlength=5))
    print("val:", X_val.shape, np.bincount(y_val, minlength=5))
    print("test:", X_test.shape, np.bincount(y_test, minlength=5))

    # 小网格，只选 C，不搞复杂 Router
    candidate_C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    rows = []
    best = None

    for C in candidate_C:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                multi_class="auto",
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )),
        ])

        clf.fit(X_train, y_train)

        pred_train = clf.predict(X_train)
        pred_val = clf.predict(X_val)

        tr = metric_row("train", y_train, pred_train)
        va = metric_row("val", y_val, pred_val)
        tr["C"] = C
        va["C"] = C
        rows.extend([tr, va])

        score = va["macro_f1"]
        if best is None or score > best["val_macro_f1"]:
            best = {
                "C": C,
                "val_macro_f1": score,
                "clf": clf,
            }

    best_C = best["C"]
    clf = best["clf"]

    print(f"[INFO] best C by val macro_f1: {best_C}")

    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    pred_test = clf.predict(X_test)

    prob_test = clf.predict_proba(X_test)
    confidence = prob_test.max(axis=1)

    final_rows = []
    for split, y_true, y_pred in [
        ("train_best", y_train, pred_train),
        ("val_best", y_val, pred_val),
        ("test_best", y_test, pred_test),
    ]:
        r = metric_row(split, y_true, y_pred)
        r["C"] = best_C
        final_rows.append(r)

    metrics = pd.DataFrame(rows + final_rows)
    metrics.to_csv(METRIC_OUT, index=False)

    pred_df = test_meta.copy()
    pred_df["model_name"] = "retfound_green_linear_probe"
    pred_df["pred_label"] = pred_test.astype(int)
    pred_df["pred_label_name"] = [CLASS_NAMES[int(x)] for x in pred_test]
    pred_df["confidence"] = confidence
    pred_df["correct"] = pred_df["pred_label"].to_numpy() == pred_df["true_label"].to_numpy()

    for i in range(5):
        pred_df[f"prob_{i}"] = prob_test[:, i]

    # 统一成后面互补性脚本好读的列顺序
    pred_df = pred_df[
        [
            "image_key",
            "image_path",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "confidence",
            "correct",
            "prob_0",
            "prob_1",
            "prob_2",
            "prob_3",
            "prob_4",
            "model_name",
        ]
    ]

    pred_df.to_csv(PRED_OUT, index=False)
    joblib.dump(clf, MODEL_OUT)

    print("\n[DONE]")
    print("model:", MODEL_OUT)
    print("metrics:", METRIC_OUT)
    print("pred:", PRED_OUT)

    print("\n=== best metrics ===")
    print(pd.DataFrame(final_rows).to_string(index=False))

    print("\n=== test label distribution ===")
    print(pd.crosstab(pred_df["true_label"], pred_df["pred_label"]))

    print("\n=== test head ===")
    print(pred_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
