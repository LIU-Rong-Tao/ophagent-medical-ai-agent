#!/usr/bin/env python3
"""Verify existing APTOS replay artifacts without model loading or inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_PROBABILITIES = [f"prob_{index}" for index in range(5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def prediction_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    case_id = "case_id" if "case_id" in frame else "image_key"
    label = "y_true" if "y_true" in frame else "true_label"
    prediction = "y_pred" if "y_pred" in frame else "pred_label"
    return case_id, label, prediction


def verify_prediction(path: Path, expected_n: int) -> None:
    frame = pd.read_csv(path)
    case_id, label, prediction = prediction_columns(frame)
    required = {case_id, label, prediction, *REQUIRED_PROBABILITIES}
    if len(frame) != expected_n or not required.issubset(frame.columns):
        raise ValueError(f"invalid prediction schema/count: {path}")
    probabilities = frame[REQUIRED_PROBABILITIES].to_numpy(float)
    if frame[case_id].astype(str).duplicated().any() or not np.isfinite(probabilities).all():
        raise ValueError(f"invalid identifiers or probabilities: {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"probabilities do not sum to one: {path}")
    if not np.array_equal(probabilities.argmax(axis=1), frame[prediction].to_numpy(int)):
        raise ValueError(f"prediction differs from probability argmax: {path}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = next(item for item in config["models"] if item["model_id"] == args.model)
    root = args.config.parents[2]
    artifact = root / model["existing_replay"]
    verify_prediction(artifact / model["validation_predictions"], 514)
    verify_prediction(artifact / model["test_predictions"], 1100)
    required = [artifact / item for item in model["required_artifacts"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing replay artifacts: " + ", ".join(missing))
    print(json.dumps({"model_id": args.model, "status": "existing_artifacts_valid"}))


if __name__ == "__main__":
    main()
