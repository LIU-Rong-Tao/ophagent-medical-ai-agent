#!/usr/bin/env python3
"""Backfill the existing forward-only cost schema for a RETFound task artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from app.ophbench_task_adapter import OphBenchLinearProbeTaskAdapter  # noqa: E402


def _update_registration(run_dir: Path, values: dict[str, object]) -> None:
    path = run_dir / "registration_record.csv"
    frame = pd.read_csv(path)
    for key, value in values.items():
        frame[key] = value
    frame.to_csv(path, index=False)


def _update_run_manifest(run_dir: Path, values: dict[str, object]) -> None:
    path = run_dir / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backfill_forward_cost(
    run_dir: Path,
    *,
    device: str = "cuda:0",
    batch_size: int = 32,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> Path:
    import torch
    from PIL import Image

    config = yaml.safe_load((run_dir / "effective_config.yaml").read_text(encoding="utf-8"))
    data_root = Path(config["data"]["root"])
    entries = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))[
        "entries"
    ]
    test_paths = [data_root / item["relative_path"] for item in entries if item["split"] == "test"]
    adapter = OphBenchLinearProbeTaskAdapter.load(
        encoder_checkpoint=config["foundation"]["encoder_checkpoint_path"],
        head_checkpoint=run_dir / "linear_probe.joblib",
        device=device,
    )
    encoder = adapter.base_adapter
    classifier = adapter.classifier

    def full_pass() -> tuple[float, int]:
        seen = 0
        elapsed = 0.0
        for start in range(0, len(test_paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in test_paths[start : start + batch_size]]
            tensors = torch.stack([encoder.preprocess(image) for image in images])
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            embeddings = encoder.encode_image(tensors).detach().cpu().numpy()
            probabilities = np.asarray(classifier.predict_proba(embeddings), dtype=float)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - begin
            if probabilities.shape[1] != 5:
                raise ValueError("RETFound task head did not return five-class probabilities")
            seen += len(images)
        return elapsed, seen

    try:
        for _ in range(warmup_runs):
            full_pass()
        measurements = []
        peak_memory = 0.0
        for repeat in range(1, timed_runs + 1):
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            elapsed, n_images = full_pass()
            ms_per_image = elapsed * 1000 / n_images
            measurements.append(ms_per_image)
            if device.startswith("cuda"):
                peak_memory = max(
                    peak_memory, torch.cuda.max_memory_allocated(device) / 1024 / 1024
                )
        median = float(np.median(measurements))
        mean = float(np.mean(measurements))
        std = float(np.std(measurements, ddof=1))
        encoder_mb = Path(config["foundation"]["encoder_checkpoint_path"]).stat().st_size / 1024 / 1024
        head_mb = (run_dir / "linear_probe.joblib").stat().st_size / 1024 / 1024
        row = {
            "artifact_id": config["identity"]["artifact_id"],
            "cost_profile_id": "rtx4090_fp32_bs32_forward_only_v1",
            "task_id": config["identity"]["task_id"],
            "dataset_id": config["identity"]["dataset_id"],
            "model_family": "retfound",
            "n_images": len(test_paths),
            "batch_size": batch_size,
            "device": device,
            "precision": "fp32",
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "estimated_forward_ms_per_image": median,
            "mean_ms_per_image": mean,
            "median_ms_per_image": median,
            "std_ms_per_image": std,
            "cv_ms_per_image": std / mean,
            "images_per_second": 1000.0 / median,
            "peak_allocated_memory_mb": peak_memory,
            "checkpoint_mb": encoder_mb + head_mb,
            "base_checkpoint_mb": encoder_mb,
            "task_head_mb": head_mb,
            "timing_scope": "forward_only",
            "cost_status": "measured",
            "timing_source": "input tensor -> RETFound encoder -> Logistic Regression -> probabilities",
        }
        output = run_dir / "forward_cost_summary.csv"
        pd.DataFrame([row]).to_csv(output, index=False)
        _update_registration(
            run_dir,
            {
                "cost_status": "measured",
                "forward_cost_ms_per_image": median,
                "throughput_images_per_second": 1000.0 / median,
                "device": device,
                "precision": "fp32",
                "batch_size": batch_size,
                "timing_scope": "forward_only",
                "forward_cost_path": str(output.resolve()),
                "base_checkpoint_mb": encoder_mb,
                "task_head_mb": head_mb,
            },
        )
        _update_run_manifest(
            run_dir,
            {
                "cost_status": "measured",
                "forward_cost_ms_per_image": median,
                "throughput_images_per_second": 1000.0 / median,
                "forward_cost_path": str(output.resolve()),
            },
        )
        return output
    except Exception as exc:
        _update_registration(run_dir, {"cost_status": "failed"})
        _update_run_manifest(run_dir, {"cost_status": "failed", "cost_error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    path = backfill_forward_cost(args.run_dir, device=args.device, batch_size=args.batch_size)
    print(path)


if __name__ == "__main__":
    main()
