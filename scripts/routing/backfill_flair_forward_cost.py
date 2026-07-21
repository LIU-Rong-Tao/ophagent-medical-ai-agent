#!/usr/bin/env python3
"""Measure forward-only cost for a FLAIR APTOS task artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.flair_task_adapter import FlairAptosTaskAdapter, preprocess_flair_image  # noqa: E402


def _update_registration(run_dir: Path, values: dict[str, object]) -> None:
    path = run_dir / "registration_record.csv"
    frame = pd.read_csv(path)
    for key, value in values.items():
        frame[key] = value
    frame.to_csv(path, index=False)


def _update_manifest(run_dir: Path, values: dict[str, object]) -> None:
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

    config = yaml.safe_load((run_dir / "effective_config.yaml").read_text(encoding="utf-8"))
    data_root = Path(config["data"]["root"])
    entries = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))[
        "entries"
    ]
    test_paths = [data_root / item["relative_path"] for item in entries if item["split"] == "test"]
    adapter = FlairAptosTaskAdapter.load(
        encoder_checkpoint=config["foundation"]["checkpoint_path"],
        task_checkpoint=run_dir / "flair_aptos_task_checkpoint.pth",
        device=device,
    )

    def full_pass() -> tuple[float, int]:
        seen = 0
        elapsed = 0.0
        for start in range(0, len(test_paths), batch_size):
            tensors = []
            for path in test_paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess_flair_image(image))
            images = torch.stack(tensors).to(device, non_blocking=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            with torch.inference_mode():
                probabilities = torch.softmax(
                    adapter.classifier(adapter.encoder(images)).float(), dim=1
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - begin
            if probabilities.shape[1] != 5:
                raise ValueError("FLAIR task adapter did not return five-class probabilities")
            seen += len(tensors)
        return elapsed, seen

    try:
        for _ in range(warmup_runs):
            full_pass()
        measurements = []
        peak_memory = 0.0
        for _ in range(timed_runs):
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            elapsed, n_images = full_pass()
            measurements.append(elapsed * 1000 / n_images)
            if device.startswith("cuda"):
                peak_memory = max(
                    peak_memory, torch.cuda.max_memory_allocated(device) / 1024 / 1024
                )
        median = float(np.median(measurements))
        mean = float(np.mean(measurements))
        std = float(np.std(measurements, ddof=1))
        base_mb = Path(config["foundation"]["checkpoint_path"]).stat().st_size / 1024 / 1024
        head_mb = (run_dir / "flair_aptos_task_checkpoint.pth").stat().st_size / 1024 / 1024
        device_name = torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU"
        profile = f"h100_fp32_bs{batch_size}_forward_only_v1"
        row = {
            "artifact_id": "aptos2019-flair-resnet50-official-lp-project-v1",
            "cost_profile_id": profile,
            "task_id": "aptos_dr_5class",
            "dataset_id": "APTOS2019",
            "model_family": "flair",
            "n_images": len(test_paths),
            "batch_size": batch_size,
            "device": device,
            "device_name": device_name,
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
            "checkpoint_mb": base_mb + head_mb,
            "base_checkpoint_mb": base_mb,
            "task_head_mb": head_mb,
            "timing_scope": "forward_only",
            "cost_status": "measured",
            "timing_source": "input tensor -> FLAIR encoder -> task head -> probabilities",
        }
        output = run_dir / "forward_cost_summary.csv"
        pd.DataFrame([row]).to_csv(output, index=False)
        updates = {
            "cost_status": "measured",
            "inference_cost_measured": True,
            "forward_cost_ms_per_image": median,
            "throughput_images_per_second": 1000.0 / median,
            "device": device,
            "device_name": device_name,
            "precision": "fp32",
            "batch_size": batch_size,
            "timing_scope": "forward_only",
            "forward_cost_path": str(output.resolve()),
        }
        _update_registration(run_dir, updates)
        _update_manifest(run_dir, updates)
        return output
    except Exception as exc:
        _update_registration(run_dir, {"cost_status": "failed", "inference_cost_measured": False})
        _update_manifest(run_dir, {"cost_status": "failed", "cost_error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(backfill_forward_cost(args.run_dir, device=args.device, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
