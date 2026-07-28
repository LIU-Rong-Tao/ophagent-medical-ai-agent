"""Run the fixed rule/4B/27B controller comparison without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Project bootstrap must precede application imports.
from app.local_controller_benchmark import (  # noqa: E402
    LOCAL_CONTROLLER_BENCHMARK_SCHEMA,
    LocalTransformersInference,
    LocalTransformersRuntimeConfig,
    build_benchmark_manifest,
    build_controller_decision_report,
    build_local_controller,
    build_rule_controller,
    load_evaluation_suite,
    load_prompt_suite,
    run_controller_benchmark,
    sha256_file,
    verify_prompt_eval_isolation,
    write_benchmark_result,
)


DEFAULT_EVAL = (
    PROJECT_ROOT
    / "experiments"
    / "opening_risk_routing_closure"
    / "configs"
    / "controllers"
    / "local_controller_eval_v1.json"
)
DEFAULT_PROMPT = (
    PROJECT_ROOT
    / "experiments"
    / "opening_risk_routing_closure"
    / "configs"
    / "controllers"
    / "local_controller_prompt_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "opening_risk_routing_closure"
    / "outputs"
    / "local_controller_benchmark_v1"
)
MODEL_SPECS = {
    "4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_path": Path(
            "/training_data/lizekun/model_cache/qwen3.5-4b"
        ),
    },
    "27b": {
        "model_id": "Qwen/Qwen3.5-27B",
        "model_path": Path(
            "/training_data/lizekun/model_cache/qwen3.5-27b"
        ),
    },
}
EXPECTED_REUSED_RESULTS = {
    "rule_controller": ("not_applicable", "rule_controller"),
    "qwen3_5_4b_zero_shot": ("zero_shot", "local_llm_controller"),
    "qwen3_5_4b_few_shot": ("few_shot", "local_llm_controller"),
    "qwen3_5_27b_zero_shot": ("zero_shot", "local_llm_controller"),
    "qwen3_5_27b_few_shot": ("few_shot", "local_llm_controller"),
}


def _parse_csv(value: str, *, allowed: set[str]) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(selected) - allowed
    if unknown or not selected:
        raise argparse.ArgumentTypeError(
            "invalid values: " + ",".join(sorted(unknown))
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        default="rule",
        help="comma-separated: rule,4b,27b",
    )
    parser.add_argument(
        "--prompt-modes",
        default="zero_shot,few_shot",
        help="comma-separated: zero_shot,few_shot",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate frozen configs and model snapshots without loading weights",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="rebuild the decision report and manifest from existing results",
    )
    return parser.parse_args()


def _finalize_outputs(
    *,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    isolation: dict[str, Any],
    results_reused: bool = False,
    reuse_validation: dict[str, Any] | None = None,
) -> None:
    labels = [str(result.get("controller_label", "")) for result in results]
    if len(labels) != len(set(labels)) or any(
        not label
        or any(character not in "abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in label)
        for label in labels
    ):
        raise ValueError("controller result labels are missing, unsafe, or duplicated")
    result_paths = [
        args.output_dir / f"{label}_results.json" for label in labels
    ]
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "controller result files missing: " + ",".join(missing)
        )
    decision = build_controller_decision_report(results)
    decision_path = args.output_dir / "controller_decision_report.json"
    write_benchmark_result(decision_path, decision)
    manifest = build_benchmark_manifest(
        eval_path=args.eval_config,
        prompt_path=args.prompt_config,
        results=results,
        isolation=isolation,
        project_root=PROJECT_ROOT,
        result_paths=result_paths,
        results_reused=results_reused,
        reuse_validation=reuse_validation,
    )
    manifest["decision_report"] = {
        "path": decision_path.name,
        "sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
    }
    write_benchmark_result(
        args.output_dir / "benchmark_manifest.json",
        manifest,
    )


def _load_reusable_results(
    *,
    output_dir: Path,
    evaluation_suite: Any,
    eval_path: Path,
    prompt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = output_dir / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("existing benchmark manifest is required for reuse")
    manifest_bytes = manifest_path.read_bytes()
    prior_manifest = json.loads(manifest_bytes.decode("utf-8"))
    expected_eval_sha = sha256_file(eval_path)
    expected_prompt_sha = sha256_file(prompt_path)
    if prior_manifest.get("evaluation_config_sha256") != expected_eval_sha:
        raise ValueError("REUSE_EVALUATION_CONFIG_SHA_MISMATCH")
    if prior_manifest.get("prompt_config_sha256") != expected_prompt_sha:
        raise ValueError("REUSE_PROMPT_CONFIG_SHA_MISMATCH")

    result_paths = sorted(output_dir.glob("*_results.json"))
    expected_names = {
        f"{label}_results.json" for label in EXPECTED_REUSED_RESULTS
    }
    actual_names = {path.name for path in result_paths}
    if actual_names != expected_names:
        raise ValueError(
            "REUSE_RESULT_FILE_SET_MISMATCH:"
            f"expected={sorted(expected_names)};actual={sorted(actual_names)}"
        )

    prior_hashes = {
        Path(str(item.get("path", ""))).name: str(
            item.get("sha256", "")
        )
        for item in prior_manifest.get("result_artifacts", [])
        if isinstance(item, dict)
    }
    results: list[dict[str, Any]] = []
    labels: list[str] = []
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"REUSE_RESULT_NOT_OBJECT:{path.name}")
        label = str(result.get("controller_label", ""))
        expected = EXPECTED_REUSED_RESULTS.get(label)
        if expected is None or path.name != f"{label}_results.json":
            raise ValueError(f"REUSE_RESULT_LABEL_MISMATCH:{path.name}")
        expected_prompt_mode, expected_controller_type = expected
        checks = {
            "schema": (
                result.get("schema_version")
                == LOCAL_CONTROLLER_BENCHMARK_SCHEMA
            ),
            "benchmark": (
                result.get("benchmark_id")
                == evaluation_suite.benchmark_id
            ),
            "status": result.get("status") == "completed",
            "prompt_mode": (
                result.get("prompt_mode") == expected_prompt_mode
            ),
            "controller_type": (
                result.get("controller_type")
                == expected_controller_type
            ),
            "case_count": (
                int(result.get("metrics", {}).get("case_count", -1))
                == len(evaluation_suite.cases)
                == len(result.get("cases", []))
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"REUSE_RESULT_VALIDATION_FAILED:{path.name}:"
                + ",".join(failed)
            )
        if prior_hashes and (
            prior_hashes.get(path.name) != sha256_file(path)
        ):
            raise ValueError(f"REUSE_RESULT_SHA_MISMATCH:{path.name}")
        labels.append(label)
        results.append(result)
    if len(labels) != len(set(labels)):
        raise ValueError("REUSE_RESULT_LABEL_DUPLICATED")
    return results, {
        "validated": True,
        "source_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "evaluation_config_sha256": expected_eval_sha,
        "prompt_config_sha256": expected_prompt_sha,
        "required_completed_controller_labels": sorted(
            EXPECTED_REUSED_RESULTS
        ),
        "result_sha_binding_from_prior_manifest": bool(prior_hashes),
        "model_inference_rerun": False,
    }


def _blocked_result(
    *,
    benchmark_id: str,
    controller_label: str,
    prompt_mode: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": LOCAL_CONTROLLER_BENCHMARK_SCHEMA,
        "benchmark_id": benchmark_id,
        "controller_label": controller_label,
        "controller_type": "local_llm_controller",
        "prompt_mode": prompt_mode,
        "status": "blocked",
        "blocking_reason": (
            f"{type(exc).__name__}:{exc}"
        )[:2000],
        "metrics": {
            "case_count": 0,
            "runtime_error_count": 1,
        },
        "cases": [],
    }


def main() -> None:
    args = parse_args()
    selected_models = _parse_csv(
        args.models,
        allowed={"rule", "4b", "27b"},
    )
    prompt_modes = _parse_csv(
        args.prompt_modes,
        allowed={"zero_shot", "few_shot"},
    )
    if args.max_new_tokens < 16 or args.max_new_tokens > 512:
        raise ValueError("--max-new-tokens must be between 16 and 512")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    prompt_suite = load_prompt_suite(args.prompt_config)
    evaluation_suite = load_evaluation_suite(args.eval_config)
    isolation = verify_prompt_eval_isolation(
        prompt_suite,
        evaluation_suite,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_results:
        results, reuse_validation = _load_reusable_results(
            output_dir=args.output_dir,
            evaluation_suite=evaluation_suite,
            eval_path=args.eval_config,
            prompt_path=args.prompt_config,
        )
        _finalize_outputs(
            args=args,
            results=results,
            isolation=isolation,
            results_reused=True,
            reuse_validation=reuse_validation,
        )
        print(
            json.dumps(
                {
                    "status": "existing_results_summarized",
                    "controller_count": len(results),
                    "reuse_validation": reuse_validation,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    results: list[dict[str, Any]] = []

    if "rule" in selected_models:
        result = run_controller_benchmark(
            controller=build_rule_controller(),
            evaluation_suite=evaluation_suite,
            controller_label="rule_controller",
            prompt_mode="not_applicable",
        )
        write_benchmark_result(
            args.output_dir / "rule_controller_results.json",
            result,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "controller": result["controller_label"],
                    "status": result["status"],
                    "metrics": result["metrics"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for model_key in ("4b", "27b"):
        if model_key not in selected_models:
            continue
        spec = MODEL_SPECS[model_key]
        runtime: LocalTransformersInference | None = None
        try:
            runtime = LocalTransformersInference(
                LocalTransformersRuntimeConfig(
                    model_id=str(spec["model_id"]),
                    model_path=Path(spec["model_path"]),
                    device=args.device,
                    dtype=args.dtype,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            if args.verify_only:
                print(
                    json.dumps(
                        {
                            "model": model_key,
                            "status": "snapshot_verified",
                            "runtime": runtime.config.public_dict(),
                            "snapshot": runtime.snapshot,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            for prompt_mode in prompt_modes:
                label = f"qwen3_5_{model_key}_{prompt_mode}"
                controller = build_local_controller(
                    model_id=str(spec["model_id"]),
                    prompt_suite=prompt_suite,
                    prompt_mode=prompt_mode,
                    inference_callable=runtime,
                    max_new_tokens=args.max_new_tokens,
                )
                result = run_controller_benchmark(
                    controller=controller,
                    evaluation_suite=evaluation_suite,
                    controller_label=label,
                    prompt_mode=prompt_mode,
                    inference_callable=runtime,
                )
                result["runtime"] = runtime.config.public_dict()
                result["snapshot"] = runtime.snapshot
                write_benchmark_result(
                    args.output_dir / f"{label}_results.json",
                    result,
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            "controller": label,
                            "status": result["status"],
                            "blocking_reason": result["blocking_reason"],
                            "metrics": result["metrics"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        except Exception as exc:
            traceback.print_exc()
            for prompt_mode in prompt_modes:
                label = f"qwen3_5_{model_key}_{prompt_mode}"
                result = _blocked_result(
                    benchmark_id=evaluation_suite.benchmark_id,
                    controller_label=label,
                    prompt_mode=prompt_mode,
                    exc=exc,
                )
                write_benchmark_result(
                    args.output_dir / f"{label}_results.json",
                    result,
                )
                results.append(result)
        finally:
            if runtime is not None:
                runtime.close()

    if not args.verify_only:
        _finalize_outputs(
            args=args,
            results=results,
            isolation=isolation,
        )
        print(
            json.dumps(
                {
                    "status": "finished",
                    "output_dir": str(args.output_dir),
                    "controller_count": len(results),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
