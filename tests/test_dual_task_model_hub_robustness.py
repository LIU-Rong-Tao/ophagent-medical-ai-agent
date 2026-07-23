from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/routing/analyze_dual_task_model_hub_robustness.py"
)
SPEC = importlib.util.spec_from_file_location("dual_task_robustness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bootstrap_metric_rows_are_paired_and_deterministic() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    scout = np.array([0, 1, 1, 0, 2, 1])
    final = np.array([0, 0, 1, 1, 2, 1])
    kwargs = {
        "task_id": "synthetic",
        "split": "val",
        "candidate_id": "route",
        "truth": truth,
        "scout": scout,
        "final": final,
        "qwk_enabled": False,
        "repeats": 200,
        "seed": 42,
    }
    first = MODULE.bootstrap_metric_rows(**kwargs)
    second = MODULE.bootstrap_metric_rows(**kwargs)

    assert first == second
    accuracy_delta = next(
        row
        for row in first
        if row["estimate"] == "route_minus_scout"
        and row["metric"] == "accuracy"
    )
    assert np.isclose(accuracy_delta["point"], 2 / 6)
    assert accuracy_delta["ci_lower"] <= accuracy_delta["point"]
    assert accuracy_delta["ci_upper"] >= accuracy_delta["point"]


def test_proxy_masks_count_corrections_and_introductions() -> None:
    truth = np.array([4, 4, 0, 3])
    scout = np.array([1, 4, 0, 1])
    final = np.array([4, 1, 0, 2])
    masks = MODULE.proxy_masks(truth, scout, final, threshold=3)

    assert masks["corrected"].tolist() == [True, False, False, False]
    assert masks["introduced"].tolist() == [False, True, False, False]
    assert masks["net_reduction"].sum() == 0
