import os
from pathlib import Path

import pytest

pytest.importorskip("ophbench")

from app.ophbench_registry_bridge import load_external_catalog


def test_real_ophbench_registry_bridge_reads_seed_catalog():
    registry_root = os.environ.get("OPHBENCH_REGISTRY_ROOT")
    if not registry_root:
        pytest.skip("OPHBENCH_REGISTRY_ROOT is required for the cross-repository integration test")

    result = load_external_catalog(registry_root=Path(registry_root))

    assert result.available is True
    assert result.model_count == 15
    assert result.checkpoint_count == 27
    retfound = next(model for model in result.models if model.model_id == "retfound")
    assert len(retfound.checkpoints) == 2
    assert retfound.runnable is False
    assert retfound.task_checkpoint is False
