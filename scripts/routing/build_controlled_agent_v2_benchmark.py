"""Build the fixed de-identified V1/V2 controlled-agent benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.controlled_agent_benchmark import (  # noqa: E402
    EVALUATION_SET_RELATIVE_PATH,
    OUTPUT_RELATIVE_DIR,
    write_controlled_agent_benchmark_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic fixed-scenario evidence for the V1/V2 "
            "controlled-agent comparison."
        )
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=PROJECT_ROOT / EVALUATION_SET_RELATIVE_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / OUTPUT_RELATIVE_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_controlled_agent_benchmark_artifacts(
        PROJECT_ROOT,
        evaluation_path=args.evaluation.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
