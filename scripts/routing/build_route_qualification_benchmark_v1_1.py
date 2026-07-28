"""Build Route Qualification Benchmark v1.1 from frozen read-only assets."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.route_qualification_benchmark import (  # noqa: E402
    write_route_qualification_benchmark_artifacts,
)


def main() -> None:
    paths = write_route_qualification_benchmark_artifacts(PROJECT_ROOT)
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
