#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_backend = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(repo_backend))

    from app.tools.benchmark_accuracy import main as benchmark_main

    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
