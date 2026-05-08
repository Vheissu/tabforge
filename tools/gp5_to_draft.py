#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Guitar Pro file into TabForge draft JSON.")
    parser.add_argument("input", type=Path, help="Path to a .gp5 file")
    parser.add_argument("-o", "--output", type=Path, help="Output .draft.json path")
    args = parser.parse_args()

    repo_backend = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(repo_backend))

    from app.services.gp_import import import_guitar_pro_file

    draft = import_guitar_pro_file(args.input)
    output = args.output or args.input.with_suffix(".draft.json")
    output.write_text(json.dumps(draft, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
