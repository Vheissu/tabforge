#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a TabForge draft against a reference draft.")
    parser.add_argument("reference", type=Path, help="Reference .draft.json")
    parser.add_argument("candidate", type=Path, help="Candidate .draft.json")
    parser.add_argument("--tolerance-beats", type=float, default=0.25)
    args = parser.parse_args()

    repo_backend = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(repo_backend))

    from app.services.evaluation import compare_drafts

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    print(json.dumps(compare_drafts(reference, candidate, args.tolerance_beats), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
