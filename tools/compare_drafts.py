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
    parser.add_argument("--duration-tolerance-beats", type=float, default=0.25)
    parser.add_argument("--strict", action="store_true", help="Include duration, position, technique, and metadata metrics.")
    parser.add_argument("--scan-window", action="store_true", help="Slide the candidate over the reference and score the best matching window.")
    parser.add_argument("--alignment-grid-beats", type=float, default=0.25)
    parser.add_argument("--max-offset-beats", type=float)
    args = parser.parse_args()

    repo_backend = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(repo_backend))

    from app.services.evaluation import compare_drafts, scan_reference_window

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if args.scan_window:
        result = scan_reference_window(
            reference,
            candidate,
            tolerance_beats=args.tolerance_beats,
            duration_tolerance_beats=args.duration_tolerance_beats,
            strict=args.strict,
            grid_beats=args.alignment_grid_beats,
            max_offset_beats=args.max_offset_beats,
        )
    else:
        result = compare_drafts(
            reference,
            candidate,
            tolerance_beats=args.tolerance_beats,
            duration_tolerance_beats=args.duration_tolerance_beats,
            strict=args.strict,
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
