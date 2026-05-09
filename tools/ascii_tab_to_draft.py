#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a six-line ASCII guitar tab into TabForge draft JSON.")
    parser.add_argument("input", type=Path, help="Path to a text file containing ASCII tab blocks")
    parser.add_argument("-o", "--output", type=Path, help="Output .draft.json path")
    parser.add_argument("--title", default="Reference Tab")
    parser.add_argument("--artist", default="Unknown")
    parser.add_argument("--tempo", type=int, default=120)
    parser.add_argument("--key")
    parser.add_argument("--tuning", default="standard")
    parser.add_argument("--time-signature", default="4/4")
    parser.add_argument("--capo-fret", type=int, default=0)
    parser.add_argument("--columns-per-beat", type=float, default=4.0)
    parser.add_argument("--default-duration-beats", type=float, default=0.25)
    parser.add_argument("--track-name", default="guitar")
    args = parser.parse_args()

    repo_backend = Path(__file__).resolve().parents[1] / "backend"
    sys.path.insert(0, str(repo_backend))

    from app.services.ascii_tab import ascii_tab_to_draft

    draft = ascii_tab_to_draft(
        args.input.read_text(encoding="utf-8"),
        title=args.title,
        artist=args.artist,
        tempo=args.tempo,
        key=args.key,
        tuning=args.tuning,
        time_signature=args.time_signature,
        capo_fret=args.capo_fret,
        columns_per_beat=args.columns_per_beat,
        default_duration_beats=args.default_duration_beats,
        track_name=args.track_name,
    )
    output = args.output or args.input.with_suffix(".draft.json")
    output.write_text(json.dumps(draft, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
