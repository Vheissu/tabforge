from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.evaluation import compare_drafts


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pair(raw: str) -> tuple[Path, Path]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("Pairs must be formatted as reference.draft.json:candidate.draft.json")
    reference, candidate = raw.split(":", 1)
    return Path(reference), Path(candidate)


def _case_passed(result: dict[str, Any], min_f1: float, min_strict_accuracy: float | None) -> bool:
    if float(result["overall"]["f1"]) < min_f1:
        return False
    if min_strict_accuracy is not None and float(result.get("strict_accuracy", 0)) < min_strict_accuracy:
        return False
    return True


def _load_manifest(path: Path) -> list[tuple[str, Path, Path]]:
    manifest = _load_json(path)
    if isinstance(manifest, dict):
        manifest_cases = manifest.get("cases", [])
    elif isinstance(manifest, list):
        manifest_cases = manifest
    else:
        raise ValueError("Benchmark manifest must be an object with cases or a list of cases.")

    cases = []
    for index, item in enumerate(manifest_cases, start=1):
        reference = Path(item["reference"])
        candidate = Path(item["candidate"])
        if not reference.is_absolute():
            reference = path.parent / reference
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        cases.append((str(item.get("name") or f"case-{index}"), reference, candidate))
    return cases


def _case_gaps(result: dict[str, Any], min_f1: float, min_strict_accuracy: float | None) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    f1 = float(result["overall"]["f1"])
    if f1 < min_f1:
        gaps.append({"metric": "overall_f1", "actual": f1, "required": min_f1})

    if min_strict_accuracy is not None:
        strict_accuracy = float(result.get("strict_accuracy", 0))
        if strict_accuracy < min_strict_accuracy:
            gaps.append({"metric": "strict_accuracy", "actual": strict_accuracy, "required": min_strict_accuracy})

    for metric in ("duration", "position", "technique", "metadata"):
        accuracy = result.get(metric, {}).get("accuracy")
        if accuracy is not None and min_strict_accuracy is not None and float(accuracy) < min_strict_accuracy:
            gaps.append({"metric": metric, "actual": float(accuracy), "required": min_strict_accuracy})
    return gaps


def _case_summary(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics", {})
    weakest_track = (diagnostics.get("weakest_tracks") or [None])[0]
    return {
        "weakest_track": weakest_track,
        "missed_reference_tracks": sorted((diagnostics.get("missed_reference") or {}).keys()),
        "extra_candidate_tracks": sorted((diagnostics.get("extra_candidate") or {}).keys()),
        "duration_mismatch_count": len(diagnostics.get("duration_mismatches") or []),
        "position_mismatch_count": len(diagnostics.get("position_mismatches") or []),
        "technique_mismatch_count": len(diagnostics.get("technique_mismatches") or []),
        "metadata_failure_count": len(diagnostics.get("metadata_failures") or []),
    }


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TabForge draft accuracy gates against reference pairs.")
    parser.add_argument("pairs", nargs="*", type=_parse_pair, help="reference.draft.json:candidate.draft.json")
    parser.add_argument("--manifest", type=Path, help="JSON manifest with cases containing name/reference/candidate.")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report.")
    parser.add_argument("--tolerance-beats", type=float, default=0.25)
    parser.add_argument("--duration-tolerance-beats", type=float, default=0.25)
    parser.add_argument("--min-f1", type=float, default=0.90)
    parser.add_argument("--min-strict-accuracy", type=float, default=0.90)
    args = parser.parse_args()

    benchmark_cases: list[tuple[str, Path, Path]] = []
    if args.manifest:
        benchmark_cases.extend(_load_manifest(args.manifest))
    benchmark_cases.extend((f"pair-{index}", reference, candidate) for index, (reference, candidate) in enumerate(args.pairs, start=1))
    if not benchmark_cases:
        parser.error("provide at least one pair or --manifest")

    cases = []
    failed = False
    for name, reference_path, candidate_path in benchmark_cases:
        result = compare_drafts(
            _load_json(reference_path),
            _load_json(candidate_path),
            tolerance_beats=args.tolerance_beats,
            duration_tolerance_beats=args.duration_tolerance_beats,
            strict=True,
        )
        passed = _case_passed(result, args.min_f1, args.min_strict_accuracy)
        failed = failed or not passed
        cases.append(
            {
                "name": name,
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "passed": passed,
                "overall_f1": result["overall"]["f1"],
                "strict_accuracy": result["strict_accuracy"],
                "gaps": _case_gaps(result, args.min_f1, args.min_strict_accuracy),
                "summary": _case_summary(result),
                "result": result,
            }
        )

    f1_values = [float(case["overall_f1"]) for case in cases]
    strict_values = [float(case["strict_accuracy"]) for case in cases]
    summary = {
        "passed": not failed,
        "minimum_f1": args.min_f1,
        "minimum_strict_accuracy": args.min_strict_accuracy,
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if case["passed"]),
        "average_f1": _average(f1_values),
        "average_strict_accuracy": _average(strict_values),
        "cases": cases,
    }
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
