from __future__ import annotations

from statistics import mean, median
from typing import Any

from app.services.fretboard import note_to_midi


def _track_by_name(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracks: dict[str, dict[str, Any]] = {}
    for track in draft.get("tracks", []):
        name = str(track.get("name"))
        existing = tracks.setdefault(name, {**track, "notes": []})
        existing.setdefault("notes", []).extend(track.get("notes", []))
    return tracks


def _event_kind(note: dict[str, Any]) -> str:
    if "drum" in note:
        return f"drum:{note['drum']}"
    midi = _event_midi(note)
    if midi is not None:
        return f"midi:{midi}"
    return f"pitch:{note.get('pitch')}"


def _event_midi(note: dict[str, Any]) -> int | None:
    if note.get("pitch_midi") is not None:
        return int(note["pitch_midi"])
    if note.get("pitch"):
        try:
            return note_to_midi(str(note["pitch"]))
        except (KeyError, ValueError):
            return None
    return None


def _match_event_pairs(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance_beats: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return _match_event_analysis(reference, candidate, tolerance_beats)["pairs"]


def _match_event_analysis(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance_beats: float,
) -> dict[str, Any]:
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []

    ordered_candidates = sorted(enumerate(candidate), key=lambda item: float(item[1].get("start_beat", 0)))
    for candidate_index, candidate_note in ordered_candidates:
        candidate_kind = _event_kind(candidate_note)
        candidate_start = float(candidate_note.get("start_beat", 0))
        best_index: int | None = None
        best_distance = tolerance_beats + 1

        for index, reference_note in enumerate(reference):
            if index in used_reference:
                continue
            if _event_kind(reference_note) != candidate_kind:
                continue
            distance = abs(float(reference_note.get("start_beat", 0)) - candidate_start)
            if distance <= tolerance_beats and distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is not None:
            used_reference.add(best_index)
            used_candidate.add(candidate_index)
            matches.append((reference[best_index], candidate_note))

    return {
        "pairs": matches,
        "unmatched_reference": [note for index, note in enumerate(reference) if index not in used_reference],
        "unmatched_candidate": [note for index, note in enumerate(candidate) if index not in used_candidate],
    }


def _match_events(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], tolerance_beats: float) -> int:
    return len(_match_event_pairs(reference, candidate, tolerance_beats))


def _score_counts(reference_count: int, candidate_count: int, matches: int) -> dict[str, Any]:
    precision = matches / candidate_count if candidate_count else 0.0
    recall = matches / reference_count if reference_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "matches": matches,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _accuracy_score(matches: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(matches / total, 4)


def _duration_metrics(
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    duration_tolerance_beats: float,
) -> dict[str, Any]:
    comparable = [
        (reference, candidate)
        for reference, candidate in matched_pairs
        if reference.get("duration") is not None and candidate.get("duration") is not None
    ]
    matches = sum(
        1
        for reference, candidate in comparable
        if abs(float(reference.get("duration", 0)) - float(candidate.get("duration", 0))) <= duration_tolerance_beats
    )
    return {
        "comparable": len(comparable),
        "matches": matches,
        "accuracy": _accuracy_score(matches, len(comparable)),
    }


def _timing_metrics(matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    deltas = [
        abs(float(reference.get("start_beat", 0)) - float(candidate.get("start_beat", 0)))
        for reference, candidate in matched_pairs
    ]
    if not deltas:
        return {
            "comparable": 0,
            "mean_delta_beats": None,
            "median_delta_beats": None,
            "max_delta_beats": None,
        }
    return {
        "comparable": len(deltas),
        "mean_delta_beats": round(mean(deltas), 4),
        "median_delta_beats": round(median(deltas), 4),
        "max_delta_beats": round(max(deltas), 4),
    }


def _position_metrics(matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    comparable = [
        (reference, candidate)
        for reference, candidate in matched_pairs
        if reference.get("string") is not None and reference.get("fret") is not None
    ]
    matches = sum(
        1
        for reference, candidate in comparable
        if int(reference.get("string")) == int(candidate.get("string", -1))
        and int(reference.get("fret")) == int(candidate.get("fret", -1))
    )
    return {
        "comparable": len(comparable),
        "matches": matches,
        "accuracy": _accuracy_score(matches, len(comparable)),
    }


def _technique_metrics(matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    comparable = [
        (reference, candidate)
        for reference, candidate in matched_pairs
        if reference.get("technique")
    ]
    matches = sum(1 for reference, candidate in comparable if reference.get("technique") == candidate.get("technique"))
    return {
        "comparable": len(comparable),
        "matches": matches,
        "accuracy": _accuracy_score(matches, len(comparable)),
    }


def _metadata_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_metadata = reference.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    reference_constraints = reference.get("constraints", {})
    candidate_constraints = candidate.get("constraints", {})

    checks = []
    if reference_metadata.get("tempo") is not None and candidate_metadata.get("tempo") is not None:
        checks.append(
            {
                "name": "tempo",
                "match": abs(int(reference_metadata["tempo"]) - int(candidate_metadata["tempo"])) <= 1,
            }
        )
    for key in ("time_signature", "capo_fret"):
        if reference_constraints.get(key) is not None and candidate_constraints.get(key) is not None:
            checks.append({"name": key, "match": reference_constraints.get(key) == candidate_constraints.get(key)})
    if reference.get("tuning", {}).get("name") and candidate.get("tuning", {}).get("name"):
        checks.append(
            {
                "name": "tuning",
                "match": reference.get("tuning", {}).get("name") == candidate.get("tuning", {}).get("name"),
            }
        )

    matches = sum(1 for check in checks if check["match"])
    return {
        "checks": checks,
        "matches": matches,
        "comparable": len(checks),
        "accuracy": _accuracy_score(matches, len(checks)),
    }


def _compact_event(note: dict[str, Any]) -> dict[str, Any]:
    keys = ("pitch", "pitch_midi", "drum", "start_beat", "duration", "string", "fret", "technique", "velocity")
    return {key: note[key] for key in keys if key in note}


def _sample_events(notes: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [_compact_event(note) for note in sorted(notes, key=lambda item: float(item.get("start_beat", 0)))[:limit]]


def _strict_diagnostics(
    track_analyses: dict[str, dict[str, Any]],
    matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metadata: dict[str, Any],
    duration_tolerance_beats: float,
) -> dict[str, Any]:
    duration_mismatches = []
    position_mismatches = []
    technique_mismatches = []

    for reference, candidate in matched_pairs:
        if reference.get("duration") is not None and candidate.get("duration") is not None:
            delta = abs(float(reference.get("duration", 0)) - float(candidate.get("duration", 0)))
            if delta > duration_tolerance_beats:
                duration_mismatches.append(
                    {
                        "reference": _compact_event(reference),
                        "candidate": _compact_event(candidate),
                        "delta_beats": round(delta, 4),
                    }
                )
        if reference.get("string") is not None and reference.get("fret") is not None:
            if int(reference.get("string")) != int(candidate.get("string", -1)) or int(reference.get("fret")) != int(candidate.get("fret", -1)):
                position_mismatches.append(
                    {
                        "reference": _compact_event(reference),
                        "candidate": _compact_event(candidate),
                    }
                )
        if reference.get("technique") and reference.get("technique") != candidate.get("technique"):
            technique_mismatches.append(
                {
                    "reference": _compact_event(reference),
                    "candidate": _compact_event(candidate),
                }
            )

    weakest_tracks = sorted(
        (
            {
                "name": name,
                "f1": analysis["score"]["f1"],
                "missed_count": len(analysis["unmatched_reference"]),
                "extra_count": len(analysis["unmatched_candidate"]),
            }
            for name, analysis in track_analyses.items()
        ),
        key=lambda item: (item["f1"], -(item["missed_count"] + item["extra_count"])),
    )

    return {
        "weakest_tracks": weakest_tracks[:5],
        "missed_reference": {
            name: _sample_events(analysis["unmatched_reference"])
            for name, analysis in track_analyses.items()
            if analysis["unmatched_reference"]
        },
        "extra_candidate": {
            name: _sample_events(analysis["unmatched_candidate"])
            for name, analysis in track_analyses.items()
            if analysis["unmatched_candidate"]
        },
        "duration_mismatches": duration_mismatches[:5],
        "position_mismatches": position_mismatches[:5],
        "technique_mismatches": technique_mismatches[:5],
        "metadata_failures": [check for check in metadata.get("checks", []) if not check.get("match")],
    }


def _strict_accuracy(result: dict[str, Any]) -> float:
    scores = [result["overall"]["f1"]]
    for key in ("duration", "position", "technique", "metadata"):
        accuracy = result.get(key, {}).get("accuracy")
        if accuracy is not None:
            scores.append(float(accuracy))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def compare_drafts(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    tolerance_beats: float = 0.25,
    duration_tolerance_beats: float = 0.25,
    strict: bool = False,
) -> dict[str, Any]:
    reference_tracks = _track_by_name(reference)
    candidate_tracks = _track_by_name(candidate)
    track_names = sorted(set(reference_tracks) | set(candidate_tracks))

    tracks: dict[str, Any] = {}
    total_reference = 0
    total_candidate = 0
    total_matches = 0
    all_matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    track_analyses: dict[str, dict[str, Any]] = {}

    for track_name in track_names:
        reference_notes = list(reference_tracks.get(track_name, {}).get("notes", []))
        candidate_notes = list(candidate_tracks.get(track_name, {}).get("notes", []))
        analysis = _match_event_analysis(reference_notes, candidate_notes, tolerance_beats)
        matched_pairs = analysis["pairs"]
        matches = len(matched_pairs)
        score = _score_counts(len(reference_notes), len(candidate_notes), matches)
        tracks[track_name] = score
        track_analyses[track_name] = {
            "score": score,
            "unmatched_reference": analysis["unmatched_reference"],
            "unmatched_candidate": analysis["unmatched_candidate"],
        }

        total_reference += len(reference_notes)
        total_candidate += len(candidate_notes)
        total_matches += matches
        all_matched_pairs.extend(matched_pairs)

    result = {
        "tolerance_beats": tolerance_beats,
        "duration_tolerance_beats": duration_tolerance_beats,
        "overall": _score_counts(total_reference, total_candidate, total_matches),
        "tracks": tracks,
    }
    if strict:
        result["timing"] = _timing_metrics(all_matched_pairs)
        result["duration"] = _duration_metrics(all_matched_pairs, duration_tolerance_beats)
        result["position"] = _position_metrics(all_matched_pairs)
        result["technique"] = _technique_metrics(all_matched_pairs)
        result["metadata"] = _metadata_metrics(reference, candidate)
        result["strict_accuracy"] = _strict_accuracy(result)
        result["diagnostics"] = _strict_diagnostics(
            track_analyses,
            all_matched_pairs,
            result["metadata"],
            duration_tolerance_beats,
        )
    return result
