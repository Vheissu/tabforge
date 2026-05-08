from __future__ import annotations

from typing import Any


def _track_by_name(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(track.get("name")): track for track in draft.get("tracks", [])}


def _event_kind(note: dict[str, Any]) -> str:
    if "drum" in note:
        return f"drum:{note['drum']}"
    if "pitch_midi" in note:
        return f"midi:{int(note['pitch_midi'])}"
    return f"pitch:{note.get('pitch')}"


def _match_events(reference: list[dict[str, Any]], candidate: list[dict[str, Any]], tolerance_beats: float) -> int:
    used_reference: set[int] = set()
    matches = 0

    for candidate_note in sorted(candidate, key=lambda note: float(note.get("start_beat", 0))):
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
            matches += 1

    return matches


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


def compare_drafts(reference: dict[str, Any], candidate: dict[str, Any], tolerance_beats: float = 0.25) -> dict[str, Any]:
    reference_tracks = _track_by_name(reference)
    candidate_tracks = _track_by_name(candidate)
    track_names = sorted(set(reference_tracks) | set(candidate_tracks))

    tracks: dict[str, Any] = {}
    total_reference = 0
    total_candidate = 0
    total_matches = 0

    for track_name in track_names:
        reference_notes = list(reference_tracks.get(track_name, {}).get("notes", []))
        candidate_notes = list(candidate_tracks.get(track_name, {}).get("notes", []))
        matches = _match_events(reference_notes, candidate_notes, tolerance_beats)
        tracks[track_name] = _score_counts(len(reference_notes), len(candidate_notes), matches)

        total_reference += len(reference_notes)
        total_candidate += len(candidate_notes)
        total_matches += matches

    return {
        "tolerance_beats": tolerance_beats,
        "overall": _score_counts(total_reference, total_candidate, total_matches),
        "tracks": tracks,
    }
