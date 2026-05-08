from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "tabforge-draft-v1"


def _notes_for_track(transcription: dict[str, Any], track_name: str) -> list[dict]:
    if track_name == "drums":
        return list(transcription.get("drums", []))
    return list(transcription.get(track_name, {}).get("notes", []))


def _track_stats(notes: list[dict]) -> dict[str, Any]:
    chord_slots: dict[float, int] = {}
    for note in notes:
        start = round(float(note.get("start_beat", 0)), 3)
        chord_slots[start] = chord_slots.get(start, 0) + 1

    max_end = 0.0
    for note in notes:
        start = float(note.get("start_beat", 0))
        duration = float(note.get("duration", 0))
        max_end = max(max_end, start + duration)

    return {
        "note_count": len(notes),
        "chord_slot_count": sum(1 for count in chord_slots.values() if count > 1),
        "last_beat": round(max_end, 3),
    }


def _quality_warnings(transcription: dict[str, Any], instruments: list[str], source_stems: dict[str, str]) -> list[dict]:
    warnings: list[dict] = []
    constraints = transcription.get("constraints", {})

    if "guitar" in instruments and "guitar" not in source_stems:
        warnings.append(
            {
                "code": "mixed_guitar_stem",
                "severity": "warning",
                "message": "Guitar was transcribed from the mixed other stem because no dedicated guitar stem was available.",
            }
        )

    if constraints.get("tempo_source") == "detected":
        warnings.append(
            {
                "code": "detected_tempo",
                "severity": "info",
                "message": "Tempo was inferred from the full mix. Set first-bar tempo for a steadier draft.",
            }
        )

    if constraints.get("time_signature_source") == "auto":
        warnings.append(
            {
                "code": "assumed_meter",
                "severity": "info",
                "message": "Meter was assumed as 4/4. Set the first-bar meter for songs in 3/4, 6/8, or 12/8.",
            }
        )

    tuning_info = transcription.get("tuning_info")
    if tuning_info:
        warnings.append(
            {
                "code": "detected_tuning",
                "severity": "info",
                "message": f"Tuning was detected as {transcription.get('tuning', 'standard')}.",
            }
        )

    return warnings


def build_tab_draft(transcription: dict[str, Any], instruments: list[str]) -> dict[str, Any]:
    source_stems = dict(transcription.get("source_stems", {}))
    tracks = []
    for instrument in instruments:
        notes = _notes_for_track(transcription, instrument)
        tracks.append(
            {
                "name": instrument,
                "source_stem": source_stems.get("guitar" if instrument == "guitar" else instrument),
                "statistics": _track_stats(notes),
                "notes": notes,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "title": transcription.get("title", "Unknown"),
            "artist": transcription.get("artist", "Unknown"),
            "tempo": transcription.get("tempo"),
            "detected_tempo": transcription.get("detected_tempo"),
            "key": transcription.get("key"),
        },
        "constraints": transcription.get("constraints", {}),
        "tuning": {
            "name": transcription.get("tuning", "standard"),
            "details": transcription.get("tuning_info"),
        },
        "sources": {
            "stems": source_stems,
        },
        "tracks": tracks,
        "quality": {
            "warnings": _quality_warnings(transcription, instruments, source_stems),
        },
    }
