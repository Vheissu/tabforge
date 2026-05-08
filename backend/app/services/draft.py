from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.fretboard import Note, get_tuning_midi, note_to_midi, optimize_positions


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


def _next_actions(warnings: list[dict], tracks: list[dict]) -> list[str]:
    actions: list[str] = []
    warning_codes = {warning.get("code") for warning in warnings}

    if "detected_tempo" in warning_codes:
        actions.append("Set the first-bar tempo when you know the BPM.")
    if "assumed_meter" in warning_codes:
        actions.append("Set the first-bar meter for songs outside straight 4/4.")
    if "mixed_guitar_stem" in warning_codes:
        actions.append("Keep the 6-stem separator enabled so guitar can use a dedicated stem.")

    for track in tracks:
        stats = track.get("statistics", {})
        name = track.get("name", "track")
        note_count = int(stats.get("note_count") or 0)
        last_beat = float(stats.get("last_beat") or 0)
        density = note_count / last_beat if last_beat > 0 else 0
        if note_count == 0:
            actions.append(f"Check the {name} stem; no notes were written.")
        elif density > 6:
            actions.append(f"Review {name}; note density is high and may include transcription noise.")

    return actions[:5]


def summarise_draft(draft: dict[str, Any]) -> dict[str, Any]:
    tracks = []
    total_notes = 0
    last_beat = 0.0

    for track in draft.get("tracks", []):
        stats = dict(track.get("statistics", {}))
        total_notes += int(stats.get("note_count") or 0)
        last_beat = max(last_beat, float(stats.get("last_beat") or 0))
        tracks.append(
            {
                "name": track.get("name"),
                "source_stem": track.get("source_stem") or track.get("source_track"),
                "statistics": stats,
            }
        )

    warnings = list(draft.get("quality", {}).get("warnings", []))
    return {
        "schema_version": draft.get("schema_version", SCHEMA_VERSION),
        "metadata": draft.get("metadata", {}),
        "constraints": draft.get("constraints", {}),
        "tuning": draft.get("tuning", {}),
        "sources": draft.get("sources", {}),
        "tracks": tracks,
        "quality": {
            "warnings": warnings,
            "next_actions": _next_actions(warnings, tracks),
        },
        "statistics": {
            "track_count": len(tracks),
            "note_count": total_notes,
            "last_beat": round(last_beat, 3),
        },
    }


def _track_notes(track: dict[str, Any]) -> list[dict[str, Any]]:
    return list(track.get("notes", []))


def _slot_key(note: dict[str, Any]) -> float:
    return round(float(note.get("start_beat", 0)), 3)


def _limit_slot_polyphony(notes: list[dict[str, Any]], max_notes_per_slot: int | None) -> list[dict[str, Any]]:
    if not max_notes_per_slot:
        return notes

    slots: dict[float, list[dict[str, Any]]] = {}
    for note in notes:
        slots.setdefault(_slot_key(note), []).append(note)

    limited: list[dict[str, Any]] = []
    for slot_notes in slots.values():
        limited.extend(
            sorted(
                slot_notes,
                key=lambda note: int(note.get("velocity", 100)),
                reverse=True,
            )[:max_notes_per_slot]
        )
    return sorted(limited, key=lambda note: (float(note.get("start_beat", 0)), str(note.get("pitch", note.get("drum", "")))))


def _positioning_tuning(tuning: str, track_name: str, capo_fret: int) -> list[int]:
    tuning_midi = get_tuning_midi(tuning, is_bass=track_name == "bass")
    if track_name == "guitar" and capo_fret > 0:
        return [pitch + capo_fret for pitch in tuning_midi]
    return tuning_midi


def _revoice_pitched_track(track: dict[str, Any], tuning: str, capo_fret: int) -> None:
    notes: list[Note] = []
    source_notes = _track_notes(track)
    for note in source_notes:
        pitch = str(note.get("pitch") or "")
        if not pitch:
            continue
        notes.append(
            Note(
                pitch=pitch,
                start_beat=float(note.get("start_beat", 0)),
                duration=float(note.get("duration", 0.25)),
                velocity=int(note.get("velocity", 100)),
            )
        )

    optimised = optimize_positions(notes, tuning=_positioning_tuning(tuning, str(track.get("name")), capo_fret))
    by_key = {(note.pitch, round(note.start_beat, 3), round(note.duration, 3)): note for note in optimised}
    revoiced: list[dict[str, Any]] = []
    for note in source_notes:
        pitch = str(note.get("pitch") or "")
        if not pitch:
            continue
        key = (pitch, round(float(note.get("start_beat", 0)), 3), round(float(note.get("duration", 0.25)), 3))
        optimised_note = by_key.get(key)
        if optimised_note and optimised_note.position:
            updated = dict(note)
            updated["pitch_midi"] = note_to_midi(pitch)
            updated["string"] = optimised_note.position.string
            updated["fret"] = optimised_note.position.fret
            revoiced.append(updated)
    track["notes"] = revoiced


def _apply_track_correction(track: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any] | None:
    if correction.get("enabled") is False:
        return None

    notes = _track_notes(track)
    min_velocity = correction.get("min_velocity")
    if min_velocity is not None:
        notes = [note for note in notes if int(note.get("velocity", 100)) >= int(min_velocity)]
    notes = _limit_slot_polyphony(notes, correction.get("max_notes_per_slot"))

    updated = dict(track)
    updated["notes"] = notes
    updated["statistics"] = _track_stats(notes)
    return updated


def _refresh_statistics(draft: dict[str, Any]) -> None:
    for track in draft.get("tracks", []):
        track["statistics"] = _track_stats(_track_notes(track))


def apply_draft_corrections(draft: dict[str, Any], corrections: dict[str, Any]) -> dict[str, Any]:
    updated = dict(draft)
    updated["metadata"] = dict(draft.get("metadata", {}))
    updated["constraints"] = dict(draft.get("constraints", {}))
    updated["tuning"] = dict(draft.get("tuning", {}))
    updated["quality"] = dict(draft.get("quality", {}))

    if corrections.get("tempo_bpm") is not None:
        updated["metadata"]["tempo"] = int(corrections["tempo_bpm"])
        updated["constraints"]["tempo_bpm"] = int(corrections["tempo_bpm"])
        updated["constraints"]["tempo_source"] = "user"
    if corrections.get("time_signature"):
        updated["constraints"]["time_signature"] = corrections["time_signature"]
        updated["constraints"]["time_signature_source"] = "user"
    if "pickup_bar_beats" in corrections:
        updated["constraints"]["pickup_bar_beats"] = corrections["pickup_bar_beats"]
    if corrections.get("triplet_feel") is not None:
        value = corrections["triplet_feel"]
        updated["constraints"]["triplet_feel"] = value == "triplet" or value is True
    if corrections.get("tuning"):
        updated["tuning"]["name"] = corrections["tuning"]
        updated["tuning"]["details"] = None
    if corrections.get("capo_fret") is not None:
        updated["constraints"]["capo_fret"] = int(corrections["capo_fret"])

    track_corrections = corrections.get("tracks", {})
    tracks: list[dict[str, Any]] = []
    for track in draft.get("tracks", []):
        track_name = str(track.get("name"))
        corrected = _apply_track_correction(dict(track), track_corrections.get(track_name, {}))
        if corrected is None:
            continue
        if track_name in {"guitar", "bass"}:
            _revoice_pitched_track(
                corrected,
                str(updated.get("tuning", {}).get("name") or "standard"),
                int(updated.get("constraints", {}).get("capo_fret") or 0),
            )
        corrected["statistics"] = _track_stats(_track_notes(corrected))
        tracks.append(corrected)
    updated["tracks"] = tracks
    _refresh_statistics(updated)

    correction_log = list(updated.get("corrections", []))
    correction_log.append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "changes": corrections,
        }
    )
    updated["corrections"] = correction_log
    return updated


def draft_to_transcription(draft: dict[str, Any]) -> dict[str, Any]:
    metadata = draft.get("metadata", {})
    constraints = draft.get("constraints", {})
    tuning = draft.get("tuning", {})
    transcription: dict[str, Any] = {
        "title": metadata.get("title", "Unknown"),
        "artist": metadata.get("artist", "Unknown"),
        "tempo": int(metadata.get("tempo") or constraints.get("tempo_bpm") or 120),
        "key": metadata.get("key"),
        "tuning": tuning.get("name", "standard"),
        "time_signature": constraints.get("time_signature", "4/4"),
        "pickup_bar_beats": constraints.get("pickup_bar_beats"),
        "triplet_feel": bool(constraints.get("triplet_feel")),
        "capo_fret": int(constraints.get("capo_fret") or 0),
    }

    for track in draft.get("tracks", []):
        name = str(track.get("name"))
        notes = _track_notes(track)
        if name == "drums":
            transcription["drums"] = notes
        elif name in {"guitar", "bass"}:
            transcription[name] = {"notes": notes}
    return transcription


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
