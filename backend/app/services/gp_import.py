from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.draft import SCHEMA_VERSION, _track_stats
from app.services.fretboard import get_tuning_midi, midi_to_note_name
from app.services.gp import DRUM_NOTE_VALUES


REVERSE_DRUM_MAP = {value: name for name, value in DRUM_NOTE_VALUES.items()}


def _duration_beats(duration) -> float:
    beats = 4 / int(duration.value)
    if getattr(duration, "isDotted", False):
        beats *= 1.5
    tuplet = getattr(duration, "tuplet", None)
    if tuplet and getattr(tuplet, "enters", 1) and getattr(tuplet, "times", 1):
        beats *= float(tuplet.times) / float(tuplet.enters)
    return beats


def _time_signature_label(header) -> str:
    signature = header.timeSignature
    return f"{signature.numerator}/{signature.denominator.value}"


def _measure_beats(header) -> float:
    signature = header.timeSignature
    return signature.numerator * (4 / signature.denominator.value)


def _tempo_value(song) -> int | None:
    tempo = getattr(song, "tempo", None)
    if tempo is None:
        return None
    try:
        return int(tempo)
    except TypeError:
        value = getattr(tempo, "value", None)
        return int(value) if value is not None else None


def _track_name(track) -> str:
    if track.isPercussionTrack:
        return "drums"
    name = track.name.lower()
    if "bass" in name:
        return "bass"
    return "guitar"


def _normalised_track_tuning(track) -> list[int]:
    return [int(string.value) for string in sorted(track.strings, key=lambda item: int(item.number), reverse=True)]


def _detect_imported_tuning(tracks: list[dict[str, Any]]) -> str:
    candidates = ("standard", "drop_d", "half_step_down", "full_step_down")
    pitched_tracks = [track for track in tracks if track.get("name") in {"guitar", "bass"}]
    for tuning in candidates:
        if pitched_tracks and all(
            track.get("tuning_midi") == get_tuning_midi(tuning, is_bass=track.get("name") == "bass")
            for track in pitched_tracks
        ):
            return tuning
    return "imported"


def _string_pitch(track, string_number: int) -> int | None:
    for string in track.strings:
        if string.number == string_number:
            return int(string.value)
    return None


def _is_tied_note(note) -> bool:
    return getattr(getattr(note, "type", None), "name", "") == "tie"


def _extend_previous_tied_note(notes: list[dict[str, Any]], string: int, pitch_midi: int, duration: float) -> bool:
    for previous in reversed(notes):
        if previous.get("string") == string and int(previous.get("pitch_midi", -1)) == pitch_midi:
            previous["duration"] = round(float(previous.get("duration", 0)) + duration, 3)
            return True
    return False


def _extract_track_notes(track, measure_starts: list[float]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for measure_index, measure in enumerate(track.measures):
        measure_start = measure_starts[measure_index]
        for voice in measure.voices:
            beat_start = measure_start
            for beat in voice.beats:
                duration = _duration_beats(beat.duration)
                for note in beat.notes:
                    if track.isPercussionTrack:
                        notes.append(
                            {
                                "drum": REVERSE_DRUM_MAP.get(int(note.value), "hihat_closed"),
                                "start_beat": round(beat_start, 3),
                                "duration": round(duration, 3),
                                "velocity": int(note.velocity),
                            }
                        )
                        continue

                    open_pitch = _string_pitch(track, int(note.string))
                    if open_pitch is None:
                        continue
                    pitch_midi = open_pitch + int(note.value) + int(track.offset or 0)
                    if _is_tied_note(note) and _extend_previous_tied_note(notes, int(note.string), pitch_midi, duration):
                        continue
                    notes.append(
                        {
                            "pitch": midi_to_note_name(pitch_midi),
                            "pitch_midi": pitch_midi,
                            "start_beat": round(beat_start, 3),
                            "duration": round(duration, 3),
                            "string": int(note.string),
                            "fret": int(note.value),
                            "velocity": int(note.velocity),
                        }
                    )
                beat_start += duration
    return notes


def import_guitar_pro_file(path: Path) -> dict[str, Any]:
    import guitarpro

    song = guitarpro.parse(str(path))
    measure_starts: list[float] = []
    current = 0.0
    for header in song.measureHeaders:
        measure_starts.append(current)
        current += _measure_beats(header)

    tracks = []
    for track in song.tracks:
        notes = _extract_track_notes(track, measure_starts)
        tracks.append(
            {
                "name": _track_name(track),
                "source_track": track.name,
                "capo_fret": int(track.offset or 0),
                "tuning_midi": _normalised_track_tuning(track),
                "strings": [{"number": int(string.number), "value": int(string.value)} for string in track.strings],
                "statistics": _track_stats(notes),
                "notes": notes,
            }
        )

    pitched_capos = [
        int(track.get("capo_fret") or 0)
        for track in tracks
        if track.get("name") in {"guitar", "bass"}
    ]
    common_capo = pitched_capos[0] if pitched_capos and all(capo == pitched_capos[0] for capo in pitched_capos) else 0

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "title": song.title or path.stem,
            "artist": song.artist or None,
            "tempo": _tempo_value(song),
            "imported_from": str(path),
        },
        "constraints": {
            "time_signature": _time_signature_label(song.measureHeaders[0]) if song.measureHeaders else "4/4",
            "capo_fret": common_capo,
        },
        "tuning": {
            "name": _detect_imported_tuning(tracks),
            "details": None,
        },
        "sources": {
            "guitar_pro_file": str(path),
        },
        "tracks": tracks,
        "quality": {
            "warnings": [],
        },
    }
