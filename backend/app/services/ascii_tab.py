from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.services.draft import SCHEMA_VERSION, _track_stats
from app.services.fretboard import get_tuning_midi, midi_to_note_name


TAB_LINE_RE = re.compile(r"^\s*([eBGDAE])\s*\|([^|\r\n]*)(?:\|.*)?$")
STRING_ORDER = ("e", "B", "G", "D", "A", "E")
STRING_NUMBERS = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}
TECHNIQUE_MARKERS = {
    "h": "hammer_on",
    "p": "pull_off",
    "/": "slide_up",
    "\\": "slide_down",
    "b": "bend",
    "r": "release",
    "~": "vibrato",
}


def _extract_blocks(text: str) -> list[list[tuple[str, str]]]:
    blocks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal current
        if len(current) == 6 and tuple(label for label, _ in current) == STRING_ORDER:
            blocks.append(current)
        current = []

    for line in text.splitlines():
        match = TAB_LINE_RE.match(line)
        if not match:
            flush()
            continue

        label, body = match.groups()
        expected_label = STRING_ORDER[len(current)] if len(current) < len(STRING_ORDER) else None
        if label != expected_label:
            flush()
            expected_label = STRING_ORDER[len(current)] if len(current) < len(STRING_ORDER) else None
        if label == expected_label:
            current.append((label, body.rstrip()))
        else:
            current = []

    flush()
    return blocks


def _is_bend_target(body: str, match: re.Match[str]) -> bool:
    before = body[match.start() - 1] if match.start() > 0 else ""
    after = body[match.end()] if match.end() < len(body) else ""
    return before == "(" or after == ")"


def _technique_for_token(body: str, match: re.Match[str]) -> str | None:
    before = body[match.start() - 1] if match.start() > 0 else ""
    after = body[match.end()] if match.end() < len(body) else ""
    if before in TECHNIQUE_MARKERS:
        return TECHNIQUE_MARKERS[before]
    if after in TECHNIQUE_MARKERS:
        return TECHNIQUE_MARKERS[after]
    return None


def _open_pitch_for_string(tuning: str, string_number: int) -> int:
    low_to_high = get_tuning_midi(tuning)
    high_to_low = list(reversed(low_to_high))
    return high_to_low[string_number - 1]


def _duration_for_note(
    note: dict[str, Any],
    notes_by_string: dict[int, list[dict[str, Any]]],
    default_duration_beats: float,
) -> float:
    string_notes = notes_by_string[int(note["string"])]
    index = string_notes.index(note)
    if index == len(string_notes) - 1:
        return default_duration_beats
    next_start = float(string_notes[index + 1]["start_beat"])
    duration = max(0.0, next_start - float(note["start_beat"]))
    return round(max(default_duration_beats, duration), 3)


def ascii_tab_to_draft(
    text: str,
    *,
    title: str = "Reference Tab",
    artist: str = "Unknown",
    tempo: int = 120,
    key: str | None = None,
    tuning: str = "standard",
    time_signature: str = "4/4",
    capo_fret: int = 0,
    columns_per_beat: float = 4.0,
    default_duration_beats: float = 0.25,
    track_name: str = "guitar",
) -> dict[str, Any]:
    if columns_per_beat <= 0:
        raise ValueError("columns_per_beat must be greater than 0")
    if default_duration_beats <= 0:
        raise ValueError("default_duration_beats must be greater than 0")

    notes: list[dict[str, Any]] = []
    current_beat = 0.0

    for block_index, block in enumerate(_extract_blocks(text)):
        max_columns = max((len(body) for _, body in block), default=0)
        for label, body in block:
            string_number = STRING_NUMBERS[label]
            open_pitch = _open_pitch_for_string(tuning, string_number)
            for match in re.finditer(r"\d{1,2}", body):
                if _is_bend_target(body, match):
                    continue
                fret = int(match.group())
                pitch_midi = open_pitch + fret
                note: dict[str, Any] = {
                    "pitch": midi_to_note_name(pitch_midi),
                    "pitch_midi": pitch_midi,
                    "start_beat": round(current_beat + (match.start() / columns_per_beat), 3),
                    "duration": default_duration_beats,
                    "string": string_number,
                    "fret": fret,
                    "velocity": 100,
                    "source_column": match.start(),
                    "source_block": block_index,
                }
                technique = _technique_for_token(body, match)
                if technique:
                    note["technique"] = technique
                notes.append(note)
        current_beat += max_columns / columns_per_beat

    notes.sort(key=lambda item: (float(item["start_beat"]), int(item["string"]), int(item["fret"])))
    notes_by_string: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        notes_by_string.setdefault(int(note["string"]), []).append(note)
    for string_notes in notes_by_string.values():
        string_notes.sort(key=lambda item: float(item["start_beat"]))
    for note in notes:
        note["duration"] = _duration_for_note(note, notes_by_string, default_duration_beats)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "title": title,
            "artist": artist,
            "tempo": tempo,
            "key": key,
        },
        "constraints": {
            "tempo_bpm": tempo,
            "tempo_source": "reference",
            "time_signature": time_signature,
            "time_signature_source": "reference",
            "capo_fret": capo_fret,
            "pickup_bar_beats": None,
            "triplet_feel": False,
        },
        "tuning": {
            "name": tuning,
            "details": None,
        },
        "sources": {
            "ascii_tab": {
                "block_count": len(_extract_blocks(text)),
                "columns_per_beat": columns_per_beat,
            }
        },
        "tracks": [
            {
                "name": track_name,
                "source_track": "ascii_tab",
                "statistics": _track_stats(notes),
                "analysis": {
                    "source": "ascii_tab",
                    "columns_per_beat": columns_per_beat,
                },
                "notes": notes,
            }
        ],
        "quality": {
            "warnings": [],
        },
    }
