from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional


@dataclass
class FretPosition:
    string: int  # 1-6 (high E to low E)
    fret: int    # 0-24


@dataclass
class Note:
    pitch: str
    start_beat: float
    duration: float
    position: Optional[FretPosition] = None
    technique: Optional[str] = None
    velocity: int = 100


STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4
BASS_STANDARD_TUNING = [28, 33, 38, 43]  # E1, A1, D2, G2

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def note_to_midi(note: str) -> int:
    note = note.strip()
    name = note[:-1].upper()
    octave = int(note[-1])
    return (octave + 1) * 12 + NOTE_TO_SEMITONE[name]


def midi_to_note_name(midi: int) -> str:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"


def get_all_positions(midi_note: int, tuning: List[int] = STANDARD_TUNING) -> List[FretPosition]:
    positions = []
    for string_idx, open_pitch in enumerate(tuning):
        fret = midi_note - open_pitch
        if 0 <= fret <= 24:
            positions.append(FretPosition(string=len(tuning) - string_idx, fret=fret))
    return positions


def _position_score(pos: FretPosition, anchor: tuple[float, float] | None) -> float:
    if anchor is None:
        return pos.fret
    anchor_string, anchor_fret = anchor
    return abs(pos.fret - anchor_fret) * 2 + abs(pos.string - anchor_string)


def _best_single_position(midi_note: int, tuning: List[int], anchor: tuple[float, float] | None) -> FretPosition | None:
    positions = get_all_positions(midi_note, tuning)
    if not positions:
        return None
    return min(positions, key=lambda position: _position_score(position, anchor))


def _best_chord_positions(notes: List[Note], tuning: List[int], anchor: tuple[float, float] | None) -> list[FretPosition | None]:
    position_options = [get_all_positions(note_to_midi(note.pitch), tuning) for note in notes]
    if any(not options for options in position_options):
        return [_best_single_position(note_to_midi(note.pitch), tuning, anchor) for note in notes]

    best_combo: tuple[FretPosition, ...] | None = None
    best_score = float("inf")
    for combo in product(*position_options):
        strings = [position.string for position in combo]
        if len(set(strings)) != len(strings):
            continue
        frets = [position.fret for position in combo]
        fret_span = max(frets) - min(frets)
        score = (
            sum(_position_score(position, anchor) for position in combo)
            + fret_span * 1.5
            + sum(frets) * 0.05
        )
        if score < best_score:
            best_score = score
            best_combo = combo

    if best_combo is not None:
        return list(best_combo)
    return [_best_single_position(note_to_midi(note.pitch), tuning, anchor) for note in notes]


def _anchor_from_positions(positions: list[FretPosition | None]) -> tuple[float, float] | None:
    assigned = [position for position in positions if position is not None]
    if not assigned:
        return None
    return (
        sum(position.string for position in assigned) / len(assigned),
        sum(position.fret for position in assigned) / len(assigned),
    )


def optimize_positions(notes: List[Note], tuning: List[int] = STANDARD_TUNING) -> List[Note]:
    if not notes:
        return notes

    anchor: tuple[float, float] | None = None
    grouped: dict[float, list[Note]] = {}
    for note in notes:
        grouped.setdefault(round(note.start_beat, 3), []).append(note)

    for start_beat in sorted(grouped):
        group = grouped[start_beat]
        positions = _best_chord_positions(group, tuning, anchor)
        for note, position in zip(group, positions):
            note.position = position
        anchor = _anchor_from_positions(positions) or anchor

    return notes


def get_tuning_midi(tuning: str, is_bass: bool = False) -> List[int]:
    base = BASS_STANDARD_TUNING if is_bass else STANDARD_TUNING
    if tuning == "drop_d":
        return [26, 33, 38, 43] if is_bass else [38, 45, 50, 55, 59, 64]
    if tuning == "half_step_down":
        return [n - 1 for n in base]
    if tuning == "full_step_down":
        return [n - 2 for n in base]
    return base
