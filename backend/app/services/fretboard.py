from __future__ import annotations

from dataclasses import dataclass
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
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def note_to_midi(note: str) -> int:
    name = note[:-1]
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
            positions.append(FretPosition(string=6 - string_idx, fret=fret))
    return positions


def optimize_positions(notes: List[Note], tuning: List[int] = STANDARD_TUNING) -> List[Note]:
    if not notes:
        return notes

    first_positions = get_all_positions(note_to_midi(notes[0].pitch), tuning)
    if first_positions:
        notes[0].position = min(first_positions, key=lambda p: p.fret)

    for i in range(1, len(notes)):
        prev_pos = notes[i - 1].position
        curr_positions = get_all_positions(note_to_midi(notes[i].pitch), tuning)

        if not curr_positions:
            continue

        if prev_pos is None:
            notes[i].position = min(curr_positions, key=lambda p: p.fret)
        else:
            def position_score(pos: FretPosition) -> float:
                fret_dist = abs(pos.fret - prev_pos.fret)
                string_dist = abs(pos.string - prev_pos.string)
                return fret_dist * 2 + string_dist

            notes[i].position = min(curr_positions, key=position_score)

    return notes


def get_tuning_midi(tuning: str, is_bass: bool = False) -> List[int]:
    base = BASS_STANDARD_TUNING if is_bass else STANDARD_TUNING
    if tuning == "drop_d" and not is_bass:
        return [38, 45, 50, 55, 59, 64]
    if tuning == "half_step_down":
        return [n - 1 for n in base]
    if tuning == "full_step_down":
        return [n - 2 for n in base]
    return base
