from __future__ import annotations

from collections import defaultdict
from typing import Any

import guitarpro
from guitarpro.models import (
    Song,
    Track,
    Measure,
    Beat,
    Note,
    Duration,
    MeasureHeader,
    TimeSignature,
    GuitarString,
)

try:
    from guitarpro.models import Tempo
except ImportError:  # pragma: no cover - compatibility with older pyguitarpro
    Tempo = None

from app.services.fretboard import get_tuning_midi

def _duration_from_beats(beats: float) -> Duration:
    if beats >= 4:
        return Duration(1)
    if beats >= 2:
        return Duration(2)
    if beats >= 1:
        return Duration(4)
    if beats >= 0.5:
        return Duration(8)
    return Duration(16)


def _group_notes_by_measure(notes: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for note in notes:
        measure_index = int(note["start_beat"] // 4)
        grouped[measure_index].append(note)
    return grouped


def _apply_technique(note: Note, technique: str | None) -> None:
    if not technique:
        return
    effect = note.effect
    if technique in {"hammer_on", "pull_off"}:
        effect.hammer = True
    elif technique in {"slide_up", "slide_down"}:
        effect.slides = [guitarpro.SlideType.shiftSlide]
    elif technique == "bend":
        bend = guitarpro.BendEffect()
        bend.type = guitarpro.BendType.bend
        bend.value = 100
        effect.bend = bend
    elif technique == "vibrato":
        effect.vibrato = True
    elif technique == "palm_mute":
        effect.palmMute = True


def _create_string_set(tuning: list[int]) -> list[GuitarString]:
    # Guitar Pro expects strings ordered from high to low.
    return [GuitarString(i + 1, pitch) for i, pitch in enumerate(reversed(tuning))]


def _populate_track(track: Track, notes: list[dict], tempo: int) -> None:
    grouped = _group_notes_by_measure(notes)
    total_measures = max(grouped.keys(), default=0) + 1

    for measure_idx in range(total_measures):
        header = MeasureHeader(number=measure_idx + 1)
        header.timeSignature = TimeSignature(4, Duration(4))
        if measure_idx == 0 and Tempo:
            header.tempo = Tempo(tempo)
        measure = Measure(track, header)
        voice = measure.voices[0]

        for note_data in grouped.get(measure_idx, []):
            beat = Beat(voice)
            beat.duration = _duration_from_beats(note_data.get("duration", 1))

            note = Note(beat)
            note.value = int(note_data.get("fret") or 0)
            note.string = int(note_data.get("string") or 1)
            note.velocity = int(note_data.get("velocity", 100))
            _apply_technique(note, note_data.get("technique"))

            beat.notes.append(note)
            voice.beats.append(beat)

        track.measures.append(measure)


def create_guitar_pro_file(transcription: dict[str, Any], output_path: str) -> None:
    song = Song()
    song.title = transcription.get("title", "Unknown")
    song.artist = transcription.get("artist", "Unknown")
    tempo_value = transcription.get("tempo", 120)
    song.tempo = Tempo(tempo_value) if Tempo else tempo_value

    tuning_name = transcription.get("tuning", "standard")

    if "guitar" in transcription:
        guitar_track = Track(song)
        guitar_track.name = "Guitar"
        guitar_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=False))
        _populate_track(guitar_track, transcription["guitar"]["notes"], tempo_value)
        song.tracks.append(guitar_track)

    if "bass" in transcription:
        bass_track = Track(song)
        bass_track.name = "Bass"
        bass_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=True))
        _populate_track(bass_track, transcription["bass"]["notes"], tempo_value)
        song.tracks.append(bass_track)

    if "drums" in transcription:
        drum_track = Track(song)
        drum_track.name = "Drums"
        drum_track.isPercussionTrack = True
        _populate_track(drum_track, transcription["drums"], tempo_value)
        song.tracks.append(drum_track)

    guitarpro.write(song, output_path)
