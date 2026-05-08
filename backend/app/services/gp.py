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
    NoteType,
    BeatStatus,
)

try:
    from guitarpro.models import Tempo
except ImportError:  # pragma: no cover - compatibility with older pyguitarpro
    Tempo = None

from app.services.fretboard import get_tuning_midi

DRUM_NOTE_VALUES = {
    "kick": 36,
    "snare": 38,
    "snare_rim": 37,
    "hihat_closed": 42,
    "hihat_open": 46,
    "hihat_pedal": 44,
    "crash": 49,
    "ride": 51,
    "tom_high": 50,
    "tom_mid": 47,
    "tom_low": 43,
}


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


def _group_notes_by_measure(notes: list[dict]) -> dict[int, dict[int, list[dict]]]:
    grouped: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for note in notes:
        measure_index = int(note["start_beat"] // 4)
        relative_beat = float(note["start_beat"]) - (measure_index * 4)
        slot = max(0, min(15, int(round(relative_beat * 4))))
        grouped[measure_index][slot].append(note)
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


def _count_measures(transcription: dict[str, Any]) -> int:
    max_beat = 0.0
    for instrument in ("guitar", "bass"):
        for note in transcription.get(instrument, {}).get("notes", []):
            max_beat = max(max_beat, float(note.get("start_beat", 0)))
    for hit in transcription.get("drums", []):
        max_beat = max(max_beat, float(hit.get("start_beat", 0)))
    return max(1, int(max_beat // 4) + 1)


def _populate_track(track: Track, notes: list[dict], total_measures: int, is_drum: bool = False) -> None:
    track.measures.clear()
    grouped = _group_notes_by_measure(notes)

    for measure_idx in range(total_measures):
        header = track.song.measureHeaders[measure_idx]
        measure = Measure(track, header)
        voice = measure.voices[0]
        measure_notes = grouped.get(measure_idx, {})

        for slot in range(16):
            beat = Beat(voice)
            beat.duration = Duration(16)
            slot_notes = measure_notes.get(slot, [])
            if not slot_notes:
                beat.status = BeatStatus.rest
                voice.beats.append(beat)
                continue

            beat.status = BeatStatus.normal
            if is_drum:
                notes_to_write = [max(slot_notes, key=lambda n: int(n.get("velocity", 100)))]
            else:
                by_string: dict[int, dict] = {}
                for note_data in slot_notes:
                    string = max(1, min(len(track.strings), int(note_data.get("string") or 1)))
                    if string not in by_string or int(note_data.get("velocity", 100)) > int(by_string[string].get("velocity", 100)):
                        by_string[string] = note_data
                notes_to_write = sorted(
                    by_string.values(),
                    key=lambda n: int(n.get("velocity", 100)),
                    reverse=True,
                )[: len(track.strings)]
            for note_data in notes_to_write:
                note = Note(beat)
                note.type = NoteType.normal
                if is_drum:
                    note.value = DRUM_NOTE_VALUES.get(str(note_data.get("drum")), 42)
                    note.string = 1
                else:
                    note.value = max(0, min(24, int(note_data.get("fret") or 0)))
                    note.string = max(1, min(len(track.strings), int(note_data.get("string") or 1)))
                note.velocity = int(note_data.get("velocity", 100))
                _apply_technique(note, note_data.get("technique"))
                beat.notes.append(note)

            voice.beats.append(beat)

        track.measures.append(measure)


def create_guitar_pro_file(transcription: dict[str, Any], output_path: str) -> None:
    song = Song()
    song.tracks.clear()
    song.title = transcription.get("title", "Unknown")
    song.artist = transcription.get("artist", "Unknown")
    tempo_value = transcription.get("tempo", 120)
    song.tempo = Tempo(tempo_value) if Tempo else tempo_value

    tuning_name = transcription.get("tuning", "standard")
    total_measures = _count_measures(transcription)
    song.measureHeaders.clear()
    for measure_idx in range(total_measures):
        header = MeasureHeader(number=measure_idx + 1)
        header.timeSignature = TimeSignature(4, Duration(4))
        if measure_idx == 0 and Tempo:
            header.tempo = Tempo(tempo_value)
        song.measureHeaders.append(header)

    if "guitar" in transcription:
        guitar_track = Track(song)
        guitar_track.name = "Guitar"
        guitar_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=False))
        _populate_track(guitar_track, transcription["guitar"]["notes"], total_measures)
        song.tracks.append(guitar_track)

    if "bass" in transcription:
        bass_track = Track(song)
        bass_track.name = "Bass"
        bass_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=True))
        _populate_track(bass_track, transcription["bass"]["notes"], total_measures)
        song.tracks.append(bass_track)

    if "drums" in transcription:
        drum_track = Track(song)
        drum_track.name = "Drums"
        drum_track.isPercussionTrack = True
        _populate_track(drum_track, transcription["drums"], total_measures, is_drum=True)
        song.tracks.append(drum_track)

    guitarpro.write(song, output_path)
