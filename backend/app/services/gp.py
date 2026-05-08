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
from app.services.constraints import beats_per_measure, parse_time_signature, pickup_time_signature

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

DRUM_STRING_VALUES = {
    "hihat_closed": 1,
    "hihat_open": 1,
    "hihat_pedal": 1,
    "crash": 2,
    "ride": 3,
    "tom_high": 4,
    "tom_mid": 5,
    "snare": 6,
    "snare_rim": 6,
    "tom_low": 6,
    "kick": 7,
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


def _measure_lengths(transcription: dict[str, Any], total_measures: int) -> list[float]:
    regular_beats = beats_per_measure(transcription.get("time_signature", "4/4"))
    pickup_beats = transcription.get("pickup_bar_beats")
    first_beats = float(pickup_beats) if pickup_beats else regular_beats
    return [first_beats if measure_idx == 0 else regular_beats for measure_idx in range(total_measures)]


def _measure_start(lengths: list[float], measure_index: int) -> float:
    return sum(lengths[:measure_index])


def _find_measure_index(start_beat: float, lengths: list[float]) -> int:
    current_start = 0.0
    for measure_index, measure_length in enumerate(lengths):
        if start_beat < current_start + measure_length:
            return measure_index
        current_start += measure_length
    return max(0, len(lengths) - 1)


def _slot_count(measure_length: float) -> int:
    return max(1, int(round(measure_length * 4)))


def _group_notes_by_measure(notes: list[dict], measure_lengths: list[float]) -> dict[int, dict[int, list[dict]]]:
    grouped: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for note in notes:
        start_beat = float(note["start_beat"])
        measure_index = _find_measure_index(start_beat, measure_lengths)
        relative_beat = start_beat - _measure_start(measure_lengths, measure_index)
        slot = max(0, min(_slot_count(measure_lengths[measure_index]) - 1, int(round(relative_beat * 4))))
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


def _create_drum_string_set() -> list[GuitarString]:
    return [GuitarString(i + 1, 35 + i) for i in range(7)]


def _count_measures(transcription: dict[str, Any]) -> int:
    max_beat = 0.0
    for instrument in ("guitar", "bass"):
        for note in transcription.get(instrument, {}).get("notes", []):
            max_beat = max(max_beat, float(note.get("start_beat", 0)) + float(note.get("duration", 0)))
    for hit in transcription.get("drums", []):
        max_beat = max(max_beat, float(hit.get("start_beat", 0)) + float(hit.get("duration", 0)))

    regular_beats = beats_per_measure(transcription.get("time_signature", "4/4"))
    pickup_beats = transcription.get("pickup_bar_beats")
    first_beats = float(pickup_beats) if pickup_beats else regular_beats
    if max_beat <= first_beats:
        return 1
    remaining = max_beat - first_beats
    return 1 + max(1, int((remaining + regular_beats - 0.001) // regular_beats))


def _populate_track(track: Track, notes: list[dict], measure_lengths: list[float], is_drum: bool = False) -> None:
    track.measures.clear()
    grouped = _group_notes_by_measure(notes, measure_lengths)

    for measure_idx, measure_length in enumerate(measure_lengths):
        header = track.song.measureHeaders[measure_idx]
        measure = Measure(track, header)
        voice = measure.voices[0]
        measure_notes = grouped.get(measure_idx, {})

        for slot in range(_slot_count(measure_length)):
            beat = Beat(voice)
            beat.duration = Duration(16)
            slot_notes = measure_notes.get(slot, [])
            if not slot_notes:
                beat.status = BeatStatus.rest
                voice.beats.append(beat)
                continue

            beat.status = BeatStatus.normal
            if is_drum:
                by_string: dict[int, dict] = {}
                for note_data in slot_notes:
                    drum = str(note_data.get("drum") or "")
                    if drum not in DRUM_NOTE_VALUES:
                        continue
                    string = DRUM_STRING_VALUES.get(drum, 1)
                    if string not in by_string or int(note_data.get("velocity", 100)) > int(by_string[string].get("velocity", 100)):
                        by_string[string] = note_data
                notes_to_write = sorted(
                    by_string.values(),
                    key=lambda n: int(n.get("velocity", 100)),
                    reverse=True,
                )
            else:
                by_string: dict[int, dict] = {}
                for note_data in slot_notes:
                    if note_data.get("string") is None or note_data.get("fret") is None:
                        continue
                    string = int(note_data["string"])
                    fret = int(note_data["fret"])
                    if not 1 <= string <= len(track.strings) or not 0 <= fret <= 24:
                        continue
                    if string not in by_string or int(note_data.get("velocity", 100)) > int(by_string[string].get("velocity", 100)):
                        by_string[string] = note_data
                notes_to_write = sorted(
                    by_string.values(),
                    key=lambda n: int(n.get("velocity", 100)),
                    reverse=True,
                )[: len(track.strings)]
            if not notes_to_write:
                beat.status = BeatStatus.rest
                voice.beats.append(beat)
                continue
            for note_data in notes_to_write:
                note = Note(beat)
                note.type = NoteType.normal
                if is_drum:
                    note.value = DRUM_NOTE_VALUES.get(str(note_data.get("drum")), 42)
                    note.string = DRUM_STRING_VALUES.get(str(note_data.get("drum")), 1)
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
    capo_fret = int(transcription.get("capo_fret") or 0)
    time_signature = transcription.get("time_signature", "4/4")
    numerator, denominator = parse_time_signature(time_signature)
    total_measures = _count_measures(transcription)
    measure_lengths = _measure_lengths(transcription, total_measures)
    song.measureHeaders.clear()
    for measure_idx in range(total_measures):
        header = MeasureHeader(number=measure_idx + 1)
        if measure_idx == 0 and transcription.get("pickup_bar_beats"):
            pickup_numerator, pickup_denominator = pickup_time_signature(float(transcription["pickup_bar_beats"]))
            header.timeSignature = TimeSignature(pickup_numerator, Duration(pickup_denominator))
        else:
            header.timeSignature = TimeSignature(numerator, Duration(denominator))
        if measure_idx == 0 and Tempo:
            header.tempo = Tempo(tempo_value)
        song.measureHeaders.append(header)

    if "guitar" in transcription:
        guitar_track = Track(song)
        guitar_track.name = "Guitar"
        guitar_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=False))
        guitar_track.offset = capo_fret
        _populate_track(guitar_track, transcription["guitar"]["notes"], measure_lengths)
        song.tracks.append(guitar_track)

    if "bass" in transcription:
        bass_track = Track(song)
        bass_track.name = "Bass"
        bass_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=True))
        _populate_track(bass_track, transcription["bass"]["notes"], measure_lengths)
        song.tracks.append(bass_track)

    if "drums" in transcription:
        drum_track = Track(song)
        drum_track.name = "Drums"
        drum_track.isPercussionTrack = True
        drum_track.strings = _create_drum_string_set()
        _populate_track(drum_track, transcription["drums"], measure_lengths, is_drum=True)
        song.tracks.append(drum_track)

    guitarpro.write(song, output_path)
