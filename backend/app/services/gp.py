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
    MidiChannel,
    Tuplet,
    NoteType,
    BeatStatus,
    BendPoint,
)

try:
    from guitarpro.models import Tempo
except ImportError:  # pragma: no cover - compatibility with older pyguitarpro
    Tempo = None

from app.services.fretboard import get_tuning_midi, note_to_midi as note_name_to_midi
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

STANDARD_DURATION_BEATS = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
TRIPLET_DURATION_BEATS = [4.0, 3.0, 2.0, 1.5, 1.0, 2 / 3, 0.5, 1 / 3]


def _duration_from_beats(beats: float) -> Duration:
    if abs(beats - (1 / 3)) < 0.02:
        return Duration(8, tuplet=Tuplet(3, 2))
    if abs(beats - (2 / 3)) < 0.02:
        return Duration(4, tuplet=Tuplet(3, 2))
    if beats >= 4:
        return Duration(1)
    if beats >= 3:
        return Duration(2, isDotted=True)
    if beats >= 2:
        return Duration(2)
    if beats >= 1.5:
        return Duration(4, isDotted=True)
    if beats >= 1:
        return Duration(4)
    if beats >= 0.75:
        return Duration(8, isDotted=True)
    if beats >= 0.5:
        return Duration(8)
    return Duration(16)


def _duration_beats(duration: Duration) -> float:
    beats = 4 / int(duration.value)
    if getattr(duration, "isDotted", False):
        beats *= 1.5
    tuplet = getattr(duration, "tuplet", None)
    if tuplet and getattr(tuplet, "enters", 1) and getattr(tuplet, "times", 1):
        beats *= float(tuplet.times) / float(tuplet.enters)
    return beats


def _duration_slot_span(duration: Duration, grid_beats: float) -> int:
    return max(1, int(round(_duration_beats(duration) / grid_beats)))


def _duration_segments(beats: float, grid_beats: float) -> list[float]:
    durations = TRIPLET_DURATION_BEATS if abs(grid_beats - (1 / 3)) < 0.02 else STANDARD_DURATION_BEATS
    remaining = max(0.0, beats)
    segments: list[float] = []
    while remaining >= grid_beats * 0.5:
        selected = next((duration for duration in durations if duration <= remaining + 0.01), grid_beats)
        segments.append(selected)
        remaining -= selected
    return segments


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


def _slot_count(measure_length: float, grid_beats: float = 0.25) -> int:
    return max(1, int(round(measure_length / grid_beats)))


def _group_notes_by_measure(
    notes: list[dict],
    measure_lengths: list[float],
    grid_beats: float = 0.25,
) -> dict[int, dict[int, list[dict]]]:
    grouped: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for note in notes:
        start_beat = float(note["start_beat"])
        measure_index = _find_measure_index(start_beat, measure_lengths)
        relative_beat = start_beat - _measure_start(measure_lengths, measure_index)
        slot = max(
            0,
            min(
                _slot_count(measure_lengths[measure_index], grid_beats) - 1,
                int(round(relative_beat / grid_beats)),
            ),
        )
        grouped[measure_index][slot].append(note)
    return grouped


def _apply_technique(note: Note, technique: str | None) -> None:
    if not technique:
        return
    effect = note.effect
    if technique in {"hammer_on", "pull_off"}:
        effect.hammer = True
    elif technique in {"slide_up", "slide_down"}:
        effect.slides = [guitarpro.SlideType.shiftSlideTo]
    elif technique == "bend":
        bend = guitarpro.BendEffect()
        bend.type = guitarpro.BendType.bend
        bend.value = 100
        bend.points = [
            BendPoint(position=0, value=0),
            BendPoint(position=guitarpro.BendEffect.maxPosition, value=100),
        ]
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


def _configure_midi(track: Track, channel: int, effect_channel: int, instrument: int) -> None:
    track.channel = MidiChannel(channel=channel, effectChannel=effect_channel, instrument=instrument)


def _string_pitch(track: Track, string_number: int) -> int | None:
    for string in track.strings:
        if string.number == string_number:
            return int(string.value)
    return None


def _note_target_midi(note_data: dict) -> int | None:
    if note_data.get("pitch_midi") is not None:
        return int(note_data["pitch_midi"])
    if note_data.get("pitch"):
        return note_name_to_midi(str(note_data["pitch"]))
    return None


def _normalise_pitched_position(track: Track, note_data: dict) -> dict:
    target_midi = _note_target_midi(note_data)
    if target_midi is None:
        return note_data

    updated = dict(note_data)
    offset = int(track.offset or 0)
    current_string = updated.get("string")
    if current_string is not None:
        open_pitch = _string_pitch(track, int(current_string))
        if open_pitch is not None:
            fret = target_midi - open_pitch - offset
            if 0 <= fret <= 24:
                updated["fret"] = fret
                return updated

    candidates = []
    for string in track.strings:
        fret = target_midi - int(string.value) - offset
        if 0 <= fret <= 24:
            candidates.append((int(string.number), fret))
    if candidates:
        string_number, fret = min(candidates, key=lambda candidate: (candidate[1], candidate[0]))
        updated["string"] = string_number
        updated["fret"] = fret
    return updated


def _with_unique_drum_strings(notes: list[dict], string_count: int) -> list[dict]:
    used_strings: set[int] = set()
    assigned: list[dict] = []
    for note_data in notes:
        drum = str(note_data.get("drum") or "")
        preferred = DRUM_STRING_VALUES.get(drum, 1)
        string = preferred
        if string in used_strings:
            string = next((candidate for candidate in range(1, string_count + 1) if candidate not in used_strings), preferred)
        used_strings.add(string)
        updated = dict(note_data)
        updated["_gp_string"] = string
        assigned.append(updated)
    return assigned


def _split_pitched_sustains(
    notes: list[dict],
    measure_lengths: list[float],
    grid_beats: float,
    split_points: list[float],
) -> list[dict]:
    segmented: list[dict] = []
    for note in notes:
        start_beat = float(note.get("start_beat", 0))
        remaining = max(grid_beats, float(note.get("duration", grid_beats)))
        current_start = start_beat
        is_tie = False

        while remaining >= grid_beats * 0.5:
            measure_index = _find_measure_index(current_start, measure_lengths)
            measure_end = _measure_start(measure_lengths, measure_index) + measure_lengths[measure_index]
            next_split = next((point for point in split_points if point > current_start + 0.001), None)
            next_boundary = min(measure_end, next_split) if next_split is not None else measure_end
            available = max(grid_beats, next_boundary - current_start)
            raw_segment = min(remaining, available)

            for duration in _duration_segments(raw_segment, grid_beats):
                segment = dict(note)
                segment["start_beat"] = round(current_start, 6)
                segment["duration"] = duration
                if is_tie:
                    segment["_tie"] = True
                segmented.append(segment)
                current_start += duration
                remaining -= duration
                is_tie = True

            if raw_segment <= 0:
                break

    return sorted(segmented, key=lambda item: (float(item.get("start_beat", 0)), str(item.get("pitch", ""))))


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


def _populate_track(
    track: Track,
    notes: list[dict],
    measure_lengths: list[float],
    is_drum: bool = False,
    grid_beats: float = 0.25,
) -> None:
    track.measures.clear()
    if not is_drum:
        split_points = sorted({round(float(note.get("start_beat", 0)), 6) for note in notes})
        notes = _split_pitched_sustains(notes, measure_lengths, grid_beats, split_points)
    grouped = _group_notes_by_measure(notes, measure_lengths, grid_beats)

    for measure_idx, measure_length in enumerate(measure_lengths):
        header = track.song.measureHeaders[measure_idx]
        measure = Measure(track, header)
        voice = measure.voices[0]
        measure_notes = grouped.get(measure_idx, {})
        occupied_slots = sorted(measure_notes)
        total_slots = _slot_count(measure_length, grid_beats)
        current_slot = 0

        while current_slot < total_slots:
            beat = Beat(voice)
            slot_notes = measure_notes.get(current_slot, [])
            next_slot = next((slot for slot in occupied_slots if slot > current_slot), None)
            remaining_slots = total_slots - current_slot

            if slot_notes:
                requested_slots = max(
                    1,
                    int(round(max(float(note.get("duration", grid_beats)) for note in slot_notes) / grid_beats)),
                )
                available_slots = remaining_slots if next_slot is None else max(1, next_slot - current_slot)
                span_slots = min(available_slots, requested_slots)
            else:
                span_slots = remaining_slots if next_slot is None else max(1, next_slot - current_slot)

            beat.duration = _duration_from_beats(span_slots * grid_beats)
            span_slots = min(remaining_slots, _duration_slot_span(beat.duration, grid_beats))

            if not slot_notes:
                beat.status = BeatStatus.rest
                voice.beats.append(beat)
                current_slot += span_slots
                continue

            beat.status = BeatStatus.normal
            if is_drum:
                by_drum: dict[int, dict] = {}
                for note_data in slot_notes:
                    drum = str(note_data.get("drum") or "")
                    if drum not in DRUM_NOTE_VALUES:
                        continue
                    drum_value = DRUM_NOTE_VALUES[drum]
                    if drum_value not in by_drum or int(note_data.get("velocity", 100)) > int(by_drum[drum_value].get("velocity", 100)):
                        by_drum[drum_value] = note_data
                notes_to_write = sorted(
                    by_drum.values(),
                    key=lambda n: int(n.get("velocity", 100)),
                    reverse=True,
                )
                notes_to_write = _with_unique_drum_strings(notes_to_write, len(track.strings))
            else:
                by_string: dict[int, dict] = {}
                for note_data in slot_notes:
                    note_data = _normalise_pitched_position(track, note_data)
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
                current_slot += span_slots
                continue
            for note_data in notes_to_write:
                note = Note(beat)
                note.type = NoteType.tie if note_data.get("_tie") else NoteType.normal
                if is_drum:
                    note.value = DRUM_NOTE_VALUES.get(str(note_data.get("drum")), 42)
                    note.string = int(note_data.get("_gp_string") or DRUM_STRING_VALUES.get(str(note_data.get("drum")), 1))
                else:
                    note.value = max(0, min(24, int(note_data.get("fret") or 0)))
                    note.string = max(1, min(len(track.strings), int(note_data.get("string") or 1)))
                note.velocity = int(note_data.get("velocity", 100))
                if note.type != NoteType.tie:
                    _apply_technique(note, note_data.get("technique"))
                beat.notes.append(note)

            voice.beats.append(beat)
            current_slot += span_slots

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
    grid_beats = 1 / 3 if transcription.get("triplet_feel") else 0.25
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
        _configure_midi(guitar_track, channel=0, effect_channel=1, instrument=27)
        guitar_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=False))
        guitar_track.offset = capo_fret
        _populate_track(guitar_track, transcription["guitar"]["notes"], measure_lengths, grid_beats=grid_beats)
        song.tracks.append(guitar_track)

    if "bass" in transcription:
        bass_track = Track(song)
        bass_track.name = "Bass"
        _configure_midi(bass_track, channel=2, effect_channel=3, instrument=33)
        bass_track.strings = _create_string_set(get_tuning_midi(tuning_name, is_bass=True))
        _populate_track(bass_track, transcription["bass"]["notes"], measure_lengths, grid_beats=grid_beats)
        song.tracks.append(bass_track)

    if "drums" in transcription:
        drum_track = Track(song)
        drum_track.name = "Drums"
        drum_track.isPercussionTrack = True
        _configure_midi(drum_track, channel=9, effect_channel=9, instrument=0)
        drum_track.strings = _create_drum_string_set()
        _populate_track(drum_track, transcription["drums"], measure_lengths, is_drum=True, grid_beats=grid_beats)
        song.tracks.append(drum_track)

    guitarpro.write(song, output_path)
