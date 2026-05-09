from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from app.services.fretboard import get_tuning_midi


@dataclass(frozen=True)
class TabEvent:
    pitch_midi: int
    start_beat: float
    duration: float
    velocity: int


@dataclass(frozen=True)
class ProcessingProfile:
    min_velocity: int
    min_duration_beats: float
    merge_gap_beats: float
    grid_beats: float
    max_notes_per_slot: int
    monophonic: bool = False


PROFILES = {
    "guitar": ProcessingProfile(
        min_velocity=24,
        min_duration_beats=0.25,
        merge_gap_beats=0.125,
        grid_beats=0.25,
        max_notes_per_slot=6,
    ),
    "bass": ProcessingProfile(
        min_velocity=28,
        min_duration_beats=0.25,
        merge_gap_beats=0.125,
        grid_beats=0.25,
        max_notes_per_slot=1,
        monophonic=True,
    ),
}


def playable_midi_range(instrument: str, tuning: str, capo_fret: int = 0) -> tuple[int, int]:
    tuning_midi = get_tuning_midi(tuning, is_bass=instrument == "bass")
    if instrument == "guitar" and capo_fret > 0:
        tuning_midi = [pitch + capo_fret for pitch in tuning_midi]
    return min(tuning_midi), max(tuning_midi) + 24


def _quantize(value: float, grid: float) -> float:
    if grid <= 0:
        return value
    return round(value / grid) * grid


def _normalise_event(event: dict, tempo: int, profile: ProcessingProfile) -> TabEvent | None:
    pitch_midi = int(event["pitch_midi"])
    velocity = int(max(1, min(127, event.get("velocity", 100))))
    if velocity < profile.min_velocity:
        return None

    start_time = float(event["start_time"])
    end_time = float(event["end_time"])
    if end_time <= start_time:
        return None

    start_beat = max(0.0, _quantize((start_time * tempo) / 60, profile.grid_beats))
    end_beat = max(
        start_beat + profile.min_duration_beats,
        _quantize((end_time * tempo) / 60, profile.grid_beats),
    )
    duration = max(profile.min_duration_beats, end_beat - start_beat)

    return TabEvent(
        pitch_midi=pitch_midi,
        start_beat=start_beat,
        duration=duration,
        velocity=velocity,
    )


def _merge_repeated_notes(events: Iterable[TabEvent], profile: ProcessingProfile) -> list[TabEvent]:
    merged: list[TabEvent] = []
    for event in sorted(events, key=lambda e: (e.pitch_midi, e.start_beat, -e.velocity)):
        if not merged:
            merged.append(event)
            continue

        previous = merged[-1]
        previous_end = previous.start_beat + previous.duration
        if event.pitch_midi == previous.pitch_midi and event.start_beat <= previous_end + profile.merge_gap_beats:
            merged[-1] = TabEvent(
                pitch_midi=previous.pitch_midi,
                start_beat=previous.start_beat,
                duration=max(previous_end, event.start_beat + event.duration) - previous.start_beat,
                velocity=max(previous.velocity, event.velocity),
            )
        else:
            merged.append(event)

    return sorted(merged, key=lambda e: (e.start_beat, e.pitch_midi))


def _slot_key(event: TabEvent, profile: ProcessingProfile) -> int:
    return int(round(event.start_beat / profile.grid_beats))


def _select_bass_note(slot_events: list[TabEvent]) -> list[TabEvent]:
    max_velocity = max(event.velocity for event in slot_events)
    plausible = [event for event in slot_events if event.velocity >= max_velocity * 0.55]
    lowest = min(plausible, key=lambda event: (event.pitch_midi, -event.velocity))
    if lowest.velocity >= max_velocity * 0.75:
        return [lowest]
    chosen = max(plausible, key=lambda event: (event.velocity, event.duration, -event.pitch_midi))
    return [chosen]


def _limit_slot_polyphony(events: Iterable[TabEvent], profile: ProcessingProfile) -> list[TabEvent]:
    slots: dict[int, dict[int, TabEvent]] = {}
    for event in events:
        slot = _slot_key(event, profile)
        slot_events = slots.setdefault(slot, {})
        previous = slot_events.get(event.pitch_midi)
        if previous is None or event.velocity > previous.velocity:
            slot_events[event.pitch_midi] = event

    limited: list[TabEvent] = []
    for slot_events in slots.values():
        values = list(slot_events.values())
        if profile.monophonic and values:
            limited.extend(_select_bass_note(values))
        else:
            limited.extend(sorted(values, key=lambda event: event.velocity, reverse=True)[: profile.max_notes_per_slot])

    return sorted(limited, key=lambda e: (e.start_beat, e.pitch_midi))


def prepare_note_events_for_tab(
    note_events: Iterable[dict],
    instrument: str,
    tempo: int,
    tuning: str,
    *,
    triplet_feel: bool = False,
    capo_fret: int = 0,
) -> list[TabEvent]:
    profile = PROFILES.get(instrument, PROFILES["guitar"])
    if triplet_feel:
        profile = replace(profile, grid_beats=1 / 3, merge_gap_beats=1 / 6)
    min_midi, max_midi = playable_midi_range(instrument, tuning, capo_fret)

    candidates: list[TabEvent] = []
    for event in note_events:
        normalised = _normalise_event(event, tempo, profile)
        if normalised is None:
            continue
        if not min_midi <= normalised.pitch_midi <= max_midi:
            continue
        candidates.append(normalised)

    merged = _merge_repeated_notes(candidates, profile)
    return _limit_slot_polyphony(merged, profile)
