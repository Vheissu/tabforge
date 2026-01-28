from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.fretboard import Note, get_tuning_midi, midi_to_note_name, optimize_positions

settings = get_settings()


def audio_to_midi(audio_path: Path, instrument: str) -> list[dict]:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("basic-pitch is not available in this environment") from exc

    model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

    freq_ranges = {
        "bass": (30, 400),
        "guitar": (80, 1200),
        "other": (80, 2000),
    }

    min_freq, max_freq = freq_ranges.get(instrument, (80, 2000))

    _, _, note_events = predict(
        str(audio_path),
        model,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        minimum_note_length=0.05,
        onset_threshold=0.5,
        frame_threshold=0.3,
    )

    normalized = []
    for event in note_events:
        if isinstance(event, dict):
            normalized.append(event)
        else:
            normalized.append({
                "start_time": float(event.start_time),
                "end_time": float(event.end_time),
                "pitch_midi": int(event.pitch_midi),
                "velocity": int(event.velocity),
            })

    return normalized


def refine_with_gemini(audio_path: Path, notes: list[dict], instrument: str, tempo: int) -> dict | None:
    if not settings.gemini_api_key:
        return None

    try:
        import google.generativeai as genai
    except Exception:
        return None

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel("gemini-3-flash")

    audio_file = genai.upload_file(str(audio_path))
    note_summary = json.dumps(notes[:50])

    prompt = (
        f"You are analyzing a {instrument} recording.\n\n"
        f"Initial MIDI transcription detected these notes (first 50):\n{note_summary}\n\n"
        f"Tempo is approximately {tempo} BPM.\n\n"
        "Listen to the audio and provide corrections/refinements:\n\n"
        "1. Identify any missed notes or incorrect pitches\n"
        "2. Detect playing techniques: hammer-ons, pull-offs, slides, bends, palm muting, vibrato\n"
        "3. Suggest optimal fret positions for playability\n"
        "4. Identify any chord voicings\n\n"
        "Respond with JSON only, no markdown."
    )

    response = model.generate_content([prompt, audio_file], generation_config={"temperature": 0.1})
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return None


def transcribe_pitched_instrument(audio_path: Path, instrument: str, tempo: int, tuning: str = "standard") -> dict[str, Any]:
    try:
        note_events = audio_to_midi(audio_path, instrument)
    except RuntimeError as exc:
        return {
            "notes": [],
            "refinement": None,
            "warning": str(exc),
        }

    notes: list[Note] = []
    for event in note_events:
        pitch = midi_to_note_name(int(event["pitch_midi"]))
        start_time = float(event["start_time"])
        end_time = float(event["end_time"])
        duration_seconds = max(0.01, end_time - start_time)
        start_beat = (start_time * tempo) / 60
        duration_beats = (duration_seconds * tempo) / 60

        notes.append(
            Note(
                pitch=pitch,
                start_beat=start_beat,
                duration=duration_beats,
                velocity=int(event.get("velocity", 100)),
            )
        )

    tuning_midi = get_tuning_midi(tuning, is_bass=instrument == "bass")
    notes = optimize_positions(notes, tuning=tuning_midi)

    refined = refine_with_gemini(audio_path, note_events, instrument, tempo)

    return {
        "notes": [
            {
                "pitch": n.pitch,
                "start_beat": n.start_beat,
                "duration": n.duration,
                "string": n.position.string if n.position else None,
                "fret": n.position.fret if n.position else None,
                "technique": n.technique,
                "velocity": n.velocity,
            }
            for n in notes
        ],
        "refinement": refined,
    }
