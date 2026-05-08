from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.fretboard import Note, get_tuning_midi, midi_to_note_name, optimize_positions

settings = get_settings()


def _frequency_range(instrument: str) -> tuple[float, float]:
    freq_ranges = {
        "bass": (30.0, 400.0),
        "guitar": (80.0, 1200.0),
        "other": (80.0, 2000.0),
    }
    return freq_ranges.get(instrument, (80.0, 2000.0))


def _audio_to_midi_with_librosa(audio_path: Path, instrument: str) -> list[dict]:
    try:
        import librosa
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("basic-pitch and librosa fallback are not available in this environment") from exc

    min_freq, max_freq = _frequency_range(instrument)
    hop_length = 512
    frame_length = 2048
    y, sample_rate = librosa.load(str(audio_path), sr=22050, mono=True)
    if y.size == 0:
        return []

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=min_freq,
        fmax=max_freq,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.frames_to_time(range(len(f0)), sr=sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = float(np.max(rms)) if rms.size else 0.0

    notes: list[dict] = []
    current_start: float | None = None
    current_midis: list[int] = []
    current_velocities: list[int] = []

    def flush(end_time: float) -> None:
        nonlocal current_start, current_midis, current_velocities
        if current_start is not None and current_midis:
            duration = end_time - current_start
            if duration >= 0.05:
                notes.append(
                    {
                        "start_time": float(current_start),
                        "end_time": float(end_time),
                        "pitch_midi": int(round(float(np.median(current_midis)))),
                        "velocity": int(round(float(np.median(current_velocities or [80])))),
                    }
                )
        current_start = None
        current_midis = []
        current_velocities = []

    for index, frequency in enumerate(f0):
        time = float(times[index])
        if not voiced_flag[index] or not np.isfinite(frequency):
            flush(time)
            continue

        midi = int(round(float(librosa.hz_to_midi(frequency))))
        frame_rms = float(rms[min(index, len(rms) - 1)]) if rms.size else 0.0
        velocity = 70 if max_rms <= 0 else int(max(1, min(127, 35 + (frame_rms / max_rms) * 92)))

        if current_start is None:
            current_start = time
            current_midis = [midi]
            current_velocities = [velocity]
            continue

        current_pitch = int(round(float(np.median(current_midis))))
        if abs(midi - current_pitch) > 1:
            flush(time)
            current_start = time
            current_midis = [midi]
            current_velocities = [velocity]
            continue

        current_midis.append(midi)
        current_velocities.append(velocity)

    flush(float(librosa.get_duration(y=y, sr=sample_rate)))
    return notes


def audio_to_midi(audio_path: Path, instrument: str) -> list[dict]:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import tensorflow as tf
    except Exception:
        return _audio_to_midi_with_librosa(audio_path, instrument)

    model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

    min_freq, max_freq = _frequency_range(instrument)

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
