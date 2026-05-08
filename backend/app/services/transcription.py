from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.fretboard import Note, get_tuning_midi, midi_to_note_name, optimize_positions
from app.services.note_processing import prepare_note_events_for_tab

settings = get_settings()


class GeminiRefinementTimeout(TimeoutError):
    pass


def _handle_refinement_timeout(_signum, _frame) -> None:
    raise GeminiRefinementTimeout("Gemini refinement timed out")


@contextmanager
def _refinement_deadline(timeout_seconds: int):
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handle_refinement_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _parse_refinement_response(text: str | None) -> dict | None:
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        stripped = stripped[start : end + 1]

    parsed = json.loads(stripped)
    return parsed if isinstance(parsed, dict) else None


def _frequency_range(instrument: str) -> tuple[float, float]:
    freq_ranges = {
        "bass": (30.0, 400.0),
        "guitar": (80.0, 1200.0),
        "other": (80.0, 2000.0),
    }
    return freq_ranges.get(instrument, (80.0, 2000.0))


def _audio_to_midi_with_librosa(audio_path: Path, instrument: str) -> list[dict]:
    return _audio_to_midi_with_spectral_fallback(audio_path, instrument)


def _audio_to_midi_with_spectral_fallback(audio_path: Path, instrument: str) -> list[dict]:
    try:
        import librosa
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("librosa spectral fallback is not available in this environment") from exc

    min_freq, max_freq = _frequency_range(instrument)
    hop_length = 512
    y, sample_rate = librosa.load(str(audio_path), sr=22050, mono=True)
    if y.size == 0:
        return []

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    max_rms = float(np.max(rms)) if rms.size else 0.0
    duration = float(librosa.get_duration(y=y, sr=sample_rate))
    if duration <= 0 or max_rms <= 0:
        return []

    envelope = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)
    onset_frames = []
    if envelope.size:
        threshold = float(np.max(envelope)) * 0.25
        wait = 4
        last_peak = -wait
        for index in range(1, len(envelope) - 1):
            if index - last_peak < wait:
                continue
            if envelope[index] >= threshold and envelope[index] >= envelope[index - 1] and envelope[index] >= envelope[index + 1]:
                onset_frames.append(index)
                last_peak = index
    onset_times = [float(t) for t in librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)]
    onset_times = [t for t in onset_times if 0 <= t < duration]

    if len(onset_times) < 4:
        step = 0.5 if instrument == "bass" else 0.25
        onset_times = [float(t) for t in np.arange(0, duration, step)]

    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=min_freq,
        fmax=max_freq,
    )

    notes: list[dict] = []
    boundaries = sorted(set([0.0, *onset_times, duration]))
    for start_time, end_time in zip(boundaries, boundaries[1:]):
        if end_time - start_time < 0.05:
            continue

        start_frame = int(librosa.time_to_frames(start_time, sr=sample_rate, hop_length=hop_length))
        end_frame = max(start_frame + 1, int(librosa.time_to_frames(end_time, sr=sample_rate, hop_length=hop_length)))
        frame_rms = rms[start_frame:min(end_frame, len(rms))]
        if frame_rms.size == 0:
            continue

        mean_rms = float(np.mean(frame_rms))
        if mean_rms < max_rms * 0.08:
            continue

        pitch_slice = pitches[:, start_frame:end_frame]
        mag_slice = magnitudes[:, start_frame:end_frame]
        if mag_slice.size == 0 or float(np.max(mag_slice)) <= 0:
            continue

        row, column = np.unravel_index(int(np.argmax(mag_slice)), mag_slice.shape)
        frequency = float(pitch_slice[row, column])
        if not np.isfinite(frequency) or frequency <= 0:
            continue

        midi = int(round(float(librosa.hz_to_midi(frequency))))
        velocity = int(max(1, min(127, 35 + (mean_rms / max_rms) * 92)))
        notes.append(
            {
                "start_time": float(start_time),
                "end_time": float(min(end_time, start_time + 2.0)),
                "pitch_midi": midi,
                "velocity": velocity,
            }
        )

    return notes[:512]


@lru_cache(maxsize=1)
def _get_basic_pitch_model():
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    return Model(ICASSP_2022_MODEL_PATH)


def _normalize_basic_pitch_event(event) -> dict:
    if isinstance(event, dict):
        velocity = event.get("velocity", event.get("amplitude", 0.8))
        if isinstance(velocity, float) and velocity <= 1.0:
            velocity = int(max(1, min(127, round(velocity * 127))))
        return {
            "start_time": float(event["start_time"]),
            "end_time": float(event["end_time"]),
            "pitch_midi": int(event["pitch_midi"]),
            "velocity": int(max(1, min(127, velocity))),
        }

    start_time, end_time, pitch_midi, amplitude, *_ = event
    return {
        "start_time": float(start_time),
        "end_time": float(end_time),
        "pitch_midi": int(pitch_midi),
        "velocity": int(max(1, min(127, round(float(amplitude) * 127)))),
    }


def audio_to_midi(audio_path: Path, instrument: str) -> list[dict]:
    try:
        from basic_pitch.inference import predict
    except Exception:
        return _audio_to_midi_with_librosa(audio_path, instrument)

    min_freq, max_freq = _frequency_range(instrument)

    _, _, note_events = predict(
        str(audio_path),
        _get_basic_pitch_model(),
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        minimum_note_length=50.0,
        onset_threshold=0.5,
        frame_threshold=0.3,
    )

    normalized = sorted((_normalize_basic_pitch_event(event) for event in note_events), key=lambda n: n["start_time"])

    if normalized:
        return normalized

    return _audio_to_midi_with_librosa(audio_path, instrument)


def refine_with_gemini(audio_path: Path, notes: list[dict], instrument: str, tempo: int) -> dict | None:
    if not settings.gemini_api_key:
        return None

    try:
        from google import genai
    except Exception:
        return None

    client = None
    try:
        timeout_seconds = settings.gemini_refinement_timeout_seconds
        with _refinement_deadline(timeout_seconds):
            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options={"timeout": timeout_seconds * 1000},
            )

            audio_file = client.files.upload(file=str(audio_path))
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

            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[prompt, audio_file],
                config={"temperature": 0.1, "response_mime_type": "application/json"},
            )
            return _parse_refinement_response(response.text)
    except Exception:
        return None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _positioning_tuning(tuning: str, instrument: str, capo_fret: int) -> list[int]:
    tuning_midi = get_tuning_midi(tuning, is_bass=instrument == "bass")
    if instrument == "guitar" and capo_fret > 0:
        return [pitch + capo_fret for pitch in tuning_midi]
    return tuning_midi


def transcribe_pitched_instrument(
    audio_path: Path,
    instrument: str,
    tempo: int,
    tuning: str = "standard",
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        note_events = audio_to_midi(audio_path, instrument)
    except RuntimeError as exc:
        return {
            "notes": [],
            "refinement": None,
            "warning": str(exc),
        }

    constraints = constraints or {}
    capo_fret = int(constraints.get("capo_fret") or 0)
    processed_events = prepare_note_events_for_tab(
        note_events,
        instrument,
        tempo,
        tuning,
        triplet_feel=bool(constraints.get("triplet_feel")),
        capo_fret=capo_fret,
    )
    notes: list[Note] = []
    for event in processed_events:
        notes.append(
            Note(
                pitch=midi_to_note_name(event.pitch_midi),
                start_beat=event.start_beat,
                duration=event.duration,
                velocity=event.velocity,
            )
        )

    tuning_midi = _positioning_tuning(tuning, instrument, capo_fret)
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
            if n.position is not None
        ],
        "refinement": refined,
    }
