from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa


def analyze_audio(audio_path: Path) -> tuple[int, str]:
    """Estimate tempo (BPM) and key using librosa."""
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Key estimation via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = chroma.sum(axis=1)
    key_index = int(np.argmax(chroma_sum))
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = keys[key_index]

    return int(round(float(tempo))), key


def detect_tuning(audio_path: Path) -> dict:
    """
    Heuristic tuning detection.
    Returns a tuning label and supporting telemetry (offset + low pitch).
    """
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=45)
    if y.size == 0 or float(np.max(np.abs(y))) <= 1e-6:
        return {
            "tuning": "standard",
            "offset_semitones": 0.0,
            "low_freq": None,
            "confidence": 0.0,
            "candidate_count": 0,
        }

    try:
        offset = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        offset = 0.0

    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=55, fmax=120)
        threshold = float(np.max(magnitudes)) * 0.25 if magnitudes.size else 0.0
        candidates = pitches[(magnitudes >= threshold) & np.isfinite(pitches) & (pitches > 0)]
        low_freq = float(np.percentile(candidates, 20)) if candidates.size else None
    except Exception:
        candidates = np.array([])
        low_freq = None

    tuning = "standard"
    confidence = 0.45 if candidates.size else 0.25
    if -1.3 <= offset <= -0.7:
        tuning = "half_step_down"
        confidence = 0.75 + max(0.0, 0.2 - abs(offset + 1.0)) * 0.75
    elif -2.4 <= offset <= -1.6:
        tuning = "full_step_down"
        confidence = 0.75 + max(0.0, 0.2 - abs(offset + 2.0)) * 0.75
    else:
        # If the low string centers around D2 but the tuning offset looks standard,
        # assume Drop D.
        if low_freq and low_freq < 78.0:
            tuning = "drop_d"
            confidence = 0.65 if abs(offset) <= 0.35 else 0.5
        elif abs(offset) <= 0.35 and candidates.size:
            confidence = 0.65

    confidence = max(0.0, min(0.95, confidence))
    return {
        "tuning": tuning,
        "offset_semitones": offset,
        "low_freq": low_freq,
        "confidence": round(confidence, 3),
        "candidate_count": int(candidates.size),
    }
