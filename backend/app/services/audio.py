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
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    try:
        offset = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        offset = 0.0

    f0 = librosa.yin(y, fmin=55, fmax=110, sr=sr)
    f0 = f0[np.isfinite(f0)]
    low_freq = float(np.median(f0)) if f0.size else None

    tuning = "standard"
    if -1.3 <= offset <= -0.7:
        tuning = "half_step_down"
    elif -2.4 <= offset <= -1.6:
        tuning = "full_step_down"
    else:
        # If the low string centers around D2 but the tuning offset looks standard,
        # assume Drop D.
        if low_freq and low_freq < 78.0:
            tuning = "drop_d"

    return {"tuning": tuning, "offset_semitones": offset, "low_freq": low_freq}
