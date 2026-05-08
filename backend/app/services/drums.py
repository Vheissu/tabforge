from __future__ import annotations

import collections
import collections.abc

# Madmom expects MutableSequence in collections (removed in Python 3.12).
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence

# NumPy 2 removed deprecated aliases used by madmom.
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

if np is not None and not hasattr(np, "float"):
    np.float = float

GP_DRUM_MAP = {
    "kick": {"line": 65, "note_head": "normal"},
    "snare": {"line": 69, "note_head": "normal"},
    "snare_rim": {"line": 69, "note_head": "x"},
    "hihat_closed": {"line": 71, "note_head": "x"},
    "hihat_open": {"line": 71, "note_head": "circle"},
    "hihat_pedal": {"line": 64, "note_head": "x"},
    "crash": {"line": 77, "note_head": "x"},
    "ride": {"line": 75, "note_head": "x"},
    "tom_high": {"line": 72, "note_head": "normal"},
    "tom_mid": {"line": 71, "note_head": "normal"},
    "tom_low": {"line": 67, "note_head": "normal"},
}


def _local_peak_frames(activations, threshold: float, wait: int = 4) -> list[tuple[int, int, float]]:
    hits: list[tuple[int, int, float]] = []
    last_peak_by_drum: dict[int, int] = {}
    for frame_idx in range(1, len(activations) - 1):
        frame = activations[frame_idx]
        for drum_idx, value in enumerate(frame[:3]):
            if frame_idx - last_peak_by_drum.get(drum_idx, -wait) < wait:
                continue
            if value < threshold:
                continue
            if value >= activations[frame_idx - 1][drum_idx] and value >= activations[frame_idx + 1][drum_idx]:
                hits.append((frame_idx, drum_idx, float(value)))
                last_peak_by_drum[drum_idx] = frame_idx
    return hits


def _transcribe_with_madmom(audio_path: str, tempo: float) -> list[dict]:
    import madmom

    proc = madmom.features.drums.DrumTrackProcessor()
    activations = proc(audio_path)

    hits = []
    threshold = 0.3
    drum_types = ["kick", "snare", "hihat_closed"]

    for frame_idx, drum_idx, value in _local_peak_frames(activations, threshold):
        time = frame_idx * 0.01
        beat = time * (tempo / 60)
        hits.append({
            "drum": drum_types[drum_idx],
            "start_beat": beat,
            "velocity": int(value * 127),
            "ghost": value < 0.5,
        })

    return hits


def _classify_drum_window(window, sr: int) -> tuple[str, int]:
    import numpy as np

    if window.size == 0:
        return "snare", 80

    spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
    frequencies = np.fft.rfftfreq(window.size, 1 / sr)
    low = float(np.sum(spectrum[(frequencies >= 35) & (frequencies < 160)]))
    mid = float(np.sum(spectrum[(frequencies >= 160) & (frequencies < 2000)]))
    high = float(np.sum(spectrum[frequencies >= 2000]))
    total = max(low + mid + high, 1e-9)
    velocity = int(max(45, min(127, 45 + (np.sqrt(float(np.mean(window ** 2))) * 600))))

    if low / total > 0.45 and low > mid * 1.2:
        return "kick", velocity
    if high / total > 0.45 and high > mid:
        return "hihat_closed", velocity
    return "snare", velocity


def _transcribe_with_librosa(audio_path: str, tempo: float) -> list[dict]:
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if y.size == 0:
        return []

    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = []
    if envelope.size:
        threshold = float(np.max(envelope)) * 0.35
        wait = 4
        last_peak = -wait
        for index in range(1, len(envelope) - 1):
            if index - last_peak < wait:
                continue
            if envelope[index] >= threshold and envelope[index] >= envelope[index - 1] and envelope[index] >= envelope[index + 1]:
                onset_frames.append(index)
                last_peak = index

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    hits = []
    for onset_time in onset_times:
        beat = float(onset_time) * (tempo / 60)
        center = int(float(onset_time) * sr)
        half_window = int(0.06 * sr)
        window = y[max(0, center - half_window): min(len(y), center + half_window)]
        drum, velocity = _classify_drum_window(window, sr)
        hits.append(
            {
                "drum": drum,
                "start_beat": beat,
                "velocity": velocity,
                "ghost": False,
            }
        )
    return hits


def transcribe_drums(audio_path: str, tempo: float) -> list[dict]:
    try:
        hits = _transcribe_with_madmom(audio_path, tempo)
        if hits:
            return hits
    except Exception:
        pass

    return _transcribe_with_librosa(audio_path, tempo)
