from __future__ import annotations

from typing import List

import collections
import collections.abc
import numpy as np

# Madmom expects MutableSequence in collections (removed in Python 3.12).
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence

# NumPy 2 removed deprecated aliases used by madmom.
if not hasattr(np, "float"):
    np.float = float

import madmom

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


def transcribe_drums(audio_path: str, tempo: float) -> List[dict]:
    proc = madmom.features.drums.DrumTrackProcessor()
    activations = proc(audio_path)

    hits = []
    threshold = 0.3

    for frame_idx, frame in enumerate(activations):
        time = frame_idx * 0.01
        drum_types = ["kick", "snare", "hihat_closed"]
        for drum_idx, drum_type in enumerate(drum_types):
            if frame[drum_idx] > threshold:
                beat = time * (tempo / 60)
                hits.append({
                    "drum": drum_type,
                    "start_beat": beat,
                    "velocity": int(frame[drum_idx] * 127),
                    "ghost": frame[drum_idx] < 0.5,
                })

    return hits
