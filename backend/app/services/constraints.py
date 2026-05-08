from __future__ import annotations

from fractions import Fraction
from typing import Any


SUPPORTED_TIME_SIGNATURES = {"4/4", "3/4", "6/8", "12/8"}


def normalise_constraints(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}

    time_signature = str(raw.get("first_bar_time_signature") or "auto")
    if time_signature == "auto":
        time_signature = "4/4"
        time_signature_source = "auto"
    elif time_signature in SUPPORTED_TIME_SIGNATURES:
        time_signature_source = "user"
    else:
        time_signature = "4/4"
        time_signature_source = "auto"

    pickup_bar_beats = raw.get("pickup_bar_beats")
    if pickup_bar_beats is not None:
        pickup_bar_beats = max(0.0, min(float(pickup_bar_beats), beats_per_measure(time_signature) - 0.25))
        if pickup_bar_beats <= 0:
            pickup_bar_beats = None

    tempo = raw.get("first_bar_tempo_bpm")
    tempo = int(tempo) if tempo is not None else None

    triplet_feel = raw.get("triplet_feel")
    triplet_feel = bool(triplet_feel) if triplet_feel is not None else False

    return {
        "time_signature": time_signature,
        "time_signature_source": time_signature_source,
        "pickup_bar_beats": pickup_bar_beats,
        "tempo_bpm": tempo,
        "tempo_source": "user" if tempo else "detected",
        "triplet_feel": triplet_feel,
        "capo_fret": int(raw.get("capo_fret") or 0),
    }


def parse_time_signature(time_signature: str) -> tuple[int, int]:
    try:
        numerator, denominator = time_signature.split("/", 1)
        return int(numerator), int(denominator)
    except ValueError:
        return 4, 4


def beats_per_measure(time_signature: str) -> float:
    numerator, denominator = parse_time_signature(time_signature)
    return numerator * (4 / denominator)


def pickup_time_signature(pickup_bar_beats: float) -> tuple[int, int]:
    signature = Fraction(pickup_bar_beats / 4).limit_denominator(16)
    return max(1, signature.numerator), signature.denominator
