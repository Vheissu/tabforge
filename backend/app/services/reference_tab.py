from __future__ import annotations

from typing import Any

from app.services.ascii_tab import ascii_tab_to_draft


def _payload_text(payload: dict[str, Any]) -> str:
    return str(payload.get("ascii_tab") or payload.get("text") or "").strip()


def has_reference_tab(payload: dict[str, Any] | None) -> bool:
    return bool(payload and _payload_text(payload))


def build_reference_tab_draft(
    payload: dict[str, Any],
    *,
    metadata: dict[str, Any],
    tempo: int,
    key: str | None,
    tuning: str,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    text = _payload_text(payload)
    if not text:
        raise ValueError("Reference tab is empty.")

    title = str(payload.get("title") or metadata.get("title") or "Reference Tab")
    artist = str(payload.get("artist") or metadata.get("artist") or "Unknown")
    reference_tempo = int(payload.get("tempo_bpm") or tempo or 120)
    reference_key = payload.get("key") or key
    reference_tuning = str(payload.get("tuning") or tuning or "standard")
    if reference_tuning == "auto":
        reference_tuning = "standard"
    track_name = str(payload.get("track_name") or "guitar")

    draft = ascii_tab_to_draft(
        text,
        title=title,
        artist=artist,
        tempo=reference_tempo,
        key=str(reference_key) if reference_key else None,
        tuning=reference_tuning,
        time_signature=str(constraints.get("time_signature") or "4/4"),
        capo_fret=int(constraints.get("capo_fret") or 0),
        columns_per_beat=float(payload.get("columns_per_beat") or 4.0),
        default_duration_beats=float(payload.get("default_duration_beats") or 0.25),
        track_name=track_name,
    )
    draft["constraints"]["tempo_source"] = "reference"
    draft["constraints"]["time_signature_source"] = constraints.get("time_signature_source") or "reference"
    draft["constraints"]["pickup_bar_beats"] = constraints.get("pickup_bar_beats")
    draft["constraints"]["triplet_feel"] = bool(constraints.get("triplet_feel"))
    draft["sources"]["youtube"] = {
        "title": metadata.get("title"),
        "artist": metadata.get("artist"),
    }
    draft["quality"]["warnings"].append(
        {
            "code": "reference_tab_used",
            "severity": "info",
            "message": "GP5 was generated from the supplied reference tab instead of blind audio transcription.",
        }
    )
    if draft["tracks"][0]["statistics"]["note_count"] == 0:
        draft["quality"]["warnings"].append(
            {
                "code": "reference_tab_empty",
                "severity": "warning",
                "message": "Reference tab did not contain a complete six-line ASCII guitar tab block.",
            }
        )
    return draft
