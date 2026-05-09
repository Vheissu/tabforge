from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional


class Instrument(str, Enum):
    guitar = "guitar"
    bass = "bass"
    drums = "drums"


class TripletFeel(str, Enum):
    auto = "auto"
    straight = "straight"
    triplet = "triplet"


class TranscriptionConstraints(BaseModel):
    first_bar_time_signature: str = Field(default="auto", pattern="^(auto|4/4|3/4|6/8|12/8)$")
    pickup_bar_beats: Optional[float] = Field(default=None, ge=0, le=8)
    first_bar_tempo_bpm: Optional[int] = Field(default=None, ge=40, le=260)
    triplet_feel: TripletFeel = TripletFeel.auto
    capo_fret: int = Field(default=0, ge=0, le=12)

    def to_worker_payload(self) -> dict:
        return {
            "first_bar_time_signature": self.first_bar_time_signature,
            "pickup_bar_beats": self.pickup_bar_beats,
            "first_bar_tempo_bpm": self.first_bar_tempo_bpm,
            "triplet_feel": None if self.triplet_feel == TripletFeel.auto else self.triplet_feel == TripletFeel.triplet,
            "capo_fret": self.capo_fret,
        }


class ReferenceTabRequest(BaseModel):
    ascii_tab: str = Field(min_length=1, max_length=120_000)
    title: Optional[str] = Field(default=None, max_length=200)
    artist: Optional[str] = Field(default=None, max_length=200)
    tempo_bpm: Optional[int] = Field(default=None, ge=40, le=260)
    key: Optional[str] = Field(default=None, max_length=16)
    tuning: Optional[str] = Field(default=None, pattern="^(standard|drop_d|half_step_down|full_step_down|auto)$")
    track_name: Instrument = Instrument.guitar
    columns_per_beat: float = Field(default=4.0, ge=1.0, le=16.0)
    default_duration_beats: float = Field(default=0.25, gt=0, le=4.0)

    def to_worker_payload(self) -> dict:
        return {
            "ascii_tab": self.ascii_tab,
            "title": self.title,
            "artist": self.artist,
            "tempo_bpm": self.tempo_bpm,
            "key": self.key,
            "tuning": self.tuning,
            "track_name": self.track_name.value,
            "columns_per_beat": self.columns_per_beat,
            "default_duration_beats": self.default_duration_beats,
        }


class TrackCorrection(BaseModel):
    enabled: bool = True
    min_velocity: Optional[int] = Field(default=None, ge=1, le=127)
    max_notes_per_slot: Optional[int] = Field(default=None, ge=1, le=7)


class DraftCorrectionRequest(BaseModel):
    tempo_bpm: Optional[int] = Field(default=None, ge=40, le=260)
    time_signature: Optional[str] = Field(default=None, pattern="^(4/4|3/4|6/8|12/8)$")
    pickup_bar_beats: Optional[float] = Field(default=None, ge=0, le=8)
    tuning: Optional[str] = Field(default=None, pattern="^(standard|drop_d|half_step_down|full_step_down)$")
    capo_fret: Optional[int] = Field(default=None, ge=0, le=12)
    triplet_feel: Optional[TripletFeel] = None
    tracks: dict[str, TrackCorrection] = Field(default_factory=dict)


class TranscriptionRequest(BaseModel):
    youtube_url: HttpUrl
    instruments: List[Instrument] = Field(
        default_factory=lambda: [Instrument.guitar, Instrument.bass, Instrument.drums],
        min_length=1,
    )
    tuning: Optional[str] = "auto"
    constraints: TranscriptionConstraints = Field(default_factory=TranscriptionConstraints)
    reference_tab: Optional[ReferenceTabRequest] = None


class JobStatus(str, Enum):
    pending = "pending"
    extracting = "extracting"
    separating = "separating"
    analyzing = "analyzing"
    transcribing = "transcribing"
    generating = "generating"
    completed = "completed"
    failed = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    message: Optional[str] = None
    download_url: Optional[str] = None
    title: Optional[str] = None
