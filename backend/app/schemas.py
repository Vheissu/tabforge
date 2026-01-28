from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional


class Instrument(str, Enum):
    guitar = "guitar"
    bass = "bass"
    drums = "drums"


class TranscriptionRequest(BaseModel):
    youtube_url: HttpUrl
    instruments: List[Instrument] = Field(default_factory=lambda: [Instrument.guitar, Instrument.bass, Instrument.drums])
    tuning: Optional[str] = "auto"


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
