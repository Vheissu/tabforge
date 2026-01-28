# TabForge - Technical Specification

> Open-source AI-powered music transcription that generates Guitar Pro tablature from YouTube videos.

## Project Overview

An open-source application that accepts a YouTube video URL, extracts the audio, uses AI to transcribe the music, and generates Guitar Pro (.gp5) tablature files for guitar, bass, and drums.

## Goals

1. Accept YouTube URLs as input and extract high-quality audio
2. Separate audio into individual instrument stems (guitar, bass, drums, vocals)
3. Transcribe each instrument track using AI models
4. Generate properly formatted Guitar Pro files with accurate timing, notes, and fret positions
5. Provide a clean web interface built with Aurelia 2 and a REST API
6. Containerize the entire stack with Docker for easy deployment

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Docker Compose                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Aurelia 2 UI   │───▶│   FastAPI        │───▶│   Celery Worker  │  │
│  │   (nginx)        │    │   (API Gateway)  │    │   (GPU-enabled)  │  │
│  │   Port: 80       │    │   Port: 8000     │    │                  │  │
│  └──────────────────┘    └────────┬─────────┘    └────────┬─────────┘  │
│                                   │                       │             │
│                          ┌────────▼─────────┐    ┌────────▼─────────┐  │
│                          │   PostgreSQL     │    │   Redis          │  │
│                          │   Port: 5432     │    │   Port: 6379     │  │
│                          └──────────────────┘    └──────────────────┘  │
│                                                                          │
│                          ┌──────────────────┐                           │
│                          │   MinIO (S3)     │                           │
│                          │   Port: 9000     │                           │
│                          └──────────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Processing Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  YouTube URL    │────▶│  Audio Extractor │────▶│  Audio Splitter │
└─────────────────┘     │  (yt-dlp + Deno) │     │  (Demucs v4)    │
                        └──────────────────┘     └─────────────────┘
                                                          │
                              ┌────────────────────────────┤
                              ▼                            ▼
                    ┌─────────────────┐          ┌─────────────────┐
                    │  Guitar/Bass    │          │  Drums Stem     │
                    │  Stem           │          │                 │
                    └────────┬────────┘          └────────┬────────┘
                             │                            │
                             ▼                            ▼
                    ┌─────────────────┐          ┌─────────────────┐
                    │  Basic Pitch    │          │  Onset/Hit      │
                    │  (Audio→MIDI)   │          │  Detection      │
                    └────────┬────────┘          └────────┬────────┘
                             │                            │
                             ▼                            ▼
                    ┌─────────────────┐          ┌─────────────────┐
                    │  Gemini 3 Flash │          │  Drum Mapping   │
                    │  Refinement     │          │                 │
                    └────────┬────────┘          └────────┬────────┘
                             │                            │
                             ▼                            ▼
                    ┌─────────────────┐          ┌─────────────────┐
                    │  Fret Position  │          │  GP Drum Track  │
                    │  Optimizer      │          │  Generator      │
                    └────────┬────────┘          └────────┬────────┘
                             │                            │
                             └────────────┬───────────────┘
                                          ▼
                              ┌─────────────────────┐
                              │  PyGuitarPro        │
                              │  (.gp5 Generator)   │
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │  .gp5 File Output   │
                              └─────────────────────┘
```

## Tech Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | Aurelia 2 | Latest | Web interface |
| **API Framework** | FastAPI | 0.115+ | REST API endpoints |
| **Task Queue** | Celery | 5.4+ | Background job processing |
| **Message Broker** | Redis | 7.4+ | Task queue backend |
| **Audio Download** | yt-dlp | 2025.12+ | YouTube audio extraction |
| **JS Runtime** | Deno | Latest | Required by yt-dlp for YouTube |
| **Source Separation** | Demucs v4 | htdemucs_ft | Stem isolation |
| **Pitch Detection** | Basic Pitch | Latest | Audio to MIDI |
| **AI Transcription** | Gemini 3 Flash | gemini-3-flash | Note refinement and technique detection |
| **GP File Generation** | PyGuitarPro | 0.10.1 | Create .gp5 files |
| **Database** | PostgreSQL | 16+ | Job metadata, user data |
| **Object Storage** | MinIO | Latest | Audio and output file storage |
| **Containerization** | Docker + Compose | 27+ | Deployment |

## Component Details

### 1. Audio Extraction Service

**Technology:** yt-dlp (latest) with Deno runtime

**Critical Note:** As of late 2025, yt-dlp requires a JavaScript runtime (Deno recommended) for YouTube support. This must be included in the Docker image.

```python
import subprocess
import json
from pathlib import Path

def extract_audio(youtube_url: str, output_dir: Path) -> dict:
    """
    Extract audio from YouTube URL using yt-dlp.
    Returns metadata about the downloaded file.
    """
    output_template = str(output_dir / "%(id)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", output_template,
        "--print-json",
        "--no-warnings",
        youtube_url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    
    return {
        "id": info["id"],
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "artist": info.get("artist") or info.get("uploader", "Unknown"),
        "audio_path": output_dir / f"{info['id']}.wav"
    }
```

**Constraints:**
- Maximum video duration: 10 minutes (configurable)
- Rate limiting to prevent abuse
- Cache downloaded audio by video ID

### 2. Audio Source Separation

**Technology:** Demucs v4 (Hybrid Transformer)

**Model:** `htdemucs_ft` (fine-tuned, best quality)

The Demucs repository has moved to `adefossez/demucs` (the original author's fork) as `facebookresearch/demucs` is no longer actively maintained.

```python
import torch
from demucs.api import Separator
from pathlib import Path

def separate_stems(audio_path: Path, output_dir: Path) -> dict:
    """
    Separate audio into stems using Demucs v4.
    Returns paths to separated stem files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    separator = Separator(
        model="htdemucs_ft",
        device=device,
        segment=7.8 if device == "cpu" else None  # Reduce for CPU
    )
    
    origin, separated = separator.separate_audio_file(str(audio_path))
    
    stems = {}
    for stem_name, stem_tensor in separated.items():
        stem_path = output_dir / f"{stem_name}.wav"
        separator.save_audio(stem_tensor, str(stem_path), samplerate=44100)
        stems[stem_name] = stem_path
    
    return stems  # Returns: drums, bass, other, vocals
```

**Output Stems:**
- `drums.wav` - Isolated drum track
- `bass.wav` - Isolated bass track  
- `other.wav` - Contains guitars, keys, synths
- `vocals.wav` - Isolated vocals (discarded or optional)

**Hardware Requirements:**
- GPU: NVIDIA with 8GB+ VRAM (strongly recommended)
- CPU fallback: Works but 10-20x slower
- RAM: 16GB minimum

### 3. Music Transcription Engine

#### 3a. Audio to MIDI (Basic Pitch)

```python
from basic_pitch.inference import predict, predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import tensorflow as tf

# Load model once at startup
basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

def audio_to_midi(audio_path: Path, instrument: str) -> tuple:
    """
    Convert audio to MIDI using Basic Pitch.
    Returns model output, MIDI data, and note events.
    """
    # Set frequency range based on instrument
    freq_ranges = {
        "bass": (30, 400),      # E1 to ~G4
        "guitar": (80, 1200),   # E2 to ~D6
        "other": (80, 2000),    # Wide range for mixed content
    }
    
    min_freq, max_freq = freq_ranges.get(instrument, (80, 2000))
    
    model_output, midi_data, note_events = predict(
        str(audio_path),
        basic_pitch_model,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        minimum_note_length=0.05,  # 50ms minimum
        onset_threshold=0.5,
        frame_threshold=0.3,
    )
    
    return model_output, midi_data, note_events
```

#### 3b. AI-Enhanced Refinement (Gemini 3 Flash)

Use Gemini 3 Flash for speed and cost efficiency. It has native audio understanding and can refine transcriptions.

```python
import google.generativeai as genai
from pathlib import Path
import json

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def refine_with_gemini(
    audio_path: Path,
    midi_notes: list,
    instrument: str,
    tempo: int
) -> dict:
    """
    Use Gemini 3 Flash to refine transcription and detect techniques.
    """
    model = genai.GenerativeModel("gemini-3-flash")
    
    # Upload audio file
    audio_file = genai.upload_file(str(audio_path))
    
    # Prepare note context
    note_summary = json.dumps(midi_notes[:50])  # First 50 notes as context
    
    prompt = f"""
    You are analyzing a {instrument} recording. 
    
    Initial MIDI transcription detected these notes (first 50):
    {note_summary}
    
    Tempo is approximately {tempo} BPM.
    
    Listen to the audio and provide corrections/refinements:
    
    1. Identify any missed notes or incorrect pitches
    2. Detect playing techniques:
       - Hammer-ons and pull-offs
       - Slides (up/down)
       - Bends (quarter, half, full, etc.)
       - Palm muting
       - Vibrato
    3. Suggest optimal fret positions for playability
    4. Identify any chord voicings
    
    Respond with JSON only, no markdown:
    {{
        "tempo_correction": null or int,
        "time_signature": "4/4",
        "key": "E minor",
        "corrections": [
            {{
                "original_note_index": 0,
                "corrected_pitch": "E2",
                "technique": "palm_mute",
                "suggested_fret": 0,
                "suggested_string": 6
            }}
        ],
        "additional_notes": [
            {{
                "start_time": 1.5,
                "pitch": "G2",
                "duration": 0.25,
                "technique": "hammer_on"
            }}
        ]
    }}
    """
    
    response = model.generate_content(
        [prompt, audio_file],
        generation_config={"temperature": 0.1}
    )
    
    return json.loads(response.text)
```

**Why Gemini 3 Flash over Pro:**
- 10x faster inference
- Significantly cheaper ($0.075/M input tokens vs $3.00/M for Pro)
- Native audio understanding (same as Pro)
- Sufficient quality for this task

#### 3c. Fret Position Optimizer

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FretPosition:
    string: int  # 1-6 (high E to low E)
    fret: int    # 0-24

@dataclass
class Note:
    pitch: str           # e.g., "E2", "A3"
    start_beat: float
    duration: float
    position: Optional[FretPosition] = None
    technique: Optional[str] = None

# Standard tuning MIDI values (low to high string)
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

def midi_to_note_name(midi: int) -> str:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"

def get_all_positions(midi_note: int, tuning: List[int] = STANDARD_TUNING) -> List[FretPosition]:
    """Get all possible fret positions for a MIDI note."""
    positions = []
    for string_idx, open_pitch in enumerate(tuning):
        fret = midi_note - open_pitch
        if 0 <= fret <= 24:
            positions.append(FretPosition(string=6 - string_idx, fret=fret))
    return positions

def optimize_positions(notes: List[Note], tuning: List[int] = STANDARD_TUNING) -> List[Note]:
    """
    Optimize fret positions for playability using a greedy algorithm
    that minimizes hand movement.
    """
    if not notes:
        return notes
    
    # Start with the first note - prefer lower frets
    first_positions = get_all_positions(note_to_midi(notes[0].pitch), tuning)
    if first_positions:
        notes[0].position = min(first_positions, key=lambda p: p.fret)
    
    # For subsequent notes, minimize distance from previous position
    for i in range(1, len(notes)):
        prev_pos = notes[i - 1].position
        curr_positions = get_all_positions(note_to_midi(notes[i].pitch), tuning)
        
        if not curr_positions:
            continue
            
        if prev_pos is None:
            notes[i].position = min(curr_positions, key=lambda p: p.fret)
        else:
            # Score based on fret distance and string distance
            def position_score(pos: FretPosition) -> float:
                fret_dist = abs(pos.fret - prev_pos.fret)
                string_dist = abs(pos.string - prev_pos.string)
                return fret_dist * 2 + string_dist  # Weight fret movement more
            
            notes[i].position = min(curr_positions, key=position_score)
    
    return notes
```

#### 3d. Drum Transcription

```python
import madmom
import numpy as np
from typing import List, Tuple

# Guitar Pro drum mapping
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
    """
    Transcribe drum audio using onset detection and classification.
    """
    # Load processors
    proc = madmom.features.drums.DrumTrackProcessor()
    
    # Process audio
    activations = proc(audio_path)
    
    # Convert activations to hits
    hits = []
    threshold = 0.3
    
    for frame_idx, frame in enumerate(activations):
        time = frame_idx * 0.01  # 10ms hop size
        
        # Check each drum type
        drum_types = ["kick", "snare", "hihat_closed"]
        for drum_idx, drum_type in enumerate(drum_types):
            if frame[drum_idx] > threshold:
                beat = time * (tempo / 60)  # Convert to beats
                hits.append({
                    "drum": drum_type,
                    "start_beat": beat,
                    "velocity": int(frame[drum_idx] * 127),
                    "ghost": frame[drum_idx] < 0.5
                })
    
    return hits
```

### 4. Guitar Pro File Generator

**Library:** PyGuitarPro 0.10.1

**Note:** PyGuitarPro only supports GP3, GP4, and GP5 formats. GPX (GP6+) is not supported for writing. GP5 format is recommended as it is widely compatible and can be opened by Guitar Pro 6, 7, 8, TuxGuitar, and Songsterr.

```python
import guitarpro
from guitarpro.models import (
    Song, Track, Measure, Voice, Beat, Note, 
    NoteEffect, BendEffect, Duration, Tempo,
    MeasureHeader, TimeSignature, GuitarString
)
from typing import List

def create_guitar_pro_file(transcription: dict, output_path: str):
    """
    Generate a Guitar Pro 5 file from transcription data.
    """
    song = Song()
    song.title = transcription["title"]
    song.artist = transcription.get("artist", "Unknown")
    song.tempo = Tempo(transcription["tempo"])
    
    # Create guitar track
    if "guitar" in transcription:
        guitar_track = create_guitar_track(transcription["guitar"])
        song.tracks.append(guitar_track)
    
    # Create bass track
    if "bass" in transcription:
        bass_track = create_bass_track(transcription["bass"])
        song.tracks.append(bass_track)
    
    # Create drum track
    if "drums" in transcription:
        drum_track = create_drum_track(transcription["drums"])
        song.tracks.append(drum_track)
    
    guitarpro.write(song, output_path)

def create_guitar_track(data: dict) -> Track:
    """Create a guitar track with proper tuning and notes."""
    track = Track()
    track.name = "Guitar"
    track.channel.instrument = 25  # Steel guitar
    track.isPercussionTrack = False
    
    # Standard tuning (high to low): E4, B3, G3, D3, A2, E2
    track.strings = [
        GuitarString(1, 64),  # E4
        GuitarString(2, 59),  # B3
        GuitarString(3, 55),  # G3
        GuitarString(4, 50),  # D3
        GuitarString(5, 45),  # A2
        GuitarString(6, 40),  # E2
    ]
    
    # Group notes by measure
    measures_data = group_notes_by_measure(data["notes"], data.get("time_signature", (4, 4)))
    
    for measure_idx, measure_notes in enumerate(measures_data):
        header = MeasureHeader()
        header.tempo = Tempo(data.get("tempo", 120))
        header.timeSignature = TimeSignature(
            data.get("time_signature", (4, 4))[0],
            Duration(data.get("time_signature", (4, 4))[1])
        )
        
        measure = Measure(track, header)
        voice = measure.voices[0]
        
        for note_data in measure_notes:
            beat = Beat(voice)
            note = Note(beat)
            note.value = note_data["fret"]
            note.string = note_data["string"]
            note.velocity = note_data.get("velocity", 100)
            
            # Apply effects
            if note_data.get("technique"):
                apply_technique(note, note_data["technique"])
            
            beat.notes.append(note)
            voice.beats.append(beat)
        
        track.measures.append(measure)
    
    return track

def apply_technique(note: Note, technique: str):
    """Apply playing technique effects to a note."""
    effect = note.effect
    
    if technique == "hammer_on":
        effect.hammer = True
    elif technique == "pull_off":
        effect.hammer = True  # GP uses same flag
    elif technique == "slide_up":
        effect.slides = [guitarpro.SlideType.shiftSlide]
    elif technique == "slide_down":
        effect.slides = [guitarpro.SlideType.shiftSlide]
    elif technique == "bend":
        bend = BendEffect()
        bend.type = guitarpro.BendType.bend
        bend.value = 100  # Full step
        effect.bend = bend
    elif technique == "vibrato":
        effect.vibrato = True
    elif technique == "palm_mute":
        effect.palmMute = True
```

### 5. API Layer (FastAPI)

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from enum import Enum
import uuid

app = FastAPI(
    title="TabForge API",
    description="AI-powered music transcription to Guitar Pro tabs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Instrument(str, Enum):
    guitar = "guitar"
    bass = "bass"
    drums = "drums"

class TranscriptionRequest(BaseModel):
    youtube_url: HttpUrl
    instruments: List[Instrument] = [Instrument.guitar, Instrument.bass, Instrument.drums]
    tuning: Optional[str] = "standard"  # standard, drop_d, half_step_down, etc.

class JobStatus(str, Enum):
    pending = "pending"
    extracting = "extracting"
    separating = "separating"
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

@app.post("/api/v1/transcribe", response_model=JobResponse)
async def create_transcription(request: TranscriptionRequest):
    """Start a new transcription job."""
    job_id = str(uuid.uuid4())
    
    # Validate URL and duration before queuing
    try:
        duration = await validate_youtube_url(str(request.youtube_url))
        if duration > 600:  # 10 minutes
            raise HTTPException(400, "Video exceeds maximum duration of 10 minutes")
    except Exception as e:
        raise HTTPException(400, str(e))
    
    # Queue the job
    task = process_transcription.delay(
        job_id=job_id,
        youtube_url=str(request.youtube_url),
        instruments=[i.value for i in request.instruments],
        tuning=request.tuning
    )
    
    # Store job metadata
    await save_job(job_id, task.id, request)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.pending,
        progress=0,
        message="Job queued for processing"
    )

@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a transcription job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    return JobResponse(**job)

@app.get("/api/v1/download/{job_id}")
async def download_file(job_id: str):
    """Download the generated Guitar Pro file."""
    job = await get_job(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(404, "File not ready or job not found")
    
    return FileResponse(
        job["file_path"],
        media_type="application/octet-stream",
        filename=f"{job['title']}.gp5"
    )
```

### 6. Celery Task Queue

```python
from celery import Celery
from celery.utils.log import get_task_logger

app = Celery(
    "tabforge",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1"
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minute hard limit
    task_soft_time_limit=1500,  # 25 minute soft limit
)

logger = get_task_logger(__name__)

@app.task(bind=True)
def process_transcription(
    self,
    job_id: str,
    youtube_url: str,
    instruments: list,
    tuning: str
):
    """Main transcription pipeline task."""
    try:
        # Stage 1: Extract audio
        self.update_state(state="EXTRACTING", meta={"progress": 5})
        metadata = extract_audio(youtube_url, TEMP_DIR / job_id)
        audio_path = metadata["audio_path"]
        
        # Stage 2: Separate stems
        self.update_state(state="SEPARATING", meta={"progress": 20})
        stems = separate_stems(audio_path, TEMP_DIR / job_id / "stems")
        
        # Stage 3: Detect tempo and key
        self.update_state(state="ANALYZING", meta={"progress": 35})
        tempo, key = analyze_audio(audio_path)
        
        # Stage 4: Transcribe each instrument
        transcription = {
            "title": metadata["title"],
            "artist": metadata["artist"],
            "tempo": tempo,
            "key": key,
        }
        
        progress_per_instrument = 40 // len(instruments)
        
        for idx, instrument in enumerate(instruments):
            self.update_state(
                state="TRANSCRIBING",
                meta={
                    "progress": 40 + (idx * progress_per_instrument),
                    "instrument": instrument
                }
            )
            
            if instrument == "drums":
                transcription["drums"] = transcribe_drums(stems["drums"], tempo)
            else:
                stem_path = stems["bass"] if instrument == "bass" else stems["other"]
                transcription[instrument] = transcribe_pitched_instrument(
                    stem_path, instrument, tempo, tuning
                )
        
        # Stage 5: Generate GP file
        self.update_state(state="GENERATING", meta={"progress": 90})
        output_path = OUTPUT_DIR / f"{job_id}.gp5"
        create_guitar_pro_file(transcription, str(output_path))
        
        # Upload to storage
        download_url = upload_to_storage(output_path, f"{job_id}.gp5")
        
        # Cleanup temp files
        cleanup_temp_files(job_id)
        
        return {
            "status": "completed",
            "progress": 100,
            "download_url": download_url,
            "title": metadata["title"]
        }
        
    except Exception as e:
        logger.exception(f"Transcription failed for job {job_id}")
        return {
            "status": "failed",
            "progress": 0,
            "message": str(e)
        }
```

### 7. Frontend (Aurelia 2)

**Project Structure:**
```
frontend/
├── src/
│   ├── main.ts
│   ├── my-app.ts
│   ├── my-app.html
│   ├── services/
│   │   └── api-service.ts
│   ├── pages/
│   │   ├── home/
│   │   │   ├── home.ts
│   │   │   └── home.html
│   │   └── job/
│   │       ├── job.ts
│   │       └── job.html
│   └── components/
│       ├── progress-bar/
│       └── instrument-selector/
├── package.json
├── tsconfig.json
└── Dockerfile
```

**API Service:**
```typescript
// src/services/api-service.ts
import { DI } from 'aurelia';

export interface TranscriptionRequest {
  youtube_url: string;
  instruments: string[];
  tuning?: string;
}

export interface JobResponse {
  job_id: string;
  status: string;
  progress: number;
  message?: string;
  download_url?: string;
}

export class ApiService {
  private baseUrl = import.meta.env.VITE_API_URL || '/api/v1';

  async createTranscription(request: TranscriptionRequest): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create transcription');
    }
    
    return response.json();
  }

  async getJobStatus(jobId: string): Promise<JobResponse> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`);
    
    if (!response.ok) {
      throw new Error('Failed to get job status');
    }
    
    return response.json();
  }

  getDownloadUrl(jobId: string): string {
    return `${this.baseUrl}/download/${jobId}`;
  }
}
```

**Home Page:**
```typescript
// src/pages/home/home.ts
import { IRouter } from '@aurelia/router';
import { ApiService, TranscriptionRequest } from '../../services/api-service';

export class Home {
  youtubeUrl = '';
  instruments = ['guitar', 'bass', 'drums'];
  selectedInstruments: string[] = ['guitar', 'bass', 'drums'];
  tuning = 'standard';
  isLoading = false;
  error = '';

  constructor(
    @IRouter private router: IRouter,
    private api: ApiService
  ) {}

  toggleInstrument(instrument: string): void {
    const index = this.selectedInstruments.indexOf(instrument);
    if (index > -1) {
      this.selectedInstruments.splice(index, 1);
    } else {
      this.selectedInstruments.push(instrument);
    }
  }

  async submit(): Promise<void> {
    if (!this.youtubeUrl || this.selectedInstruments.length === 0) {
      this.error = 'Please enter a YouTube URL and select at least one instrument';
      return;
    }

    this.isLoading = true;
    this.error = '';

    try {
      const request: TranscriptionRequest = {
        youtube_url: this.youtubeUrl,
        instruments: this.selectedInstruments,
        tuning: this.tuning,
      };

      const response = await this.api.createTranscription(request);
      this.router.load(`/job/${response.job_id}`);
    } catch (e) {
      this.error = e.message;
    } finally {
      this.isLoading = false;
    }
  }
}
```

```html
<!-- src/pages/home/home.html -->
<div class="container mx-auto max-w-2xl p-6">
  <h1 class="text-4xl font-bold mb-8 text-center">TabForge</h1>
  <p class="text-gray-600 mb-8 text-center">
    Generate Guitar Pro tabs from any YouTube video using AI
  </p>

  <form submit.trigger="submit()">
    <div class="mb-6">
      <label class="block text-sm font-medium mb-2">YouTube URL</label>
      <input
        type="url"
        value.bind="youtubeUrl"
        placeholder="https://www.youtube.com/watch?v=..."
        class="w-full p-3 border rounded-lg"
        required
      />
    </div>

    <div class="mb-6">
      <label class="block text-sm font-medium mb-2">Instruments</label>
      <div class="flex gap-4">
        <label repeat.for="inst of instruments" class="flex items-center gap-2">
          <input
            type="checkbox"
            checked.bind="selectedInstruments"
            model.bind="inst"
            change.trigger="toggleInstrument(inst)"
          />
          <span class="capitalize">${inst}</span>
        </label>
      </div>
    </div>

    <div class="mb-6">
      <label class="block text-sm font-medium mb-2">Guitar Tuning</label>
      <select value.bind="tuning" class="w-full p-3 border rounded-lg">
        <option value="standard">Standard (EADGBE)</option>
        <option value="drop_d">Drop D (DADGBE)</option>
        <option value="half_step_down">Half Step Down (Eb)</option>
        <option value="full_step_down">Full Step Down (D)</option>
      </select>
    </div>

    <div if.bind="error" class="mb-4 p-4 bg-red-100 text-red-700 rounded-lg">
      ${error}
    </div>

    <button
      type="submit"
      class="w-full p-4 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
      disabled.bind="isLoading"
    >
      <span if.bind="!isLoading">Generate Tabs</span>
      <span if.bind="isLoading">Processing...</span>
    </button>
  </form>
</div>
```

## Docker Configuration

### docker-compose.yml

```yaml
version: "3.9"

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - api
    environment:
      - VITE_API_URL=http://localhost:8000/api/v1

  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
      - minio
    environment:
      - DATABASE_URL=postgresql://tabforge:tabforge@postgres:5432/tabforge
      - REDIS_URL=redis://redis:6379/0
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data/temp:/app/temp
      - ./data/output:/app/output

  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.worker
    command: celery -A app.tasks worker --loglevel=info --concurrency=2
    depends_on:
      - redis
      - postgres
      - minio
    environment:
      - DATABASE_URL=postgresql://tabforge:tabforge@postgres:5432/tabforge
      - REDIS_URL=redis://redis:6379/0
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data/temp:/app/temp
      - ./data/output:/app/output
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7.4-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=tabforge
      - POSTGRES_PASSWORD=tabforge
      - POSTGRES_DB=tabforge
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data

volumes:
  redis_data:
  postgres_data:
  minio_data:
```

### Backend Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (required for yt-dlp YouTube support)
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Worker Dockerfile (GPU-enabled)

```dockerfile
# backend/Dockerfile.worker
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    ffmpeg \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="$DENO_INSTALL/bin:$PATH"

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy application code
COPY . .

CMD ["celery", "-A", "app.tasks", "worker", "--loglevel=info"]
```

### Frontend Dockerfile

```dockerfile
# frontend/Dockerfile
FROM node:22-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### requirements.txt

```
# API
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
python-multipart>=0.0.12

# Task Queue
celery>=5.4.0
redis>=5.2.0

# Database
sqlalchemy>=2.0.0
asyncpg>=0.30.0
alembic>=1.14.0

# Audio Processing
yt-dlp>=2025.12.0
demucs>=4.0.0
basic-pitch>=0.3.0
librosa>=0.10.0
madmom>=0.16.0

# AI
google-generativeai>=0.8.0

# Guitar Pro
pyguitarpro>=0.10.1

# Storage
minio>=7.2.0
boto3>=1.35.0

# Utilities
python-dotenv>=1.0.0
httpx>=0.28.0
```

## AI Model Comparison

| Model | Audio Support | Cost (per song ~3 min) | Speed | Best For |
|-------|---------------|------------------------|-------|----------|
| **Gemini 3 Flash** | Native | ~$0.002 | Fast | Production use, refinement |
| **Gemini 3 Pro** | Native | ~$0.02 | Medium | Complex analysis if needed |
| **Claude 3.5 Sonnet** | Via transcription | ~$0.01 | Medium | Alternative, good reasoning |
| **GPT-4o** | Native | ~$0.015 | Medium | Alternative option |

**Recommendation:** Use Gemini 3 Flash for the transcription refinement step. It has native audio understanding, is very fast, and costs a fraction of other options.

## Cost Estimates (Per Transcription)

| Component | Cost |
|-----------|------|
| Gemini 3 Flash API (3 min audio) | ~$0.002 |
| GPU compute (5 min @ $0.50/hr) | ~$0.04 |
| Storage (30 day retention) | ~$0.001 |
| **Total per song** | **~$0.05** |

## Security Considerations

1. **Rate limiting:** 10 requests/hour per IP for unauthenticated users
2. **URL validation:** Strict YouTube URL validation, reject playlists
3. **Duration limits:** Reject videos over 10 minutes
4. **File validation:** Scan generated files before serving
5. **API keys:** Store in environment variables, never commit
6. **CORS:** Configure properly for production domains

## Future Enhancements

1. **Additional output formats:** MusicXML, MIDI, PDF sheet music
2. **Tuning detection:** Auto-detect alternate tunings from audio
3. **Tab editor:** Allow users to correct AI mistakes in-browser
4. **Feedback loop:** Use corrections to improve future transcriptions
5. **Real-time preview:** Audio playback synced with tab visualization
6. **Chord detection:** Show chord diagrams above tab
7. **Section markers:** Auto-detect verse, chorus, bridge
8. **User accounts:** Save transcription history

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/tabforge.git
cd tabforge

# Create .env file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Start all services
docker compose up -d

# View logs
docker compose logs -f worker

# Access the app
open http://localhost
```

## License

MIT License - see LICENSE file for details.

---

**Note on GP5 Format:** PyGuitarPro only supports GP3, GP4, and GP5 formats. While Guitar Pro 6+ uses the GPX format (a ZIP archive with XML), there is no reliable Python library for writing GPX files. GP5 is recommended as it is universally compatible with Guitar Pro 6, 7, 8, TuxGuitar, MuseScore, and Songsterr.