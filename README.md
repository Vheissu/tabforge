# TabForge

Forge tabs from audio. An open-source application that accepts a YouTube video URL, extracts the audio, uses AI to transcribe the music, and generates Guitar Pro (.gp5) tablature files.

## Quick Start

1. Create an environment file and add your Gemini API key:

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=your_key
```

2. Start the stack (CPU worker):

```bash
docker compose up -d --build
```

GPU worker (optional, requires NVIDIA runtime):

```bash
docker compose --profile gpu up -d --build
```

3. Open the app:

```bash
open http://localhost
```

## Local Dev (optional)

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Worker:

```bash
cd backend
source .venv/bin/activate
celery -A app.tasks.celery_app worker --loglevel=info
```

## Notes

- GEMINI_API_KEY is required for AI refinement, but the system will still run without it (refinement step is skipped).
- Tuning can be set to `auto` (default) to let the server detect standard/Drop D/half-step/full-step based on pitch analysis.
- On Apple Silicon (linux/arm64) Docker builds skip basic-pitch/tensorflow because wheels are unavailable; transcription will return empty notes for pitched instruments unless you build for amd64.
- Separation is the slowest step on CPU (especially Apple Silicon). You can speed it up by setting `SEPARATION_MODEL=htdemucs` or by skipping separation entirely with `SEPARATION_ENABLED=false` (lower accuracy).
- Default maximum video duration is 10 minutes; update `MAX_DURATION_SECONDS` via env if needed.
