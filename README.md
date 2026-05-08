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
open http://localhost:8090
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
pip install -r requirements-worker.txt
celery -A app.tasks.celery_app worker --loglevel=info
```

## Notes

- GEMINI_API_KEY is required for AI refinement, but the system will still run without it (refinement step is skipped).
- `GEMINI_MODEL` defaults to `gemini-3-flash-preview`; if refinement is unavailable or exceeds `GEMINI_REFINEMENT_TIMEOUT_SECONDS`, transcription still completes without it.
- The API image is intentionally lightweight. Audio, ML, Guitar Pro, and storage dependencies live in the worker image.
- Docker stores temp audio and generated GP5 files in named volumes shared by the API and worker, so downloads still work when object storage upload is unavailable.
- Docker stores downloaded separation model weights in a named worker cache volume, so repeated worker restarts do not need to fetch them again.
- Tuning can be set to `auto` (default) to let the server detect standard/Drop D/half-step/full-step based on pitch analysis.
- Advanced transcription settings can provide the musical priors that a blind audio pass cannot reliably infer: first-bar meter, pickup length, first-bar tempo, triplet/shuffle feel, and capo fret.
- Completed jobs write both a `.gp5` file and a versioned `.draft.json` artifact. The draft captures metadata, constraints, source stems, per-track notes/statistics, and quality warnings so future editor and benchmarking work does not need to reverse-engineer GP5 files.
- Existing Guitar Pro files can be normalized into the same draft format with `python tools/gp5_to_draft.py path/to/file.gp5`. This is the start of the reference-corpus path for evaluation and future tab-prior training.
- Drafts can be scored against a reference with `python tools/compare_drafts.py reference.draft.json generated.draft.json --strict`. Strict scoring checks pitch/timing F1 plus duration, string/fret position, technique, tempo, meter, tuning, and capo.
- Reference corpora can be gated with `python tools/benchmark_accuracy.py --manifest benchmarks/manifest.json --min-f1 0.9 --min-strict-accuracy 0.9`. The same command is available inside backend containers as `python -m app.tools.benchmark_accuracy`.
- Generated draft notes include both pitch names and MIDI pitch numbers so imported Guitar Pro references and fresh transcriptions can be compared consistently.
- Gemini refinement can replace deterministic pitched notes only when it returns a valid, playable beat-based note list; rejected refinements are kept as diagnostics instead of silently changing the tab.
- Auto-tuning now records confidence and candidate counts. Low-confidence tuning is surfaced in the draft report so pitch-shifted first passes can be corrected before regeneration.
- Apple Silicon Docker workers install Basic Pitch with the linux/arm64 TensorFlow CPU runtime. If Basic Pitch is unavailable, pitched instruments fall back to simpler librosa heuristics, but that fallback is not expected to produce useful full-song tabs.
- Use Python 3.11 for local worker development if you want Basic Pitch/TensorFlow support. Newer Python versions may run the API but skip pitched-instrument transcription models.
- Separation is the slowest step on CPU (especially Apple Silicon). TabForge defaults to `SEPARATION_MODEL=htdemucs_6s` so guitar transcription can use a dedicated guitar stem when Demucs provides one. You can use `SEPARATION_MODEL=htdemucs_ft` for slower but strong 4-stem separation, `SEPARATION_MODEL=htdemucs` for speed, or `SEPARATION_ENABLED=false` to skip separation entirely (lower accuracy).
- Default maximum video duration is 10 minutes; update `MAX_DURATION_SECONDS` via env if needed.
