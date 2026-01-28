Goal (incl. success criteria):
- Implement Tabforge per tech-spec.md (API, worker, frontend, Docker), refining improvements; repo should build/run locally.
Constraints/Assumptions:
- Maintain/update CONTINUITY.md every turn.
- Use stack in tech-spec (Aurelia 2 rc0, FastAPI, Celery, Redis, Postgres, MinIO, Docker).
Key decisions:
- Frontend targets Aurelia 2 rc0; TS config uses new decorators (no experimentalDecorators/emitDecoratorMetadata).
- Frontend visual direction: industrial/forge-inspired dark UI with warm accents and bold typography.
- Default worker is CPU for local dev; GPU worker available via compose profile.
- basic-pitch/tensorflow install only on amd64 due to missing arm64 wheels; degrade gracefully when absent.
- Tuning defaults to auto detection via audio heuristics.
State:
- Transcription runs but separation is very slow on Apple Silicon; added config knobs to skip or change model.
Done:
- Implemented Home and Job pages with polling status, progress bar, and download link.
- Added global CSS theme and component styles.
- Registered ApiService singleton in Aurelia app.
- Added frontend nginx.conf for SPA routing + API proxy.
- Updated README with GEMINI_API_KEY and tuning notes.
- Ran `npm install` and `npm run build` in frontend successfully.
- Expanded .gitignore for build/dev artifacts.
- Recreated frontend/Dockerfile.
- Added build-essential/libsndfile1 and Cython/numpy install steps to backend images.
- Split requirements into base + ML; added conditional ML install by TARGETARCH.
- Made transcription return empty notes when basic-pitch unavailable.
- Added tuning detection (librosa estimate_tuning + low-pitch heuristic) and wired to pipeline.
- Patched madmom import for Python 3.12 compatibility and added np.float shim.
- Pinned numpy<2.0.
- Deferred drums import inside task to avoid API crash on import.
- Patched gp.py to allow missing Tempo class and avoid song.tempo.value usage.
- Patched separation to lazy-import demucs.api and use CLI fallback.
- Updated nginx proxy config to use Docker DNS resolver and variable upstream to avoid stale IPs.
- Rebuilt frontend image and recreated container to apply nginx changes.
- Verified http://localhost/api/v1/health returns {"status":"ok"}.
- Fixed Celery async helper to reuse a single event loop per worker process.
- Rebuilt/recreated api and worker containers with the loop fix.
- Added separation settings (SEPARATION_ENABLED/MODEL/SEGMENT) and fast-mode message; updated docs and compose env.
Now:
- Rebuild/recreate api/worker to apply separation config changes; advise user on fast-mode settings.
Next:
- Confirm separation speed with new settings or skip separation for local dev.
Open questions (UNCONFIRMED if needed):
- Preferred minimal viable subset for initial implementation? (UNCONFIRMED)
- Any repo conventions to follow? (UNCONFIRMED)
Working set (files/ids/commands):
- frontend/nginx.conf
- docker-compose.yml
- CONTINUITY.md
- backend/app/tasks.py
- backend/app/services/separation.py
- backend/app/core/config.py
