from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from celery.utils.log import get_task_logger

from app.celery_app import celery_app
from app.core.config import get_settings
from app.db.crud import update_job
from app.db.session import async_session

settings = get_settings()
logger = get_task_logger(__name__)

TEMP_DIR = Path(settings.temp_dir)
OUTPUT_DIR = Path(settings.output_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


_worker_loop: asyncio.AbstractEventLoop | None = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def _run_async(coro):
    loop = _get_worker_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    return loop.run_until_complete(coro)


async def _update_job(job_id: str, **fields) -> None:
    async with async_session() as session:
        await update_job(session, job_id, **fields)


def cleanup_temp_files(job_id: str) -> None:
    path = TEMP_DIR / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@celery_app.task(bind=True)
def process_transcription(
    self,
    job_id: str,
    youtube_url: str,
    instruments: list,
    tuning: str,
    constraints: dict | None = None,
):
    from app.services.audio import analyze_audio, detect_tuning
    from app.services.constraints import normalise_constraints
    from app.services.draft import build_tab_draft
    from app.services.gp import create_guitar_pro_file
    from app.services.separation import separate_stems
    from app.services.storage import upload_to_storage
    from app.services.transcription import transcribe_pitched_instrument
    from app.services.youtube import extract_audio

    try:
        tab_constraints = normalise_constraints(constraints)

        _run_async(_update_job(job_id, status="extracting", progress=5))
        metadata = extract_audio(youtube_url, TEMP_DIR / job_id)
        audio_path = metadata["audio_path"]

        separation_message = (
            "Separating stems" if settings.separation_enabled else "Skipping separation (fast mode)"
        )
        _run_async(_update_job(job_id, status="separating", progress=20, message=separation_message))
        stems = separate_stems(audio_path, TEMP_DIR / job_id / "stems")

        _run_async(_update_job(job_id, status="analyzing", progress=35))
        detected_tempo, key = analyze_audio(audio_path)
        tempo = tab_constraints["tempo_bpm"] or detected_tempo

        transcription: dict = {
            "title": metadata["title"],
            "artist": metadata["artist"],
            "tempo": tempo,
            "detected_tempo": detected_tempo,
            "key": key,
            "constraints": tab_constraints,
            "time_signature": tab_constraints["time_signature"],
            "pickup_bar_beats": tab_constraints["pickup_bar_beats"],
            "triplet_feel": tab_constraints["triplet_feel"],
            "capo_fret": tab_constraints["capo_fret"],
        }

        detected_tuning = tuning
        if not tuning or tuning == "auto":
            _run_async(_update_job(job_id, status="analyzing", progress=37, message="Detecting tuning"))
            tuning_info = detect_tuning(stems.get("other", audio_path))
            detected_tuning = tuning_info["tuning"]
            transcription["tuning"] = detected_tuning
            transcription["tuning_info"] = tuning_info
        else:
            transcription["tuning"] = tuning

        progress_per_instrument = max(1, 40 // max(len(instruments), 1))

        for idx, instrument in enumerate(instruments):
            _run_async(
                _update_job(
                    job_id,
                    status="transcribing",
                    progress=40 + (idx * progress_per_instrument),
                    message=f"Transcribing {instrument}",
                )
            )

            if instrument == "drums":
                from app.services.drums import transcribe_drums

                transcription["drums"] = transcribe_drums(str(stems["drums"]), tempo)
            else:
                stem_path = stems["bass"] if instrument == "bass" else stems.get("guitar", stems["other"])
                transcription[instrument] = transcribe_pitched_instrument(
                    stem_path,
                    instrument,
                    tempo,
                    detected_tuning or "standard",
                    tab_constraints,
                )

        _run_async(_update_job(job_id, status="generating", progress=90))
        output_path = OUTPUT_DIR / f"{job_id}.gp5"
        transcription["source_stems"] = {name: path.name for name, path in stems.items()}
        draft = build_tab_draft(transcription, [str(instrument) for instrument in instruments])
        (OUTPUT_DIR / f"{job_id}.draft.json").write_text(
            json.dumps(draft, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        create_guitar_pro_file(transcription, str(output_path))

        try:
            download_url = upload_to_storage(output_path, f"{job_id}.gp5")
            completion_message = "Completed"
        except Exception as upload_exc:
            logger.warning("Storage upload failed for job %s: %s", job_id, upload_exc)
            download_url = None
            completion_message = "Completed; serving local download"

        _run_async(
            _update_job(
                job_id,
                status="completed",
                progress=100,
                download_url=download_url,
                title=metadata["title"],
                message=completion_message,
            )
        )

        cleanup_temp_files(job_id)
        return {
            "status": "completed",
            "progress": 100,
            "download_url": download_url,
            "title": metadata["title"],
        }

    except Exception as exc:
        logger.exception("Transcription failed for job %s", job_id)
        _run_async(_update_job(job_id, status="failed", progress=0, message=str(exc)))
        return {"status": "failed", "progress": 0, "message": str(exc)}
