from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from app.core.config import get_settings
from app.core.limiter import rate_limit
from app.db.crud import create_job, get_job
from app.db.session import get_session
from app.schemas import JobResponse, JobStatus, TranscriptionRequest
from app.services.youtube import validate_youtube_url
from app.tasks import celery_app

settings = get_settings()
router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/transcribe", response_model=JobResponse, dependencies=[Depends(rate_limit)])
async def create_transcription(
    request_data: TranscriptionRequest,
    request: Request,
    session=Depends(get_session),
) -> JobResponse:
    try:
        validate_youtube_url(str(request_data.youtube_url), settings.max_duration_seconds)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())

    task = celery_app.send_task(
        "app.tasks.process_transcription",
        kwargs={
            "job_id": job_id,
            "youtube_url": str(request_data.youtube_url),
            "instruments": [i.value for i in request_data.instruments],
            "tuning": request_data.tuning or "auto",
        },
    )

    await create_job(
        session,
        job_id=job_id,
        youtube_url=str(request_data.youtube_url),
        instruments=[i.value for i in request_data.instruments],
        tuning=request_data.tuning or "auto",
        request_ip=request.client.host if request.client else None,
        celery_task_id=task.id,
    )

    return JobResponse(job_id=job_id, status=JobStatus.pending, progress=0, message="Job queued")


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, session=Depends(get_session)) -> JobResponse:
    job = await get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        progress=job.progress,
        message=job.message,
        download_url=job.download_url,
        title=job.title,
    )


@router.get("/download/{job_id}")
async def download_file(job_id: str, session=Depends(get_session)):
    job = await get_job(session, job_id)
    if not job or job.status != JobStatus.completed.value:
        raise HTTPException(status_code=404, detail="File not ready or job not found")

    if job.download_url:
        return RedirectResponse(job.download_url)

    local_path = Path(settings.output_dir) / f"{job_id}.gp5"
    if not local_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=local_path,
        media_type="application/octet-stream",
        filename=f"{job.title or job_id}.gp5",
    )
