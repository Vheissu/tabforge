from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job


async def create_job(
    session: AsyncSession,
    job_id: str,
    youtube_url: str,
    instruments: list,
    tuning: str,
    request_ip: str | None,
    celery_task_id: str | None,
) -> Job:
    job = Job(
        id=job_id,
        youtube_url=youtube_url,
        instruments=instruments,
        tuning=tuning,
        status="pending",
        progress=0,
        request_ip=request_ip,
        celery_task_id=celery_task_id,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, job_id: str) -> Job | None:
    result = await session.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def update_job(
    session: AsyncSession,
    job_id: str,
    **fields,
) -> Job | None:
    job = await get_job(session, job_id)
    if not job:
        return None
    for key, value in fields.items():
        setattr(job, key, value)
    await session.commit()
    await session.refresh(job)
    return job
