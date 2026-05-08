from __future__ import annotations

from app.core.config import get_settings
from celery import Celery


def _default_result_backend(redis_url: str) -> str:
    if "/" not in redis_url:
        return redis_url

    prefix, database = redis_url.rsplit("/", 1)
    if database.isdigit():
        return f"{prefix}/1"
    return redis_url


settings = get_settings()

celery_app = Celery(
    "tabforge",
    broker=settings.redis_url,
    backend=settings.redis_result_url or _default_result_backend(settings.redis_url),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.task_time_limit_seconds,
    task_soft_time_limit=settings.task_soft_time_limit_seconds,
)
