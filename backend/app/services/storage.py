from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from minio import Minio

from app.core.config import get_settings

settings = get_settings()


def get_client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket() -> None:
    client = get_client()
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)


def upload_to_storage(file_path: Path, object_name: str) -> str:
    ensure_bucket()
    client = get_client()
    client.fput_object(settings.minio_bucket, object_name, str(file_path))
    return client.presigned_get_object(settings.minio_bucket, object_name, expires=timedelta(days=1))
