from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TabForge API"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://tabforge:tabforge@postgres:5432/tabforge"
    redis_url: str = "redis://redis:6379/0"
    redis_result_url: str | None = None

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket: str = "tabforge"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3-flash-preview"
    gemini_refinement_timeout_seconds: int = 45

    max_duration_seconds: int = 600
    rate_limit_per_hour: int = 10
    task_time_limit_seconds: int = 1800
    task_soft_time_limit_seconds: int = 1500

    separation_enabled: bool = True
    separation_model: str = "htdemucs_ft"
    separation_segment_seconds: float = 7.8

    temp_dir: str = "./temp"
    output_dir: str = "./output"

    allow_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
