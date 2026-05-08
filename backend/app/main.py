from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.routes import router
from app.core.config import get_settings
from app.db.models import Base
from app.db.session import engine

settings = get_settings()


def _cors_origins(raw_origins: str) -> list[str]:
    if raw_origins.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await _init_db(engine)
    yield


app = FastAPI(
    title="TabForge API",
    description="AI-powered music transcription to Guitar Pro tabs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(settings.allow_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


async def _init_db(db_engine: AsyncEngine) -> None:
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
