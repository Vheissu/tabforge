from __future__ import annotations

import time
from fastapi import HTTPException, Request
import redis.asyncio as redis

from app.core.config import get_settings

settings = get_settings()

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


async def rate_limit(request: Request) -> None:
    """Simple per-IP rate limit using Redis INCR with 1h TTL."""
    client = get_redis()
    ip = request.client.host if request.client else "unknown"
    key = f"rate:{ip}:{int(time.time() // 3600)}"
    count = await client.incr(key)
    if count == 1:
        await client.expire(key, 3600)
    if count > settings.rate_limit_per_hour:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
