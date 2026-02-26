from __future__ import annotations

from redis import Redis
from rq import Queue

from app.core.settings import Settings


def get_redis_conn(settings: Settings) -> Redis:
    return Redis.from_url(settings.redis_url)


def get_rq_queue(settings: Settings, redis_conn: Redis, queue_name: str) -> Queue:
    return Queue(
        name=queue_name,
        connection=redis_conn,
        default_timeout=settings.job_timeout_seconds,
    )
