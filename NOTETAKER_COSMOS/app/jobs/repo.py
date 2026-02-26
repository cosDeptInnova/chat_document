from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from redis import Redis

from app.core.settings import Settings


@dataclass(frozen=True)
class JobStatus:
    status: str
    queue: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class JobRepo:
    def __init__(self, redis_conn: Redis, settings: Settings):
        self.redis = redis_conn
        self.settings = settings

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def init_job(self, job_id: str, queue: str, input_path: str, opts: Dict[str, Any]) -> None:
        now = time.time()
        self.redis.hset(
            self._key(job_id),
            mapping={
                "status": "queued",
                "queue": queue,
                "created_at": now,
                "input_path": input_path,
                "opts_json": __import__("json").dumps(opts),
            },
        )
        self.redis.expire(self._key(job_id), self.settings.result_ttl_seconds)

    def mark_started(self, job_id: str) -> None:
        self.redis.hset(self._key(job_id), mapping={"status": "running", "started_at": time.time()})

    def mark_finished(self, job_id: str) -> None:
        self.redis.hset(self._key(job_id), mapping={"status": "finished", "finished_at": time.time()})

    def mark_failed(self, job_id: str, error: str) -> None:
        self.redis.hset(
            self._key(job_id),
            mapping={"status": "failed", "finished_at": time.time(), "error": error},
        )

    def get_status(self, job_id: str) -> JobStatus:
        d = self.redis.hgetall(self._key(job_id))
        if not d:
            return JobStatus(status="not_found")

        def _f(k: str) -> Optional[float]:
            v = d.get(k.encode("utf-8"))
            if not v:
                return None
            try:
                return float(v.decode("utf-8"))
            except Exception:
                return None

        def _s(k: str) -> Optional[str]:
            v = d.get(k.encode("utf-8"))
            return v.decode("utf-8") if v else None

        return JobStatus(
            status=_s("status") or "unknown",
            queue=_s("queue"),
            created_at=_f("created_at"),
            started_at=_f("started_at"),
            finished_at=_f("finished_at"),
            error=_s("error"),
        )
