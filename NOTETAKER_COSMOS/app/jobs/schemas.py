from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class SubmitResponse(BaseModel):
    job_id: str
    status: str
    queue: str
    result: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    job_id: str
    status: str
    queue: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
