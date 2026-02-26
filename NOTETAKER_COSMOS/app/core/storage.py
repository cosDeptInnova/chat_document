from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import UploadFile

from app.core.settings import Settings
from app.utils.ids import new_job_id
from app.utils.jsonx import read_json, write_json


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = settings.data_dir
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def new_job_id(self) -> str:
        return new_job_id()

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def save_upload(self, file: UploadFile, dest: Path) -> None:
        # stream to disk
        with dest.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    def result_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "result.json"

    def write_result_json(self, job_id: str, obj: Dict[str, Any]) -> None:
        write_json(self.result_path(job_id), obj)

    def read_result_json(self, job_id: str) -> Dict[str, Any]:
        p = self.result_path(job_id)
        if not p.exists():
            return {}
        return read_json(p)
