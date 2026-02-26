from __future__ import annotations

import secrets


def new_job_id() -> str:
    # 128-bit urlsafe id
    return secrets.token_urlsafe(16)
