from __future__ import annotations

from pathlib import Path
from typing import Optional

CONTENT_TYPE_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
}


def pick_extension(filename: Optional[str], content_type: Optional[str], allowed_exts: list[str]) -> str:
    """
    Decide una extensión fiable para guardar el upload.
    - Usa el sufijo del filename si es válido
    - Si no, usa content-type (audio/mpeg => .mp3)
    - Si no, usa .bin (ffmpeg suele poder detectar igual)
    """
    fn = filename or ""
    ext = Path(fn).suffix.lower() if fn else ""

    if ext in allowed_exts:
        return ext

    ct = (content_type or "").split(";")[0].strip().lower()
    ext2 = CONTENT_TYPE_TO_EXT.get(ct, "")

    if ext2 in allowed_exts:
        return ext2

    # Sin extensión o desconocida: guarda como .bin y deja que ffmpeg lo intente
    return ext if ext else ".bin"
