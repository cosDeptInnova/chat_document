from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_wav_16k_mono(input_path: Path, output_path: Path, ffmpeg_path: str = "ffmpeg") -> None:
    """
    Converts arbitrary input audio/video to 16kHz mono WAV PCM.
    Robust for mp3/m4a/mp4/webm etc. Uses ffmpeg sniffing (works even if input is .bin).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",  # drop video/subtitles/data streams
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if p.returncode != 0:
        # show last part of stderr (usually includes codec/format errors)
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-4000:]}")
