from __future__ import annotations

from typing import List, Dict, Protocol


class Diarizer(Protocol):
    def diarize(self, wav_path: str) -> List[Dict]:
        """
        Returns list of diarization segments:
        [
          {"start": 0.12, "end": 3.40, "speaker": "SPEAKER_00"},
          ...
        ]
        """
