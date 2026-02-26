from __future__ import annotations

from typing import Any, Dict, List, Optional


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_speaker_to_span(
    start: float, end: float, diar: List[Dict], idx_hint: int = 0
) -> tuple[Optional[str], int]:
    """
    Two-pointer scan over diarization segments.
    Returns (best_speaker, next_idx_hint).
    """
    best_speaker = None
    best_ov = 0.0

    i = idx_hint
    n = len(diar)

    # advance to first diar segment that could overlap
    while i < n and diar[i]["end"] <= start:
        i += 1

    j = i
    while j < n and diar[j]["start"] < end:
        ov = _overlap(start, end, diar[j]["start"], diar[j]["end"])
        if ov > best_ov:
            best_ov = ov
            best_speaker = diar[j]["speaker"]
        j += 1

    return best_speaker, i


def merge_asr_with_diarization(asr: Dict[str, Any], diar_segments: List[Dict]) -> Dict[str, Any]:
    """
    Adds speaker labels to ASR segments and words by maximum overlap.
    """
    if not diar_segments:
        return asr

    merged = dict(asr)
    merged["diarization"] = diar_segments

    out_segments: List[Dict[str, Any]] = []
    diar_i = 0

    for seg in asr.get("segments", []):
        s0 = float(seg["start"])
        s1 = float(seg["end"])
        speaker, diar_i = _assign_speaker_to_span(s0, s1, diar_segments, diar_i)

        seg2 = dict(seg)
        seg2["speaker"] = speaker

        # word-level speaker
        if "words" in seg2 and isinstance(seg2["words"], list):
            wi = diar_i
            new_words = []
            for w in seg2["words"]:
                w0 = float(w["start"])
                w1 = float(w["end"])
                ws, wi = _assign_speaker_to_span(w0, w1, diar_segments, wi)
                w2 = dict(w)
                w2["speaker"] = ws
                new_words.append(w2)
            seg2["words"] = new_words

        out_segments.append(seg2)

    merged["segments"] = out_segments
    return merged
