from __future__ import annotations

import os
from typing import Dict, List, Optional

from app.core.logging import logger


class PyannoteDiarizer:
    """
    Offline diarization using pyannote.audio.

    - pipeline_path debe apuntar a un DIRECTORIO de pipeline descargado (recomendado),
      o a un repo clonado localmente (pyannote/speaker-diarization-...).
    - Para evitar problemas con torchcodec/ffmpeg en Windows, cargamos el wav a memoria
      y se lo pasamos al pipeline como {"waveform": ..., "sample_rate": ...}.
    - Diarización en CPU (para no consumir VRAM; tus GPUs están para LLMs/ASR).
    """

    def __init__(self, pipeline_path: Optional[str], device: str = "cpu", enabled: bool = True):
        self.enabled = enabled
        self.pipeline_path = pipeline_path
        # Forzamos CPU por requisitos de VRAM/convivencia con LLMs
        self.device = "cpu"
        self._pipeline = None
        self._torch = None  # se setea en _load()

        if self.enabled and not self.pipeline_path:
            raise RuntimeError(
                "Diarization enabled but PYANNOTE_PIPELINE_PATH is not set. "
                "Provide a local path to a pyannote speaker diarization pipeline directory."
            )

    def _load(self):
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
            import torch
        except Exception as e:
            raise RuntimeError(
                "pyannote.audio not installed. Install requirements_diarization.txt"
            ) from e

        self._torch = torch

        # Limita CPU (evita que pyannote se coma la máquina)
        num_threads = int(os.getenv("PYANNOTE_NUM_THREADS", "4"))
        try:
            torch.set_num_threads(num_threads)
            logger.info("pyannote torch.set_num_threads(%s)", num_threads)
        except Exception:
            pass

        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        logger.info("Loading pyannote pipeline from local path: %s", self.pipeline_path)

        # En pyannote.audio >=4, use_auth_token fue renombrado a token. :contentReference[oaicite:2]{index=2}
        # Si pipeline_path es local, token normalmente no hace falta, pero ayuda si el pipeline
        # referencia repos gated y estás descargando la primera vez.
        kwargs = {}
        if token:
            kwargs["token"] = token

        self._pipeline = Pipeline.from_pretrained(self.pipeline_path, **kwargs)

        # Forzamos CPU siempre (requisito VRAM)
        try:
            self._pipeline.to(torch.device("cpu"))
            logger.info("pyannote pipeline loaded on CPU")
        except Exception:
            logger.info("pyannote pipeline loaded (device move skipped)")

    def _load_wav_in_memory(self, wav_path: str) -> Dict:
        """
        Carga WAV a memoria como {'waveform': Tensor(channels, time), 'sample_rate': int}
        usando soundfile -> torchaudio -> wave (fallback).
        """
        torch = self._torch
        assert torch is not None

        # 1) soundfile (rápido para wav)
        try:
            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # (time, channels)
            audio = np.transpose(audio, (1, 0))  # (channels, time)
            waveform = torch.from_numpy(audio)
            return {"waveform": waveform, "sample_rate": int(sr)}
        except Exception:
            pass

        # 2) torchaudio
        try:
            import torchaudio

            waveform, sr = torchaudio.load(wav_path)  # (channels, time)
            # asegura float32
            if waveform.dtype != torch.float32:
                waveform = waveform.float()
            return {"waveform": waveform, "sample_rate": int(sr)}
        except Exception:
            pass

        # 3) wave (stdlib) fallback para WAV PCM
        import wave
        import numpy as np

        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
            scale = 32768.0
        elif sampwidth == 4:
            dtype = np.int32
            scale = 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sampwidth}")

        data = np.frombuffer(raw, dtype=dtype).astype("float32") / scale
        data = data.reshape(-1, n_channels).T  # (channels, time)
        waveform = torch.from_numpy(data)
        return {"waveform": waveform, "sample_rate": int(sr)}

    def diarize(self, wav_path: str) -> List[Dict]:
        if not self.enabled:
            return []

        self._load()

        try:
            # Evita dependencia de torchcodec/ffmpeg: pasamos audio en memoria
            file_dict = self._load_wav_in_memory(wav_path)
            output = self._pipeline(file_dict)

            # pyannote.audio v4 devuelve DiarizeOutput con speaker_diarization :contentReference[oaicite:3]{index=3}
            if hasattr(output, "speaker_diarization"):
                annotation = output.speaker_diarization
            elif hasattr(output, "diarization"):
                annotation = output.diarization
            else:
                # compat antigua: output ya es Annotation
                annotation = output

            if not hasattr(annotation, "itertracks"):
                raise RuntimeError(
                    f"Unexpected diarization output type: {type(output)} / annotation: {type(annotation)}"
                )

            segments: List[Dict] = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
                )

            segments.sort(key=lambda x: (x["start"], x["end"]))
            return segments

        except Exception as e:
            raise RuntimeError(f"pyannote diarization failed: {e}") from e
