"""faster-whisper wrapper. Model loads once in the FastAPI lifespan; blocking
the event loop during transcription is fine with a single caller."""

import os
import time
from pathlib import Path

# ~/.cache/huggingface is root-owned on this server (old docker run), so the
# default download location fails with EACCES. Must be set before importing
# faster_whisper, which reads it at import time.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "faster-whisper"))

import numpy as np
from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def load() -> WhisperModel:
    """Load the model and warm it with one second of silence so the first real
    turn does not pay CUDA initialisation."""
    global _model
    _model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
    warmup = np.zeros(16000, dtype=np.float32)
    segments, _ = _model.transcribe(warmup, language="en")
    list(segments)  # generator — consume to actually run it
    return _model


def transcribe(pcm: bytes) -> tuple[str, int]:
    """PCM16 16 kHz mono -> (transcript, elapsed ms)."""
    start = time.perf_counter()
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = _model.transcribe(audio, language="en", beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return text, elapsed_ms
