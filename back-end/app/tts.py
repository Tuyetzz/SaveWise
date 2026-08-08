"""OpenAI TTS with an on-disk cache. The question set is finite and
deterministic, so after one run every question is cached and TTS leaves the
critical path on its own."""

import hashlib
import os
from pathlib import Path

from openai import OpenAI

_client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

CACHE_DIR = Path("data/tts_cache")

# response_format="pcm": 24 kHz, 16-bit, mono, no container.
SAMPLE_RATE = 24000
CHUNK_BYTES = 1920  # ~40 ms at 24 kHz 16-bit mono

VOICE = "coral"
INSTRUCTIONS = (
    "You are a calm, experienced first responder speaking to a survivor "
    "trapped under rubble. Warm, steady, reassuring tone — like a paramedic "
    "keeping someone safe and calm. Speak slowly and clearly with short "
    "natural pauses. Quiet confidence; never robotic, never panicked."
)


def speak(text: str) -> bytes:
    # Voice and instructions are part of the key — changing them must not
    # serve stale audio.
    key = hashlib.sha1(f"{VOICE}|{INSTRUCTIONS}|{text}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.pcm"
    if cache_path.exists():
        return cache_path.read_bytes()
    response = _client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=VOICE,
        input=text,
        instructions=INSTRUCTIONS,
        response_format="pcm",
    )
    pcm = response.content
    cache_path.write_bytes(pcm)
    return pcm
