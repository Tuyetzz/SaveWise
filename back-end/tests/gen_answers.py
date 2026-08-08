"""Generate spoken casualty answers as PCM16 16 kHz mono, using the project's
own TTS module (24 kHz) resampled down. Run from back-end/ with uv run."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from app import tts  # noqa: E402

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/answers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANSWERS = {
    "q_trapped": "My legs are pinned under something heavy. I cannot move them at all.",
    "q_breathing": "Yes, I can breathe okay, but there is a lot of dust down here.",
    "q_respiratory_rate": "I counted like you said, and I got six breaths in fifteen seconds.",
    "q_radial_pulse": "Yes, I can feel my pulse. It is beating quite fast.",
    "q_obeys_commands": "I am alert. I am just really scared.",
    "q_can_walk": "I do not think so. My ankle is crushed and it hurts a lot.",
    "q_injuries": "My left ankle is crushed, and my head is bleeding a little.",
    "q_people_in_building": "There were four of us in the house when it came down.",
    "q_others_last_seen": "My two kids were upstairs in their bedroom, and my husband was in the kitchen.",
}


def resample_24k_to_16k(pcm24: bytes) -> bytes:
    audio = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
    n_out = int(len(audio) * 16000 / 24000)
    x_out = np.linspace(0, len(audio) - 1, n_out)
    resampled = np.interp(x_out, np.arange(len(audio)), audio)
    return resampled.astype(np.int16).tobytes()


for qid, text in ANSWERS.items():
    pcm24 = tts.speak(text)
    pcm16k = resample_24k_to_16k(pcm24)
    out = OUT_DIR / f"{qid}.pcm"
    out.write_bytes(pcm16k)
    print(f"{qid}: {len(pcm16k) / 32000:.1f} s -> {out}")
