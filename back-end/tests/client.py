"""Milestone test client for /ws/converse.

Connects, sends start, answers each question with a pre-generated PCM16 16 kHz
segment (answers/<question_id>.pcm), prints every protocol message. If no
answer file exists, sends 2 s of silence."""

import asyncio
import json
import sys
from pathlib import Path

import websockets

ANSWER_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/answers")
URL = "ws://127.0.0.1:8000/api/ws/converse"
SILENCE_2S = b"\x00" * (2 * 16000 * 2)


async def run() -> None:
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({"t": "start"}))
        audio_bytes = 0
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                audio_bytes += len(msg)
                continue
            data = json.loads(msg)
            t = data.get("t")
            if t == "question":
                print(f"<- question {data['question_id']} seq={data['seq']}: {data['text']!r}")
                audio_bytes = 0
            elif t == "question_end":
                print(f"<- question_end ({audio_bytes} audio bytes, "
                      f"{audio_bytes / 48000:.1f} s @ 24 kHz)")
                qid = current_qid[0]
                answer = ANSWER_DIR / f"{qid}.pcm"
                pcm = answer.read_bytes() if answer.exists() else SILENCE_2S
                print(f"-> answer for {qid}: {len(pcm) / 32000:.1f} s")
                await ws.send(pcm)
            elif t == "transcript":
                print(f"<- transcript seq={data['seq']}: {data['text']!r}")
            elif t == "fields":
                print(f"<- fields known={data['known']} unknown={data['unknown']}")
            elif t == "complete":
                print(f"<- complete interview_id={data['interview_id']}")
                return
            elif t == "error":
                print(f"<- error: {data['message']}")
            if t == "question":
                current_qid[0] = data["question_id"]


current_qid = [None]

asyncio.run(run())
