"""FastAPI app: /health and the /ws/converse voice loop.

Turn ordering is strictly sequential by design — receive segment, transcribe,
extract, pick next question, speak it. Half-duplex, one interview at a time."""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import select

from app import extraction, questions, stt, triage, tts
from app.db import (
    Assessment,
    CaseEvent,
    Interview,
    Session,
    Turn,
    create_all,
    engine,
)

AUDIO_DIR = Path("data/audio")
MAX_SEGMENT_BYTES = 20 * 16000 * 2  # 20 s of PCM16 @ 16 kHz mono
MAX_ASKS = 3  # per question: initial ask + repeats; then the case is no-response


@asynccontextmanager
async def lifespan(app: FastAPI):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    tts.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    create_all()
    print("loading whisper model...")
    app.state.whisper = stt.load()
    print("whisper ready")
    yield


app = FastAPI(lifespan=lifespan)

# The dashboard runs on another port; no auth by design (trusted LAN, POC).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _case_summary(s: Session, interview: Interview) -> dict:
    latest = s.exec(
        select(Assessment)
        .where(Assessment.interview_id == interview.id)
        .order_by(Assessment.turn_seq.desc())
    ).first()
    fields = latest.fields if latest else {}
    result = triage.classify(fields, interview.status)
    last_turn = s.exec(
        select(Turn)
        .where(Turn.interview_id == interview.id)
        .order_by(Turn.seq.desc())
    ).first()

    # Responder actions: latest override/workflow event wins, notes accumulate.
    events = s.exec(
        select(CaseEvent)
        .where(CaseEvent.interview_id == interview.id)
        .order_by(CaseEvent.created_at, CaseEvent.id)
    ).all()
    override: str | None = None
    workflow = "outstanding"
    notes = [e for e in events if e.kind == "note"]
    for e in events:
        if e.kind == "override":
            override = None if e.value == "auto" else e.value
        elif e.kind == "workflow":
            workflow = e.value or "outstanding"

    category = override or result["category"]
    score = result["score"]
    if override:
        score = score - triage.BASE[result["category"]] + triage.BASE[category]

    return {
        "id": interview.id,
        "status": interview.status,
        "started_at": interview.started_at.isoformat(),
        "ended_at": interview.ended_at.isoformat() if interview.ended_at else None,
        "last_heard_at": (
            last_turn.created_at.isoformat() if last_turn else interview.started_at.isoformat()
        ),
        "turns": last_turn.seq if last_turn else 0,
        "fields": fields,
        "reasons": result["reasons"],
        "unknown_gates": result["unknown_gates"],
        "auto_category": result["category"],
        "category": category,
        "overridden": override is not None,
        "score": score,
        "workflow": workflow,
        "notes_count": len(notes),
        "latest_note": notes[-1].note if notes else None,
    }


@app.get("/api/dashboard")
def dashboard():
    with Session(engine) as s:
        cases = [_case_summary(s, i) for i in s.exec(select(Interview)).all()]
    # Rescued cases sink below everything still outstanding.
    cases.sort(
        key=lambda c: (c["workflow"] == "rescued", -c["score"], c["last_heard_at"])
    )

    counts = {"immediate": 0, "delayed": 0, "minor": 0}
    for c in cases:
        if c["workflow"] != "rescued":
            counts[c["category"]] += 1
    unaccounted = []
    for c in cases:
        people = c["fields"].get("people_in_building")
        if people is not None and people > 1:
            unaccounted.append(
                {
                    "case_id": c["id"],
                    "others": people - 1,
                    "last_seen": c["fields"].get("others_last_seen"),
                    "category": c["category"],
                }
            )
    return {
        "cases": cases,
        "counts": counts,
        "active": sum(1 for c in cases if c["status"] == "active"),
        "no_response": sum(1 for c in cases if c["status"] == "no_response"),
        "rescued": sum(1 for c in cases if c["workflow"] == "rescued"),
        "people_reported": sum(
            c["fields"].get("people_in_building") or 0 for c in cases
        ),
        "others_unaccounted": sum(u["others"] for u in unaccounted),
        "unaccounted": unaccounted,
    }


class EventIn(BaseModel):
    kind: str  # override | workflow | note
    value: str | None = None
    note: str | None = None


_EVENT_VALUES = {
    "override": {"immediate", "delayed", "minor", "auto"},
    "workflow": {"outstanding", "dispatched", "rescued"},
}


@app.post("/api/cases/{case_id}/events")
def add_event(case_id: str, event: EventIn):
    if event.kind in _EVENT_VALUES:
        if event.value not in _EVENT_VALUES[event.kind]:
            raise HTTPException(400, f"invalid value for {event.kind}: {event.value}")
    elif event.kind == "note":
        if not (event.note and event.note.strip()):
            raise HTTPException(400, "note text required")
    else:
        raise HTTPException(400, f"unknown event kind: {event.kind}")

    with Session(engine) as s:
        if s.get(Interview, case_id) is None:
            raise HTTPException(404, "case not found")
        s.add(
            CaseEvent(
                interview_id=case_id,
                kind=event.kind,
                value=event.value,
                note=event.note.strip() if event.note else None,
            )
        )
        s.commit()
        interview = s.get(Interview, case_id)
        summary = _case_summary(s, interview)
    print(f"[{case_id}] responder event: {event.kind}={event.value or event.note!r}")
    return summary


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str):
    with Session(engine) as s:
        interview = s.get(Interview, case_id)
        if interview is None:
            return {"error": "not found"}
        turns = s.exec(
            select(Turn).where(Turn.interview_id == case_id).order_by(Turn.seq)
        ).all()
        assessments = s.exec(
            select(Assessment)
            .where(Assessment.interview_id == case_id)
            .order_by(Assessment.turn_seq)
        ).all()
        events = s.exec(
            select(CaseEvent)
            .where(CaseEvent.interview_id == case_id)
            .order_by(CaseEvent.created_at, CaseEvent.id)
        ).all()
        summary = _case_summary(s, interview)
    return {
        **summary,
        "events": [
            {
                "kind": e.kind,
                "value": e.value,
                "note": e.note,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
        "timeline": [
            {
                "seq": t.seq,
                "question": t.question_text,
                "answer": t.transcript,
                "stt_ms": t.stt_ms,
                "fields_after": next(
                    (a.fields for a in assessments if a.turn_seq == t.seq), None
                ),
            }
            for t in turns
        ],
    }


async def send_question(
    ws: WebSocket, question: questions.Question, seq: int, attempt: int = 1
) -> None:
    await ws.send_text(
        json.dumps(
            {
                "t": "question",
                "question_id": question.id,
                "seq": seq,
                "text": question.text,
                "sample_rate": tts.SAMPLE_RATE,
                "attempt": attempt,
            }
        )
    )
    try:
        pcm = tts.speak(question.text)
        for i in range(0, len(pcm), tts.CHUNK_BYTES):
            await ws.send_bytes(pcm[i : i + tts.CHUNK_BYTES])
    except Exception as exc:
        print(f"[tts] failed for {question.id}: {exc}")
        await ws.send_text(json.dumps({"t": "error", "message": f"tts failed: {exc}"}))
    await ws.send_text(json.dumps({"t": "question_end"}))


@app.websocket("/ws/converse")
async def converse(ws: WebSocket):
    await ws.accept()
    interview_id = uuid.uuid4().hex[:8]
    fields: dict = {f: None for f in questions.FIELD_ORDER}
    asked: list[str] = []
    current_q: questions.Question | None = None
    seq = 0
    asks = 0
    started = False
    final_status: str | None = None  # None -> abandoned
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                ctrl = json.loads(msg["text"])
                if ctrl.get("t") == "start" and not started:
                    started = True
                    with Session(engine) as s:
                        s.add(Interview(id=interview_id))
                        s.commit()
                    print(f"[{interview_id}] interview started")
                    current_q = questions.next_question(fields, asked)
                    asked.append(current_q.id)
                    asks = 1
                    await send_question(ws, current_q, seq + 1)
                elif ctrl.get("t") == "repeat":
                    # Client heard nothing for ~5 s. Re-ask, up to MAX_ASKS,
                    # then the case is no-response.
                    if not started or current_q is None:
                        await ws.send_text(
                            json.dumps({"t": "error", "message": "send start first"})
                        )
                    elif asks < MAX_ASKS:
                        asks += 1
                        print(
                            f"[{interview_id}] no response, repeating "
                            f"{current_q.id} (attempt {asks}/{MAX_ASKS})"
                        )
                        await send_question(ws, current_q, seq + 1, attempt=asks)
                    else:
                        final_status = "no_response"
                        print(
                            f"[{interview_id}] no response after {asks} asks "
                            f"of {current_q.id} — marking case no-response"
                        )
                        await ws.send_text(
                            json.dumps(
                                {"t": "no_response", "interview_id": interview_id}
                            )
                        )
                        break
                elif ctrl.get("t") == "end":
                    break

            elif msg.get("bytes"):
                pcm = msg["bytes"]
                if not started or current_q is None:
                    await ws.send_text(
                        json.dumps({"t": "error", "message": "send start first"})
                    )
                    continue
                if len(pcm) > MAX_SEGMENT_BYTES:
                    await ws.send_text(
                        json.dumps(
                            {"t": "error", "message": "segment longer than 20 s ignored"}
                        )
                    )
                    continue

                seq += 1
                audio_path = AUDIO_DIR / f"turn_{interview_id}_{seq}.pcm"
                audio_path.write_bytes(pcm)

                transcript, stt_ms = stt.transcribe(pcm)
                print(f"[{interview_id}] turn {seq} stt {stt_ms} ms: {transcript!r}")
                await ws.send_text(
                    json.dumps({"t": "transcript", "seq": seq, "text": transcript})
                )
                with Session(engine) as s:
                    s.add(
                        Turn(
                            interview_id=interview_id,
                            seq=seq,
                            question_id=current_q.id,
                            question_text=current_q.text,
                            transcript=transcript,
                            audio_path=str(audio_path),
                            stt_ms=stt_ms,
                        )
                    )
                    s.commit()

                fields = extraction.extract(transcript, current_q.id, fields)
                known = {k: v for k, v in fields.items() if v is not None}
                unknown = [k for k, v in fields.items() if v is None]
                print(f"[{interview_id}] turn {seq} fields known={known} unknown={unknown}")
                await ws.send_text(
                    json.dumps({"t": "fields", "known": known, "unknown": unknown})
                )
                with Session(engine) as s:
                    s.add(
                        Assessment(
                            interview_id=interview_id, turn_seq=seq, fields=fields
                        )
                    )
                    s.commit()

                next_q = questions.next_question(fields, asked)
                if next_q is None:
                    final_status = "complete"
                    await ws.send_text(
                        json.dumps({"t": "complete", "interview_id": interview_id})
                    )
                    print(f"[{interview_id}] interview complete after turn {seq}")
                    break
                current_q = next_q
                asked.append(next_q.id)
                asks = 1
                await send_question(ws, next_q, seq + 1)
    except WebSocketDisconnect:
        pass
    finally:
        if started:
            # The only UPDATE in the application: closing out the interview.
            status = final_status or "abandoned"
            with Session(engine) as s:
                interview = s.get(Interview, interview_id)
                interview.status = status
                interview.ended_at = datetime.utcnow()
                s.add(interview)
                s.commit()
            if status == "abandoned":
                print(f"[{interview_id}] interview abandoned at turn {seq}")
