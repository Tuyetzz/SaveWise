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

from app import extraction, interviewer, questions, rover, stt, triage, tts
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


def run() -> None:
    """`uv run dev` entrypoint. No --reload: it would re-load the Whisper
    model on every save.

    Loopback only: the nginx proxy at hackathon.marcusnguyen.dev is the sole
    public entry point — the backend must not be reachable on :8000 from
    outside the box."""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)


app = FastAPI(lifespan=lifespan)

# The dashboard runs on another port; no auth by design (trusted LAN, POC).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Rover relays: phone video -> detection app, operator commands -> Pi.
app.include_router(rover.router)


# Every route (HTTP and WS) lives under /api — the reverse proxy at
# hackathon.marcusnguyen.dev routes /api/* here and everything else to the
# Next.js frontend.
@app.get("/api/health")
def health():
    return {"status": "ok"}


class DashboardHub:
    """Fan-out of case-change notifications to open dashboards. Payloads are
    intentionally just pointers ({t, case_id}) — clients refetch, so the
    HTTP endpoints stay the single source of truth."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def broadcast(self, payload: dict) -> None:
        message = json.dumps(payload)
        for ws in list(self.clients):
            try:
                await ws.send_text(message)
            except Exception:
                self.clients.discard(ws)


hub = DashboardHub()


@app.websocket("/api/ws/dashboard")
async def dashboard_ws(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # nothing expected; detects disconnect
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(ws)


def _case_summary(
    s: Session, interview: Interview, fields_override: dict | None = None
) -> dict:
    """fields_override replays the summary as of an earlier assessment
    snapshot (used to compute a case's previous queue position)."""
    if fields_override is not None:
        fields = fields_override
    else:
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


def _ranked(cases: list[dict]) -> list[dict]:
    # Rescued cases sink below everything still outstanding.
    return sorted(
        cases,
        key=lambda c: (c["workflow"] == "rescued", -c["score"], c["last_heard_at"]),
    )


@app.get("/api/dashboard")
def dashboard():
    with Session(engine) as s:
        cases = _ranked([_case_summary(s, i) for i in s.exec(select(Interview)).all()])

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
async def add_event(case_id: str, event: EventIn):
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
    await hub.broadcast({"t": "case_updated", "case_id": case_id})
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


def _audio_ms(audio_path: str | None) -> int | None:
    """Duration of a stored answer segment. Segments are raw PCM16 mono
    @ 16 kHz, so ms = bytes / 32."""
    if not audio_path:
        return None
    try:
        return Path(audio_path).stat().st_size // 32
    except OSError:
        return None


@app.get("/api/cases/{case_id}/reasoning")
def case_reasoning(case_id: str):
    """The full decision chain for the latest assessment: what was heard,
    what the LLM extracted, which deterministic rule fired, and how the case
    moved in the queue. Everything is recomputed from stored turns and
    assessment snapshots — nothing here is narrated after the fact."""
    with Session(engine) as s:
        interview = s.get(Interview, case_id)
        if interview is None:
            raise HTTPException(404, "case not found")

        turns = s.exec(
            select(Turn).where(Turn.interview_id == case_id).order_by(Turn.seq)
        ).all()
        assessments = s.exec(
            select(Assessment)
            .where(Assessment.interview_id == case_id)
            .order_by(Assessment.turn_seq)
        ).all()
        latest = assessments[-1] if assessments else None
        previous = assessments[-2] if len(assessments) >= 2 else None
        fields = latest.fields if latest else {f: None for f in questions.FIELD_ORDER}
        prev_fields = previous.fields if previous else None

        summaries = [_case_summary(s, i) for i in s.exec(select(Interview)).all()]
        ranked = _ranked(summaries)
        current_position = next(
            i + 1 for i, c in enumerate(ranked) if c["id"] == case_id
        )
        summary = ranked[current_position - 1]

        # Previous position: replay the queue with this case's previous
        # assessment snapshot, everyone else unchanged.
        prev_position = None
        moved_ahead_of: list[str] = []
        prev_result = None
        if previous is not None:
            prev_summary = _case_summary(s, interview, fields_override=prev_fields)
            prev_ranked = _ranked(
                [c for c in summaries if c["id"] != case_id] + [prev_summary]
            )
            prev_position = next(
                i + 1 for i, c in enumerate(prev_ranked) if c["id"] == case_id
            )
            above_before = {c["id"] for c in prev_ranked[: prev_position - 1]}
            below_now = {c["id"] for c in ranked[current_position:]}
            moved_ahead_of = [
                c["id"] for c in ranked[current_position:] if c["id"] in above_before
            ] if above_before & below_now else []
            prev_result = triage.classify(prev_fields, "complete")

    result = triage.classify(fields, interview.status)

    field_entries = {}
    for name in questions.FIELD_ORDER:
        value = fields.get(name)
        prev_value = (prev_fields or {}).get(name)
        field_entries[name] = {
            "value": value,
            "known": value is not None,
            # On the very first assessment everything the LLM filled in is new.
            "changed_this_turn": (
                value != prev_value if previous is not None else value is not None
            ),
        }
    known_count = sum(1 for e in field_entries.values() if e["known"])

    return {
        "case": {
            "id": interview.id,
            "status": interview.status,
            "started_at": interview.started_at.isoformat(),
            "ended_at": interview.ended_at.isoformat() if interview.ended_at else None,
            "workflow": summary["workflow"],
        },
        "assessment": {
            "version": latest.turn_seq if latest else 0,
            "source": "llm_extraction",
            "created_at": latest.created_at.isoformat() if latest else None,
            "fields": field_entries,
            "chief_complaint": fields.get("injuries"),
            "completeness_pct": round(100 * known_count / len(questions.FIELD_ORDER)),
            "unknown_fields": [n for n, e in field_entries.items() if not e["known"]],
        },
        "transcript": {
            "turns": [
                {
                    "index": t.seq,
                    "question_text": t.question_text,
                    "transcript": t.transcript,
                    "audio_duration_ms": _audio_ms(t.audio_path),
                }
                for t in turns
            ]
        },
        "decision": {
            "category": result["category"],
            "previous_category": prev_result["category"] if prev_result else None,
            "changed_category": (
                prev_result is not None
                and prev_result["category"] != result["category"]
            ),
            "rule_fired": result["rule_fired"],
            "trace": result["trace"],
            "unknown_escalated": result["rule_fired"] == "unknown_escalated",
            "unknown_gates": result["unknown_gates"],
            "overridden": summary["overridden"],
            "displayed_category": summary["category"],
        },
        "ranking": {
            "previous_position": prev_position,
            "current_position": current_position,
            "delta": (prev_position - current_position) if prev_position else None,
            "queue_size": len(ranked),
            "urgency_score": summary["score"],
            "moved_ahead_of": moved_ahead_of,
        },
    }


async def send_question(
    ws: WebSocket,
    question: questions.Question,
    seq: int,
    text: str,
    attempt: int = 1,
) -> None:
    """text is the phrasing actually spoken this turn (LLM-generated, or the
    preset fallback) — not necessarily question.text."""
    await ws.send_text(
        json.dumps(
            {
                "t": "question",
                "question_id": question.id,
                "seq": seq,
                "text": text,
                "sample_rate": tts.SAMPLE_RATE,
                "attempt": attempt,
            }
        )
    )
    try:
        pcm = tts.speak(text)
        for i in range(0, len(pcm), tts.CHUNK_BYTES):
            await ws.send_bytes(pcm[i : i + tts.CHUNK_BYTES])
    except Exception as exc:
        print(f"[tts] failed for {question.id}: {exc}")
        await ws.send_text(json.dumps({"t": "error", "message": f"tts failed: {exc}"}))
    await ws.send_text(json.dumps({"t": "question_end"}))


async def send_closing(ws: WebSocket, text: str) -> None:
    """Spoken sign-off: confirms the request was received and responders are
    notified. Same audio path as questions, but no answer is expected."""
    await ws.send_text(
        json.dumps({"t": "closing", "text": text, "sample_rate": tts.SAMPLE_RATE})
    )
    try:
        pcm = tts.speak(text)
        for i in range(0, len(pcm), tts.CHUNK_BYTES):
            await ws.send_bytes(pcm[i : i + tts.CHUNK_BYTES])
    except Exception as exc:
        print(f"[tts] closing failed: {exc}")


@app.websocket("/api/ws/converse")
async def converse(ws: WebSocket):
    await ws.accept()
    interview_id = uuid.uuid4().hex[:8]
    fields: dict = {f: None for f in questions.FIELD_ORDER}
    asked: list[str] = []  # one entry per ask — a circled-back id appears twice
    history: list[dict] = []  # [{"question": spoken text, "answer": transcript}]
    current_q: questions.Question | None = None
    current_text: str | None = None  # what was actually spoken this turn
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
                    await hub.broadcast(
                        {"t": "case_updated", "case_id": interview_id}
                    )
                    current_q = questions.next_question(fields, asked)
                    asked.append(current_q.id)
                    asks = 1
                    current_text = interviewer.phrase(current_q, history, fields)
                    await send_question(ws, current_q, seq + 1, current_text)
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
                            f"[{interview_id}] no response, re-asking "
                            f"{current_q.id} (attempt {asks}/{MAX_ASKS})"
                        )
                        # Rephrased, not repeated verbatim: simpler and more
                        # reassuring each time the survivor stays silent.
                        current_text = interviewer.phrase(
                            current_q, history, fields, attempt=asks
                        )
                        await send_question(
                            ws, current_q, seq + 1, current_text, attempt=asks
                        )
                    else:
                        final_status = "no_response"
                        print(
                            f"[{interview_id}] no response after {asks} asks "
                            f"of {current_q.id} — marking case no-response"
                        )
                        # They may hear even if they can't speak — say that
                        # help has been alerted before moving on.
                        await send_closing(ws, interviewer.NO_RESPONSE_CLOSING)
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
                            question_text=current_text or current_q.text,
                            transcript=transcript,
                            audio_path=str(audio_path),
                            stt_ms=stt_ms,
                        )
                    )
                    s.commit()
                history.append(
                    {"question": current_text or current_q.text, "answer": transcript}
                )

                fields = extraction.extract(
                    transcript, current_q.id, fields, question_text=current_text
                )
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
                await hub.broadcast(
                    {"t": "case_updated", "case_id": interview_id, "turn_seq": seq}
                )

                next_q = questions.next_question(fields, asked)
                if next_q is None:
                    final_status = "complete"
                    await send_closing(
                        ws, interviewer.closing(history, fields)
                    )
                    await ws.send_text(
                        json.dumps({"t": "complete", "interview_id": interview_id})
                    )
                    print(f"[{interview_id}] interview complete after turn {seq}")
                    break
                current_q = next_q
                # Second time round for this goal? Tell the phraser so it can
                # acknowledge ("I know I asked before...") instead of pressing.
                revisit = asked.count(next_q.id) + 1
                asked.append(next_q.id)
                asks = 1
                current_text = interviewer.phrase(
                    next_q, history, fields, attempt=revisit
                )
                await send_question(ws, next_q, seq + 1, current_text)
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
            await hub.broadcast({"t": "case_updated", "case_id": interview_id})
            if status == "abandoned":
                print(f"[{interview_id}] interview abandoned at turn {seq}")
