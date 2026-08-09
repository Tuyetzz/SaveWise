# DisasterResponse — Phase 1 voice interview backend

Audio in → transcript → clinical field extraction → next question out as audio,
over one WebSocket. Deterministic code decides WHAT happens: which field to ask
for next (`questions.py`, including one circle-back for triage-critical fields
still unknown after the first pass) and triage classification (`triage.py`).
The LLM only extracts fields (`extraction.py`) and words each question for the
actual survivor (`interviewer.py`) — acknowledging what they just said,
simplifying on silence or confusion, never choosing goals or triage. If
phrasing generation fails, the preset question text ships instead; the
interview never blocks on a model.

## Setup

```bash
cp .env.example .env   # put the OpenAI key in OPEN_AI_KEY
uv sync
```

## Run

```bash
uv run dev
```

Serves on `127.0.0.1:8000` — loopback only. The nginx proxy at
hackathon.marcusnguyen.dev is the sole public entry point: it terminates TLS
(Let's Encrypt) and routes `/api/*` here, everything else to the frontend on
:3308. Phones just use the domain; no certificate warnings, everything
same-origin.

If you ever need direct LAN access without the proxy (old workflow), run
uvicorn with `--host 0.0.0.0` and self-signed TLS, and accept the certificate
warning at `https://<server-ip>:8000/api/health` on the phone first.

No `--reload` — it reloads the Whisper model on every save. First start
downloads distil-large-v3 (~1.5 GB) to `~/.cache/faster-whisper`.

## Protocol

All routes — HTTP and WebSocket — live under `/api`: the reverse proxy routes
`/api/*` to this backend (port 8000) and everything else to the frontend
(port 3308).

`/api/ws/converse`, one connection per interview. Client sends `{"t":"start"}`,
then one binary PCM16 16 kHz mono frame per complete utterance (client does the
endpointing). Server replies per turn: `question` + binary PCM (24 kHz) +
`question_end`, then after each answer `transcript`, `fields`. When the
interview ends (`complete` or `no_response`), a spoken `closing` + audio is
sent first — confirming first responders are notified and coming — so the
survivor always hears that the request was received.

`/api/ws/dashboard`, any number of connections. Server pushes
`{"t": "case_updated", "case_id": ...}` whenever an interview starts, an
assessment lands, a status changes, or a responder acts. Payloads are pointers,
not state — clients refetch, so the HTTP endpoints stay the source of truth.

`GET /api/cases/{id}/reasoning` returns the full decision chain for the latest
assessment: verbatim transcript turns (with audio duration), per-field
extraction state (`value`/`known`/`changed_this_turn`), the classifier's own
`trace` (every condition evaluated, the values it read, whether it matched —
emitted by `triage.classify` as it runs, not reconstructed), and the queue
effect (previous vs current position, additive score, which cases it moved
ahead of). Backs the dashboard's reasoning panel.

## Reset / inspect

- Reset everything: `rm -rf data/` — schema and folders recreate on next start.
- On any model/schema change: delete `data/triage.db` (no migrations, by design).
- Audit trail: `sqlite3 data/triage.db "select turn_seq, fields from assessment where interview_id='...' order by turn_seq;"`

## Tests

```bash
uv run pytest
```
