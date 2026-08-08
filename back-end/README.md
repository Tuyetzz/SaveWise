# DisasterResponse — Phase 1 voice interview backend

Audio in → transcript → clinical field extraction → next question out as audio,
over one WebSocket. The LLM extracts fields only; question order and (later)
triage classification are deterministic code.

## Setup

```bash
cp .env.example .env   # put the OpenAI key in OPEN_AI_KEY
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile <key.pem> --ssl-certfile <cert.pem>
```

No `--reload` — it reloads the Whisper model on every save. First start
downloads distil-large-v3 (~1.5 GB) to `~/.cache/faster-whisper`.

Before demo day: visit `https://<server-ip>:8000/health` on the phone once and
accept the certificate warning, otherwise the `wss://` handshake fails opaquely.

## Protocol

`/ws/converse`, one connection per interview. Client sends `{"t":"start"}`,
then one binary PCM16 16 kHz mono frame per complete utterance (client does the
endpointing). Server replies per turn: `question` + binary PCM (24 kHz) +
`question_end`, then after each answer `transcript`, `fields`, and eventually
`complete`.

## Reset / inspect

- Reset everything: `rm -rf data/` — schema and folders recreate on next start.
- On any model/schema change: delete `data/triage.db` (no migrations, by design).
- Audit trail: `sqlite3 data/triage.db "select turn_seq, fields from assessment where interview_id='...' order by turn_seq;"`

## Tests

```bash
uv run pytest
```
