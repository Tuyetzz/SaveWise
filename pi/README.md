# pi — rover bridge (backend WebSocket → GPIO → Arduino)

The Pi camera never worked (mismatched components), so the Pi's only job now
is **driving**: it keeps an outbound WebSocket open to the backend and mirrors
the operator's current command onto three GPIO pins. The Arduino
(`arduino/RoverRemote/RoverRemote.ino`) reads those pins and runs the motors.
Vision moved off the Pi entirely — the phone mounted on the rover streams its
camera to the backend, and `detection_app` consumes that stream (see below).

```
phone on the rover (/rover)   camera ─► backend /api/ws/video/upload ─► /api/ws/video/feed ─┬─► detection_app
                                                                                            └─► admin console (raw view)
detection_app --publish       boxes drawn ─► /api/ws/video/annotated/upload ─► .../annotated/feed ─► admin console (detection view)
admin console (/control)      drive pad ─► backend /api/ws/rover/control ─► /api/ws/rover/agent ─► THIS APP ─► GPIO ─► Arduino
```

## Wiring (Pi ↔ Arduino)

| Pi (BCM)   | Arduino | Meaning        |
| ---------- | ------- | -------------- |
| GPIO17     | A0      | command bit 0  |
| GPIO27     | A1      | command bit 1  |
| GPIO22     | A2      | command bit 2  |
| GND        | GND     | **required** — common ground |

Command code (bit2 bit1 bit0): `0` stop · `1` forward · `2` backward ·
`3` left · `4` right · `5` autonomous (Arduino's own obstacle-avoidance).
The Arduino uses input pullups, so an unplugged cable or powered-off Pi reads
`7` → stop. The Pi's 3.3 V high is a valid HIGH for the 5 V Arduino; never
wire an Arduino *output* back to a Pi pin without a level shifter.

Motor/ultrasonic wiring is unchanged from `RoverTestV2.ino` — flash
`arduino/RoverRemote/RoverRemote.ino` over it.

## Setup (on the Pi)

```bash
cd pi
python3 -m venv --system-site-packages .venv   # keeps apt's gpiozero visible
.venv/bin/pip install -r requirements.txt
```

`gpiozero` + `lgpio` are the only GPIO stack that works on the Pi 5 —
`RPi.GPIO` and `pigpio` do not; don't swap them in.

## Run

```bash
.venv/bin/python rover_pi.py                  # connects to the default backend
.venv/bin/python rover_pi.py --url ws://192.168.1.10:8000/api/ws/rover/agent
python3 rover_pi.py --dry-run -v              # anywhere, no GPIO: logs writes
```

Default URL is `wss://hackathon.marcusnguyen.dev/api/ws/rover/agent`
(override with `--url` or `ROVER_AGENT_URL`).

Failsafes, in layers: the backend re-sends the current command every 0.5 s and
this app stops the motors if that stream goes quiet for 2 s; any disconnect
stops before reconnecting; and if the Pi dies outright the Arduino's pullups
read "stop".

## Full-system demo checklist

1. **Backend**: `cd back-end && uv run dev`
2. **Frontend**: `cd front-end && bun run dev` (or the deployed site)
3. **Pi**: `python rover_pi.py` (this folder)
4. **Phone** on the rover: open `/rover`, tap **Start camera** — this is the
   rover's camera and the triage interview mic in one (no preview on the
   phone; it's pure input)
5. **Admin laptop**: open `/control` — live view (raw or detection boxes)
   plus the drive pad; the "pi" dot turns green when step 3 is up
6. **Detection** (on the server, next to the backend — `uv sync` once in
   `detection_app/`, then):
   `uv run python -m rescue_vision --source ws://127.0.0.1:8000/api/ws/video/feed --publish ws://127.0.0.1:8000/api/ws/video/annotated/upload`
   — the `--publish` half is what feeds the console's Detection view
7. Sanity check everything at `GET /api/rover/status`
