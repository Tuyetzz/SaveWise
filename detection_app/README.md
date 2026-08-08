# Rescue Rover Vision

Logs every human an autonomous rescue rover passes during a sweep, with a
confidence score and a photo of each person.

The rover **drives itself** — straight lines, turning left when its own front
sensor hits a blockage. This repo never steers it and contains no motor code.
It watches through a forward-facing camera and writes a journey report.

> ⚠️ Work lives on the **`feat/vision-subsystem`** branch, not `master`.
> `git checkout feat/vision-subsystem`

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.\demo.bat              # live webcam; or `.\demo.bat clip` for the test fixture
```

Press **q** or **Esc** to stop. Model weights download themselves on first run.

## What you get

`output/report.html` — open it in any browser. One card per person: their best
photo, when they were seen, how long for, peak confidence. Self-contained, so
you can email it.

Also `output/sightings.jsonl` (one `rescue.sighting.v1` record per person) and
`output/sightings/` (one JPEG each). Add `--raw-log` for per-frame rows when
debugging a missed detection.

## Code map

| File | What it does |
|---|---|
| `frame_source.py` | Video file / webcam / picamera2, behind one interface |
| `detector.py` | Two-tier YOLO26 cascade + `ScriptedDetector` test double |
| `geometry.py` | Bearing and coarse distance from a bounding box |
| `tracking.py` | ByteTrack IDs, smoothing, "is this a confirmed human yet" |
| `sightings.py` | Turns per-frame detections into one record per person |
| `annotate.py` / `palette.py` | The video overlay and per-person colours |
| `report.py` | The HTML report |
| `pipeline.py` | Wires the above together, one frame at a time |
| `cli.py` | Flags, and the summary printed on exit |
| `config.py` / `types.py` | Every tunable in one frozen dataclass; shared data types |
| `events.py` / `preview.py` | Optional `--raw-log` rows; optional MJPEG stream |

## Adding a feature

Two rules make the codebase testable without a camera, a model, or a rover —
please keep them:

1. **`geometry.py` and `tracking.py` are pure.** No file I/O, no `time.time()` —
   the clock is passed in as a parameter. That's why the maths has tests.
2. **`pipeline.py` depends on the `Detector` *protocol*, not on Ultralytics.**
   `ScriptedDetector` replays canned detections, so `test_pipeline_smoke.py`
   runs the whole loop offline. Keep new stages reachable from that test.

```powershell
.venv\Scripts\python.exe -m pytest -q          # 167 tests, no hardware needed
.venv\Scripts\python.exe -m pytest tests/test_sightings.py -v
```

Always call `.venv\Scripts\python.exe` explicitly. A conda `(base)` env is often
active and has none of the dependencies.

## Gotchas that will cost you an hour

- **One person = one sighting.** Detection drops ~23% of frames on the real
  camera. Closing a record on the first missing frame logged one person **9.4
  times**; `sighting_gap_s` is the grace period that fixes it. Don't remove it.
- **Retention ≠ visibility.** `TrackStore.tracks()` includes people the detector
  has stopped seeing. Anything that logs or draws must use `visible_tracks()`.
- **Colour is keyed on sighting ID**, never on position in the frame — a person
  leaving must not repaint everyone else.
- **YOLO26 is NMS-free**, so `iou=` does nothing. Tune with `conf=`.
- **`lap` must be pip-installed.** Ultralytics fetches it over the network at
  the first `track()` call, which fails on an offline Pi — and fails late.
- **Never `cv2.imshow` on the Pi** (headless). Use `--mjpeg-port` instead.

## Where the detail lives

- **`PRD.md`** — the spec. Read the changelogs (§0, §0b, §0c) first; several
  decisions were reversed and each records why. Also has measured FPS and
  camera-noise numbers.
- **`CLAUDE.md`** — architecture and constraints, written for AI assistants but
  just as useful for a human.
- **`docs/superpowers/specs/`** — why things are built the way they are.

## Known limits

Fixed 53.5° forward camera, no scanning: **anyone outside that cone is never
seen.** Coverage depends on the rover's route.

No odometry: the log records *when* someone was seen and at what angle, never
*where*. Don't let a feature imply otherwise.

`PiCameraSource` has never run against real hardware. Expect it to need fixing
on first Pi boot.
