# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A person-detection subsystem for an autonomous rescue rover, implementing `PRD.md` v3.

**The rover drives itself** — straight lines, turning left when its own front sensor finds a
blockage. This codebase never steers it and cannot move anything. It observes and records:
who was seen during the journey, when, with what confidence, plus one saved image per person.

This changed in v3 (PRD §0c). Earlier versions computed a turn command to orient the rover
toward a detected person; that proved too hard to control, so `control.py`, `selection.py` and
`rover.py` were deleted. If you find a reference to `Command`, `RoverController`, `KP`, or
`is_target`, it is stale — remove it.

`PRD.md` is the source of truth. Read §0/§0b/§0c (changelogs) first: several decisions were
reversed, and each reversal records why.

## Commands

```bash
.venv/Scripts/python.exe -m pytest -q                    # full suite, no hardware needed
.venv/Scripts/python.exe -m pytest tests/test_geometry.py -v
.venv/Scripts/python.exe -m pytest tests/test_sightings.py::test_peak_confidence_is_the_maximum_not_the_last_value -v
```

```powershell
.\demo.bat            # live webcam  (PowerShell needs the .\ prefix)
.\demo.bat clip       # bundled 3-person fixture
```

Always invoke `.venv/Scripts/python.exe` explicitly — a conda `(base)` environment is usually
active and lacks every dependency.

## Architecture

```
FrameSource ──> Detector ──> geometry → tracking ──> SightingRecorder
 file/cam/pi    cascade                               journey log + best frames
```

Two ideas carry the design:

**The core is pure.** `geometry.py` and `tracking.py` do no I/O and never call `time.time()` —
the clock is a parameter. That is what makes bearing signs and sighting accumulation testable
with no camera and no model. Keep it that way; if you need time in the core, pass it in.

**The pipeline depends on a `Detector` Protocol, not on Ultralytics.** `ScriptedDetector` in
`detector.py` replays canned detections, so `test_pipeline_smoke.py` exercises the whole loop
offline. Any new pipeline stage must stay reachable from that test.

**Detection is a two-tier cascade** (PRD §6.5): `yolo26n` @480 px every frame feeding ByteTrack,
plus `yolo26s` @640 px on candidate frames only, rate-limited. A track is promoted to
"confirmed human" after `N_CONFIRM` confirm hits and only confirmed tracks reach the log.

## Output contract

- `output/sightings.jsonl` — `rescue.sighting.v1`, **one record per person encountered**, not
  per frame. Written when a track disappears; `close()` finalises anyone still visible.
- `output/sightings/` — exactly one JPEG per sighting, the peak-confidence frame.
- `output/events.jsonl` — `rescue.detection.v2` per-frame rows, only with `--raw-log`.

Do not reintroduce per-frame logging as a default. Driving past one person for five seconds at
10 FPS produced ~50 near-identical rows; collapsing that was the point of v3.

## Things that will bite

- **Retention ≠ visibility.** `TrackStore.tracks()` includes tracks the detector has stopped
  reporting (kept so ByteTrack can re-associate IDs). Anything that logs or displays must use
  `visible_tracks(frame_index)`. Getting this wrong previously fabricated 42 detection rows for
  a person who had left the frame.
- **Two confidence fields.** `confidence` is live per-frame and goes to logs; `display_confidence`
  is sampled at 1 Hz and goes on-screen, because a number redrawn at 10 Hz is unreadable.
- **`distance_m` is null when `distance_valid` is false.** A prone person fails the aspect-ratio
  check — and prone people are the actual rescue target. Never gate anything on a rejected estimate.
- **YOLO26 is NMS-free**, so `iou=` does nothing. Tune with `conf=` only.
- **`lap` must be installed explicitly.** Ultralytics AutoUpdates it over the network at the
  first `track()` call, which fails on an offline Pi — and fails late, at the first tracked frame.
- **`RPi.GPIO` and `pigpio` do not work on Pi 5.** Irrelevant now that motor code is gone, but
  do not reintroduce them.
- **Never call `cv2.imshow` on the Pi** (headless). Use `--mjpeg-port`.
- **Sensor noise, not resolution, limits detection** on the camera module: gain noise costs ~36%
  of detections, blur 13–16%, resolution almost nothing. PRD §6.6 carries the measurements and a
  flagged caveat that its short-exposure/high-gain trade may be tuned the wrong way.

## Unverified code

`PiCameraSource` in `frame_source.py` is written from the spec and **has never been executed** —
`picamera2` cannot be imported on Windows. Its import is lazy so the package still loads. Expect
it to need fixing on first Pi boot, and do not describe it as working.

## Sign convention

Negative `bearing_deg` = person left of centre. Measured off the rover's heading at the moment
of the sighting; with no odometry it cannot be converted into a position. The log says *when*
someone was seen and at what angle — never *where*.
