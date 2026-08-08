# Design: Rescue Rover Vision Subsystem

**Date:** 2026-08-08
**Status:** Approved
**Implements:** `PRD.md` v2.1

This document records the *software* design. `PRD.md` remains the product spec
and the authority on behaviour, constants, and sign conventions. Where this
document and the PRD disagree, the PRD wins — except for the deltas listed in
§5, which were decided after the PRD was written and are reflected back into it
as changelog entries C10–C13.

## 1. Organizing principle

The PRD names two seams: `FrameSource` and `RoverController`. This design adds a
third, which the PRD implies but never states:

> **The decision-making core is pure. It performs no I/O and reads no clock.**

Bearing, distance, distance validity, EMA smoothing, track promotion, target
selection, and the P-controller are all pure functions over plain data
structures. Time is passed in as a parameter, never read from `time.time()`
inside the core.

This matters for one concrete reason. The PRD's own risk table says the three
most likely ways this demo fails are an inverted turn sign, controller
oscillation, and a bad distance estimate on a lying-down person. All three live
in the pure core. Making that core testable without a camera, a model, or a
motor means those failures get caught by `pytest` on a laptop rather than on the
floor at the demo.

```
    impure edge              pure core                 impure edge
  ┌─────────────┐    ┌───────────────────────┐    ┌──────────────────┐
  │ FrameSource │───▶│ geometry → tracking   │───▶│ RoverController  │
  │ file/cam/pi │    │ → selection → control │    │ console/gpiozero │
  └─────────────┘    └───────────┬───────────┘    └──────────────────┘
        ▲                        │
  ┌─────┴────────┐         ┌─────▼────────┐
  │   Detector   │         │ EventWriter  │
  │ (ultralytics)│         │ JSONL+frames │
  └──────────────┘         └──────────────┘
```

## 2. Module layout

```
rescue_vision/
  config.py         Appendix A constants as a frozen dataclass; CLI overrides
  geometry.py       bearing_deg, distance_m, distance_valid          [pure]
  tracking.py       per-track state, EMA smoothing, N_CONFIRM promotion [pure]
  selection.py      target priority + TARGET_HOLD stickiness         [pure]
  control.py        deadband, KP, MIN_TURN, drive rule, mixing       [pure]
  detector.py       two-tier cascade, model loading, confirm rate limit
  frame_source.py   FrameSource ABC + VideoFile / Webcam / PiCamera
  rover.py          RoverController ABC + ConsoleRover / GpioZeroRover + watchdog
  events.py         rescue.detection.v1 JSONL writer, frame saving, disk cap
  annotate.py       bbox / track ID / bearing / distance overlay drawing
  pipeline.py       orchestration; depends on a Detector *protocol*
  cli.py            arg parsing, wiring, signal handling, finally: stop()
scripts/
  export_models.py  one-time ONNX export
tests/
  test_geometry.py  test_tracking.py  test_selection.py  test_control.py
  test_pipeline_smoke.py
```

`pipeline.py` depending on a `Detector` protocol rather than the concrete
ultralytics-backed class is what makes the smoke test possible: the test injects
a stub returning scripted boxes, exercising real tracking, selection, control,
and JSONL emission with no model and no camera.

## 3. Unit responsibilities

| Unit | Does | Depends on |
|---|---|---|
| `geometry` | Maps a bbox + frame size to bearing and distance, and decides whether distance is trustworthy | config constants only |
| `tracking` | Holds per-track-ID state; applies EMA; promotes a track to "confirmed" after `N_CONFIRM` confirm hits | geometry outputs |
| `selection` | Picks exactly one target per frame, with stickiness | track states, injected `now` |
| `control` | Target bearing/distance → `turn`, `drive`; mixes to left/right | injected config |
| `detector` | Runs scan every frame with tracking; runs confirm when escalation and rate-limit allow | ultralytics |
| `rover` | Applies commands to motors; guarantees stop | gpiozero (lazy) |
| `events` | Serializes to JSONL; saves annotated frames under a disk cap | filesystem |

Each is independently testable. `control` in particular can be exercised over a
swept range of bearings to assert no oscillation and correct sign, which the PRD
calls the single most common failure.

## 4. Key behaviours

**Bearing** uses the `tan` form, not the linear approximation (PRD §6.7) — the
edges of the frame are exactly where a person first appears during a sweep.

**Distance validity** returns a reason, not just a boolean, so the annotated
frame and the log can say *why* an estimate was rejected. Debugging "distance is
wrong" is much faster when the frame says `clipped_bottom` or `not_upright`.

**Sign convention** is asserted in tests, not just documented: a person on the
left produces negative `bearing_deg`, and a negative bearing produces a negative
`turn_command`. Per PRD Appendix B, if the physical rover turns the wrong way,
the fix is rewiring `MOTOR_PINS`, not negating `KP`.

**Safety** is layered: `stop()` in a `finally`, `stop()` on `KeyboardInterrupt`
and on any uncaught exception, plus a watchdog thread in the `RoverController`
base class that stops the motors if `drive()` has not been called within
`WATCHDOG_TIMEOUT`. The watchdog lives in the base class so both backends get it
and neither can forget it.

**Frame-level resilience** (NFR4): the per-frame body is wrapped so a single bad
frame logs and continues rather than killing the run.

## 5. Deltas from PRD v2

| # | Delta | Rationale |
|---|---|---|
| D1 | ONNX Runtime replaces NCNN | NCNN on ARM64 is unreliable under current Ultralytics — see PRD C10 |
| D2 | Export uses `half=False, simplify=True` | ONNX Runtime CPU has no fp16 fast path — PRD C11 |
| D3 | `WebcamSource` added | No test clip available; `--source 0` gives an immediate dev target |
| D4 | `lap>=0.5.12` pinned explicitly | Ultralytics AutoUpdates it on first `track()`; that network call fails on an offline Pi, and fails late |
| D5 | `MOTOR_PINS` left as placeholders; `STBY` supported as optional | Driver not yet chosen; one config shape serves both L298N and TB6612FNG |

D4 was found by running the code during setup, not by reading docs.

## 6. Verification

**Can be verified on this machine (Windows, no hardware):**
- All pure-core unit tests.
- The end-to-end smoke test with a stub detector: asserts JSONL conforms to
  `rescue.detection.v1`, exactly one row per frame has `is_target: true`, and
  the rover receives a `stop()` on exit.
- A real run against a webcam or video file, producing annotated output.

**Cannot be verified here, and will be delivered untested:**
- `PiCameraSource` — `picamera2` cannot be imported on Windows.
- `GpioZeroRover` — no GPIO.

Both use lazy imports so the package still imports on Windows. They are expected
to need fixes on first Pi boot, most likely the gpiochip number (PRD §6.9) and
the pin mapping. This is stated plainly rather than hidden: the Pi path is
written from the spec, not validated.

## 7. Out of scope

Per PRD §5: path planning, obstacle avoidance, SLAM, absolute localization,
multi-camera fusion, model training, and long-occlusion re-identification. Also
out of scope for this iteration: the UDP/socket event transport (PRD §6.4
mentions it as optional; JSONL file only was chosen).
