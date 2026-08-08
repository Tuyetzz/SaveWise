# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**There is no code yet.** The repository contains only `PRD.md` (the full spec, v2) and
`hardware/camera_module.jpg`. It is not a git repository.

`PRD.md` is the source of truth for design decisions. Read §6 before implementing anything;
read §0 (changelog) to know which v1 decisions were reversed. When a design question comes up
that the PRD already answers, follow the PRD rather than re-deciding — several of its choices
exist specifically to avoid known failure modes, and the reasoning is recorded inline.

Open questions the PRD has *not* resolved are listed in §10 (demo lighting, motor pin
assignments, L298N vs TB6612FNG, chassis turning characteristics for `KP`). Ask rather than
guess on those.

## What is being built

A person-detection subsystem for an autonomous rescue rover: camera → detect → track → compute
bearing + coarse distance → emit a turn command that orients the rover toward the person. Path
planning, obstacle avoidance, and localization are explicitly out of scope (§5).

## Architecture

The whole design hinges on one idea: **the detection/control core is byte-for-byte identical
between the Windows dev machine and the Pi 5.** Only the two ends of the pipe swap out, via two
interfaces:

- `FrameSource` — `cv2.VideoCapture("clip.mp4")` on Windows, `picamera2` on the Pi.
- `RoverController` (ABC, §6.9) — `ConsoleRover` (logs to stdout/JSONL, the default) on
  Windows, `GpioZeroRover` on the Pi. `GpioZeroRover` must import `gpiozero` **lazily** so the
  module still imports on Windows.

Both are selected by CLI flag. Never let platform conditionals leak into the detection,
tracking, bearing, distance, or control code.

Pipeline stages, in order (§6.5, §6.7, §6.8):

1. **Scan pass** — `YOLO26n` @ 480 px, conf 0.25, every frame. Also feeds the tracker.
2. **Tracking** — ByteTrack (`model.track(..., persist=True, tracker="bytetrack.yaml")`)
   applied to the *scan* pass. Track IDs are the unit of alerting — one alert per new confirmed
   track, never one per frame.
3. **Confirm pass** — `YOLO26s` @ 640 px, conf 0.45, only when scan returned ≥1 person, rate-
   limited to once per `CONFIRM_MIN_INTERVAL`. A track is promoted to "confirmed human" after
   `N_CONFIRM` confirm hits.
4. **Bearing + distance** — `tan`-based bearing (not the linear approximation), pinhole
   distance from bbox height, EMA-smoothed per track ID.
5. **Target selection + P-control** — one target per frame, sticky for `TARGET_HOLD`;
   proportional turn with deadband and a `MIN_TURN` stiction floor.

Output is newline-delimited JSON, one object per detection per frame, schema
`rescue.detection.v1` (§6.4). Every row repeats `turn_command`/`drive_command` so a consumer
reading a single line has everything it needs.

All tunables live in one config module — see Appendix A of the PRD for the full list and
starting values. Don't scatter magic numbers into the pipeline.

## Commands

Nothing to build or test yet. When the Pi environment is set up (§6.6):

```bash
sudo apt install -y python3-picamera2 python3-opencv libcamera-apps python3-venv
python3 -m venv --system-site-packages venv   # --system-site-packages is REQUIRED
source venv/bin/activate
pip install ultralytics ncnn gpiozero lgpio   # no --break-system-packages inside a venv
libcamera-hello --list-cameras                # verify camera
```

Do **not** `pip install picamera2` or `opencv-python-headless` — apt provides both, and the
venv sees them via `--system-site-packages`. The PyPI `picamera2` build routinely fails on the
Pi.

One-time model export (the only step needing internet):

```python
from ultralytics import YOLO
YOLO("yolo26n.pt").export(format="ncnn", imgsz=480, half=True)
YOLO("yolo26s.pt").export(format="ncnn", imgsz=640, half=True)
```

## Constraints that bite

- **`RPi.GPIO` and `pigpio` do not work on Pi 5** (RP1 southbridge). Use `gpiozero` backed by
  `lgpio`. Verify the gpiochip number with `ls /dev/gpiochip*` — it has moved between kernel
  releases.
- **YOLO26 is NMS-free**, so the `iou=` threshold does nothing. Tune recall/precision with
  `conf=` only.
- **Disk, not RAM, is the binding constraint** — 16 GB SD card, and `ultralytics` pulls in
  PyTorch (~1.5–2.5 GB on arm64) even when NCNN does the inference. Saved detection frames are
  the unbounded item: rate-limit to 1 per track per second, cap the output dir at ~500 MB. Run
  `df -h` after each phase, target ≥2 GB free.
- **Sign conventions** (Appendix B): negative `bearing_deg` = person left of centre; positive
  `turn_command` = rover rotates clockwise/right. A correct system has `turn_command` taking the
  *same* sign as `bearing_deg`. If the rover turns away from people, swap the motor pin pairs in
  `MOTOR_PINS` — do not negate `KP`, that leaves the convention lying to the next reader.
- **Motors default to stopped.** `stop()` on `KeyboardInterrupt`, on any uncaught exception, and
  in a `finally`. A watchdog thread stops the drive if `drive()` hasn't been called within
  `WATCHDOG_TIMEOUT`. Bench-test with wheels off the ground, every time.
- **A single failed frame must not crash the pipeline** — log and continue (NFR4).
- **Headless Pi** — never call `cv2.imshow` on the Pi path.
- **Fixed short exposure is a requirement, not polish.** The OV5647 is rolling-shutter;
  auto-exposure indoors picks 20–30 ms and smears a turning rover's frames. Disable `AeEnable`,
  set `ExposureTime`, raise `AnalogueGain`.
- Camera intrinsics (`HFOV_DEG = 53.5`) are for Camera Module v1 / OV5647. A v2/v3 module
  changes them and every bearing and distance silently goes wrong.
- `distance_m` is ±25–30% at best and invalid in the cases tabulated in §6.7 — notably a low
  bbox aspect ratio, which means the person is lying down. Lying-down people are the actual
  rescue target, so never gate an *alert* on distance; gate only the approach behaviour.
- Confidence values differ between `.pt` on Windows and NCNN on the Pi. Tune `conf` coarsely on
  the clip, re-check on the Pi.
