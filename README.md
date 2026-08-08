# Rescue Rover Vision Subsystem

Person detection, bearing, coarse distance, and a normalized turn command for
an autonomous rescue rover. Implements `PRD.md` v2.1.

Answers *"is there a person, and which way should I turn to face them?"* It does
not answer *"where should I go next?"* — path planning and obstacle avoidance
are out of scope.

## Setup (Windows dev)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Model weights download automatically on first run. Nothing to fetch by hand.

## Run

Webcam, console motor backend (nothing physically moves):

```
.venv\Scripts\python.exe -m rescue_vision --source 0 --display
```

Recorded clip, writing an annotated MP4:

```
.venv\Scripts\python.exe -m rescue_vision --source clip.mp4 --save-video out.mp4
```

Output lands in `output/`: `events.jsonl` (schema `rescue.detection.v1`) and
`detections/` frames.

## Tests

```
.venv\Scripts\python.exe -m pytest -q
```

All tests run without a camera, a model, or motors.

## Deploying to the Pi 5

1. Export the models **on the desktop**: `python scripts/export_models.py`
2. Copy `yolo26n.onnx` and `yolo26s.onnx` to the Pi.
3. Set up the Pi per PRD §6.6 — a `--system-site-packages` venv, and `lap`
   installed explicitly (Ultralytics otherwise downloads it at the first
   `track()` call, which fails offline).
4. Bench-test **with the wheels off the ground**:

```
python -m rescue_vision --source picamera --rover gpiozero \
    --scan-model yolo26n.onnx --confirm-model yolo26s.onnx --mjpeg-port 8080
```

Never pass `--display` on the headless Pi — use `--mjpeg-port` and open the
printed URL from a laptop on the same network.

> `GpioZeroRover` and `PiCameraSource` are written from the spec and have
> **never been run against hardware**. Expect the gpiochip number and the pin
> mapping in `rescue_vision/rover.py` to need adjustment on first boot.

## Architecture

A pure decision core with impure edges:

```
FrameSource ──> Detector ──> geometry → tracking → selection → control ──> RoverController
 file/cam/pi    cascade                     │                               console/gpiozero
                                            └──> EventWriter (JSONL + frames)
```

`geometry`, `tracking`, `selection`, and `control` perform no I/O and never read
a clock — time is passed in. That is what makes the sign conventions and the
controller testable on a laptop instead of on the floor at the demo.

## Sign conventions

| Quantity | Negative | Positive |
|---|---|---|
| `bearing_deg` | Person is **left** of centre | Person is **right** of centre |
| `turn_command` | Rover rotates **counter-clockwise / left** | Rover rotates **clockwise / right** |
| `drive_command` | Reverse | Forward |

`turn_command` takes the **same sign** as `bearing_deg`. If the rover turns away
from people, swap the motor pin pairs in `MOTOR_PINS` — do not negate `kp`, which
would leave this table lying to the next reader.

## Tuning on the day

Tunables are in `rescue_vision/config.py`; the ones you will actually touch are
exposed as CLI flags:

- `--kp` — proportional gain. Carpet and hard floor differ enough to matter.
- `--deadband-deg` — widen if the rover hunts around centre.
- `--min-turn` — raise if the motors buzz without turning.

### FPS

Measured end-to-end on an Intel Core Ultra 5 226V desktop with the `.pt`
backend — a Pi 5 is roughly 4–6× slower:

| Configuration | ms/frame | FPS |
|---|---|---|
| Default cascade (scan 480 + confirm 640 @ 0.15 s) | 108 | 9.2 |
| `--confirm-min-interval 0.5` | 58 | 17.2 |
| `--no-confirm` | 46 | 21.6 |
| `--no-confirm --scan-imgsz 320` | 27 | 37.1 |

The confirm tier is ~60% of the frame budget. **Expect to run the Pi with
`--confirm-min-interval 0.5` at minimum** — the default configuration will not
reach the PRD's 10 FPS target there. Raise that flag before you shrink
`--scan-imgsz`: a smaller scan input is what starts costing you detections of
distant and prone people, which is the failure you cannot recover from.
