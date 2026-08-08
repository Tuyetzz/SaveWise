# Rescue Rover Vision Subsystem

Logs every human seen during an autonomous rover's journey, with confidence
scores. Implements `PRD.md` v3.

The rover drives itself — straight lines, turning left when its own front
sensor finds a blockage. **This subsystem never steers it.** It watches and
records: who was seen, when, how confident the detector was, and one saved image
of each person's best moment.

Answers *"who did the rover pass, and how sure are we?"*

> **Coverage limitation, by design.** The camera is fixed forward with a 53.5°
> field of view and never scans. Anyone outside that cone as the rover drives
> past is never seen. Coverage is a property of the route, not of the vision.

## Setup (Windows dev)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Model weights download automatically on first run. Nothing to fetch by hand.

## Demo on Windows

In **PowerShell** — note the leading `.\`, without which PowerShell will not run
a script from the current directory:

```powershell
.\demo.bat            # live webcam
.\demo.bat clip       # the bundled 3-person test clip
.\demo.bat my.mp4     # any video file
```

In **cmd.exe** the bare name works:

```
demo.bat clip
```

Press **q** or **Esc** in the preview window to quit. Nothing physically moves;
there is no flag in this program that can command the rover.

The launcher passes `--confirm-min-interval 0.5`, which roughly doubles FPS for
a smoother live picture. Drop it for maximum detection precision.

If you use conda, ignore the active `(base)` environment — `demo.bat` calls
`.venv\Scripts\python.exe` by absolute path, so it always uses the project venv
regardless of what conda has activated.

## The journey log

A summary prints on exit:

```
JOURNEY COMPLETE - 3 human sighting(s)

    #    seen at      for  peak conf   bearing   nearest
  ---  ---------  -------  ---------  --------  --------
    1       2.0s     2.5s       0.90     +3.4d      4.7m
    2       2.5s     2.2s       0.89    -23.1d      5.3m
    3       3.7s     1.9s       0.86     -4.8d      5.2m
```

`output/` then contains:

- **`sightings.jsonl`** — one `rescue.sighting.v1` record per person
  encountered, with first/last seen, duration, peak and mean confidence,
  bearing range and closest trustworthy distance.
- **`sightings/`** — exactly one JPEG per sighting: that person's
  highest-confidence frame.

**One record per person, not per frame.** Driving past someone for five seconds
at 10 FPS is one row, not fifty. Add `--raw-log` to also write per-frame
`rescue.detection.v2` rows to `events.jsonl` when you need to debug why someone
was missed.

**"Sighting", not "person", is deliberate.** If tracking drops and re-acquires
the same individual they appear twice — visible in the log as two entries
seconds apart at similar bearings. No merge heuristic, because a wrong merge of
two different people would be invisible.

**There is no position.** Without odometry the log records *when* someone was
seen and at what angle off the rover's heading at that moment — never where.

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
4. Run it:

```
python -m rescue_vision --source picamera \
    --scan-model yolo26n.onnx --confirm-model yolo26s.onnx \
    --confirm-min-interval 0.5 --mjpeg-port 8080
```

Never pass `--display` on the headless Pi — use `--mjpeg-port` and open the
printed URL from a laptop on the same network.

No motor wiring, no bench test, no wheels-off-the-ground step: this program
cannot move anything. The rover's own driving logic is a separate concern.

> `PiCameraSource` is written from the spec and has **never been run against
> hardware** — it is the one remaining unverified module. Expect the picamera2
> configuration in `rescue_vision/frame_source.py` to need adjustment on first
> boot.

## Architecture

```
FrameSource ──> Detector ──> geometry → tracking ──> SightingRecorder
 file/cam/pi    cascade                                journey log + best frames
```

`geometry` and `tracking` perform no I/O and never read a clock — time is
passed in. That is what lets the bearing conventions and the sighting
accumulation be tested on a laptop with no camera and no model.

## Camera module vs laptop webcam

Measured on 36 clip frames, `yolo26n` @480 px, against synthetic degradation:

| Condition | Detections kept | Mean conf |
|---|---|---|
| Baseline | 100% | 0.72 |
| Resolution quartered | 80% | 0.73 |
| Darkness alone | 95–97% | 0.68–0.72 |
| **Darkness + gain noise** | **64%** | 0.74 |
| Motion blur | 84–87% | 0.63–0.75 |
| Realistic Pi combination | 77% | 0.71 |

Resolution barely matters. **Sensor noise from high analogue gain is the
dominant cost** — bigger than darkness, blur, and resolution combined. What
degrades is recall, not confidence, and tracking absorbs much of it since
alerts fire per track rather than per frame.

If detections are sparse on the camera module:

```
--scan-conf 0.15        recover recall; N_CONFIRM still guards precision
```

and try a longer exposure at lower gain than PRD §6.6 suggests — see the
measured caveat there.

## Sign convention

| Quantity | Negative | Positive |
|---|---|---|
| `bearing_deg` | Person is **left** of centre | Person is **right** of centre |

Measured off the rover's heading at the moment of the sighting. With no
odometry it cannot be turned into a position.

## Tuning on the day

Tunables live in `rescue_vision/config.py`; the ones you will actually touch are
exposed as CLI flags:

- `--scan-conf 0.15` — recover recall on a noisy camera module.
- `--confirm-min-interval 0.5` — the main FPS lever.
- `--raw-log` — per-frame rows, for diagnosing a missed person.

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
