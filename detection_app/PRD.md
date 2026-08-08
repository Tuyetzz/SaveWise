# PRD: Real-Time Human Detection & Bearing Subsystem for Autonomous Rescue Rover

**Status:** Draft v2.1 — Hackathon build
**Owner:** _fill in_
**Last updated:** 2026-08-08

---

## 0. Changelog (v1 → v2)

Read this section first — several v1 decisions changed and one contradiction was resolved.

| # | Change | Why |
|---|---|---|
| C1 | **Camera is now fixed forward-facing on the chassis.** The rover's own yaw does the 360° sweep. v1 described a separately rotating camera platform. | Resolves the contradiction between §1 ("camera continuously revolves") and the actual build. Removes the need for a slip ring / pan servo, and makes bearing math trivial (camera axis == chassis axis). **Confirm this matches your hardware before building.** |
| C2 | **Distance estimation moved from "out of scope" into scope** (§5, FR8). Mono-camera pinhole estimate from bounding-box height. | Required to decide "approach vs. stop". Explicitly labeled a *coarse* estimate (±25–30%), not localization. |
| C3 | **Multi-object tracking added** (FR6). ByteTrack via Ultralytics, stable IDs across frames. | Needed to avoid re-alerting on the same person every frame, and to smooth bearing so the rover doesn't jitter. |
| C4 | **Rover turn command is now an output of this subsystem** (§6.8). v1 put all turning logic out of scope. | The subsystem now emits a normalized turn command; the *drive/path-planning* logic stays out of scope. The boundary moved, it didn't disappear. |
| C5 | **§6.4 interface finalized** — was "TBD with nav team". Now a concrete JSON event schema + a `RoverController` abstract interface. | Unblocks coding. |
| C6 | **Motor hardware decided:** L298N / TB6612FNG H-bridge driven by `gpiozero`. `RPi.GPIO` explicitly banned. | `RPi.GPIO` does not work on Pi 5 (RP1 southbridge). This is the most common Pi rover setup. |
| C7 | **Added §6.10 dev/test path on Windows 11** using a recorded video clip. | Lets the vision half be validated before the Pi hardware is ready. |
| C8 | **Fixed two real bugs in the v1 setup commands** (§6.6). `--break-system-packages` inside an activated venv, and `pip install picamera2`. | Both would have wasted debugging time on the day. |
| C9 | **Added an explicit disk budget** (§9). | `ultralytics` pulls in PyTorch (~1.5–2.5 GB on arm64). On a 16 GB card this is the single most likely thing to break the build. v1 under-weighted it. |

## 0b. Changelog (v2 → v2.1)

| # | Change | Why |
|---|---|---|
| C10 | **Inference backend changed from NCNN to ONNX Runtime.** All `*_ncnn_model/` references become `*.onnx`; `ncnn` drops out of the pip line and `onnxruntime` replaces it. | NCNN on ARM64 is currently unreliable under Ultralytics: recent versions raise `NotImplementedError` in `AutoBackend` ("NCNN inference is not supported on ARM64"), certain `ultralytics`+`torch` pairings break the export outright, and YOLO26n→NCNN export on Pi has been reported to die with `Illegal instruction (core dumped)`. NCNN is faster *when it works*; ONNX has far fewer ways to fail. Speed is recoverable (§6.5); a backend that won't load on demo day is not. |
| C11 | **Export flags changed to `half=False, simplify=True`.** v2 specified `half=True`. | Correct for NCNN, wrong for ONNX. ONNX Runtime's CPU execution provider has no fp16 fast path — fp16 weights get upcast at runtime or run slower. Use int8/dynamic quantization as the speed lever instead. |
| C12 | **Added a webcam frame source** alongside file and picamera2. | No test clip available at build time. `--source 0` gives an immediate dev target. |
| C13 | **Confirmed: camera is fixed forward-facing.** Resolves the open item flagged in C1 and §10. | Verified against the actual hardware. |

---

## 0c. Changelog (v2.1 → v3) — the subsystem no longer steers

| # | Change | Why |
|---|---|---|
| C14 | **The rover drives itself.** Straight lines, turning left when its own front sensor detects a blockage. This subsystem never commands motion. | Steering the rover toward a detected person proved too difficult to control in practice. |
| C15 | **Turn control, target selection and motor control are removed** — §6.8 and §6.9 are void. `control.py`, `selection.py` and `rover.py` are deleted, along with `KP`, `DEADBAND_DEG`, `MIN_TURN`, `TARGET_HOLD`, `SEARCH_HOLD`, `STOP_DISTANCE_M`, `APPROACH_SPEED` and `WATCHDOG_TIMEOUT`. | Nothing consumes them. This also removes the repo's riskiest code: the never-executed `GpioZeroRover`. |
| C16 | **The deliverable is a journey log**, schema `rescue.sighting.v1` — one record per person encountered, not one per frame. §6.4's per-frame `rescue.detection.v1` becomes an opt-in debug log at `v2`, minus the command fields. | At 10 FPS, driving past one person for 5 s wrote ~50 near-identical rows. The unit a reader cares about is the person. |
| C17 | **One saved image per sighting**, the highest-confidence frame. Frame rate-limiting and the output disk cap are deleted. | Retires §9's *"saved detection frames — unbounded, this is the one that bites"*, the largest risk to the 16 GB card. Three people means three JPEGs. |
| C18 | **There is no target.** Every confirmed person is recorded equally. | Target selection existed only to decide where to steer. |

**New coverage limitation, inherent to this design:** with a fixed 53.5°
forward camera and no scanning, anyone outside that forward cone is never seen.
Coverage is now a property of the rover's path, not of the vision system. Say
this out loud in the demo before a judge asks.

## 1. Overview

A camera-based subsystem mounted on an autonomously moving rover. The
camera is **fixed, facing forward** on the chassis; the rover sweeps its
surroundings by rotating its own body, scanning in real time for the
presence of humans (simulating post-disaster search-and-rescue scanning).

When a human is confirmed, the subsystem emits a detection event
containing the person's **bearing** (angle off the rover's centreline),
a **coarse distance estimate**, and a **normalized turn command** that
steers the rover to face them.

This PRD covers the detection subsystem and the turn command it produces.
It does **not** cover the rover's path planning, obstacle avoidance, or
autonomous navigation behaviour.

## 2. Problem Statement

In a disaster-response scenario, manually scanning a wide area for
survivors is slow and puts rescuers at risk. A rover that can move through
a space, sweep its surroundings, and flag likely human presence gives a
fast, low-risk way to narrow down where a human search team should focus
first.

For this hackathon, the subsystem is a proof-of-concept demonstrating that
this detect-and-orient loop works end-to-end on cheap, self-contained
hardware (no cloud, no external accelerator).

## 3. Goals & Success Metrics

| Goal | Metric | Target for demo |
|---|---|---|
| Detect a human in the camera's field of view | Detection triggers within camera FOV | Detects a person standing/lying within ~3–5 m in a demo environment |
| Run in real time while the rover turns | Sustained inference FPS | ≥10 FPS sustained on Pi 5 CPU |
| Low false-negative rate | Missed-detection rate on test walkthrough | Person detected in ≥90% of pass-by trials |
| Usable output for the rover | Detection event → signal | Every confirmed detection produces a timestamped event + annotated frame + turn command |
| **Correct orientation** | Bearing error after rover settles | Person within ±10° of frame centre within 2 s of first confirmed detection |
| **Stable tracking** | Track ID persistence | A single person keeps one track ID for ≥90% of the frames they are visible |
| **No oscillation** | Settling behaviour | Rover does not hunt/oscillate around centre — enters deadband and stops |

_Note: numeric targets are placeholders — tighten them after a first
benchmark pass on the actual hardware._

## 4. Target Users / Stakeholders

- **Primary user:** the rover's autonomy/navigation subsystem, which
  consumes detection events to decide what to do next (stop, approach,
  alert, mark location).
- **Secondary user:** hackathon judges/demo audience, who need to see
  detections clearly (annotated video feed or on-screen log).
- **Team:** whoever builds the navigation/drive subsystem consumes the
  interface in §6.4 and §6.8.

## 5. Scope

### In scope
- Capturing live video from the camera module
- Running person detection on the video stream in real time
- **Multi-object tracking** with stable IDs across frames
- **Bearing estimation** (horizontal angle to each tracked person)
- **Coarse distance estimation** from bounding-box geometry
- Emitting detection events (presence, confidence, track ID, bearing,
  distance, timestamp, annotated frame)
- **Emitting a normalized turn command** to orient the rover toward the
  highest-priority detected person
- A hardware abstraction layer for motor control, with a Pi 5 backend and
  a console/no-op backend for desktop testing
- Basic logging/visual output for demo purposes

### Out of scope
- Path planning, obstacle avoidance, SLAM, waypoint navigation
- Absolute localization of the detected person (GPS / map coordinates)
- Closed-loop odometry or IMU fusion — turn control is **vision-only
  proportional control**, no wheel encoders assumed
- Multi-camera fusion
- Cloud offload / network dependency (must run fully on-device)
- Training a custom model from scratch (fine-tuning is a stretch goal)
- Person re-identification across occlusions longer than the tracker's
  buffer

> **Scope boundary, stated plainly:** this subsystem answers *"is there a
> person, and which way should I turn to face them?"* It does not answer
> *"where should I go next?"*

## 6. Requirements

### 6.1 Functional Requirements

- **FR1:** Capture video frames continuously from the OV5647 camera module
  while the rover moves and turns.
- **FR2:** Run person detection on incoming frames without pausing capture.
- **FR3:** Output a discrete "human detected" event including confidence
  and timestamp when a person is confirmed.
- **FR4:** Save or expose an annotated frame (bounding box, track ID,
  bearing, distance drawn) for each detection.
- **FR5:** Provide a simple interface the rover's control logic can poll or
  subscribe to for detection events (§6.4).
- **FR6:** Assign and maintain a **stable track ID** per person across
  frames; do not re-fire a "new detection" event for an already-tracked
  person.
- **FR7:** Compute **bearing in degrees** for each tracked person, signed
  relative to the camera's optical axis (negative = left, positive =
  right).
- **FR8:** Compute a **coarse distance estimate** in metres for each
  tracked person, with an explicit validity flag when the estimate is
  unreliable (see §6.7).
- **FR9:** Select **one target** per frame (§6.8 priority rule) and emit a
  normalized turn command in `[-1.0, +1.0]` to orient toward it.
- **FR10:** Emit an explicit `stop` command when no person is tracked, when
  the target is inside the bearing deadband, or when the target is closer
  than the stop distance.
- **FR11:** Implement a **motion watchdog** — if no command is issued
  within `WATCHDOG_TIMEOUT` (default 0.5 s), motors stop automatically.
- **FR12:** Support two frame sources behind one interface: live
  `picamera2` capture (Pi) and a video file (desktop testing).

### 6.2 Non-Functional Requirements

- **NFR1 (Latency):** End-to-end frame capture → detection event under
  200 ms, so a person isn't missed as the rover sweeps past.
- **NFR2 (Throughput):** Sustain ≥10 FPS on Raspberry Pi 5 CPU-only
  inference (no accelerator).
- **NFR3 (Resource footprint):** Must fit a 16 GB SD card — see the disk
  budget in §9. RAM (16 GB) is not the binding constraint; **disk is**.
- **NFR4 (Reliability):** A single failed frame must not crash the
  pipeline. Log and continue.
- **NFR5 (Standalone operation):** No internet or cloud inference at run
  time. Model download/export happens once, ahead of the demo.
- **NFR6 (Safety):** Motors must default to stopped. Any uncaught
  exception, `KeyboardInterrupt`, or watchdog expiry stops the drive.
- **NFR7 (Portability):** The vision pipeline must run unchanged on
  Windows 11 (file input, console motor backend) and Raspberry Pi OS Lite
  (camera input, GPIO motor backend).

### 6.3 Data / Model Requirements

- Person-detection model runs fully offline on-device.
- Small enough for real-time CPU inference on Pi 5 (nano/small class).
- Stock COCO-trained weights acceptable; COCO class `0` = person.
- **Chosen models:** `YOLO26n` (scan pass) + `YOLO26s` (confirm pass), both
  exported to **ONNX** (changed in v2.1 — see C10). See §6.6.

### 6.4 Interface Requirements

The subsystem exposes detection events as **newline-delimited JSON**
(JSONL) — written to a log file, and optionally to a local UDP/UNIX socket
for the nav process to subscribe to. One object per confirmed detection
per frame.

```json
{
  "schema": "rescue.detection.v1",
  "timestamp": 1754640000.123,
  "frame_index": 412,
  "track_id": 3,
  "confidence": 0.87,
  "bbox_xyxy": [312, 118, 466, 502],
  "bearing_deg": -12.4,
  "distance_m": 3.2,
  "distance_valid": true,
  "is_target": true,
  "turn_command": -0.31,
  "drive_command": 0.0,
  "annotated_frame": "detections/frame_000412.jpg"
}
```

Field notes:

- `bearing_deg` — negative = person is left of centre, positive = right.
- `distance_valid` — `false` when the bbox is clipped by a frame edge or
  the person is judged non-upright; consumers must not trust `distance_m`
  when this is `false`.
- `is_target` — exactly one detection per frame has this `true` (the
  selected target); others are reported for situational awareness.
- `turn_command` / `drive_command` — normalized `[-1, +1]`. Repeated on
  every event row so a consumer reading a single line has everything.
- `annotated_frame` — relative path, or `null` if frame saving is
  rate-limited (see §9 disk budget).

### 6.5 Detection Pipeline

**Two-tier cascade, not a single model.**

| Tier | Model | Input size | Conf | Role |
|---|---|---|---|---|
| Scan (every frame) | `YOLO26n` | 480 px | 0.25 | Fast, recall-favouring first pass |
| Confirm (candidates only) | `YOLO26s` | 640 px | 0.45 | Slower, accurate second pass — this is what triggers an alert |

**Escalation rule (was ambiguous in v1, now specified):** run the confirm
pass on a frame if the scan pass returns ≥1 person box with confidence
≥ 0.25. Skip the confirm pass entirely if the scan pass is empty. To
protect FPS, the confirm pass is rate-limited to at most once every
`CONFIRM_MIN_INTERVAL` (default 0.15 s); between confirms, tracked targets
are carried forward by the tracker rather than re-confirmed.

**A track is promoted to "confirmed human"** once it has been confirmed by
the `YOLO26s` pass on `N_CONFIRM` frames (default 2) — this suppresses
single-frame false positives. Once promoted, a track stays confirmed until
the tracker drops it.

**Tracking:** ByteTrack via Ultralytics (`model.track(..., persist=True,
tracker="bytetrack.yaml")`), applied to the **scan** pass so every frame
feeds the tracker. Track IDs are the unit of alerting — one alert per new
confirmed track, not one per frame.

**Why YOLO26 over the alternatives:**

- **YOLO26** (Ultralytics, released 14 Jan 2026) is purpose-built for
  CPU/edge inference: it is natively **NMS-free** end-to-end, which removes
  a post-processing stage and cuts latency, and it benchmarks faster than
  YOLO11 on identical hardware at slightly higher mAP. Best current fit for
  "no accelerator, real CPU constraints."
- **YOLO11n** is the fallback if YOLO26-specific bugs appear — more
  battle-tested, a bit slower.
- **YOLOv8l / larger** ruled out — multi-second per-frame latency on Pi 5
  CPU, unusable on a moving platform.
- **MediaPipe person detector** — lighter, reasonable fallback if the
  cascade misses target FPS, but less robust in cluttered/occluded scenes,
  which matters more here than raw speed.
- **Why a cascade:** nano alone maximizes speed but produces more false
  positives on atypical poses; small/medium alone is too slow every frame.
  Running the accurate model only on flagged frames gets close to nano
  speed with better-than-nano precision.

> **NMS-free caveat worth knowing:** because YOLO26 is end-to-end, the
> usual `iou=` NMS threshold has no effect. Tune recall/precision with
> `conf=` only. Don't waste demo-day time tweaking an IoU knob that isn't
> wired to anything.

**Why ONNX (changed in v2.1 — was NCNN):** NCNN is Tencent's ARM-optimized
inference framework and is genuinely the fastest Ultralytics export format on
Raspberry Pi *when it loads*. It frequently does not. Recent Ultralytics
releases raise `NotImplementedError` in `AutoBackend` on ARM64 with "NCNN
inference is not supported on ARM64"; specific `ultralytics`+`torch` version
pairs break the export step; and exporting YOLO26n to NCNN on a Pi has been
reported to fail with `Illegal instruction (core dumped)`.

ONNX Runtime is slower than a working NCNN but loads reliably on arm64 and is a
first-class Ultralytics export target. The trade is deliberate: lost speed is
recoverable by dropping `SCAN_IMGSZ` or the confirm tier (§9 risk table); a
backend that refuses to load 30 minutes before a demo is not recoverable.

Model paths live in config, so if NCNN turns out to work on your specific image,
switching back is a one-line change — the loader takes a path and Ultralytics
infers the format from the extension.

### 6.6 Environment Setup

**On the Pi 5 (Raspberry Pi OS Lite, headless):**

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv libcamera-apps python3-venv

# --system-site-packages is REQUIRED so the venv can see the apt-installed
# picamera2 and opencv. Do not pip-install those two.
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Inside an activated venv, do NOT pass --break-system-packages.
# That flag is for installing into the system interpreter and is both
# unnecessary and misleading here. (v1 of this PRD had it — it was wrong.)
pip install ultralytics onnxruntime lap gpiozero lgpio

# Confirm the camera is detected:
libcamera-hello --list-cameras
```

**`lap` is not optional and is easy to miss.** Ultralytics does not declare it
as a hard dependency; it silently AutoUpdates it from PyPI the first time
`model.track()` runs. On a Pi with no network at run time (NFR5) that fails —
and it fails *late*, at the first tracked frame rather than at startup, which
makes it look like a tracker bug. Verified on Windows during setup: the first
`track()` call triggered a `lap>=0.5.12` download. Install it up front.

Two deliberate omissions from that `pip` line, both disk-driven:

- **`picamera2`** — provided by apt as `python3-picamera2`. The PyPI build
  needs libcamera bindings compiled and routinely fails on the Pi.
- **`opencv-python-headless`** — apt's `python3-opencv` is already there
  and is visible via `--system-site-packages`. Installing both wastes
  ~100 MB on a card that can't spare it.

**Export both models to ONNX (one-time, do this before demo day):**

```python
from ultralytics import YOLO

# Scan model — nano, small input, runs every frame
YOLO("yolo26n.pt").export(format="onnx", imgsz=480, half=False, simplify=True)
# -> creates yolo26n.onnx

# Confirm model — small, larger input, runs only on candidates
YOLO("yolo26s.pt").export(format="onnx", imgsz=640, half=False, simplify=True)
# -> creates yolo26s.onnx
```

`half=False` is deliberate (C11). ONNX Runtime's CPU execution provider has no
fp16 fast path; exporting fp16 gains nothing and can cost. If you need more
speed, reach for int8/dynamic quantization or a smaller `imgsz`, not fp16.

`yolo26n.pt` and `yolo26s.pt` download automatically from Ultralytics
assets on first use — this is the **only** step that needs internet.

**Prefer exporting on the desktop and copying the `.onnx` files to the Pi.**
ONNX artifacts are portable, and desktop export avoids the ARM-specific export
failures described in §6.5 as well as a slow, memory-hungry export on the SD
card.

**Camera configuration — motion blur is a requirement, not polish:**

```python
picam2.set_controls({
    "AeEnable": False,        # auto-exposure will pick long exposures indoors
    "ExposureTime": 4000,     # microseconds; 4 ms. Lower = less blur, darker
    "AnalogueGain": 4.0,      # compensate the brightness lost above
})
```

The OV5647 is a rolling-shutter sensor. Auto-exposure indoors will happily
choose 20–30 ms, which smears a person into the background while the rover
is turning. Fix the exposure short and raise gain to compensate; accept the
extra sensor noise.

> **Measured caveat — this trade may be tuned the wrong way (2026-08-08).**
> Detection was measured on 36 clip frames with `yolo26n` @480 px, conf 0.25,
> against synthetic degradations:
>
> | Degradation | Detections retained | Mean conf |
> |---|---|---|
> | Resolution quartered | 80% | 0.73 |
> | Darkness alone (30–50%) | 95–97% | 0.68–0.72 |
> | **Darkness + gain noise** | **64%** | 0.74 |
> | Motion blur 9–15 px | 84–87% | 0.63–0.75 |
> | Realistic Pi (dark+noise+blur) | 77% | 0.71 |
>
> Blur costs 13–16%. Gain noise costs 36%. So "short exposure, high gain"
> buys a cheap fix for a *more expensive* problem. **Test both settings in the
> real demo lighting** rather than accepting 4 ms / 4× as given — a longer
> exposure at lower gain may well detect better even with visible smear.
>
> Also note resolution is a non-issue: the sensor has far more pixels than the
> model consumes. And what degrades is *recall*, not confidence — surviving
> detections stay around 0.71. Lower `SCAN_CONF` to ~0.15 to recover recall;
> `N_CONFIRM` and the confirm tier still guard precision.
>
> Caveats on the method: blur was modelled as a linear horizontal smear, which
> likely understates rolling-shutter skew during fast rotation, and Gaussian
> noise is only an approximation of sensor noise. Trust the direction more than
> the exact magnitudes.

### 6.7 Bearing & Distance Estimation

**Camera intrinsics (Camera Module v1 / OV5647):** horizontal FOV
**53.5°**, vertical FOV 41.4°. Put these in config — if you swap to a
Camera Module v2 or v3, both numbers change and every estimate below
silently goes wrong.

**Bearing** — from the horizontal position of the bbox centre:

```
cx          = (x1 + x2) / 2
bearing_deg = ((cx / frame_width) - 0.5) * HFOV_DEG
```

Negative = left of centre. This is a small-angle-friendly linear
approximation; a `tan`-based version is correct everywhere:

```
f_px        = frame_width / (2 * tan(radians(HFOV_DEG) / 2))
bearing_deg = degrees(atan((cx - frame_width / 2) / f_px))
```

Use the `tan` version — it costs nothing and is right everywhere.

Note the two forms agree **exactly** at the frame centre and at both edges;
that is forced by construction, since the `tan` form is defined to return
±HFOV/2 at the edges. They diverge in *between*, peaking near the middle of
each half — about 0.8° at three-quarters across a 640 px frame with this HFOV.
So the linear approximation's error is worst mid-frame, not at the edges as
v1/v2 of this PRD implied. Verified by
`tests/test_geometry.py::test_tan_and_linear_forms_agree_at_centre_and_edges`.

**Distance** — pinhole model on bounding-box height:

```
f_px        = frame_width / (2 * tan(radians(HFOV_DEG) / 2))
distance_m  = (ASSUMED_HUMAN_HEIGHT_M * f_px) / bbox_height_px
```

with `ASSUMED_HUMAN_HEIGHT_M = 1.7`.

**This estimate is coarse and fails in specific, predictable ways.** Set
`distance_valid = false` when any of these hold:

| Condition | Why it breaks |
|---|---|
| bbox touches top or bottom frame edge (within 2 px) | Person is clipped; bbox height understates true height, so distance is over-estimated |
| bbox aspect ratio `h/w < 1.2` | Person is likely lying down or crouching — the 1.7 m height assumption is invalid. **This matters: lying-down people are the actual rescue target.** |
| bbox height < 40 px | Quantization error dominates; estimate is noise |
| `distance_m` outside [0.3, 30] m | Physically implausible for this camera and scene |

Expected accuracy when valid: **±25–30%** for an upright, fully visible
adult. Good enough for "approach vs. stop", not good enough for anything
you'd put on a map. Do not let this creep into being treated as
localization — §5 keeps localization out of scope for a reason.

**Smoothing:** apply an exponential moving average per track ID
(`alpha = 0.4`) to both bearing and distance before they reach the
controller. Raw per-frame bearing is noisy enough to make the rover
visibly twitch.

### 6.8 Target Selection & Turn Control

**Target selection**, when multiple people are tracked, in order:

1. Confirmed tracks only (promoted per §6.5).
2. Among those, the **nearest** with `distance_valid = true`.
3. If none have a valid distance, the one with the **largest bbox area**
   (a reasonable proxy for nearest).
4. Tie-break on lowest track ID for determinism.

Once selected, a target is **sticky** for `TARGET_HOLD` (default 1.0 s)
even if another briefly scores higher — this stops the rover flip-flopping
between two people.

**Turn control** — proportional, with a deadband:

```
error_deg = target.bearing_deg           # 0 == centred

if |error_deg| <= DEADBAND_DEG:          # default 5°
    turn = 0.0                           # on target, stop turning
else:
    turn = clamp(KP * error_deg, -1.0, +1.0)     # KP default 0.02
    if 0 < |turn| < MIN_TURN:            # default 0.25
        turn = copysign(MIN_TURN, turn)  # overcome motor stiction
```

Sign convention: **positive `turn` = rover rotates clockwise / to its
right.** Write this on the chassis with a marker. Getting it backwards is
the single most common way this demo fails, and it looks identical to a
detection bug from the outside.

Why each constant exists:

- **`DEADBAND_DEG`** stops the classic P-controller hunt around zero. With
  no deadband and no derivative term, the rover oscillates forever.
- **`MIN_TURN`** exists because a DC motor at 8% duty cycle doesn't turn,
  it just buzzes. Below the stiction floor, a small error produces no
  motion but the controller thinks it commanded some.
- **`KP = 0.02`** maps a 25° error to a 0.5 turn command. Tune on the
  actual chassis — carpet and hard floor behave differently enough to
  matter.

**Drive control** (deliberately minimal — approach behaviour is a stretch
goal, not a demo requirement):

```
if target is None:                         drive = 0.0
elif |error_deg| > DEADBAND_DEG:           drive = 0.0   # turn in place first
elif distance_valid and distance_m < STOP_DISTANCE_M:   # default 1.5 m
                                           drive = 0.0
else:                                      drive = APPROACH_SPEED  # default 0.3
```

Turn-in-place-then-advance is chosen over simultaneous turn-and-drive
because it is far easier to debug on the day and looks more deliberate in
a demo.

### 6.9 Motor Control Hardware & Abstraction

**Chosen backend: L298N or TB6612FNG H-bridge driven by `gpiozero`.** This
is the most common Pi rover setup by a wide margin, and `gpiozero` is the
Raspberry Pi Foundation's recommended library for new projects.

**Pi 5 constraints — these are hard:**

- **`RPi.GPIO` does not work on Pi 5.** The RP1 southbridge needs different
  kernel drivers. `pigpio` is also unsupported. Use `gpiozero` backed by
  `lgpio`.
- `gpiozero`'s pin factory may need pinning depending on kernel version —
  the RP1 has appeared as both `/dev/gpiochip4` and `/dev/gpiochip0` across
  kernel releases. Verify with `ls /dev/gpiochip*` on your actual image
  rather than trusting a tutorial.
- PWM here is **software-generated**, so expect mild speed jitter under CPU
  load. Fine for a rover; it would not be fine for precise servos.

**Considered and rejected for this build:**

| Option | Why not |
|---|---|
| I²C motor HAT (Waveshare / Adafruit PCA9685) | Genuinely better — hardware PWM, immune to CPU load. Rejected only because it's extra hardware to source. **Swap to this if you already own one**; only the backend class changes. |
| Offload to Pico/ESP32 over serial | Most robust, best safety story. Rejected as two codebases under hackathon time pressure. |
| ROS 2 `/cmd_vel` | Only worth it if the stack is already ROS. It isn't. |

**Abstraction.** All motor access goes through one interface so the vision
half can be tested with no hardware at all:

```python
class RoverController(ABC):
    @abstractmethod
    def drive(self, turn: float, forward: float) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    def close(self) -> None: ...
```

| Implementation | Use |
|---|---|
| `ConsoleRover` | Logs commands to stdout/JSONL. Windows 11 clip testing. **Default.** |
| `GpioZeroRover` | Real L298N/TB6612 via `gpiozero.Motor`. Pi 5 only; imports `gpiozero` lazily so the module still imports on Windows. |

Differential mixing, with clamping applied **after** mixing so a
simultaneous turn+forward request can't silently saturate one side:

```
left  = clamp(forward + turn, -1.0, +1.0)
right = clamp(forward - turn, -1.0, +1.0)
```

**Safety requirements on any backend:**

- `stop()` on `KeyboardInterrupt`, on any uncaught exception, and in a
  `finally` block. Motors must never be left running by a crash.
- Watchdog thread: if `drive()` hasn't been called within
  `WATCHDOG_TIMEOUT` (0.5 s), call `stop()`. A vision pipeline that hangs
  must not leave a rover driving into a wall.
- **Bench-test with the wheels off the ground first.** Every time.

**Pin configuration** (edit to match your wiring — placeholders):

```python
MOTOR_PINS = {
    "left":  {"forward": 17, "backward": 18, "enable": 12},
    "right": {"forward": 22, "backward": 23, "enable": 13},
}
```

### 6.10 Development & Test Path (Windows 11)

The vision pipeline is validated on a recorded clip on Windows 11 before it
ever runs on the Pi.

| | Windows 11 (dev/test) | Raspberry Pi 5 (deploy) |
|---|---|---|
| Frame source | `cv2.VideoCapture(file)` or webcam index | `picamera2` |
| Models | `yolo26n.pt` / `yolo26s.pt` direct | `yolo26n.onnx` / `yolo26s.onnx` |
| Motor backend | `ConsoleRover` | `GpioZeroRover` |
| Output | Annotated MP4 + JSONL | JSONL + rate-limited frames |
| Display | Optional `cv2.imshow` | Headless — never call `imshow` |

Both paths sit behind a `FrameSource` interface, selected by CLI flag, so
the detection, tracking, bearing, distance, and control code is byte-for-
byte identical between them. The only things that change are the two ends
of the pipe.

Note that `.pt` on Windows and ONNX on the Pi will give **slightly
different confidence values** for the same frame — a different backend and
graph simplification. Don't tune `conf` thresholds to three decimal places on
Windows and expect them to transfer. Tune coarsely on the clip, re-check on
the Pi.

**Acceptance test for the clip:** the pipeline must report at least one
confirmed track with a plausible bearing sign — person visibly on the left
of frame yields negative `bearing_deg` — and must produce an annotated
output video plus a JSONL log with no unhandled exceptions.

## 7. Technical Constraints & Assumptions

- **Hardware:** Raspberry Pi 5 Model B, 16 GB RAM, standalone (no Hailo/
  Coral accelerator) — CPU-only inference.
- **Storage:** 16 GB SD card, Raspberry Pi OS **Lite** (headless). This is
  the binding constraint — see the disk budget in §9.
- **Camera:** Camera Module v1 (OV5647), CSI ribbon, **fixed forward-facing
  on the chassis** (changed in v2 — see C1).
- **Motors:** DC motors via L298N/TB6612FNG H-bridge, `gpiozero` + `lgpio`.
- **Assumption:** no wheel encoders, no IMU. Turn control is vision-only
  closed loop — the rover knows it has turned far enough only because the
  person moved toward frame centre. If the person leaves the frame
  entirely, the controller has no memory of which way they went.
  _Mitigation: on target loss, continue the last commanded turn direction
  for `SEARCH_HOLD` (default 0.7 s) before stopping._
- **Assumption:** camera capture and rover motion are not synchronized —
  the camera does not pause for frames. Short fixed exposure is therefore a
  requirement (§6.6).
- **Assumption:** demo runs in a controlled indoor/outdoor test area, not
  an actual disaster site.
- **Assumption:** one to three people in frame at once. The target
  selection rule in §6.8 is not designed for crowds.

## 8. Milestones / Timeline

_Fill in actual hours/dates for your event._

| Phase | Deliverable | Gate |
|---|---|---|
| 0. Clip test (Win 11) | Detection + tracking + bearing/distance running on a recorded clip, `ConsoleRover` logging turn commands | Annotated MP4 + JSONL, bearing sign correct |
| 1. Pi setup | Camera verified, both models exported to NCNN, disk budget checked | `libcamera-hello --list-cameras` OK; `df -h` shows ≥2 GB free |
| 2. Core loop | Real-time detection on live camera feed | ≥10 FPS sustained |
| 3. Motors (bench) | `GpioZeroRover` driving motors **with wheels off the ground** | Correct direction per turn sign |
| 4. Closed loop | Rover on the floor, turns to face a person | Person centred within ±10°, no oscillation |
| 5. Polish | Annotated output, false-positive tuning, disk cleanup | — |
| 6. Demo | End-to-end: rover sweeps, flags a person, turns to face them | — |

Phase 3's "wheels off the ground" gate is not optional padding. A sign
error in §6.8 sends the rover away from the person at full speed, and on
the floor that is how you lose a chassis.

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **16 GB SD card runs out of space** — `ultralytics` pulls PyTorch (~1.5–2.5 GB on arm64) | Build breaks mid-hackathon. **Most likely failure mode.** | Budget explicitly (below). Cap saved frames. `df -h` at the end of every phase. |
| Turn-direction sign inverted | Rover turns away from the person; looks exactly like a detection bug | Bench-test wheels-up before floor testing; label the chassis |
| P-controller oscillation / hunting | Rover twitches, never settles; looks broken in the demo | Deadband + `MIN_TURN` + EMA smoothing (§6.7, §6.8) |
| CPU-only inference too slow during sweeps | Missed detections | **Raise `CONFIRM_MIN_INTERVAL` first** — measurements below show the confirm tier is ~60% of the frame budget. Shrink `SCAN_IMGSZ` only after that. (v1/v2 had this order backwards.) |
| Motion blur from continuous rotation | Missed or low-confidence detections | Fixed short exposure + raised gain (§6.6) |
| Distance estimate wildly wrong for lying-down people | Rover mis-judges approach — and lying-down people are the actual rescue target | `distance_valid = false` on low aspect ratio; never gate an *alert* on distance, only the approach behaviour |
| Stock COCO model misses atypical poses | Lower real-world accuracy than benchmarks suggest | Test against lying/occluded/partially-visible poses during prep, not just standing people |
| Tracker ID switches when people cross | Duplicate alerts, target flip-flop | `TARGET_HOLD` stickiness; accept ID switches as a known limitation at this scope |
| Vision pipeline hangs while motors run | Rover drives into something | Watchdog (FR11) |
| No accelerator hardware | Lower accuracy ceiling vs. NFR targets | Scope FPS/latency targets to CPU reality; accelerator is documented future work, not a blocker |

**Measured throughput (2026-08-08).** End-to-end pipeline on an Intel Core
Ultra 5 226V desktop, `.pt` backend, 640×480 clip, frame saving off:

| Configuration | ms/frame | FPS |
|---|---|---|
| Cascade: scan 480 + confirm 640 @ 0.15 s (PRD default) | 108 | **9.2** |
| Cascade with `CONFIRM_MIN_INTERVAL = 0.5` | 58 | 17.2 |
| Scan tier only (confirm disabled) | 46 | 21.6 |
| Scan only @ 320 px | 27 | 37.1 |

**Read this as a warning.** The PRD default manages 9.2 FPS on a desktop CPU
roughly 4–6× faster than a Pi 5. NFR2 asks for ≥10 FPS *on the Pi*, and the
default configuration will not get there — expect ~2 FPS unmodified.

The confirm tier is ~60% of the frame budget, so it is the first thing to trim,
not the scan resolution. ONNX should recover some of the gap versus `.pt`, but
plan on running with `--confirm-min-interval 0.5` or `--no-confirm` on the Pi
and re-benchmarking there. These numbers are the desktop baseline to compare
the Pi against, not a prediction of it.

**Disk budget (16 GB card, Raspberry Pi OS Lite).** Check this in Phase 1,
not on demo day:

| Item | Approx |
|---|---|
| Raspberry Pi OS Lite + apt packages (picamera2, opencv, libcamera) | ~2.5 GB |
| PyTorch + torchvision (arm64 CPU wheels, pulled in by `ultralytics`) | ~1.5–2.5 GB |
| `ultralytics` + `onnxruntime` + remaining deps | ~0.5 GB |
| `.pt` weights (`yolo26n` + `yolo26s`) | ~0.1 GB |
| ONNX exported models | ~0.1 GB |
| Saved detection frames | **unbounded — this is the one that bites** |
| **Free headroom target** | **≥2 GB** |

Rate-limit saved frames to at most 1 per confirmed track per second, cap
the output directory at ~500 MB, and delete oldest-first when it fills.

Note that exporting to ONNX does **not** let you skip PyTorch — the
`ultralytics` package imports torch regardless of which backend runs
inference. The export format buys you speed, not disk.

## 10. Open Questions

**Resolved in v2:**

- ~~Event format/protocol for the nav subsystem?~~ → JSONL, schema in §6.4.
- ~~What does "detected" trigger physically?~~ → Turn to face the person,
  stop at 1.5 m. Approach behaviour is minimal by design (§6.8).
- ~~Camera on a rotating platform or fixed?~~ → Fixed forward; the rover's
  own yaw sweeps. **Confirmed against the actual hardware, 2026-08-08.**
- ~~Inference backend?~~ → ONNX Runtime, not NCNN. See C10.

**Still open:**

- Is there a fixed demo environment / lighting condition to test against
  ahead of time? Directly affects the fixed exposure value in §6.6 — that
  number needs to be set in the actual demo lighting.
- Confirm exact hackathon timeline/deadlines for §8.
- Actual motor pin assignments and whether the chassis uses L298N or
  TB6612FNG (§6.9).
- Does the demo need the rover to *approach* the person, or is turning to
  face them sufficient? Currently scoped as turn-only with a minimal
  approach; a real approach behaviour needs obstacle handling, which is
  out of scope.
- Wheel base / turning characteristics — needed to pick a sensible starting
  `KP`, which is currently a guess.

---

## Appendix A: Configuration Constants

Every tunable in one place. These are starting values, not tuned ones.

```python
# --- Camera (OV5647 / Camera Module v1) ---
HFOV_DEG              = 53.5     # CHANGE if using Camera Module v2/v3
VFOV_DEG              = 41.4
FRAME_WIDTH           = 640
FRAME_HEIGHT          = 480
EXPOSURE_TIME_US      = 4000     # tune in actual demo lighting
ANALOGUE_GAIN         = 4.0

# --- Detection ---
SCAN_MODEL            = "yolo26n.onnx"         # "yolo26n.pt" on Windows
CONFIRM_MODEL         = "yolo26s.onnx"         # "yolo26s.pt" on Windows
SCAN_IMGSZ            = 480
CONFIRM_IMGSZ         = 640
SCAN_CONF             = 0.25
CONFIRM_CONF          = 0.45
PERSON_CLASS_ID       = 0
CONFIRM_MIN_INTERVAL  = 0.15     # seconds between confirm passes
N_CONFIRM             = 2        # confirms before a track is "human"

# --- Tracking ---
TRACKER_CFG           = "bytetrack.yaml"
EMA_ALPHA             = 0.4      # bearing/distance smoothing

# --- Distance ---
ASSUMED_HUMAN_HEIGHT_M = 1.7
MIN_BBOX_HEIGHT_PX     = 40
MIN_ASPECT_RATIO       = 1.2     # h/w below this -> not upright
DISTANCE_RANGE_M       = (0.3, 30.0)

# --- Control / Safety ---
# REMOVED IN v3 (C15). The rover drives itself, so nothing in this subsystem
# can command motion. DEADBAND_DEG, KP, MIN_TURN, TARGET_HOLD, SEARCH_HOLD,
# STOP_DISTANCE_M, APPROACH_SPEED and WATCHDOG_TIMEOUT are all gone.

# --- Output ---
SAVE_FRAMES           = True     # one best frame per sighting
JPEG_QUALITY          = 90
CONFIDENCE_SAMPLE_INTERVAL = 1.0 # display only; the log stays per-frame
# FRAME_SAVE_INTERVAL and MAX_OUTPUT_DIR_MB removed in v3 (C17): saving one
# image per sighting bounds disk by people seen rather than frames run.
```

## Appendix B: Sign Conventions

One quantity survives v3. `turn_command` and `drive_command` were removed with
the control layer (C15) — this subsystem no longer commands the rover.

| Quantity | Negative | Positive |
|---|---|---|
| `bearing_deg` | Person is **left** of frame centre | Person is **right** of frame centre |

Bearing is measured off the rover's heading **at the moment of the sighting**.
With no odometry and no record of the rover's own turns, it cannot be converted
into a position — the log says a person was seen 12° to the left 47 seconds in,
not where that person was.
