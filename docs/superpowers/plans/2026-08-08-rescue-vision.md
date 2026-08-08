# Rescue Rover Vision Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only person-detection subsystem that emits, per frame, each tracked person's bearing and coarse distance plus a normalized turn command to orient a rescue rover toward the highest-priority person.

**Architecture:** A pure decision core (geometry → tracking → selection → control) with no I/O and no clock reads, bracketed by impure edges: a `FrameSource` (video file / webcam / picamera2), a two-tier Ultralytics detector cascade, a JSONL event writer, and a `RoverController` (console / gpiozero). The pipeline depends on a `Detector` *protocol*, so an end-to-end smoke test can inject scripted detections and exercise every stage with no model, no camera, and no motors.

**Tech Stack:** Python 3.13, Ultralytics 8.4 (YOLO26n + YOLO26s), ONNX Runtime (Pi) / PyTorch `.pt` (Windows), OpenCV, `gpiozero` + `lgpio` (Pi only), pytest.

## Global Constraints

- Source of truth for all behaviour and constants: `PRD.md` v2.1. Constant values are copied verbatim from PRD Appendix A.
- **Sign convention (PRD Appendix B):** negative `bearing_deg` = person left of centre; positive `turn_command` = rover rotates clockwise/right. `turn_command` takes the **same sign** as `bearing_deg`.
- **Pure core:** `geometry.py`, `tracking.py`, `selection.py`, `control.py` must not import `cv2`, `ultralytics`, `gpiozero`, or call `time.time()`. Time is a parameter.
- **Lazy imports:** `picamera2` and `gpiozero` are imported inside functions, never at module top level. The whole package must import cleanly on Windows.
- **Person only:** every inference call passes `classes=[0]`.
- **Motors default to stopped.** `stop()` on `KeyboardInterrupt`, on any uncaught exception, and in a `finally`.
- Python: 3.13. Run everything through `.venv/Scripts/python.exe` on Windows.
- Every commit message ends with the trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- Never commit `*.pt`, `*.onnx`, `detections/`, or `*.jsonl` (already in `.gitignore`).

## File Structure

| File | Responsibility |
|---|---|
| `rescue_vision/config.py` | All PRD Appendix A constants as a frozen dataclass |
| `rescue_vision/types.py` | Shared dataclasses: `BBox`, `RawDetection`, `Measurement`, `TrackState`, `Command` |
| `rescue_vision/geometry.py` | Bearing, focal length, distance, distance validity, IoU |
| `rescue_vision/tracking.py` | Per-track state, EMA smoothing, confirm matching, promotion |
| `rescue_vision/selection.py` | Target priority rule + `TARGET_HOLD` stickiness |
| `rescue_vision/control.py` | Deadband P-controller, drive rule, differential mixing |
| `rescue_vision/rover.py` | `RoverController` ABC + watchdog, `ConsoleRover`, `GpioZeroRover` |
| `rescue_vision/events.py` | `rescue.detection.v1` JSONL writer, frame saving, disk cap |
| `rescue_vision/annotate.py` | Overlay drawing |
| `rescue_vision/frame_source.py` | `FrameSource` ABC + video file / webcam / picamera2 |
| `rescue_vision/detector.py` | `Detector` protocol + `CascadeDetector` |
| `rescue_vision/pipeline.py` | Per-frame orchestration and the run loop |
| `rescue_vision/cli.py` | Arg parsing, wiring, signal handling |
| `scripts/export_models.py` | One-time ONNX export |

Tasks 1–5 build the pure core bottom-up. Tasks 6–9 build the edges. Tasks 10–12 wire it together. Each task ends green.

---

### Task 1: Package skeleton, config, and shared types

**Files:**
- Create: `rescue_vision/__init__.py`, `rescue_vision/config.py`, `rescue_vision/types.py`
- Test: `tests/test_config.py`, `tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config` (frozen dataclass, all PRD Appendix A fields); `BBox(x1,y1,x2,y2)` with properties `width`, `height`, `cx`, `cy`, `area`, `aspect_ratio`; `RawDetection(bbox, confidence, track_id)`; `Measurement(bearing_deg, distance_m, distance_valid, invalid_reason)`; `TrackState`; `Command(turn, drive)`.

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from rescue_vision.config import Config
from rescue_vision.types import BBox


def test_config_defaults_match_prd_appendix_a():
    c = Config()
    assert c.hfov_deg == 53.5
    assert c.assumed_human_height_m == 1.7
    assert c.deadband_deg == 5.0
    assert c.kp == 0.02
    assert c.min_turn == 0.25
    assert c.stop_distance_m == 1.5
    assert c.watchdog_timeout == 0.5
    assert c.person_class_id == 0


def test_config_is_immutable():
    import dataclasses
    import pytest
    c = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.kp = 0.5


def test_bbox_geometry_properties():
    b = BBox(10.0, 20.0, 40.0, 120.0)
    assert b.width == 30.0
    assert b.height == 100.0
    assert b.cx == 25.0
    assert b.cy == 70.0
    assert b.area == 3000.0
    assert b.aspect_ratio == pytest.approx(100.0 / 30.0)


def test_bbox_aspect_ratio_of_zero_width_is_zero_not_a_crash():
    assert BBox(5.0, 5.0, 5.0, 50.0).aspect_ratio == 0.0
```

Add `import pytest` at the top of the file (used by two tests).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/__init__.py`:
```python
"""Real-time human detection and bearing subsystem for an autonomous rescue rover."""

__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

`rescue_vision/types.py`:
```python
"""Shared data structures passed between pipeline stages.

Absolute imports mean `rescue_vision.types` never shadows the stdlib `types`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates, xyxy order."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Height / width. Zero for a degenerate box rather than raising."""
        return self.height / self.width if self.width > 0 else 0.0

    def as_xyxy_ints(self) -> list[int]:
        return [int(round(v)) for v in (self.x1, self.y1, self.x2, self.y2)]


@dataclass(frozen=True)
class RawDetection:
    """One person box straight out of the detector, before any smoothing."""

    bbox: BBox
    confidence: float
    track_id: int | None = None


@dataclass(frozen=True)
class Measurement:
    """Geometry derived from a single bbox."""

    bearing_deg: float
    distance_m: float
    distance_valid: bool
    invalid_reason: str | None = None


@dataclass
class TrackState:
    """Smoothed, accumulated state for one track ID. Mutable by design."""

    track_id: int
    bbox: BBox
    confidence: float
    bearing_deg: float
    distance_m: float
    distance_valid: bool
    invalid_reason: str | None = None
    confirm_count: int = 0
    confirmed: bool = False
    last_seen_frame: int = -1


@dataclass(frozen=True)
class Command:
    """Normalized rover command, both components in [-1, +1]."""

    turn: float
    drive: float
```

`rescue_vision/config.py`:
```python
"""All tunables in one place. Values copied verbatim from PRD Appendix A.

These are starting values, not tuned ones. Tune on the real chassis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Camera (OV5647 / Camera Module v1) ---
    # CHANGE hfov_deg if using Camera Module v2/v3 or every bearing and
    # distance below silently goes wrong.
    hfov_deg: float = 53.5
    vfov_deg: float = 41.4
    frame_width: int = 640
    frame_height: int = 480
    exposure_time_us: int = 4000  # tune in actual demo lighting
    analogue_gain: float = 4.0

    # --- Detection ---
    scan_model: str = "yolo26n.pt"      # "yolo26n.onnx" on the Pi
    confirm_model: str = "yolo26s.pt"   # "yolo26s.onnx" on the Pi
    scan_imgsz: int = 480
    confirm_imgsz: int = 640
    scan_conf: float = 0.25
    confirm_conf: float = 0.45
    person_class_id: int = 0
    confirm_min_interval: float = 0.15
    n_confirm: int = 2
    confirm_iou_match: float = 0.5  # IoU to associate a confirm box to a track

    # --- Tracking ---
    tracker_cfg: str = "bytetrack.yaml"
    ema_alpha: float = 0.4
    track_max_age_frames: int = 30  # drop tracks unseen this long

    # --- Distance ---
    assumed_human_height_m: float = 1.7
    min_bbox_height_px: float = 40.0
    min_aspect_ratio: float = 1.2
    distance_min_m: float = 0.3
    distance_max_m: float = 30.0
    edge_margin_px: float = 2.0

    # --- Control ---
    deadband_deg: float = 5.0
    kp: float = 0.02
    min_turn: float = 0.25
    target_hold: float = 1.0
    search_hold: float = 0.7
    stop_distance_m: float = 1.5
    approach_speed: float = 0.3

    # --- Safety ---
    watchdog_timeout: float = 0.5

    # --- Output ---
    save_frames: bool = True
    frame_save_interval: float = 1.0
    max_output_dir_mb: int = 500
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/ tests/
git commit -m "feat: add package skeleton, config, and shared types"
```

---

### Task 2: Geometry — bearing, distance, validity, IoU

**Files:**
- Create: `rescue_vision/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `Config`, `BBox`, `Measurement` from Task 1.
- Produces:
  - `focal_px(frame_width: int, hfov_deg: float) -> float`
  - `bearing_deg(cx: float, frame_width: int, hfov_deg: float) -> float`
  - `distance_m(bbox_height_px: float, f_px: float, assumed_height_m: float) -> float`
  - `distance_validity(bbox: BBox, frame_w: int, frame_h: int, dist_m: float, cfg: Config) -> tuple[bool, str | None]`
  - `measure(bbox: BBox, frame_w: int, frame_h: int, cfg: Config) -> Measurement`
  - `iou(a: BBox, b: BBox) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_geometry.py`:
```python
import math

import pytest

from rescue_vision.config import Config
from rescue_vision.geometry import (
    bearing_deg,
    distance_m,
    distance_validity,
    focal_px,
    iou,
    measure,
)
from rescue_vision.types import BBox

CFG = Config()


def test_bearing_is_zero_at_frame_centre():
    assert bearing_deg(320.0, 640, 53.5) == pytest.approx(0.0)


def test_bearing_is_negative_left_of_centre():
    """PRD Appendix B: negative == person is LEFT of centre."""
    assert bearing_deg(100.0, 640, 53.5) < 0.0


def test_bearing_is_positive_right_of_centre():
    assert bearing_deg(540.0, 640, 53.5) > 0.0


def test_bearing_at_frame_edge_is_half_the_hfov():
    """The tan form must return exactly +HFOV/2 at the right edge."""
    assert bearing_deg(640.0, 640, 53.5) == pytest.approx(53.5 / 2, abs=1e-6)
    assert bearing_deg(0.0, 640, 53.5) == pytest.approx(-53.5 / 2, abs=1e-6)


def test_bearing_is_antisymmetric_about_centre():
    left = bearing_deg(320.0 - 150.0, 640, 53.5)
    right = bearing_deg(320.0 + 150.0, 640, 53.5)
    assert left == pytest.approx(-right)


def test_tan_form_differs_from_linear_approximation_at_the_edge():
    """Guards against someone 'simplifying' back to the linear form.

    PRD 6.7 requires the tan form because the edge is where a person first
    appears during a sweep.
    """
    linear = ((640.0 / 640) - 0.5) * 53.5
    assert abs(bearing_deg(640.0, 640, 53.5) - linear) > 0.5


def test_focal_px_matches_pinhole_formula():
    expected = 640 / (2 * math.tan(math.radians(53.5) / 2))
    assert focal_px(640, 53.5) == pytest.approx(expected)


def test_distance_of_a_person_filling_the_frame_height():
    f = focal_px(640, 53.5)
    assert distance_m(480.0, f, 1.7) == pytest.approx(1.7 * f / 480.0)


def test_distance_halves_when_bbox_height_doubles():
    f = focal_px(640, 53.5)
    assert distance_m(200.0, f, 1.7) == pytest.approx(2 * distance_m(400.0, f, 1.7))


def test_distance_of_zero_height_bbox_is_infinite_not_a_crash():
    assert math.isinf(distance_m(0.0, 500.0, 1.7))


def test_distance_invalid_when_bbox_clipped_at_bottom():
    bbox = BBox(300.0, 100.0, 340.0, 479.0)  # within edge_margin_px of h=480
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "clipped_bottom"


def test_distance_invalid_when_bbox_clipped_at_top():
    bbox = BBox(300.0, 1.0, 340.0, 300.0)
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "clipped_top"


def test_distance_invalid_for_lying_down_person():
    """A prone person is the actual rescue target -- must be flagged, not trusted."""
    bbox = BBox(100.0, 200.0, 400.0, 260.0)  # 300 wide, 60 tall -> ratio 0.2
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "not_upright"


def test_distance_invalid_when_bbox_too_short():
    bbox = BBox(300.0, 200.0, 320.0, 230.0)  # 30 px tall, below the 40 px floor
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is False
    assert reason == "bbox_too_small"


def test_distance_invalid_when_out_of_physical_range():
    bbox = BBox(300.0, 100.0, 340.0, 300.0)
    valid, reason = distance_validity(bbox, 640, 480, 99.0, CFG)
    assert valid is False
    assert reason == "implausible_distance"


def test_distance_valid_for_upright_fully_visible_person():
    bbox = BBox(300.0, 100.0, 360.0, 400.0)  # 60x300, ratio 5.0, no edge contact
    valid, reason = distance_validity(bbox, 640, 480, 3.0, CFG)
    assert valid is True
    assert reason is None


def test_measure_combines_bearing_and_distance():
    bbox = BBox(100.0, 100.0, 160.0, 400.0)
    m = measure(bbox, 640, 480, CFG)
    assert m.bearing_deg < 0.0  # left of centre
    assert m.distance_m > 0.0
    assert m.distance_valid is True


def test_iou_of_identical_boxes_is_one():
    b = BBox(0.0, 0.0, 10.0, 10.0)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou(BBox(0.0, 0.0, 10.0, 10.0), BBox(50.0, 50.0, 60.0, 60.0)) == 0.0


def test_iou_of_half_overlapping_boxes():
    a = BBox(0.0, 0.0, 10.0, 10.0)
    b = BBox(5.0, 0.0, 15.0, 10.0)
    assert iou(a, b) == pytest.approx(50.0 / 150.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.geometry'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/geometry.py`:
```python
"""Pinhole geometry: bbox -> bearing and coarse distance. PRD 6.7.

Pure module. No I/O, no clock, no cv2.
"""

from __future__ import annotations

import math

from rescue_vision.config import Config
from rescue_vision.types import BBox, Measurement


def focal_px(frame_width: int, hfov_deg: float) -> float:
    """Focal length in pixels implied by the frame width and horizontal FOV."""
    return frame_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def bearing_deg(cx: float, frame_width: int, hfov_deg: float) -> float:
    """Signed horizontal angle from the optical axis to a bbox centre.

    Negative = left of centre, positive = right (PRD Appendix B).

    Uses the tan form, not the linear approximation. It costs nothing and is
    correct at the frame edges, which is exactly where a person first appears
    during a sweep.
    """
    f_px = focal_px(frame_width, hfov_deg)
    return math.degrees(math.atan((cx - frame_width / 2.0) / f_px))


def distance_m(bbox_height_px: float, f_px: float, assumed_height_m: float) -> float:
    """Coarse pinhole distance from bbox height. +-25-30% at best when valid."""
    if bbox_height_px <= 0:
        return math.inf
    return (assumed_height_m * f_px) / bbox_height_px


def distance_validity(
    bbox: BBox, frame_w: int, frame_h: int, dist_m: float, cfg: Config
) -> tuple[bool, str | None]:
    """Decide whether a distance estimate is trustworthy. PRD 6.7 table.

    Returns a reason string rather than a bare bool so the annotated frame and
    the log can say *why* an estimate was rejected.
    """
    if bbox.y1 <= cfg.edge_margin_px:
        return False, "clipped_top"
    if bbox.y2 >= frame_h - cfg.edge_margin_px:
        return False, "clipped_bottom"
    if bbox.aspect_ratio < cfg.min_aspect_ratio:
        # Likely lying down or crouching -- the 1.7 m assumption is invalid.
        return False, "not_upright"
    if bbox.height < cfg.min_bbox_height_px:
        return False, "bbox_too_small"
    if not (cfg.distance_min_m <= dist_m <= cfg.distance_max_m):
        return False, "implausible_distance"
    return True, None


def measure(bbox: BBox, frame_w: int, frame_h: int, cfg: Config) -> Measurement:
    """Full geometry for one bbox."""
    f_px = focal_px(frame_w, cfg.hfov_deg)
    bearing = bearing_deg(bbox.cx, frame_w, cfg.hfov_deg)
    dist = distance_m(bbox.height, f_px, cfg.assumed_human_height_m)
    valid, reason = distance_validity(bbox, frame_w, frame_h, dist, cfg)
    return Measurement(
        bearing_deg=bearing,
        distance_m=dist,
        distance_valid=valid,
        invalid_reason=reason,
    )


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. Used to match confirm-pass boxes to tracks."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_geometry.py -v`
Expected: PASS, 19 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/geometry.py tests/test_geometry.py
git commit -m "feat: add pinhole bearing and distance geometry with validity rules"
```

---

### Task 3: Tracking — EMA smoothing, confirm matching, promotion

**Files:**
- Create: `rescue_vision/tracking.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `Config`, `BBox`, `RawDetection`, `TrackState` (Task 1); `measure`, `iou` (Task 2).
- Produces:
  - `ema(previous: float | None, new: float, alpha: float) -> float`
  - `TrackStore(cfg: Config)` with methods:
    - `update(detections: list[RawDetection], frame_w: int, frame_h: int, frame_index: int) -> None`
    - `apply_confirmations(confirm_boxes: list[BBox]) -> set[int]`
    - `prune(frame_index: int) -> None`
    - `tracks() -> list[TrackState]`
    - `confirmed_tracks() -> list[TrackState]`

- [ ] **Step 1: Write the failing test**

`tests/test_tracking.py`:
```python
import pytest

from rescue_vision.config import Config
from rescue_vision.tracking import TrackStore, ema
from rescue_vision.types import BBox, RawDetection

CFG = Config()


def det(x1, y1, x2, y2, track_id, conf=0.9):
    return RawDetection(BBox(x1, y1, x2, y2), conf, track_id)


def test_ema_with_no_previous_value_returns_the_new_value():
    assert ema(None, 10.0, 0.4) == 10.0


def test_ema_blends_toward_the_new_value():
    assert ema(0.0, 10.0, 0.4) == pytest.approx(4.0)


def test_ema_repeated_converges_on_the_new_value():
    v = 0.0
    for _ in range(50):
        v = ema(v, 10.0, 0.4)
    assert v == pytest.approx(10.0, abs=1e-6)


def test_update_creates_a_track_for_a_new_id():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=7)], 640, 480, 0)
    tracks = store.tracks()
    assert len(tracks) == 1
    assert tracks[0].track_id == 7


def test_detections_without_a_track_id_are_ignored():
    """The tracker owns identity. An untracked box has nothing to accumulate onto."""
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=None)], 640, 480, 0)
    assert store.tracks() == []


def test_bearing_is_smoothed_rather_than_jumping():
    """Raw per-frame bearing is noisy enough to make the rover visibly twitch."""
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0)
    first = store.tracks()[0].bearing_deg
    store.update([det(600.0, 100.0, 660.0, 400.0, track_id=1)], 640, 480, 1)
    second = store.tracks()[0].bearing_deg
    raw_second = 0.0
    from rescue_vision.geometry import bearing_deg
    raw_second = bearing_deg(630.0, 640, CFG.hfov_deg)
    assert first < second < raw_second


def test_a_track_starts_unconfirmed():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0)
    assert store.tracks()[0].confirmed is False
    assert store.confirmed_tracks() == []


def test_track_is_promoted_after_n_confirm_matching_confirm_passes():
    store = TrackStore(CFG)
    box = BBox(300.0, 100.0, 360.0, 400.0)
    store.update([RawDetection(box, 0.9, 1)], 640, 480, 0)

    matched = store.apply_confirmations([box])
    assert matched == {1}
    assert store.tracks()[0].confirm_count == 1
    assert store.tracks()[0].confirmed is False  # n_confirm is 2

    store.apply_confirmations([box])
    assert store.tracks()[0].confirmed is True
    assert [t.track_id for t in store.confirmed_tracks()] == [1]


def test_confirmation_does_not_match_a_distant_box():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0)
    matched = store.apply_confirmations([BBox(0.0, 0.0, 20.0, 20.0)])
    assert matched == set()
    assert store.tracks()[0].confirm_count == 0


def test_once_confirmed_a_track_stays_confirmed():
    """PRD 6.5: promoted tracks stay confirmed until the tracker drops them."""
    store = TrackStore(CFG)
    box = BBox(300.0, 100.0, 360.0, 400.0)
    store.update([RawDetection(box, 0.9, 1)], 640, 480, 0)
    store.apply_confirmations([box])
    store.apply_confirmations([box])
    assert store.tracks()[0].confirmed is True

    for i in range(1, 5):
        store.update([RawDetection(box, 0.9, 1)], 640, 480, i)
    assert store.tracks()[0].confirmed is True


def test_prune_drops_tracks_unseen_for_longer_than_max_age():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0)
    store.prune(frame_index=CFG.track_max_age_frames + 1)
    assert store.tracks() == []


def test_prune_keeps_a_recently_seen_track():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 10)
    store.prune(frame_index=11)
    assert len(store.tracks()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.tracking'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/tracking.py`:
```python
"""Per-track accumulation: EMA smoothing and confirm-pass promotion. PRD 6.5, 6.7.

Pure module. No I/O, no clock.
"""

from __future__ import annotations

from rescue_vision.config import Config
from rescue_vision.geometry import iou, measure
from rescue_vision.types import BBox, RawDetection, TrackState


def ema(previous: float | None, new: float, alpha: float) -> float:
    """Exponential moving average. First sample passes through unchanged."""
    if previous is None:
        return new
    return alpha * new + (1.0 - alpha) * previous


class TrackStore:
    """Holds smoothed state per track ID and decides when a track is human.

    Track IDs come from ByteTrack via the scan pass. Detections without a track
    ID are ignored: identity is the tracker's job, and there is nothing to
    accumulate state onto without it.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._tracks: dict[int, TrackState] = {}

    def update(
        self,
        detections: list[RawDetection],
        frame_w: int,
        frame_h: int,
        frame_index: int,
    ) -> None:
        """Fold this frame's scan detections into per-track smoothed state."""
        for d in detections:
            if d.track_id is None:
                continue
            m = measure(d.bbox, frame_w, frame_h, self._cfg)
            existing = self._tracks.get(d.track_id)
            if existing is None:
                self._tracks[d.track_id] = TrackState(
                    track_id=d.track_id,
                    bbox=d.bbox,
                    confidence=d.confidence,
                    bearing_deg=m.bearing_deg,
                    distance_m=m.distance_m,
                    distance_valid=m.distance_valid,
                    invalid_reason=m.invalid_reason,
                    last_seen_frame=frame_index,
                )
                continue

            a = self._cfg.ema_alpha
            existing.bearing_deg = ema(existing.bearing_deg, m.bearing_deg, a)
            # Only smooth distance across frames where it is trustworthy;
            # blending in a garbage estimate poisons the average.
            if m.distance_valid:
                base = existing.distance_m if existing.distance_valid else None
                existing.distance_m = ema(base, m.distance_m, a)
            else:
                existing.distance_m = m.distance_m
            existing.distance_valid = m.distance_valid
            existing.invalid_reason = m.invalid_reason
            existing.bbox = d.bbox
            existing.confidence = d.confidence
            existing.last_seen_frame = frame_index

    def apply_confirmations(self, confirm_boxes: list[BBox]) -> set[int]:
        """Match confirm-pass boxes to tracks by IoU and promote on N_CONFIRM.

        The confirm model returns its own boxes with no track IDs, so
        association is by overlap. Each track may be credited at most once per
        confirm pass.
        """
        matched: set[int] = set()
        for track in self._tracks.values():
            best = max(
                (iou(track.bbox, cb) for cb in confirm_boxes), default=0.0
            )
            if best >= self._cfg.confirm_iou_match:
                track.confirm_count += 1
                if track.confirm_count >= self._cfg.n_confirm:
                    track.confirmed = True
                matched.add(track.track_id)
        return matched

    def prune(self, frame_index: int) -> None:
        """Drop tracks the tracker has stopped reporting."""
        max_age = self._cfg.track_max_age_frames
        self._tracks = {
            tid: t
            for tid, t in self._tracks.items()
            if frame_index - t.last_seen_frame <= max_age
        }

    def tracks(self) -> list[TrackState]:
        return list(self._tracks.values())

    def confirmed_tracks(self) -> list[TrackState]:
        return [t for t in self._tracks.values() if t.confirmed]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tracking.py -v`
Expected: PASS, 12 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/tracking.py tests/test_tracking.py
git commit -m "feat: add track store with EMA smoothing and confirm promotion"
```

---

### Task 4: Target selection with stickiness

**Files:**
- Create: `rescue_vision/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `Config`, `TrackState` (Task 1).
- Produces: `TargetSelector(cfg: Config)` with `select(tracks: list[TrackState], now: float) -> TrackState | None`.

- [ ] **Step 1: Write the failing test**

`tests/test_selection.py`:
```python
from rescue_vision.config import Config
from rescue_vision.selection import TargetSelector
from rescue_vision.types import BBox, TrackState

CFG = Config()


def track(track_id, distance_m=5.0, distance_valid=True, confirmed=True,
          bbox=None):
    return TrackState(
        track_id=track_id,
        bbox=bbox or BBox(300.0, 100.0, 360.0, 400.0),
        confidence=0.9,
        bearing_deg=0.0,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=confirmed,
    )


def test_no_tracks_selects_nothing():
    assert TargetSelector(CFG).select([], now=0.0) is None


def test_unconfirmed_tracks_are_never_selected():
    sel = TargetSelector(CFG)
    assert sel.select([track(1, confirmed=False)], now=0.0) is None


def test_nearest_valid_distance_wins():
    sel = TargetSelector(CFG)
    chosen = sel.select([track(1, distance_m=8.0), track(2, distance_m=3.0)], now=0.0)
    assert chosen.track_id == 2


def test_largest_bbox_wins_when_no_distance_is_valid():
    small = BBox(300.0, 100.0, 320.0, 200.0)
    large = BBox(100.0, 50.0, 300.0, 450.0)
    sel = TargetSelector(CFG)
    chosen = sel.select(
        [
            track(1, distance_valid=False, bbox=small),
            track(2, distance_valid=False, bbox=large),
        ],
        now=0.0,
    )
    assert chosen.track_id == 2


def test_a_track_with_valid_distance_beats_one_without():
    sel = TargetSelector(CFG)
    chosen = sel.select(
        [track(1, distance_valid=False), track(2, distance_m=20.0)], now=0.0
    )
    assert chosen.track_id == 2


def test_lowest_track_id_breaks_ties_for_determinism():
    sel = TargetSelector(CFG)
    chosen = sel.select([track(5, distance_m=4.0), track(2, distance_m=4.0)], now=0.0)
    assert chosen.track_id == 2


def test_target_is_sticky_within_target_hold():
    """Stops the rover flip-flopping between two people."""
    sel = TargetSelector(CFG)
    assert sel.select([track(1, distance_m=5.0)], now=0.0).track_id == 1
    chosen = sel.select(
        [track(1, distance_m=5.0), track(2, distance_m=1.0)], now=0.5
    )
    assert chosen.track_id == 1


def test_target_switches_after_target_hold_expires():
    sel = TargetSelector(CFG)
    sel.select([track(1, distance_m=5.0)], now=0.0)
    chosen = sel.select(
        [track(1, distance_m=5.0), track(2, distance_m=1.0)], now=1.5
    )
    assert chosen.track_id == 2


def test_stickiness_does_not_resurrect_a_vanished_target():
    sel = TargetSelector(CFG)
    sel.select([track(1)], now=0.0)
    chosen = sel.select([track(2)], now=0.1)
    assert chosen.track_id == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.selection'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/selection.py`:
```python
"""Pick exactly one target per frame. PRD 6.8.

Pure module: the clock is passed in, never read.
"""

from __future__ import annotations

from rescue_vision.config import Config
from rescue_vision.types import TrackState


class TargetSelector:
    """Applies the PRD 6.8 priority rule with TARGET_HOLD stickiness."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._current_id: int | None = None
        self._selected_at: float = 0.0

    def select(self, tracks: list[TrackState], now: float) -> TrackState | None:
        candidates = [t for t in tracks if t.confirmed]
        if not candidates:
            self._current_id = None
            return None

        by_id = {t.track_id: t for t in candidates}

        # Stickiness: hold the previous target briefly even if another now
        # scores higher, so the rover does not flip-flop between two people.
        if self._current_id in by_id:
            if now - self._selected_at < self._cfg.target_hold:
                return by_id[self._current_id]

        best = min(candidates, key=self._priority)
        if best.track_id != self._current_id:
            self._current_id = best.track_id
            self._selected_at = now
        return best

    @staticmethod
    def _priority(t: TrackState) -> tuple[int, float, int]:
        """Sort key: valid-distance first, then nearest, then largest, then id.

        Lower is better. Tracks with a valid distance form tier 0 and are
        ranked by distance; the rest form tier 1 and are ranked by descending
        bbox area, a reasonable proxy for nearest.
        """
        if t.distance_valid:
            return (0, t.distance_m, t.track_id)
        return (1, -t.bbox.area, t.track_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_selection.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/selection.py tests/test_selection.py
git commit -m "feat: add sticky target selection"
```

---

### Task 5: Control — deadband P-controller, drive rule, differential mixing

**Files:**
- Create: `rescue_vision/control.py`
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: `Config`, `TrackState`, `Command` (Task 1).
- Produces:
  - `clamp(value: float, lo: float, hi: float) -> float`
  - `turn_command(bearing_deg: float, cfg: Config) -> float`
  - `drive_command(target: TrackState | None, turn: float, cfg: Config) -> float`
  - `compute_command(target: TrackState | None, cfg: Config) -> Command`
  - `mix(turn: float, forward: float) -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

`tests/test_control.py`:
```python
import pytest

from rescue_vision.config import Config
from rescue_vision.control import (
    clamp,
    compute_command,
    drive_command,
    mix,
    turn_command,
)
from rescue_vision.types import BBox, TrackState

CFG = Config()


def target(bearing_deg=0.0, distance_m=5.0, distance_valid=True):
    return TrackState(
        track_id=1,
        bbox=BBox(300.0, 100.0, 360.0, 400.0),
        confidence=0.9,
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=True,
    )


def test_clamp_bounds_both_ways():
    assert clamp(5.0, -1.0, 1.0) == 1.0
    assert clamp(-5.0, -1.0, 1.0) == -1.0
    assert clamp(0.5, -1.0, 1.0) == 0.5


def test_turn_is_zero_inside_the_deadband():
    """No deadband and no derivative term means the rover oscillates forever."""
    assert turn_command(0.0, CFG) == 0.0
    assert turn_command(4.9, CFG) == 0.0
    assert turn_command(-4.9, CFG) == 0.0


def test_turn_takes_the_same_sign_as_bearing():
    """PRD Appendix B. Getting this backwards looks identical to a detection bug."""
    assert turn_command(25.0, CFG) > 0.0
    assert turn_command(-25.0, CFG) < 0.0


def test_turn_is_proportional_to_error_above_the_stiction_floor():
    assert turn_command(25.0, CFG) == pytest.approx(0.5)


def test_small_error_is_lifted_to_min_turn_to_beat_stiction():
    """A DC motor at 8 percent duty cycle buzzes rather than turning."""
    raw = CFG.kp * 6.0  # 0.12, below MIN_TURN of 0.25
    assert raw < CFG.min_turn
    assert turn_command(6.0, CFG) == pytest.approx(CFG.min_turn)
    assert turn_command(-6.0, CFG) == pytest.approx(-CFG.min_turn)


def test_turn_is_clamped_to_unit_range():
    assert turn_command(1000.0, CFG) == 1.0
    assert turn_command(-1000.0, CFG) == -1.0


def test_turn_never_oscillates_across_a_bearing_sweep():
    """Sweeping the error toward zero must decrease |turn| monotonically and
    land at exactly zero, never overshooting into the opposite sign."""
    previous = 1.1
    for bearing in [40.0, 30.0, 20.0, 10.0, 6.0, 4.0, 0.0]:
        t = turn_command(bearing, CFG)
        assert t >= 0.0
        assert t <= previous
        previous = t
    assert turn_command(0.0, CFG) == 0.0


def test_no_target_means_no_drive():
    assert drive_command(None, turn=0.0, cfg=CFG) == 0.0


def test_no_drive_while_still_turning():
    """Turn in place first -- easier to debug and looks more deliberate."""
    assert drive_command(target(bearing_deg=30.0), turn=0.6, cfg=CFG) == 0.0


def test_no_drive_when_closer_than_stop_distance():
    t = target(bearing_deg=0.0, distance_m=1.0)
    assert drive_command(t, turn=0.0, cfg=CFG) == 0.0


def test_approaches_when_centred_and_far_enough():
    t = target(bearing_deg=0.0, distance_m=5.0)
    assert drive_command(t, turn=0.0, cfg=CFG) == pytest.approx(CFG.approach_speed)


def test_approaches_when_distance_is_invalid_but_centred():
    """Never gate motion on a distance we have already flagged untrustworthy."""
    t = target(bearing_deg=0.0, distance_valid=False, distance_m=0.1)
    assert drive_command(t, turn=0.0, cfg=CFG) == pytest.approx(CFG.approach_speed)


def test_compute_command_with_no_target_is_a_full_stop():
    cmd = compute_command(None, CFG)
    assert cmd.turn == 0.0
    assert cmd.drive == 0.0


def test_compute_command_turns_toward_an_off_centre_target():
    cmd = compute_command(target(bearing_deg=-20.0), CFG)
    assert cmd.turn < 0.0
    assert cmd.drive == 0.0


def test_mix_pure_forward_drives_both_sides_equally():
    assert mix(turn=0.0, forward=0.5) == (0.5, 0.5)


def test_mix_pure_right_turn_is_opposite_on_each_side():
    left, right = mix(turn=0.5, forward=0.0)
    assert left == 0.5
    assert right == -0.5


def test_mix_clamps_after_mixing_not_before():
    """Clamping before mixing would let one side silently saturate."""
    left, right = mix(turn=0.8, forward=0.8)
    assert left == 1.0
    assert right == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_control.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.control'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/control.py`:
```python
"""Proportional turn control with a deadband, plus differential mixing. PRD 6.8.

Pure module. No I/O, no clock.

Sign convention (PRD Appendix B): positive turn == rover rotates clockwise /
to its right, and turn takes the SAME sign as bearing. If the physical rover
turns away from people, swap the motor pin pairs in MOTOR_PINS -- do not negate
KP, which would leave this convention lying to the next reader.
"""

from __future__ import annotations

import math

from rescue_vision.config import Config
from rescue_vision.types import Command, TrackState


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def turn_command(bearing_deg: float, cfg: Config) -> float:
    """Deadband proportional control toward frame centre."""
    if abs(bearing_deg) <= cfg.deadband_deg:
        return 0.0
    turn = clamp(cfg.kp * bearing_deg, -1.0, 1.0)
    if 0.0 < abs(turn) < cfg.min_turn:
        # Below the stiction floor a DC motor buzzes without turning, while
        # the controller believes it commanded motion.
        turn = math.copysign(cfg.min_turn, turn)
    return turn


def drive_command(target: TrackState | None, turn: float, cfg: Config) -> float:
    """Turn in place first, then advance. PRD 6.8."""
    if target is None:
        return 0.0
    if abs(target.bearing_deg) > cfg.deadband_deg:
        return 0.0
    # Only an estimate we trust may hold the rover back. A distance already
    # flagged invalid must not gate behaviour on a number we know is wrong.
    if target.distance_valid and target.distance_m < cfg.stop_distance_m:
        return 0.0
    return cfg.approach_speed


def compute_command(target: TrackState | None, cfg: Config) -> Command:
    """Full command for one frame."""
    if target is None:
        return Command(turn=0.0, drive=0.0)
    turn = turn_command(target.bearing_deg, cfg)
    return Command(turn=turn, drive=drive_command(target, turn, cfg))


def mix(turn: float, forward: float) -> tuple[float, float]:
    """Differential mixing. Clamp AFTER mixing so neither side saturates
    silently on a simultaneous turn+forward request."""
    return (
        clamp(forward + turn, -1.0, 1.0),
        clamp(forward - turn, -1.0, 1.0),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_control.py -v`
Expected: PASS, 17 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/control.py tests/test_control.py
git commit -m "feat: add deadband P-controller and differential mixing"
```

---

### Task 6: Rover controllers with watchdog

**Files:**
- Create: `rescue_vision/rover.py`
- Test: `tests/test_rover.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `mix` (Task 5).
- Produces:
  - `RoverController` ABC: public `drive(turn: float, forward: float) -> None`, `stop() -> None`, `close() -> None`; subclasses implement `_apply(left: float, right: float) -> None` and `_stop() -> None`.
  - `ConsoleRover(cfg, sink=print)` with attribute `commands: list[tuple[float, float]]`.
  - `GpioZeroRover(cfg, pins: dict, stby_pin: int | None = None)`.
  - `MOTOR_PINS: dict` module constant.

- [ ] **Step 1: Write the failing test**

`tests/test_rover.py`:
```python
import time

from rescue_vision.config import Config
from rescue_vision.rover import ConsoleRover

CFG = Config()


def test_drive_mixes_to_left_and_right():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.drive(turn=0.0, forward=0.5)
    assert rover.commands[-1] == (0.5, 0.5)
    rover.close()


def test_stop_sends_zero_to_both_sides():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.drive(turn=0.5, forward=0.5)
    rover.stop()
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_watchdog_stops_the_motors_when_drive_goes_quiet():
    """A vision pipeline that hangs must not leave a rover driving into a wall."""
    cfg = Config(watchdog_timeout=0.1)
    rover = ConsoleRover(cfg, sink=lambda _: None)
    rover.drive(turn=0.0, forward=0.5)
    assert rover.commands[-1] == (0.5, 0.5)
    time.sleep(0.35)
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_watchdog_does_not_fire_while_drive_is_called_regularly():
    cfg = Config(watchdog_timeout=0.3)
    rover = ConsoleRover(cfg, sink=lambda _: None)
    for _ in range(6):
        rover.drive(turn=0.0, forward=0.4)
        time.sleep(0.05)
    assert rover.commands[-1] == (0.4, 0.4)
    rover.close()


def test_close_stops_the_motors():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.drive(turn=0.0, forward=0.5)
    rover.close()
    assert rover.commands[-1] == (0.0, 0.0)


def test_context_manager_stops_on_exception():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    try:
        with rover:
            rover.drive(turn=0.0, forward=0.9)
            raise RuntimeError("pipeline blew up")
    except RuntimeError:
        pass
    assert rover.commands[-1] == (0.0, 0.0)


def test_console_rover_writes_to_its_sink():
    lines = []
    rover = ConsoleRover(CFG, sink=lines.append)
    rover.drive(turn=0.25, forward=0.0)
    rover.close()
    assert any("turn" in line for line in lines)


def test_gpiozero_rover_import_does_not_require_gpiozero():
    """The module must import cleanly on Windows -- gpiozero is imported lazily."""
    from rescue_vision.rover import GpioZeroRover

    assert GpioZeroRover is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rover.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.rover'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/rover.py`:
```python
"""Motor abstraction. PRD 6.9.

Safety contract for every backend:
  - motors default to stopped
  - stop() on KeyboardInterrupt, on any uncaught exception, and in a finally
  - a watchdog stops the drive if drive() goes quiet for WATCHDOG_TIMEOUT

The watchdog lives in the base class so both backends get it and neither can
forget it.

BENCH-TEST WITH THE WHEELS OFF THE GROUND FIRST. EVERY TIME.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable

from rescue_vision.config import Config
from rescue_vision.control import mix

log = logging.getLogger(__name__)

# Placeholder wiring -- EDIT TO MATCH YOUR CHASSIS (PRD 6.9, open question).
# If the rover turns away from people, swap the forward/backward pins here
# rather than negating KP.
MOTOR_PINS: dict[str, dict[str, int]] = {
    "left": {"forward": 17, "backward": 18, "enable": 12},
    "right": {"forward": 22, "backward": 23, "enable": 13},
}


class RoverController(ABC):
    """Base controller: mixing, watchdog, and guaranteed stop."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._closed = False
        self._last_drive = 0.0
        self._watchdog_fired = True  # start stopped
        self._timer = threading.Timer(cfg.watchdog_timeout, self._on_watchdog)
        self._timer.daemon = True
        self._timer.start()

    def drive(self, turn: float, forward: float) -> None:
        """Apply a normalized command and pet the watchdog."""
        left, right = mix(turn, forward)
        with self._lock:
            if self._closed:
                return
            self._watchdog_fired = False
            self._restart_timer()
            self._apply(left, right)

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._timer.cancel()
            self._stop()

    def __enter__(self) -> "RoverController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _restart_timer(self) -> None:
        self._timer.cancel()
        self._timer = threading.Timer(self._cfg.watchdog_timeout, self._on_watchdog)
        self._timer.daemon = True
        self._timer.start()

    def _on_watchdog(self) -> None:
        with self._lock:
            if self._closed or self._watchdog_fired:
                return
            self._watchdog_fired = True
            log.warning(
                "watchdog: no drive() within %.2fs, stopping motors",
                self._cfg.watchdog_timeout,
            )
            self._stop()

    @abstractmethod
    def _apply(self, left: float, right: float) -> None:
        """Send per-side speeds in [-1, +1] to the hardware."""

    @abstractmethod
    def _stop(self) -> None:
        """Bring both sides to a halt. Must be safe to call repeatedly."""


class ConsoleRover(RoverController):
    """Logs commands instead of moving anything. Default, and the Windows path."""

    def __init__(self, cfg: Config, sink: Callable[[str], None] = print) -> None:
        self.commands: list[tuple[float, float]] = []
        self._sink = sink
        super().__init__(cfg)

    def _apply(self, left: float, right: float) -> None:
        self.commands.append((left, right))
        self._sink(f"[rover] left={left:+.2f} right={right:+.2f}")

    def _stop(self) -> None:
        self.commands.append((0.0, 0.0))
        self._sink("[rover] stop  turn=0.00 forward=0.00")


class GpioZeroRover(RoverController):
    """Real L298N / TB6612FNG via gpiozero. Pi 5 only.

    UNTESTED: written from the spec, never executed against hardware. Expect
    the gpiochip number and pin mapping to need adjustment on first Pi boot
    (PRD 6.9). RPi.GPIO and pigpio do NOT work on Pi 5 -- gpiozero backed by
    lgpio is the supported path.
    """

    def __init__(
        self,
        cfg: Config,
        pins: dict[str, dict[str, int]] | None = None,
        stby_pin: int | None = None,
    ) -> None:
        from gpiozero import DigitalOutputDevice, Motor  # lazy: Pi only

        pins = pins or MOTOR_PINS
        self._left = Motor(
            forward=pins["left"]["forward"],
            backward=pins["left"]["backward"],
            enable=pins["left"]["enable"],
            pwm=True,
        )
        self._right = Motor(
            forward=pins["right"]["forward"],
            backward=pins["right"]["backward"],
            enable=pins["right"]["enable"],
            pwm=True,
        )
        # The TB6612FNG ignores every input until STBY is driven high. The
        # L298N has no such pin; leave stby_pin as None for it.
        self._stby = DigitalOutputDevice(stby_pin) if stby_pin is not None else None
        if self._stby is not None:
            self._stby.on()
        super().__init__(cfg)

    @staticmethod
    def _drive_one(motor, speed: float) -> None:
        if speed > 0:
            motor.forward(min(1.0, speed))
        elif speed < 0:
            motor.backward(min(1.0, -speed))
        else:
            motor.stop()

    def _apply(self, left: float, right: float) -> None:
        self._drive_one(self._left, left)
        self._drive_one(self._right, right)

    def _stop(self) -> None:
        self._left.stop()
        self._right.stop()

    def close(self) -> None:
        super().close()
        self._left.close()
        self._right.close()
        if self._stby is not None:
            self._stby.off()
            self._stby.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_rover.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/rover.py tests/test_rover.py
git commit -m "feat: add rover controllers with base-class watchdog"
```

---

### Task 7: Event writer — JSONL schema, frame saving, disk cap

**Files:**
- Create: `rescue_vision/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `Config`, `TrackState`, `Command` (Task 1).
- Produces:
  - `build_event(track, target_id, command, frame_index, timestamp, annotated_frame) -> dict`
  - `EventWriter(jsonl_path: Path, frames_dir: Path, cfg: Config)` with `emit(rows: list[dict]) -> None`, `should_save_frame(track_id: int, now: float) -> bool`, `save_frame(image, track_id: int, frame_index: int, now: float) -> str | None`, `enforce_disk_cap() -> None`, `close() -> None`.

- [ ] **Step 1: Write the failing test**

`tests/test_events.py`:
```python
import json

import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.events import EventWriter, build_event
from rescue_vision.types import BBox, Command, TrackState

CFG = Config()


def track(track_id=3, bearing_deg=-12.4, distance_m=3.2, distance_valid=True):
    return TrackState(
        track_id=track_id,
        bbox=BBox(312.0, 118.0, 466.0, 502.0),
        confidence=0.87,
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=True,
    )


def test_event_matches_the_prd_schema():
    ev = build_event(
        track=track(),
        target_id=3,
        command=Command(turn=-0.31, drive=0.0),
        frame_index=412,
        timestamp=1754640000.123,
        annotated_frame="detections/frame_000412.jpg",
    )
    assert ev["schema"] == "rescue.detection.v1"
    assert ev["frame_index"] == 412
    assert ev["track_id"] == 3
    assert ev["bbox_xyxy"] == [312, 118, 466, 502]
    assert ev["bearing_deg"] == pytest.approx(-12.4)
    assert ev["is_target"] is True
    assert ev["turn_command"] == pytest.approx(-0.31)
    assert ev["annotated_frame"] == "detections/frame_000412.jpg"


def test_non_target_track_is_reported_but_not_flagged():
    ev = build_event(track(track_id=9), 3, Command(0.0, 0.0), 1, 0.0, None)
    assert ev["is_target"] is False
    assert ev["annotated_frame"] is None


def test_commands_are_repeated_on_every_row():
    """A consumer reading a single line must have everything it needs."""
    ev = build_event(track(track_id=9), 3, Command(-0.5, 0.3), 1, 0.0, None)
    assert ev["turn_command"] == pytest.approx(-0.5)
    assert ev["drive_command"] == pytest.approx(0.3)


def test_invalid_distance_is_reported_with_its_reason():
    t = track(distance_valid=False)
    t.invalid_reason = "not_upright"
    ev = build_event(t, 3, Command(0.0, 0.0), 1, 0.0, None)
    assert ev["distance_valid"] is False
    assert ev["invalid_reason"] == "not_upright"


def test_emit_writes_one_json_object_per_line(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl", tmp_path / "frames", CFG)
    writer.emit([build_event(track(), 3, Command(0.0, 0.0), 1, 0.0, None)])
    writer.emit([build_event(track(), 3, Command(0.0, 0.0), 2, 0.0, None)])
    writer.close()

    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["frame_index"] == 1
    assert json.loads(lines[1])["frame_index"] == 2


def test_frame_saving_is_rate_limited_per_track(tmp_path):
    """Saved frames are the one unbounded item in the 16 GB disk budget."""
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    assert writer.should_save_frame(track_id=1, now=100.0) is True
    writer.save_frame(np.zeros((10, 10, 3), np.uint8), 1, 0, now=100.0)
    assert writer.should_save_frame(track_id=1, now=100.5) is False
    assert writer.should_save_frame(track_id=1, now=101.5) is True
    writer.close()


def test_different_tracks_have_independent_rate_limits(tmp_path):
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    writer.save_frame(np.zeros((10, 10, 3), np.uint8), 1, 0, now=100.0)
    assert writer.should_save_frame(track_id=2, now=100.1) is True
    writer.close()


def test_save_frame_returns_a_relative_path_and_writes_a_file(tmp_path):
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    rel = writer.save_frame(np.zeros((20, 20, 3), np.uint8), 7, 412, now=1.0)
    writer.close()
    assert rel is not None
    assert "412" in rel
    assert (tmp_path / rel).exists()


def test_save_frame_returns_none_when_saving_is_disabled(tmp_path):
    cfg = Config(save_frames=False)
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", cfg)
    assert writer.save_frame(np.zeros((10, 10, 3), np.uint8), 1, 0, now=1.0) is None
    writer.close()


def test_disk_cap_deletes_oldest_frames_first(tmp_path):
    cfg = Config(max_output_dir_mb=0)  # cap of zero forces eviction
    frames = tmp_path / "frames"
    writer = EventWriter(tmp_path / "e.jsonl", frames, cfg)
    for i in range(5):
        writer.save_frame(np.zeros((50, 50, 3), np.uint8), 1, i, now=float(i) * 2)
    writer.enforce_disk_cap()
    writer.close()
    assert len(list(frames.glob("*.jpg"))) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.events'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/events.py`:
```python
"""Detection event output: rescue.detection.v1 JSONL plus annotated frames.

PRD 6.4 for the schema, PRD 9 for the disk budget. Saved frames are the one
unbounded item on a 16 GB card, so they are both rate-limited and capped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2

from rescue_vision.config import Config
from rescue_vision.types import Command, TrackState

log = logging.getLogger(__name__)

SCHEMA = "rescue.detection.v1"


def build_event(
    track: TrackState,
    target_id: int | None,
    command: Command,
    frame_index: int,
    timestamp: float,
    annotated_frame: str | None,
) -> dict[str, Any]:
    """One JSONL row. Commands are repeated on every row so a consumer reading
    a single line has everything it needs."""
    return {
        "schema": SCHEMA,
        "timestamp": round(timestamp, 3),
        "frame_index": frame_index,
        "track_id": track.track_id,
        "confidence": round(track.confidence, 3),
        "bbox_xyxy": track.bbox.as_xyxy_ints(),
        "bearing_deg": round(track.bearing_deg, 2),
        "distance_m": (
            round(track.distance_m, 2) if track.distance_valid else None
        ),
        "distance_valid": track.distance_valid,
        "invalid_reason": track.invalid_reason,
        "is_target": track.track_id == target_id,
        "turn_command": round(command.turn, 3),
        "drive_command": round(command.drive, 3),
        "annotated_frame": annotated_frame,
    }


class EventWriter:
    """Appends JSONL rows and saves rate-limited annotated frames."""

    def __init__(self, jsonl_path: Path, frames_dir: Path, cfg: Config) -> None:
        self._cfg = cfg
        self._frames_dir = Path(frames_dir)
        self._root = Path(jsonl_path).parent
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._root.mkdir(parents=True, exist_ok=True)
        self._fh = open(jsonl_path, "a", encoding="utf-8")
        self._last_saved: dict[int, float] = {}

    def emit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def should_save_frame(self, track_id: int, now: float) -> bool:
        if not self._cfg.save_frames:
            return False
        last = self._last_saved.get(track_id)
        return last is None or (now - last) >= self._cfg.frame_save_interval

    def save_frame(self, image, track_id: int, frame_index: int, now: float) -> str | None:
        """Write an annotated frame. Returns a path relative to the output root."""
        if not self.should_save_frame(track_id, now):
            return None
        path = self._frames_dir / f"frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(path), image):
            log.warning("failed to write %s", path)
            return None
        self._last_saved[track_id] = now
        self.enforce_disk_cap()
        try:
            return str(path.relative_to(self._root)).replace("\\", "/")
        except ValueError:
            return str(path)

    def enforce_disk_cap(self) -> None:
        """Delete oldest frames first until the directory is under the cap."""
        files = sorted(
            self._frames_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime
        )
        total = sum(p.stat().st_size for p in files)
        cap = self._cfg.max_output_dir_mb * 1024 * 1024
        while files and total > cap:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            try:
                oldest.unlink()
            except OSError as exc:
                log.warning("could not delete %s: %s", oldest, exc)

    def close(self) -> None:
        self._fh.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_events.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/events.py tests/test_events.py
git commit -m "feat: add JSONL event writer with frame rate limiting and disk cap"
```

---

### Task 8: Annotation overlay

**Files:**
- Create: `rescue_vision/annotate.py`
- Test: `tests/test_annotate.py`

**Interfaces:**
- Consumes: `TrackState`, `Command` (Task 1).
- Produces: `draw_overlay(frame, tracks: list[TrackState], target_id: int | None, command: Command, fps: float) -> np.ndarray` — returns a new array, never mutates the input.

- [ ] **Step 1: Write the failing test**

`tests/test_annotate.py`:
```python
import numpy as np

from rescue_vision.annotate import draw_overlay
from rescue_vision.types import BBox, Command, TrackState


def track(track_id=1, confirmed=True):
    return TrackState(
        track_id=track_id,
        bbox=BBox(100.0, 50.0, 200.0, 350.0),
        confidence=0.9,
        bearing_deg=-12.4,
        distance_m=3.2,
        distance_valid=True,
        confirmed=confirmed,
    )


def test_overlay_returns_a_new_array_and_leaves_the_input_untouched():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(frame, [track()], 1, Command(-0.31, 0.0), 12.0)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    assert frame.sum() == 0


def test_overlay_draws_something_for_each_track():
    frame = np.zeros((480, 640, 3), np.uint8)
    one = draw_overlay(frame, [track(1)], 1, Command(0.0, 0.0), 12.0)
    two = draw_overlay(
        frame,
        [track(1), TrackState(2, BBox(400.0, 60.0, 480.0, 300.0), 0.8, 10.0, 4.0, True, confirmed=True)],
        1,
        Command(0.0, 0.0),
        12.0,
    )
    assert two.sum() > one.sum()


def test_overlay_handles_no_tracks():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = draw_overlay(frame, [], None, Command(0.0, 0.0), 12.0)
    assert out.shape == frame.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.annotate'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/annotate.py`:
```python
"""Annotated frame drawing for the demo feed. PRD FR4."""

from __future__ import annotations

import cv2
import numpy as np

from rescue_vision.types import Command, TrackState

_TARGET_COLOUR = (0, 255, 0)
_CONFIRMED_COLOUR = (0, 200, 255)
_TENTATIVE_COLOUR = (128, 128, 128)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    frame: np.ndarray,
    tracks: list[TrackState],
    target_id: int | None,
    command: Command,
    fps: float,
) -> np.ndarray:
    """Draw boxes, track IDs, bearing and distance. Returns a new image."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Centreline: the reference the bearing is measured against.
    cv2.line(out, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)

    for t in tracks:
        if t.track_id == target_id:
            colour = _TARGET_COLOUR
        elif t.confirmed:
            colour = _CONFIRMED_COLOUR
        else:
            colour = _TENTATIVE_COLOUR

        x1, y1, x2, y2 = t.bbox.as_xyxy_ints()
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        if t.distance_valid:
            dist = f"{t.distance_m:.1f}m"
        else:
            dist = f"?({t.invalid_reason or 'invalid'})"
        label = f"#{t.track_id} {t.bearing_deg:+.1f}deg {dist}"
        cv2.putText(out, label, (x1, max(14, y1 - 6)), _FONT, 0.45, colour, 1,
                    cv2.LINE_AA)

    status = (
        f"turn={command.turn:+.2f} drive={command.drive:+.2f} "
        f"fps={fps:.1f} tracks={len(tracks)}"
    )
    cv2.putText(out, status, (8, h - 10), _FONT, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotate.py -v`
Expected: PASS, 3 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/annotate.py tests/test_annotate.py
git commit -m "feat: add annotated frame overlay"
```

---

### Task 9: Frame sources

**Files:**
- Create: `rescue_vision/frame_source.py`
- Test: `tests/test_frame_source.py`

**Interfaces:**
- Consumes: `Config` (Task 1).
- Produces:
  - `FrameSource` ABC: `read() -> np.ndarray | None`, `width: int`, `height: int`, `close() -> None`, plus `__iter__`.
  - `VideoFileSource(path: str)`, `WebcamSource(index: int, cfg: Config)`, `PiCameraSource(cfg: Config)`.
  - `create_frame_source(spec: str, cfg: Config) -> FrameSource` — `"picamera"` selects the Pi source, an all-digit string selects a webcam index, anything else is a file path.

- [ ] **Step 1: Write the failing test**

`tests/test_frame_source.py`:
```python
import cv2
import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.frame_source import (
    VideoFileSource,
    create_frame_source,
)

CFG = Config()


@pytest.fixture
def clip(tmp_path):
    """A 6-frame 64x48 synthetic clip."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for i in range(6):
        frame = np.full((48, 64, 3), i * 10, np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_video_file_source_reports_its_dimensions(clip):
    src = VideoFileSource(str(clip))
    assert src.width == 64
    assert src.height == 48
    src.close()


def test_video_file_source_yields_frames_then_none(clip):
    src = VideoFileSource(str(clip))
    frames = list(src)
    src.close()
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)


def test_read_returns_none_at_end_of_stream(clip):
    src = VideoFileSource(str(clip))
    for _ in range(6):
        src.read()
    assert src.read() is None
    src.close()


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(str(tmp_path / "nope.mp4"))


def test_factory_selects_a_file_source_for_a_path(clip):
    src = create_frame_source(str(clip), CFG)
    assert isinstance(src, VideoFileSource)
    src.close()


def test_factory_selects_a_webcam_for_a_digit_string(monkeypatch):
    """No real camera on CI, so intercept the constructor and assert dispatch."""
    import rescue_vision.frame_source as fs

    captured = {}

    class FakeWebcam:
        def __init__(self, index, cfg):
            captured["index"] = index

    monkeypatch.setattr(fs, "WebcamSource", FakeWebcam)
    fs.create_frame_source("2", CFG)
    assert captured["index"] == 2


def test_factory_selects_the_pi_camera_for_the_picamera_spec(monkeypatch):
    import rescue_vision.frame_source as fs

    captured = {}

    class FakePiCamera:
        def __init__(self, cfg):
            captured["built"] = True

    monkeypatch.setattr(fs, "PiCameraSource", FakePiCamera)
    fs.create_frame_source("picamera", CFG)
    assert captured["built"] is True


def test_factory_rejects_a_missing_path():
    with pytest.raises(FileNotFoundError):
        create_frame_source("definitely_not_here.mp4", CFG)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_frame_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.frame_source'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/frame_source.py`:
```python
"""Frame capture behind one interface. PRD FR12, 6.10.

Everything downstream of this module is identical on Windows and the Pi.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from rescue_vision.config import Config

log = logging.getLogger(__name__)


class FrameSource(ABC):
    """A stream of BGR frames."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Next frame, or None when the stream is exhausted."""

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    def close(self) -> None:
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Recorded clip. The Windows dev path."""

    def __init__(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"video file not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video file: {path}")
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def close(self) -> None:
        self._cap.release()


class WebcamSource(FrameSource):
    """Laptop webcam. Dev convenience when no clip is available."""

    def __init__(self, index: int, cfg: Config) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def close(self) -> None:
        self._cap.release()


class PiCameraSource(FrameSource):
    """picamera2 capture on the Pi 5.

    UNTESTED: written from the spec, never executed. picamera2 cannot be
    imported on Windows, so the import is lazy.

    Auto-exposure indoors picks 20-30 ms and smears a person into the
    background while the rover turns. Fixed short exposure plus raised gain is
    a requirement, not polish (PRD 6.6).
    """

    def __init__(self, cfg: Config) -> None:
        from picamera2 import Picamera2  # lazy: Pi only

        self._cam = Picamera2()
        video_cfg = self._cam.create_video_configuration(
            main={"size": (cfg.frame_width, cfg.frame_height), "format": "RGB888"}
        )
        self._cam.configure(video_cfg)
        self._cam.start()
        self._cam.set_controls(
            {
                "AeEnable": False,
                "ExposureTime": cfg.exposure_time_us,
                "AnalogueGain": cfg.analogue_gain,
            }
        )
        self._w = cfg.frame_width
        self._h = cfg.frame_height

    def read(self) -> np.ndarray | None:
        return self._cam.capture_array()

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


def create_frame_source(spec: str, cfg: Config) -> FrameSource:
    """Build a source from a CLI spec.

    "picamera"  -> PiCameraSource (Pi only)
    "0", "1"    -> WebcamSource at that index
    anything else -> VideoFileSource for that path
    """
    if spec == "picamera":
        return PiCameraSource(cfg)
    if spec.isdigit():
        return WebcamSource(int(spec), cfg)
    return VideoFileSource(spec)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_frame_source.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/frame_source.py tests/test_frame_source.py
git commit -m "feat: add file, webcam, and picamera2 frame sources"
```

---

### Task 10: Detector protocol and two-tier cascade

**Files:**
- Create: `rescue_vision/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `Config`, `BBox`, `RawDetection` (Task 1).
- Produces:
  - `Detector` Protocol: `scan(frame) -> list[RawDetection]`, `confirm(frame) -> list[BBox]`, `should_confirm(has_candidates: bool, now: float) -> bool`.
  - `CascadeDetector(cfg: Config)` implementing it.
  - `ScriptedDetector(script: list[list[RawDetection]], confirm_all: bool = True)` — a test double, exported from the module so the smoke test and future benchmarks can share it.

- [ ] **Step 1: Write the failing test**

`tests/test_detector.py`:
```python
import numpy as np

from rescue_vision.config import Config
from rescue_vision.detector import ScriptedDetector
from rescue_vision.types import BBox, RawDetection

CFG = Config()
FRAME = np.zeros((480, 640, 3), np.uint8)


def test_scripted_detector_replays_its_script():
    d1 = [RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]
    d2 = [RawDetection(BBox(5.0, 0.0, 15.0, 30.0), 0.9, 1)]
    det = ScriptedDetector([d1, d2])
    assert det.scan(FRAME) == d1
    assert det.scan(FRAME) == d2


def test_scripted_detector_returns_nothing_past_the_end_of_its_script():
    det = ScriptedDetector([[RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]])
    det.scan(FRAME)
    assert det.scan(FRAME) == []


def test_scripted_confirm_echoes_the_last_scan_boxes():
    boxes = [RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]
    det = ScriptedDetector([boxes])
    det.scan(FRAME)
    assert det.confirm(FRAME) == [boxes[0].bbox]


def test_scripted_confirm_returns_nothing_when_confirm_all_is_false():
    boxes = [RawDetection(BBox(0.0, 0.0, 10.0, 30.0), 0.9, 1)]
    det = ScriptedDetector([boxes], confirm_all=False)
    det.scan(FRAME)
    assert det.confirm(FRAME) == []


def test_confirm_is_skipped_when_the_scan_pass_found_nothing():
    """PRD 6.5 escalation rule -- no candidates means no confirm pass."""
    det = ScriptedDetector([])
    assert det.should_confirm(has_candidates=False, now=100.0) is False


def test_confirm_runs_on_the_first_candidate_frame():
    det = ScriptedDetector([])
    assert det.should_confirm(has_candidates=True, now=100.0) is True


def test_confirm_is_rate_limited_to_protect_fps():
    det = ScriptedDetector([], confirm_min_interval=0.15)
    assert det.should_confirm(True, now=100.0) is True
    assert det.should_confirm(True, now=100.05) is False
    assert det.should_confirm(True, now=100.20) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.detector'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/detector.py`:
```python
"""Two-tier detection cascade. PRD 6.5.

Tier 1 (scan): nano model, small input, every frame, feeds the tracker.
Tier 2 (confirm): small model, larger input, only when the scan pass found a
candidate and the rate limit allows.

The pipeline depends on the Detector Protocol, not on CascadeDetector, so the
whole loop can be tested with no model and no camera.

NMS-free caveat: YOLO26 is end-to-end, so the usual iou= NMS threshold has no
effect. Tune recall and precision with conf= only.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from rescue_vision.config import Config
from rescue_vision.types import BBox, RawDetection

log = logging.getLogger(__name__)


class Detector(Protocol):
    """What the pipeline needs from a detector."""

    def scan(self, frame: np.ndarray) -> list[RawDetection]: ...

    def confirm(self, frame: np.ndarray) -> list[BBox]: ...

    def should_confirm(self, has_candidates: bool, now: float) -> bool: ...


class _RateLimitedConfirm:
    """Shared escalation policy: confirm only on candidates, and not too often."""

    def __init__(self, confirm_min_interval: float) -> None:
        self._interval = confirm_min_interval
        self._last_confirm: float | None = None

    def should_confirm(self, has_candidates: bool, now: float) -> bool:
        if not has_candidates:
            return False
        if self._last_confirm is not None and (now - self._last_confirm) < self._interval:
            return False
        self._last_confirm = now
        return True


class CascadeDetector(_RateLimitedConfirm):
    """Ultralytics-backed cascade. Loads two models up front."""

    def __init__(self, cfg: Config) -> None:
        from ultralytics import YOLO  # heavy import, kept out of module scope

        super().__init__(cfg.confirm_min_interval)
        self._cfg = cfg
        log.info("loading scan model %s", cfg.scan_model)
        self._scan_model = YOLO(cfg.scan_model)
        log.info("loading confirm model %s", cfg.confirm_model)
        self._confirm_model = YOLO(cfg.confirm_model)

    def scan(self, frame: np.ndarray) -> list[RawDetection]:
        """Every-frame pass. Runs through the tracker so IDs stay stable."""
        results = self._scan_model.track(
            frame,
            persist=True,
            tracker=self._cfg.tracker_cfg,
            imgsz=self._cfg.scan_imgsz,
            conf=self._cfg.scan_conf,
            classes=[self._cfg.person_class_id],
            verbose=False,
        )
        return _to_detections(results, with_ids=True)

    def confirm(self, frame: np.ndarray) -> list[BBox]:
        """Candidate-only pass. Boxes carry no track IDs; the caller matches
        them to tracks by IoU."""
        results = self._confirm_model.predict(
            frame,
            imgsz=self._cfg.confirm_imgsz,
            conf=self._cfg.confirm_conf,
            classes=[self._cfg.person_class_id],
            verbose=False,
        )
        return [d.bbox for d in _to_detections(results, with_ids=False)]


def _to_detections(results, with_ids: bool) -> list[RawDetection]:
    """Convert an Ultralytics Results list into our own types."""
    out: list[RawDetection] = []
    if not results:
        return out
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    if with_ids and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int).tolist()
    else:
        ids = [None] * len(xyxy)

    for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
        out.append(
            RawDetection(
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                track_id=tid,
            )
        )
    return out


class ScriptedDetector(_RateLimitedConfirm):
    """Test double: replays a fixed list of per-frame detections.

    Lets the full pipeline be exercised with no model, no camera and no
    network -- which is the only way the smoke test can run in CI.
    """

    def __init__(
        self,
        script: list[list[RawDetection]],
        confirm_all: bool = True,
        confirm_min_interval: float = 0.0,
    ) -> None:
        super().__init__(confirm_min_interval)
        self._script = script
        self._index = 0
        self._confirm_all = confirm_all
        self._last_scan: list[RawDetection] = []

    def scan(self, frame: np.ndarray) -> list[RawDetection]:
        if self._index >= len(self._script):
            self._last_scan = []
            return []
        self._last_scan = self._script[self._index]
        self._index += 1
        return self._last_scan

    def confirm(self, frame: np.ndarray) -> list[BBox]:
        if not self._confirm_all:
            return []
        return [d.bbox for d in self._last_scan]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_detector.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add rescue_vision/detector.py tests/test_detector.py
git commit -m "feat: add detector protocol, cascade, and scripted test double"
```

---

### Task 11: Pipeline orchestration and end-to-end smoke test

**Files:**
- Create: `rescue_vision/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces:
  - `FrameResult(frame_index, tracks, target, command, rows, annotated)`
  - `Pipeline(detector, rover, writer, cfg, clock=time.monotonic)` with `process_frame(frame, frame_index) -> FrameResult` and `run(source, max_frames=None, on_frame=None) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline_smoke.py`:
```python
import json

import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.detector import ScriptedDetector
from rescue_vision.events import EventWriter
from rescue_vision.pipeline import Pipeline
from rescue_vision.rover import ConsoleRover
from rescue_vision.types import BBox, RawDetection

CFG = Config(save_frames=False, n_confirm=1)
FRAME = np.zeros((480, 640, 3), np.uint8)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.1
        return self.t


def person_at(cx: float, track_id: int = 1) -> RawDetection:
    """An upright, fully visible person centred on cx."""
    return RawDetection(BBox(cx - 30.0, 100.0, cx + 30.0, 400.0), 0.9, track_id)


def build(tmp_path, script, cfg=CFG):
    detector = ScriptedDetector(script)
    rover = ConsoleRover(cfg, sink=lambda _: None)
    writer = EventWriter(tmp_path / "events.jsonl", tmp_path / "frames", cfg)
    pipeline = Pipeline(detector, rover, writer, cfg, clock=FakeClock())
    return pipeline, rover, writer


def test_person_on_the_left_yields_negative_bearing_and_a_left_turn(tmp_path):
    """PRD 6.10 acceptance test: bearing sign must be correct."""
    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 3)
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert result.target is not None
    assert result.target.bearing_deg < 0.0
    assert result.command.turn < 0.0  # same sign as bearing


def test_person_on_the_right_yields_positive_bearing_and_a_right_turn(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(540.0)]] * 3)
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert result.target.bearing_deg > 0.0
    assert result.command.turn > 0.0


def test_centred_person_produces_no_turn(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(320.0)]] * 3)
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert abs(result.target.bearing_deg) <= CFG.deadband_deg
    assert result.command.turn == 0.0


def test_no_detections_commands_a_full_stop(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[], [], []])
    result = None
    for i in range(3):
        result = pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    assert result.target is None
    assert result.command.turn == 0.0
    assert result.command.drive == 0.0


def test_exactly_one_row_per_frame_is_flagged_as_target(tmp_path):
    script = [[person_at(200.0, 1), person_at(450.0, 2)]] * 3
    pipeline, rover, writer = build(tmp_path, script)
    for i in range(3):
        pipeline.process_frame(FRAME, i)
    writer.close()
    rover.close()

    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert lines
    by_frame: dict[int, int] = {}
    for line in lines:
        row = json.loads(line)
        assert row["schema"] == "rescue.detection.v1"
        by_frame[row["frame_index"]] = by_frame.get(row["frame_index"], 0) + int(
            row["is_target"]
        )
    assert set(by_frame.values()) == {1}


def test_a_failing_frame_does_not_kill_the_run(tmp_path):
    """NFR4: log and continue."""

    class ExplodingDetector(ScriptedDetector):
        def scan(self, frame):
            if self._index == 1:
                self._index += 1
                raise RuntimeError("simulated inference failure")
            return super().scan(frame)

    detector = ExplodingDetector([[person_at(320.0)]] * 4)
    rover = ConsoleRover(CFG, sink=lambda _: None)
    writer = EventWriter(tmp_path / "e.jsonl", tmp_path / "frames", CFG)
    pipeline = Pipeline(detector, rover, writer, CFG, clock=FakeClock())

    processed = pipeline.run(iter([FRAME] * 4))
    writer.close()
    rover.close()
    assert processed == 4


def test_run_stops_the_rover_when_the_stream_ends(tmp_path):
    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 3)
    pipeline.run(iter([FRAME] * 3))
    writer.close()
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_run_stops_the_rover_on_an_unhandled_source_error(tmp_path):
    """Motors must never be left running by a crash."""

    def exploding_source():
        yield FRAME
        raise RuntimeError("camera died")

    pipeline, rover, writer = build(tmp_path, [[person_at(100.0)]] * 3)
    with pytest.raises(RuntimeError):
        pipeline.run(exploding_source())
    writer.close()
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.pipeline'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/pipeline.py`:
```python
"""Per-frame orchestration: detect -> track -> select -> control -> emit.

Depends on the Detector Protocol, never on a concrete model, so the whole loop
is testable with no model, camera, or motors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from rescue_vision.annotate import draw_overlay
from rescue_vision.config import Config
from rescue_vision.control import compute_command
from rescue_vision.detector import Detector
from rescue_vision.events import EventWriter, build_event
from rescue_vision.rover import RoverController
from rescue_vision.selection import TargetSelector
from rescue_vision.tracking import TrackStore
from rescue_vision.types import Command, TrackState

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame_index: int
    tracks: list[TrackState]
    target: TrackState | None
    command: Command
    rows: list[dict] = field(default_factory=list)
    annotated: np.ndarray | None = None


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        rover: RoverController,
        writer: EventWriter,
        cfg: Config,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._detector = detector
        self._rover = rover
        self._writer = writer
        self._cfg = cfg
        self._clock = clock
        self._tracks = TrackStore(cfg)
        self._selector = TargetSelector(cfg)
        self._fps = 0.0
        self._last_frame_at: float | None = None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        now = self._clock()
        h, w = frame.shape[:2]

        detections = self._detector.scan(frame)
        self._tracks.update(detections, w, h, frame_index)

        if self._detector.should_confirm(bool(detections), now):
            self._tracks.apply_confirmations(self._detector.confirm(frame))

        self._tracks.prune(frame_index)

        tracks = self._tracks.tracks()
        target = self._selector.select(tracks, now)
        command = compute_command(target, self._cfg)
        self._rover.drive(turn=command.turn, forward=command.drive)

        self._update_fps(now)
        annotated = draw_overlay(
            frame, tracks, target.track_id if target else None, command, self._fps
        )

        rows: list[dict] = []
        confirmed = [t for t in tracks if t.confirmed]
        for t in confirmed:
            saved = self._writer.save_frame(annotated, t.track_id, frame_index, now)
            rows.append(
                build_event(
                    track=t,
                    target_id=target.track_id if target else None,
                    command=command,
                    frame_index=frame_index,
                    timestamp=now,
                    annotated_frame=saved,
                )
            )
        if rows:
            self._writer.emit(rows)

        return FrameResult(frame_index, tracks, target, command, rows, annotated)

    def run(
        self,
        source: Iterable[np.ndarray],
        max_frames: int | None = None,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> int:
        """Drive the pipeline over a frame stream. Always stops the rover."""
        processed = 0
        try:
            for index, frame in enumerate(source):
                if max_frames is not None and index >= max_frames:
                    break
                processed += 1
                try:
                    result = self.process_frame(frame, index)
                except Exception:
                    # NFR4: a single failed frame must not crash the pipeline.
                    log.exception("frame %d failed; continuing", index)
                    continue
                if on_frame is not None:
                    on_frame(result)
        finally:
            self._rover.stop()
        return processed

    def _update_fps(self, now: float) -> None:
        if self._last_frame_at is not None:
            dt = now - self._last_frame_at
            if dt > 0:
                instant = 1.0 / dt
                self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant
        self._last_frame_at = now
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_smoke.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, all tests green

- [ ] **Step 6: Commit**

```bash
git add rescue_vision/pipeline.py tests/test_pipeline_smoke.py
git commit -m "feat: add pipeline orchestration with end-to-end smoke test"
```

---

### Task 12: CLI, export script, and README

**Files:**
- Create: `rescue_vision/cli.py`, `rescue_vision/__main__.py`, `scripts/export_models.py`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_parser() -> argparse.ArgumentParser`, `config_from_args(args) -> Config`, `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
from rescue_vision.cli import build_parser, config_from_args


def test_source_is_required():
    import pytest

    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_select_the_console_rover():
    """Motors must never engage by accident on a dev machine."""
    args = build_parser().parse_args(["--source", "clip.mp4"])
    assert args.rover == "console"


def test_model_paths_are_overridable_for_the_pi():
    args = build_parser().parse_args(
        ["--source", "picamera", "--scan-model", "yolo26n.onnx",
         "--confirm-model", "yolo26s.onnx"]
    )
    cfg = config_from_args(args)
    assert cfg.scan_model == "yolo26n.onnx"
    assert cfg.confirm_model == "yolo26s.onnx"


def test_kp_and_deadband_are_tunable_from_the_command_line():
    """These get tuned on the chassis, so they must not need a code edit."""
    args = build_parser().parse_args(
        ["--source", "0", "--kp", "0.05", "--deadband-deg", "8"]
    )
    cfg = config_from_args(args)
    assert cfg.kp == 0.05
    assert cfg.deadband_deg == 8.0


def test_no_save_frames_flag_disables_frame_output():
    args = build_parser().parse_args(["--source", "0", "--no-save-frames"])
    assert config_from_args(args).save_frames is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rescue_vision.cli'`

- [ ] **Step 3: Write minimal implementation**

`rescue_vision/cli.py`:
```python
"""Command-line entry point. Wires the two swappable ends of the pipe."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from rescue_vision.config import Config
from rescue_vision.events import EventWriter
from rescue_vision.frame_source import create_frame_source
from rescue_vision.pipeline import Pipeline
from rescue_vision.rover import ConsoleRover, GpioZeroRover

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rescue_vision",
        description="Person detection, bearing, and turn command for a rescue rover.",
    )
    p.add_argument(
        "--source",
        required=True,
        help='Frame source: a video path, a webcam index ("0"), or "picamera".',
    )
    p.add_argument(
        "--rover",
        choices=["console", "gpiozero"],
        default="console",
        help="Motor backend. Defaults to console so motors never move by accident.",
    )
    p.add_argument("--scan-model", default=None, help="Override the scan model path.")
    p.add_argument("--confirm-model", default=None, help="Override the confirm model path.")
    p.add_argument("--output-dir", default="output", help="Where JSONL and frames go.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--display", action="store_true",
                   help="Show a preview window. NEVER use this on the headless Pi.")
    p.add_argument("--save-video", default=None, help="Write an annotated MP4 here.")
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--deadband-deg", type=float, default=None)
    p.add_argument("--min-turn", type=float, default=None)
    p.add_argument("--stby-pin", type=int, default=None,
                   help="TB6612FNG standby pin. Omit for an L298N.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    if args.scan_model:
        overrides["scan_model"] = args.scan_model
    if args.confirm_model:
        overrides["confirm_model"] = args.confirm_model
    if args.kp is not None:
        overrides["kp"] = args.kp
    if args.deadband_deg is not None:
        overrides["deadband_deg"] = args.deadband_deg
    if args.min_turn is not None:
        overrides["min_turn"] = args.min_turn
    if args.no_save_frames:
        overrides["save_frames"] = False
    return dataclasses.replace(Config(), **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = config_from_args(args)

    out = Path(args.output_dir)
    writer = EventWriter(out / "events.jsonl", out / "detections", cfg)

    if args.rover == "gpiozero":
        rover = GpioZeroRover(cfg, stby_pin=args.stby_pin)
    else:
        rover = ConsoleRover(cfg)

    # Imported here so --help works without paying the ultralytics import cost.
    from rescue_vision.detector import CascadeDetector

    source = create_frame_source(args.source, cfg)
    detector = CascadeDetector(cfg)
    pipeline = Pipeline(detector, rover, writer, cfg)

    video_writer = None
    if args.save_video:
        import cv2

        video_writer = cv2.VideoWriter(
            args.save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            15.0,
            (source.width, source.height),
        )

    def on_frame(result) -> None:
        if result.annotated is None:
            return
        if video_writer is not None:
            video_writer.write(result.annotated)
        if args.display:
            import cv2

            cv2.imshow("rescue_vision", result.annotated)
            cv2.waitKey(1)

    try:
        count = pipeline.run(source, max_frames=args.max_frames, on_frame=on_frame)
        log.info("processed %d frames", count)
        return 0
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130
    finally:
        # Order matters: motors first, always.
        rover.close()
        source.close()
        writer.close()
        if video_writer is not None:
            video_writer.release()
        if args.display:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
```

`rescue_vision/__main__.py`:
```python
import sys

from rescue_vision.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

`scripts/export_models.py`:
```python
"""One-time ONNX export. PRD 6.6.

Run this on the DESKTOP and copy the .onnx files to the Pi. ONNX artifacts are
portable, and desktop export avoids the ARM-specific export failures described
in PRD 6.5 as well as a slow, memory-hungry export on the SD card.

half=False is deliberate: ONNX Runtime's CPU execution provider has no fp16
fast path, so fp16 export gains nothing and can cost.
"""

from __future__ import annotations

from ultralytics import YOLO

EXPORTS = [
    ("yolo26n.pt", 480),  # scan: runs every frame
    ("yolo26s.pt", 640),  # confirm: candidates only
]


def main() -> None:
    for weights, imgsz in EXPORTS:
        print(f"exporting {weights} at imgsz={imgsz} ...")
        path = YOLO(weights).export(
            format="onnx", imgsz=imgsz, half=False, simplify=True
        )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
```

`README.md`:
```markdown
# Rescue Rover Vision Subsystem

Person detection, bearing, coarse distance, and a normalized turn command for
an autonomous rescue rover. Implements `PRD.md` v2.1.

## Setup (Windows dev)

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

Model weights download automatically on first run.

## Run

Webcam, console motor backend (nothing moves):

    .venv\Scripts\python.exe -m rescue_vision --source 0 --display

Recorded clip, writing an annotated MP4:

    .venv\Scripts\python.exe -m rescue_vision --source clip.mp4 --save-video out.mp4

Output lands in `output/`: `events.jsonl` (schema `rescue.detection.v1`) and
`detections/` frames.

## Tests

    .venv\Scripts\python.exe -m pytest -q

## Deploying to the Pi 5

1. Export the models on the desktop: `python scripts/export_models.py`
2. Copy `yolo26n.onnx` and `yolo26s.onnx` to the Pi.
3. Set up the Pi per PRD 6.6 (`--system-site-packages` venv; `lap` is required).
4. Bench-test with the wheels OFF THE GROUND:

       python -m rescue_vision --source picamera --rover gpiozero \
           --scan-model yolo26n.onnx --confirm-model yolo26s.onnx

Never pass `--display` on the headless Pi.

`GpioZeroRover` and `PiCameraSource` are written from the spec and have never
been run against hardware. Expect the gpiochip number and pin mapping in
`rescue_vision/rover.py` to need adjustment on first boot.

## Sign conventions

Negative `bearing_deg` = person left of centre. Positive `turn_command` = rover
rotates clockwise/right. `turn_command` takes the same sign as `bearing_deg`.
If the rover turns away from people, swap the motor pin pairs in `MOTOR_PINS` —
do not negate `kp`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Verify `--help` works without loading ultralytics**

Run: `.venv/Scripts/python.exe -m rescue_vision --help`
Expected: usage text printed, exit 0

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add rescue_vision/cli.py rescue_vision/__main__.py scripts/ README.md tests/test_cli.py
git commit -m "feat: add CLI, ONNX export script, and README"
```

---

### Task 13: Real-model acceptance run

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Consumes: the complete package.
- Produces: evidence that the PRD 6.10 acceptance test passes with real inference.

- [ ] **Step 1: Run against a real frame source**

Run:
```
.venv/Scripts/python.exe -m rescue_vision --source 0 --max-frames 120 \
    --save-video output/acceptance.mp4 -v
```

If no webcam is available, substitute a video file path.

- [ ] **Step 2: Check the acceptance criteria (PRD 6.10)**

Confirm all of:
- The run completes with no unhandled exceptions.
- `output/events.jsonl` exists and every line parses as JSON with
  `"schema": "rescue.detection.v1"`.
- At least one confirmed track appears when a person is in frame.
- A person visibly on the LEFT of frame produces a negative `bearing_deg`.
- `output/acceptance.mp4` plays and shows boxes with track IDs.

Verify the bearing sign from the log rather than by eye:
```
.venv/Scripts/python.exe -c "import json;rows=[json.loads(l) for l in open('output/events.jsonl')];print('rows',len(rows));print('bearing range',min(r['bearing_deg'] for r in rows),max(r['bearing_deg'] for r in rows));print('targets per frame OK',all(sum(1 for r in rows if r['frame_index']==f and r['is_target'])<=1 for f in {r['frame_index'] for r in rows}))"
```

- [ ] **Step 3: Measure throughput**

Run the same command and note the `fps` value in the annotated video's status
line. Record it — it is the desktop baseline that the Pi number gets compared
against, and PRD NFR2 wants ≥10 FPS on the Pi.

- [ ] **Step 4: Commit any fixes**

If the run surfaced bugs, fix them, re-run the suite, and commit:

```bash
git add -A
git commit -m "fix: address issues found in acceptance run"
```

---

## Self-Review

**Spec coverage.** Every functional requirement maps to a task: FR1/FR12 → Task 9; FR2 → Task 10; FR3/FR5 → Task 7; FR4 → Task 8; FR6 → Task 3; FR7 → Task 2; FR8 → Task 2; FR9 → Tasks 4, 5; FR10 → Task 5; FR11 → Task 6. Non-functional: NFR4 → Task 11 (`test_a_failing_frame_does_not_kill_the_run`); NFR5 → `lap` pinned in `requirements.txt`; NFR6 → Task 6 and Task 11's crash tests; NFR7 → lazy imports throughout, asserted in Task 6. Design deltas D1–D5 → Task 12 (`export_models.py`), Task 9 (`WebcamSource`), Task 6 (`stby_pin`).

**Known gaps, deliberately left.** `SEARCH_HOLD` (PRD §7 — keep turning briefly after target loss) is defined in `Config` but not consumed by any task. It is a mitigation for a failure mode that only appears on a moving rover, and cannot be tested or tuned without hardware. Implement it after the first floor test, when the actual loss behaviour can be observed. `TARGET_HOLD` stickiness is implemented (Task 4) because it is testable in pure code.

**Type consistency.** `TrackState` field names are identical across Tasks 1, 3, 4, 5, 7, 8, 11. `Command(turn, drive)` is used uniformly — note `build_event` maps it to the PRD's `turn_command`/`drive_command` JSON keys. `BBox.as_xyxy_ints()` is defined in Task 1 and used in Tasks 7 and 8. `should_confirm(has_candidates, now)` has one signature, shared by `CascadeDetector` and `ScriptedDetector` via `_RateLimitedConfirm`.
