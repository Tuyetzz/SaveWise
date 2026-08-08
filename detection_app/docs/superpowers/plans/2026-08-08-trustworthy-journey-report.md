# Trustworthy Journey Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the journey log count each person exactly once, give each person a stable colour across the whole sweep, and render the result as a self-contained HTML report.

**Architecture:** Three strictly ordered, independently shippable changes to the existing observer pipeline. Task 1 fixes a measured correctness defect in `SightingRecorder`; Task 2 makes that fix visible by keying box colour on sighting identity; Task 3 renders the finished sightings as an offline HTML artifact.

**Tech Stack:** Python 3.13, OpenCV, stdlib `base64`/`html`. No new dependencies.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-08-08-trustworthy-journey-report-design.md`.
- **Colour follows the entity, never its rank.** Colour is keyed on `sighting_id`, never on a track's index in the current frame. A person leaving must not repaint the survivors.
- **Identity is never colour-alone.** Every box label carries `P<n>`. Validated: 3 colours pass all-pairs separation, 6 hard-fail the normal-vision floor.
- `geometry.py` and `tracking.py` stay pure — no I/O, no clock reads. Time is a parameter.
- **cv2 Hershey fonts are ASCII-only.** Video labels must not contain `·` or any non-ASCII character; it renders as `?`. The HTML report may use any character.
- Run everything through `.venv/Scripts/python.exe`.
- Every commit message ends with: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| File | Change |
|---|---|
| `rescue_vision/config.py` | Add `sighting_gap_s` |
| `rescue_vision/sightings.py` | Grace period in `finalise_absent`; expose `sighting_id_for` and `journey_duration_s` |
| `rescue_vision/palette.py` | **New** — the validated colour slots, shared by overlay and report |
| `rescue_vision/annotate.py` | Colour by sighting id, `P<n>` label, dark outline |
| `rescue_vision/pipeline.py` | Pass the track→sighting map to `draw_overlay` |
| `rescue_vision/report.py` | **New** — self-contained HTML |
| `rescue_vision/cli.py` | Write the report on exit |

---

### Task 1: Sighting continuity

**Files:**
- Modify: `rescue_vision/config.py`, `rescue_vision/sightings.py`
- Test: `tests/test_sightings.py`

**Interfaces:**
- Consumes: `Config`, `Sighting`, `TrackState`.
- Produces: `Config.sighting_gap_s: float = 1.5`; `finalise_absent` gains grace-period behaviour; `SightingRecorder.journey_duration_s` property.

- [ ] **Step 1: Write the failing regression test**

Append to `tests/test_sightings.py`:
```python
def test_a_dropped_frame_does_not_split_one_person_into_two_sightings(tmp_path):
    """The measured defect: at camera-module noise levels the detector misses
    ~23% of frames, and closing on the first miss logged one person ~9 times."""
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.finalise_absent({1}, now=0.0)
    r.finalise_absent(set(), now=0.2)      # missed frame, inside the grace window
    r.observe([track()], FRAME, now=0.4)   # same ByteTrack id returns
    r.finalise_absent({1}, now=0.4)
    r.close()
    assert len(rows(tmp_path)) == 1


def test_a_long_absence_still_closes_the_sighting(tmp_path):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.finalise_absent(set(), now=5.0)
    assert len(rows(tmp_path)) == 1
    r.observe([track()], FRAME, now=6.0)
    r.close()
    assert len(rows(tmp_path)) == 2


def test_duration_reports_time_actually_seen_not_the_grace_period(tmp_path):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.observe([track()], FRAME, now=2.0)
    r.finalise_absent(set(), now=9.0)
    assert rows(tmp_path)[0]["duration_s"] == pytest.approx(2.0)


@pytest.mark.parametrize("drop_rate", [0.0, 0.1, 0.23, 0.4])
def test_one_person_is_one_sighting_at_every_realistic_dropout_rate(
    tmp_path, drop_rate
):
    """The demo's central claim, as a test."""
    import random

    rng = random.Random(0)
    r = rec(tmp_path)
    for i in range(60):
        now = i * 0.1
        seen = rng.random() >= drop_rate
        tracks = [track()] if seen else []
        r.observe(tracks, FRAME, now)
        r.finalise_absent({t.track_id for t in tracks}, now)
    r.close()
    assert len(rows(tmp_path)) == 1


def test_journey_duration_covers_the_whole_run(tmp_path):
    r = rec(tmp_path)
    r.observe([], FRAME, now=100.0)
    r.observe([track()], FRAME, now=104.5)
    r.close()
    assert r.journey_duration_s == pytest.approx(4.5)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sightings.py -q`
Expected: the dropout parametrisation fails at 0.1/0.23/0.4, plus `AttributeError` on `journey_duration_s`.

- [ ] **Step 3: Add the config constant**

In `rescue_vision/config.py`, under `# --- Output ---`:
```python
    # Grace before a sighting closes. Sized against ByteTrack's default
    # track_buffer (30 frames, ~2-3s at our rates): a track returning inside
    # that window still carries the SAME id, so the returning detection rejoins
    # its sighting. A longer grace than the buffer would be actively harmful --
    # the person comes back with a NEW id and we would be merging strangers.
    sighting_gap_s: float = 1.5
```

- [ ] **Step 4: Implement the grace period**

In `rescue_vision/sightings.py`, replace `finalise_absent` and add the duration property:
```python
    def finalise_absent(self, visible_track_ids: Iterable[int], now: float) -> None:
        """Close sightings whose track has been gone longer than the grace period.

        Closing on the first missing frame was wrong: the detector legitimately
        drops frames (~23% under camera-module noise), and each miss split one
        person into another sighting -- ~9.4 records for a single person at that
        rate. ByteTrack holds the id across the gap; this waits for it.
        """
        if self._start is None:
            return
        visible = set(visible_track_ids)
        elapsed = now - self._start
        for track_id, s in list(self._open.items()):
            if track_id in visible:
                continue
            if elapsed - s.last_seen_s >= self._cfg.sighting_gap_s:
                self._finalise(track_id)

    @property
    def journey_duration_s(self) -> float:
        """Wall time from the first observed frame to the most recent one."""
        return self._last_seen_any
```

Track the run length. In `__init__` add `self._last_seen_any = 0.0`, and in `observe`, immediately after the `if self._start is None:` guard:
```python
        self._last_seen_any = now - self._start
```

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sightings.py -q`
Expected: PASS, all tests green.

- [ ] **Step 6: Confirm the defect is gone end to end**

Run: `.venv/Scripts/python.exe -m pytest -q` then
`.\demo.bat clip`
Expected: full suite green; the clip still reports 3 sightings.

- [ ] **Step 7: Commit**

```bash
git add rescue_vision/config.py rescue_vision/sightings.py tests/test_sightings.py
git commit -m "fix: one person is one sighting across detector dropouts"
```

---

### Task 2: Per-person colour

**Files:**
- Create: `rescue_vision/palette.py`
- Modify: `rescue_vision/sightings.py`, `rescue_vision/annotate.py`, `rescue_vision/pipeline.py`
- Test: `tests/test_palette.py`, `tests/test_annotate.py`

**Interfaces:**
- Consumes: `SightingRecorder` from Task 1.
- Produces:
  - `palette.PERSON_COLOURS_HEX: list[str]` (8 slots, fixed order)
  - `palette.colour_for(sighting_id: int) -> str`
  - `palette.hex_to_bgr(hex_colour: str) -> tuple[int, int, int]`
  - `SightingRecorder.sighting_id_for(track_id: int) -> int | None`
  - `annotate.draw_overlay(frame, tracks, fps, sightings_count, sighting_ids: dict[int, int])`
  - `annotate.format_label(track, sighting_id: int | None) -> str`

- [ ] **Step 1: Write the failing palette test**

`tests/test_palette.py`:
```python
from rescue_vision.palette import PERSON_COLOURS_HEX, colour_for, hex_to_bgr


def test_the_first_three_slots_are_the_validated_all_pairs_set():
    """3 colours pass all-pairs separation; 6 hard-fail the normal-vision
    floor. The PRD assumes 1-3 people, so these three carry the real case."""
    assert PERSON_COLOURS_HEX[:3] == ["#2a78d6", "#eb6834", "#1baf7a"]


def test_each_person_gets_a_distinct_colour():
    assert len({colour_for(i) for i in range(1, 9)}) == 8


def test_colour_is_stable_for_a_given_sighting():
    assert colour_for(2) == colour_for(2)


def test_colour_follows_the_sighting_id_not_call_order():
    """A person leaving must not repaint the survivors."""
    first = colour_for(3)
    colour_for(1)
    colour_for(2)
    assert colour_for(3) == first


def test_slots_wrap_after_eight_people():
    assert colour_for(9) == colour_for(1)


def test_hex_converts_to_bgr_for_opencv():
    assert hex_to_bgr("#2a78d6") == (214, 120, 42)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_palette.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'rescue_vision.palette'`

- [ ] **Step 3: Write the palette**

`rescue_vision/palette.py`:
```python
"""Per-person colours, shared by the video overlay and the HTML report.

Colour is keyed on SIGHTING ID, never on a track's position in the current
frame: colour follows the entity, never its rank. A person leaving must not
repaint the survivors.

Slot order is a validated categorical theme, not a preference. Checked with the
dataviz validator on the light surface:

  first 3 slots, --pairs all : PASS  (CVD dE 9.2, normal-vision dE 24.0)
  first 6 slots, --pairs all : FAIL  (magenta<->orange dE 12.9, below the 15
                                      floor -- indistinguishable even to a
                                      full-colour-vision viewer)

The PRD assumes 1-3 people, so the first three slots carry the real case. Past
three, colour cannot carry identity alone -- which is why every label also
prints P<n>.
"""

from __future__ import annotations

PERSON_COLOURS_HEX: list[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#008300",  # green
]

# Not yet through the confirm cascade, so not yet counted as a person.
TENTATIVE_HEX = "#808080"


def colour_for(sighting_id: int) -> str:
    """Fixed-order slot for a sighting. Wraps past eight; P<n> disambiguates."""
    return PERSON_COLOURS_HEX[(sighting_id - 1) % len(PERSON_COLOURS_HEX)]


def hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    """OpenCV wants BGR, and hex is RGB."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_palette.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Write the failing annotate tests**

Replace the label tests in `tests/test_annotate.py` and add colour tests:
```python
def test_label_carries_the_person_number_and_confidence():
    assert format_label(track(1), sighting_id=2) == "P2  confidence_score = 0.90"


def test_label_is_pure_ascii_because_hershey_fonts_cannot_render_more():
    label = format_label(track(1), sighting_id=2)
    assert label.isascii()


def test_an_unconfirmed_track_has_no_person_number_yet():
    t = track(1, confirmed=False)
    assert format_label(t, sighting_id=None) == "confidence_score = 0.90"


def test_each_person_is_drawn_in_their_own_colour():
    frame = np.zeros((480, 640, 3), np.uint8)
    a, b = track(1), track(2)
    b.bbox = BBox(300.0, 50.0, 400.0, 350.0)
    out = draw_overlay(frame, [a, b], 12.0, 2, {1: 1, 2: 2})

    from rescue_vision.palette import hex_to_bgr, colour_for

    pixels = out.reshape(-1, 3)
    for sid in (1, 2):
        want = np.array(hex_to_bgr(colour_for(sid)))
        assert (pixels == want).all(axis=1).any(), f"P{sid} colour missing"


def test_a_persons_colour_does_not_change_when_someone_else_leaves():
    """Colour follows the entity, never its rank."""
    frame = np.zeros((480, 640, 3), np.uint8)
    from rescue_vision.palette import colour_for, hex_to_bgr

    second = track(2)
    second.bbox = BBox(300.0, 50.0, 400.0, 350.0)

    both = draw_overlay(frame, [track(1), second], 12.0, 2, {1: 1, 2: 2})
    alone = draw_overlay(frame, [second], 12.0, 2, {2: 2})

    want = np.array(hex_to_bgr(colour_for(2)))
    for img in (both, alone):
        assert (img.reshape(-1, 3) == want).all(axis=1).any()
```

- [ ] **Step 6: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotate.py -q`
Expected: FAIL — `format_label()` takes 1 positional argument, `draw_overlay()` signature mismatch.

- [ ] **Step 7: Update annotate**

In `rescue_vision/annotate.py`, replace the colour constants and both functions:
```python
from rescue_vision.palette import TENTATIVE_HEX, colour_for, hex_to_bgr

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_OUTLINE = (20, 20, 20)


def format_label(t: TrackState, sighting_id: int | None) -> str:
    """`P2  confidence_score = 0.89`, or confidence alone before confirmation.

    ASCII only: cv2's Hershey fonts render anything else as '?'. The person
    number means identity is never carried by colour alone -- necessary past
    three people, and for the ~1 in 12 men with a colour vision deficiency.
    """
    conf = f"confidence_score = {t.display_confidence:.2f}"
    return f"P{sighting_id}  {conf}" if sighting_id is not None else conf


def draw_overlay(
    frame: np.ndarray,
    tracks: list[TrackState],
    fps: float,
    sightings_count: int = 0,
    sighting_ids: dict[int, int] | None = None,
) -> np.ndarray:
    """One coloured box per person. Returns a new image.

    Grey means "seen but not yet confirmed"; a colour means "counted, and in
    the report". Colour is keyed on sighting id so it stays with the person for
    the whole sweep.
    """
    sighting_ids = sighting_ids or {}
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)

    ordered = sorted(tracks, key=lambda t: t.confirmed)
    occupied: list[tuple[int, int, int, int]] = []

    for t in ordered:
        sid = sighting_ids.get(t.track_id) if t.confirmed else None
        colour = hex_to_bgr(colour_for(sid) if sid is not None else TENTATIVE_HEX)

        x1, y1, x2, y2 = t.bbox.as_xyxy_ints()
        # Dark outline first: the palette's contrast is measured against a
        # controlled chart surface, but ours is whatever the camera sees. A
        # blue box on a blue door would otherwise vanish.
        cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), _OUTLINE, 4)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = format_label(t, sid)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        lx, ly = place_label(x1, y1 - 4, tw, th, w, h, occupied)
        occupied.append((lx, ly - th - 4, lx + tw + 4, ly))

        cv2.rectangle(out, (lx - 1, ly - th - 5), (lx + tw + 5, ly + 1), _OUTLINE, -1)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 4, ly), colour, -1)
        cv2.putText(
            out, label, (lx + 2, ly - 3), _FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        if abs(lx - x1) > 2 or abs(ly - (y1 - 4)) > 2:
            cv2.line(out, (lx, ly), (x1, y1), colour, 1)

    people = sum(1 for t in tracks if t.confirmed)
    status = f"fps={fps:.1f} people={people} sightings={sightings_count}"
    cv2.putText(out, status, (8, h - 10), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
```

Delete the now-unused `_CONFIRMED_COLOUR` and `_TENTATIVE_COLOUR` constants.

- [ ] **Step 8: Expose the track→sighting mapping**

In `rescue_vision/sightings.py`:
```python
    def sighting_id_for(self, track_id: int) -> int | None:
        """The open sighting for this track, if it has one yet."""
        s = self._open.get(track_id)
        return s.sighting_id if s else None
```

In `rescue_vision/pipeline.py`, `process_frame` — build the map before drawing, and note that `observe` must run first so a newly confirmed person already has a sighting id on the frame they appear:
```python
        self._recorder.observe(tracks, frame, now)
        sighting_ids = {
            t.track_id: sid
            for t in tracks
            if (sid := self._recorder.sighting_id_for(t.track_id)) is not None
        }
        self._update_fps(now)
        annotated = draw_overlay(
            frame, tracks, self._fps, len(self._recorder.summary()), sighting_ids
        )
        self._recorder.attach_frame(tracks, annotated)
        self._recorder.finalise_absent({t.track_id for t in tracks}, now)
```

This reorders `observe` before `draw_overlay`, so the best-frame capture needs
splitting out. In `sightings.py`, remove the frame copy from `observe` and add:
```python
    def attach_frame(self, tracks: list[TrackState], annotated: np.ndarray) -> None:
        """Store the annotated frame for any track whose peak was just set.

        Split from observe() because the overlay cannot be drawn until sighting
        ids exist, and ids are assigned by observe().
        """
        if not self._cfg.save_frames:
            return
        for t in tracks:
            s = self._open.get(t.track_id)
            if s is not None and t.confidence >= s.peak_confidence:
                self._best_frame[t.track_id] = annotated.copy()
```

- [ ] **Step 9: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all green. Fix any `draw_overlay` call sites in `test_pipeline_smoke.py`.

- [ ] **Step 10: Look at the output**

Run: `.venv/Scripts/python.exe -m rescue_vision --source tests/fixtures/pan_clip.mp4 --confirm-min-interval 0.5`
Then open `output/sightings/sighting_001.jpg`. Confirm three distinct box colours, `P1`/`P2`/`P3` labels, and that the dark outline keeps every box legible against the bus.

- [ ] **Step 11: Commit**

```bash
git add rescue_vision/palette.py rescue_vision/annotate.py rescue_vision/sightings.py rescue_vision/pipeline.py tests/
git commit -m "feat: stable per-person colour keyed on sighting identity"
```

---

### Task 3: The HTML report

**Files:**
- Create: `rescue_vision/report.py`
- Modify: `rescue_vision/cli.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Sighting` list and `journey_duration_s` from Task 1, `colour_for` from Task 2.
- Produces: `report.build_report(sightings: list[Sighting], out_dir: Path, duration_s: float) -> Path`.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
from rescue_vision.report import build_report
from rescue_vision.types import Sighting


def sighting(sighting_id=1, distance=3.2, frame=None):
    return Sighting(
        sighting_id=sighting_id,
        track_id=sighting_id,
        first_seen_s=12.4,
        last_seen_s=17.9,
        frames_seen=48,
        peak_confidence=0.93,
        confidence_sum=48 * 0.81,
        bearing_at_peak_deg=-8.3,
        closest_distance_m=distance,
        best_frame_path=frame,
    )


def test_report_is_written_and_names_the_people_found(tmp_path):
    path = build_report([sighting(1), sighting(2)], tmp_path, duration_s=134.0)
    html = path.read_text(encoding="utf-8")
    assert path.name == "report.html"
    assert "2" in html
    assert "P1" in html and "P2" in html


def test_an_empty_journey_still_produces_a_report(tmp_path):
    html = build_report([], tmp_path, duration_s=60.0).read_text(encoding="utf-8")
    assert "No humans" in html


def test_each_card_carries_its_person_colour(tmp_path):
    from rescue_vision.palette import colour_for

    html = build_report([sighting(1), sighting(2)], tmp_path, 10.0).read_text("utf-8")
    assert colour_for(1) in html
    assert colour_for(2) in html


def test_images_are_inlined_so_the_file_stands_alone(tmp_path):
    import cv2
    import numpy as np

    frames = tmp_path / "sightings"
    frames.mkdir()
    cv2.imwrite(str(frames / "s1.jpg"), np.zeros((20, 20, 3), np.uint8))
    html = build_report(
        [sighting(1, frame="sightings/s1.jpg")], tmp_path, 10.0
    ).read_text("utf-8")
    assert "data:image/jpeg;base64," in html
    assert "sightings/s1.jpg" not in html.split("base64,")[0][-200:]


def test_a_missing_image_degrades_instead_of_crashing(tmp_path):
    html = build_report(
        [sighting(1, frame="sightings/gone.jpg")], tmp_path, 10.0
    ).read_text("utf-8")
    assert "P1" in html


def test_unknown_distance_is_not_reported_as_a_number(tmp_path):
    html = build_report([sighting(1, distance=None)], tmp_path, 10.0).read_text("utf-8")
    assert "not measurable" in html


def test_report_states_the_coverage_limitation(tmp_path):
    """Judges reward a team that knows what it did not search."""
    html = build_report([sighting(1)], tmp_path, 10.0).read_text("utf-8")
    assert "53.5" in html
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'rescue_vision.report'`

- [ ] **Step 3: Write the report**

`rescue_vision/report.py`:
```python
"""Self-contained HTML journey report.

Images are inlined as base64 so the file opens offline from any browser with no
server and no sibling files -- it can be handed to someone or emailed as-is.
"""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path

from rescue_vision.palette import colour_for
from rescue_vision.types import Sighting

log = logging.getLogger(__name__)


def _inline_image(out_dir: Path, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    path = out_dir / rel_path
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        log.warning("could not inline %s: %s", path, exc)
        return None
    return f"data:image/jpeg;base64,{data}"


def _card(s: Sighting, out_dir: Path) -> str:
    colour = colour_for(s.sighting_id)
    src = _inline_image(out_dir, s.best_frame_path)
    img = (
        f'<img src="{src}" alt="Person {s.sighting_id}">'
        if src
        else '<div class="noimg">no image</div>'
    )
    dist = (
        f"{s.closest_distance_m:.1f} m"
        if s.closest_distance_m is not None
        else "not measurable"
    )
    side = "left" if s.bearing_at_peak_deg < 0 else "right"
    return f"""
    <article class="card" style="--accent:{colour}">
      <div class="thumb">{img}</div>
      <div class="body">
        <h2><span class="swatch"></span>P{s.sighting_id}</h2>
        <dl>
          <dt>First seen</dt><dd>{s.first_seen_s:.1f}s into the sweep</dd>
          <dt>Visible for</dt><dd>{s.duration_s:.1f}s ({s.frames_seen} frames)</dd>
          <dt>Peak confidence</dt><dd><strong>{s.peak_confidence:.0%}</strong></dd>
          <dt>Mean confidence</dt><dd>{s.mean_confidence:.0%}</dd>
          <dt>Direction</dt>
          <dd>{abs(s.bearing_at_peak_deg):.0f}&deg; to the {side}</dd>
          <dt>Nearest approach</dt><dd>{dist}</dd>
        </dl>
      </div>
    </article>"""


_CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;
      --line:rgba(11,11,11,.12)}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;
      --ink:#fff;--muted:#c3c2b7;--line:rgba(255,255,255,.14)}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem;background:var(--bg);color:var(--ink);
     font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:60rem;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem}
.lede{color:var(--muted);margin:0 0 .35rem}
.count{font-size:3.5rem;font-weight:650;line-height:1;margin:1.25rem 0 .25rem}
.caveat{color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);
        margin-top:2rem;padding-top:1rem}
.cards{display:grid;gap:1rem;margin-top:1.75rem;
       grid-template-columns:repeat(auto-fill,minmax(19rem,1fr))}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
      overflow:hidden;border-left:5px solid var(--accent)}
.thumb{background:#000;aspect-ratio:4/3;display:grid;place-items:center}
.thumb img{width:100%;height:100%;object-fit:contain;display:block}
.noimg{color:#777;font-size:.85rem}
.body{padding:.9rem 1.1rem 1.1rem}
h2{font-size:1.05rem;margin:0 0 .6rem;display:flex;align-items:center;gap:.5rem}
.swatch{width:.85rem;height:.85rem;border-radius:3px;background:var(--accent);
        flex:none}
dl{display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;margin:0;
   font-size:.9rem}
dt{color:var(--muted)}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
"""


def build_report(sightings: list[Sighting], out_dir: Path, duration_s: float) -> Path:
    """Render the journey report. Returns the path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(sightings)

    if n:
        headline = f'<div class="count">{n}</div><p class="lede">' + (
            "human found" if n == 1 else "humans found"
        ) + "</p>"
        cards = "".join(_card(s, out_dir) for s in sightings)
    else:
        headline = '<div class="count">0</div><p class="lede">No humans detected</p>'
        cards = ""

    mins, secs = divmod(int(duration_s), 60)
    doc = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rover journey report</title>
<style>{_CSS}</style>
<body><div class="wrap">
  <h1>Rover journey report</h1>
  <p class="lede">Autonomous sweep, {mins}m {secs:02d}s</p>
  {headline}
  <div class="cards">{cards}</div>
  <p class="caveat">
    Each person above was counted once, however many frames they appeared in.
    Times are measured from the start of the sweep and bearings are relative to
    the rover's heading at that moment &mdash; the rover carries no odometry, so
    this report records <em>when</em> someone was seen, never <em>where</em>.
    The camera is fixed forward with a 53.5&deg; field of view and does not
    scan, so anyone outside that cone as the rover passed was not searched.
  </p>
</div></body></html>"""

    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
```

Note `html` is imported for future escaping needs; if the linter objects, drop
the import — no user-controlled strings reach the template today.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Wire it into the CLI**

In `rescue_vision/cli.py`, inside the `finally` block, immediately after
`print(format_summary(...))`:
```python
        from rescue_vision.report import build_report

        report = build_report(
            recorder.summary(), out, recorder.journey_duration_s
        )
        print(f"  Report: {report.resolve()}\n")
```

- [ ] **Step 6: Verify end to end**

Run: `.\demo.bat clip`, then open `output/report.html` in a browser.
Expected: three cards, each with a photo, a colour stripe matching that person's
box colour in the video, and the coverage caveat at the foot.

- [ ] **Step 7: Full suite and commit**

```bash
.venv/Scripts/python.exe -m pytest -q
git add rescue_vision/report.py rescue_vision/cli.py tests/test_report.py
git commit -m "feat: self-contained HTML journey report"
```

---

## Self-Review

**Spec coverage.** Design §1 (continuity) → Task 1, including the parametrised
dropout regression the spec names as the deliverable's credibility. §2 (colour) →
Task 2, covering entity-keyed colour, the `P<n>` label, and the dark outline. §3
(report) → Task 3, covering self-containment, colour swatches, and the coverage
caveat. Out-of-scope items (posture, maps, cross-gap merging) have no task, as
intended.

**Ordering hazard, handled.** Task 2 must reorder `observe()` before
`draw_overlay()`, because the overlay needs sighting ids and `observe()` assigns
them. That breaks the existing best-frame capture, which lived inside `observe`
and expected the annotated frame. Task 2 Step 8 splits it into `attach_frame()`.
An implementer who skips that step will silently save unannotated frames — the
Step 10 eyeball check catches it.

**Type consistency.** `format_label` gains a second required parameter, so all
call sites move together in Task 2 Steps 5–7. `draw_overlay`'s new `sighting_ids`
argument is keyword-optional, so `test_pipeline_smoke.py` compiles unchanged but
should still be checked in Step 9. `Sighting.best_frame_path` is the field name
used by both `sightings.py` and `report.py`.
