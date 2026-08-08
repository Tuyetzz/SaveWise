"""Journey sightings log: one record per person encountered.

The rover drives itself -- straight lines, turning left when its own front
sensor finds a blockage. This subsystem never steers it. The deliverable is a
record of who was seen along the way and how confident the detector was.

Why sightings rather than frames: at 10 FPS, driving past one person for five
seconds produces ~50 near-identical detection rows. The unit a reader cares
about is the person encountered, so state is accumulated while a track is
visible and one record is written when it disappears.

Why one image per sighting: the highest-confidence frame is held in memory and
written once on finalisation. Three people means three JPEGs. This retires
PRD 9's "saved detection frames -- unbounded, this is the one that bites",
which was the largest risk to the 16 GB card.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from rescue_vision.config import Config
from rescue_vision.types import Sighting, TrackState

log = logging.getLogger(__name__)

SCHEMA = "rescue.sighting.v1"


class SightingRecorder:
    """Accumulates one record per person seen during the journey."""

    def __init__(
        self,
        cfg: Config,
        out_dir: Path,
        journey_start: float | None = None,
    ) -> None:
        """`journey_start` defaults to the first observed frame.

        Latching rather than defaulting to zero matters: the pipeline clock is
        `time.monotonic()`, which counts from boot, so a fixed zero would report
        a person "seen at 19610s". Latching on the first frame also excludes the
        seconds spent loading models before the journey really begins.
        """
        self._cfg = cfg
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self._out / "sightings"
        self._jsonl = self._out / "sightings.jsonl"
        self._start = journey_start
        self._open: dict[int, Sighting] = {}
        self._best_frame: dict[int, np.ndarray] = {}
        self._done: list[Sighting] = []
        self._next_id = 1
        self._last_seen_any = 0.0

    def observe(
        self, tracks: list[TrackState], annotated_frame: np.ndarray, now: float
    ) -> None:
        """Fold this frame's confirmed tracks into their open sightings."""
        if self._start is None:
            self._start = now
        self._last_seen_any = now - self._start
        for t in tracks:
            if not t.confirmed:
                continue
            elapsed = now - self._start
            s = self._open.get(t.track_id)
            if s is None:
                s = Sighting(
                    sighting_id=self._next_id,
                    track_id=t.track_id,
                    first_seen_s=elapsed,
                    last_seen_s=elapsed,
                    bearing_min_deg=t.bearing_deg,
                    bearing_max_deg=t.bearing_deg,
                )
                self._next_id += 1
                self._open[t.track_id] = s

            s.last_seen_s = elapsed
            s.frames_seen += 1
            s.confidence_sum += t.confidence
            s.bearing_min_deg = min(s.bearing_min_deg, t.bearing_deg)
            s.bearing_max_deg = max(s.bearing_max_deg, t.bearing_deg)

            if t.confidence > s.peak_confidence:
                s.peak_confidence = t.confidence
                s.peak_confidence_at_s = elapsed
                s.bearing_at_peak_deg = t.bearing_deg
                if self._cfg.save_frames:
                    # Keep the best look at this person, not the latest one.
                    self._best_frame[t.track_id] = annotated_frame.copy()

            # Only an estimate that passed the PRD 6.7 validity rules may
            # count. A prone person fails the aspect-ratio check, and prone
            # people are the actual rescue target -- never quietly trust one.
            if t.distance_valid:
                s.distance_valid_frames += 1
                if s.closest_distance_m is None or t.distance_m < s.closest_distance_m:
                    s.closest_distance_m = t.distance_m

    def finalise_absent(self, visible_track_ids: Iterable[int], now: float) -> None:
        """Close sightings whose track has been gone longer than the grace period.

        Closing on the first missing frame was wrong: the detector legitimately
        drops frames (~23% under camera-module noise), and each miss split one
        person into another sighting -- ~9.4 records for a single person at that
        rate, which makes the log worthless. ByteTrack holds the id across the
        gap; this waits for it rather than throwing that away.
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

    def close(self) -> None:
        """Finalise everything still open -- the journey ended mid-sighting."""
        for track_id in list(self._open):
            self._finalise(track_id)

    def summary(self) -> list[Sighting]:
        return list(self._done)

    def _finalise(self, track_id: int) -> None:
        s = self._open.pop(track_id)
        frame = self._best_frame.pop(track_id, None)
        if frame is not None:
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            path = self._frames_dir / f"sighting_{s.sighting_id:03d}.jpg"
            ok = cv2.imwrite(
                str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality]
            )
            if ok:
                s.best_frame_path = f"sightings/{path.name}"
            else:
                log.warning("failed to write %s", path)
        self._done.append(s)
        self._append(_to_row(s))

    def _append(self, row: dict[str, Any]) -> None:
        # Append per sighting rather than buffering to the end, so a crash
        # mid-journey still leaves everything found so far on disk.
        with open(self._jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


def _to_row(s: Sighting) -> dict[str, Any]:
    """One JSONL row. Times are seconds since journey start, never wall clock:
    without odometry the log can say *when*, not *where*."""
    return {
        "schema": SCHEMA,
        "sighting_id": s.sighting_id,
        "first_seen_s": round(s.first_seen_s, 2),
        "last_seen_s": round(s.last_seen_s, 2),
        "duration_s": round(s.duration_s, 2),
        "frames_seen": s.frames_seen,
        "peak_confidence": round(s.peak_confidence, 3),
        "mean_confidence": round(s.mean_confidence, 3),
        "peak_confidence_at_s": round(s.peak_confidence_at_s, 2),
        "bearing_at_peak_deg": round(s.bearing_at_peak_deg, 2),
        "bearing_range_deg": [
            round(s.bearing_min_deg, 2),
            round(s.bearing_max_deg, 2),
        ],
        "closest_distance_m": (
            round(s.closest_distance_m, 2) if s.closest_distance_m is not None else None
        ),
        "distance_valid_frames": s.distance_valid_frames,
        "best_frame": s.best_frame_path,
    }
