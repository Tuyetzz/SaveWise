"""Per-frame orchestration: detect -> track -> confirm -> record.

The rover drives itself, so nothing here steers anything. Each frame is
observed, folded into the journey's sightings log, and forgotten.

Depends on the Detector Protocol, never on a concrete model, so the whole loop
is testable with no model and no camera.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from rescue_vision.annotate import draw_overlay
from rescue_vision.config import Config
from rescue_vision.detector import Detector
from rescue_vision.events import RawEventWriter, build_event
from rescue_vision.sightings import SightingRecorder
from rescue_vision.tracking import TrackStore
from rescue_vision.types import TrackState

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame_index: int
    tracks: list[TrackState]
    sightings_so_far: int = 0
    rows: list[dict] = field(default_factory=list)
    annotated: np.ndarray | None = None


class Pipeline:
    def __init__(
        self,
        detector: Detector,
        recorder: SightingRecorder,
        cfg: Config,
        raw_writer: RawEventWriter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._detector = detector
        self._recorder = recorder
        self._raw = raw_writer
        self._cfg = cfg
        self._clock = clock
        self._tracks = TrackStore(cfg)
        self._fps = 0.0
        self._last_frame_at: float | None = None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        now = self._clock()
        h, w = frame.shape[:2]

        detections = self._detector.scan(frame)
        self._tracks.update(detections, w, h, frame_index, now)

        if self._detector.should_confirm(bool(detections), now):
            self._tracks.apply_confirmations(self._detector.confirm(frame))

        self._tracks.prune(frame_index)

        # Only tracks seen on THIS frame count. Retained-but-unseen tracks stay
        # in the store purely so ByteTrack can re-associate their IDs; treating
        # one as present would log a person who has already left the frame.
        tracks = self._tracks.visible_tracks(frame_index)

        self._update_fps(now)
        annotated = draw_overlay(
            frame, tracks, self._fps, len(self._recorder.summary())
        )

        self._recorder.observe(tracks, annotated, now)
        self._recorder.finalise_absent({t.track_id for t in tracks}, now)

        rows: list[dict] = []
        if self._raw is not None:
            rows = [build_event(t, frame_index, now) for t in tracks]
            if rows:
                self._raw.emit(rows)

        return FrameResult(
            frame_index=frame_index,
            tracks=tracks,
            sightings_so_far=len(self._recorder.summary()),
            rows=rows,
            annotated=annotated,
        )

    def run(
        self,
        source: Iterable[np.ndarray],
        max_frames: int | None = None,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> int:
        """Drive the pipeline over a frame stream, closing the log on the way out."""
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
            # Anyone still visible when the journey ends still gets a record.
            self._recorder.close()
        return processed

    def _update_fps(self, now: float) -> None:
        if self._last_frame_at is not None:
            dt = now - self._last_frame_at
            if dt > 0:
                instant = 1.0 / dt
                self._fps = (
                    instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant
                )
        self._last_frame_at = now
