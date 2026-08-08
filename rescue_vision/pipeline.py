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
    """One frame in, one command and zero or more event rows out."""

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
        self._tracks.update(detections, w, h, frame_index, now)

        if self._detector.should_confirm(bool(detections), now):
            self._tracks.apply_confirmations(self._detector.confirm(frame))

        self._tracks.prune(frame_index)

        # Only tracks seen on THIS frame may steer the rover, be drawn, or be
        # logged. Retained-but-unseen tracks stay in the store purely so
        # ByteTrack can re-associate their IDs.
        tracks = self._tracks.visible_tracks(frame_index)
        target = self._selector.select(tracks, now)
        command = compute_command(target, self._cfg)
        self._rover.drive(turn=command.turn, forward=command.drive)

        self._update_fps(now)
        annotated = draw_overlay(
            frame, tracks, target.track_id if target else None, command, self._fps
        )

        # Only confirmed tracks are reported. One alert per confirmed track,
        # never one per raw detection (PRD FR6).
        rows: list[dict] = []
        for t in tracks:
            if not t.confirmed:
                continue
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
                self._fps = (
                    instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant
                )
        self._last_frame_at = now
