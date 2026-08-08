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
    """One JSONL row.

    Commands are repeated on every row so a consumer reading a single line has
    everything it needs. `confidence` is the live per-frame score, not the 1 Hz
    display sample -- logs are for analysis and must stay precise.

    `distance_m` is null when `distance_valid` is false, so a consumer cannot
    accidentally use a number this subsystem has already rejected.
    """
    return {
        "schema": SCHEMA,
        "timestamp": round(timestamp, 3),
        "frame_index": frame_index,
        "track_id": track.track_id,
        "confidence": round(track.confidence, 3),
        "bbox_xyxy": track.bbox.as_xyxy_ints(),
        "bearing_deg": round(track.bearing_deg, 2),
        "distance_m": round(track.distance_m, 2) if track.distance_valid else None,
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

    def save_frame(
        self, image, track_id: int, frame_index: int, now: float
    ) -> str | None:
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
        files = sorted(self._frames_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
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
