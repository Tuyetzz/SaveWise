"""Optional per-frame raw detection log. Off by default.

The journey deliverable is `sightings.jsonl` -- one record per person
encountered (see `sightings.py`). This module is the debug companion: every
detection on every frame, enabled with `--raw-log`.

Turn it on when a person was missed and you need frame-level evidence of why.
Leave it off otherwise: a two-minute journey past three people produces
hundreds of rows nobody reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rescue_vision.config import Config
from rescue_vision.types import TrackState

log = logging.getLogger(__name__)

# v2: the rover no longer takes commands from this subsystem, so is_target,
# turn_command and drive_command are gone. Fields were removed, so the schema
# version is bumped rather than reused.
SCHEMA = "rescue.detection.v2"


def build_event(
    track: TrackState,
    frame_index: int,
    timestamp: float,
) -> dict[str, Any]:
    """One raw JSONL row.

    `confidence` is the live per-frame score, not the 1 Hz display sample --
    logs are for analysis and must stay precise.

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
        "confirmed": track.confirmed,
    }


class RawEventWriter:
    """Appends per-frame detection rows. Only constructed when --raw-log is on."""

    def __init__(self, jsonl_path: Path, cfg: Config) -> None:
        self._cfg = cfg
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
