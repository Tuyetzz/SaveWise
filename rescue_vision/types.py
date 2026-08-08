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
    """Smoothed, accumulated state for one track ID. Mutable by design.

    Two confidence fields, deliberately:

    - `confidence` is the live per-frame score. It goes into the JSONL, where
      precision matters more than legibility.
    - `display_confidence` is sampled once per second and held between samples.
      It goes on the annotated frame, where a value flickering at 10 Hz is
      unreadable. See Amendment A in the implementation plan.
    """

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
    display_confidence: float = 0.0
    confidence_sampled_at: float | None = None


@dataclass(frozen=True)
class Command:
    """Normalized rover command, both components in [-1, +1]."""

    turn: float
    drive: float
