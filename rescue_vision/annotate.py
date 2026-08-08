"""Annotated frame drawing for the demo feed. PRD FR4.

Measured cost: ~0.3 ms per frame on a desktop CPU for three boxes, so roughly
1-2 ms on a Pi 5. Against a ~100 ms inference budget this is free -- the box
and the confidence score are never what slows the pipeline down.
"""

from __future__ import annotations

import cv2
import numpy as np

from rescue_vision.types import Command, TrackState

_TARGET_COLOUR = (0, 255, 0)  # green: the selected target
_CONFIRMED_COLOUR = (0, 200, 255)  # amber: confirmed human, not the target
_TENTATIVE_COLOUR = (128, 128, 128)  # grey: seen, not yet confirmed
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def format_label(t: TrackState) -> str:
    """One-line label: identity, class, confidence, bearing, distance.

    The confidence shown is `display_confidence`, sampled once per second
    (Amendment A). Whether the track has cleared the confirm cascade is carried
    by the box COLOUR rather than the number.
    """
    if t.distance_valid:
        dist = f"{t.distance_m:.1f}m"
    else:
        dist = f"dist?{t.invalid_reason or 'invalid'}"
    return (
        f"#{t.track_id} person {t.display_confidence:.2f} "
        f"{t.bearing_deg:+.1f}deg {dist}"
    )


def draw_overlay(
    frame: np.ndarray,
    tracks: list[TrackState],
    target_id: int | None,
    command: Command,
    fps: float,
) -> np.ndarray:
    """Draw boxes, track IDs, confidence, bearing and distance. Returns a new image."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Centreline: the reference that bearing is measured against.
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

        label = format_label(t)
        # Filled strip behind the text so it stays readable over a busy scene.
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        ly = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 4, ly), colour, -1)
        cv2.putText(
            out, label, (x1 + 2, ly - 3), _FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )

    status = (
        f"turn={command.turn:+.2f} drive={command.drive:+.2f} "
        f"fps={fps:.1f} tracks={len(tracks)}"
    )
    cv2.putText(out, status, (8, h - 10), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
