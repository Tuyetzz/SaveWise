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
    """The box label: confidence and nothing else.

    Track ID, bearing, and distance are deliberately NOT drawn -- they crowd
    the frame and none of them is what a viewer needs to read at a glance.
    All three are still written to the JSONL every frame, including
    `invalid_reason` when a distance estimate was rejected, so nothing is lost
    for analysis; this is a display choice only.

    The value is `display_confidence`, sampled once per second (Amendment A),
    because a number redrawn at 10 Hz is unreadable. Whether the track has
    cleared the confirm cascade is carried by the box COLOUR.
    """
    return f"confidence_score = {t.display_confidence:.2f}"


def place_label(
    x: int,
    y: int,
    tw: int,
    th: int,
    frame_w: int,
    frame_h: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """Find a spot for a label strip that is on-screen and not already taken.

    Two people standing close together is the normal multi-person case, and
    naive placement overdraws one label with the other -- the confidence score
    of the occluded track becomes unreadable. Equally, a person at the frame
    edge is where a sweep first finds them, and their label would otherwise run
    off-screen.

    Returns the (x, baseline_y) for a strip of size tw x th. The strip occupies
    (x, y - th) to (x + tw, y).
    """
    pad = 4
    x = max(0, min(x, frame_w - tw - pad))
    y = max(th + pad, y)

    # Step the label downward until it clears everything already drawn. Bounded
    # so a dense crowd degrades to overlapping labels rather than an infinite
    # loop or labels marching off the bottom of the frame.
    step = th + pad + 2
    for _ in range(8):
        rect = (x, y - th - pad, x + tw + pad, y)
        if not any(_overlaps(rect, o) for o in occupied):
            return x, y
        if y + step > frame_h:
            break
        y += step
    return x, y


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


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

    # Draw the target last so its label wins any remaining contest for space.
    ordered = sorted(tracks, key=lambda t: t.track_id == target_id)
    occupied: list[tuple[int, int, int, int]] = []

    for t in ordered:
        if t.track_id == target_id:
            colour = _TARGET_COLOUR
        elif t.confirmed:
            colour = _CONFIRMED_COLOUR
        else:
            colour = _TENTATIVE_COLOUR

        x1, y1, x2, y2 = t.bbox.as_xyxy_ints()
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = format_label(t)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        lx, ly = place_label(x1, y1 - 4, tw, th, w, h, occupied)
        occupied.append((lx, ly - th - 4, lx + tw + 4, ly))

        # Filled strip behind the text so it stays readable over a busy scene.
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 4, ly), colour, -1)
        cv2.putText(
            out, label, (lx + 2, ly - 3), _FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        # Tie the label back to its box when displacement moved it away.
        if abs(lx - x1) > 2 or abs(ly - (y1 - 4)) > 2:
            cv2.line(out, (lx, ly), (x1, y1), colour, 1)

    status = (
        f"turn={command.turn:+.2f} drive={command.drive:+.2f} "
        f"fps={fps:.1f} tracks={len(tracks)}"
    )
    cv2.putText(out, status, (8, h - 10), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
