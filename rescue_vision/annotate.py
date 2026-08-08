"""Annotated frame drawing for the demo feed. PRD FR4.

Measured cost: ~0.3 ms per frame on a desktop CPU for three boxes, so roughly
1-2 ms on a Pi 5. Against a ~100 ms inference budget this is free -- the box
and the confidence score are never what slows the pipeline down.
"""

from __future__ import annotations

import cv2
import numpy as np

from rescue_vision.palette import TENTATIVE_HEX, colour_for, hex_to_bgr
from rescue_vision.types import TrackState

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_OUTLINE = (20, 20, 20)


def format_label(t: TrackState, sighting_id: int | None = None) -> str:
    """`P2  confidence_score = 0.89`, or confidence alone before confirmation.

    ASCII only: cv2's Hershey fonts render anything else (including a middle
    dot) as '?'. The HTML report is free to use whatever it likes.

    The person number means identity is never carried by colour alone --
    necessary past three people, where the palette can no longer separate them,
    and for the ~1 in 12 men with a colour vision deficiency.

    Bearing and distance are deliberately not drawn. They crowd the frame and
    are not what a viewer reads at a glance; both still reach the sightings log.

    The value is `display_confidence`, sampled once per second, because a
    number redrawn at 10 Hz is unreadable.
    """
    conf = f"confidence_score = {t.display_confidence:.2f}"
    return f"P{sighting_id}  {conf}" if sighting_id is not None else conf


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
    fps: float,
    sightings_count: int = 0,
    sighting_ids: dict[int, int] | None = None,
) -> np.ndarray:
    """One coloured box per person. Returns a new image.

    Grey means "seen but not yet confirmed"; a colour means "counted, and in
    the report". Colour is keyed on sighting id so it stays with the same
    person for the whole sweep -- only stable because a sighting now survives
    brief detection dropouts.
    """
    sighting_ids = sighting_ids or {}
    out = frame.copy()
    h, w = out.shape[:2]

    # Centreline: the reference that bearing is measured against.
    cv2.line(out, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)

    # Confirmed people last, so their labels win any contest for space.
    ordered = sorted(tracks, key=lambda t: t.confirmed)
    occupied: list[tuple[int, int, int, int]] = []

    for t in ordered:
        sid = sighting_ids.get(t.track_id) if t.confirmed else None
        colour = hex_to_bgr(colour_for(sid) if sid is not None else TENTATIVE_HEX)

        x1, y1, x2, y2 = t.bbox.as_xyxy_ints()
        # Dark outline first. The palette's contrast is measured against a
        # controlled chart surface; ours is whatever the camera sees, so a blue
        # box on a blue door would otherwise vanish.
        cv2.rectangle(out, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), _OUTLINE, 4)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = format_label(t, sid)
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        lx, ly = place_label(x1, y1 - 4, tw, th, w, h, occupied)
        occupied.append((lx, ly - th - 4, lx + tw + 4, ly))

        # Filled strip behind the text so it stays readable over a busy scene.
        cv2.rectangle(out, (lx - 1, ly - th - 5), (lx + tw + 5, ly + 1), _OUTLINE, -1)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 4, ly), colour, -1)
        cv2.putText(
            out, label, (lx + 2, ly - 3), _FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        # Tie the label back to its box when displacement moved it away.
        if abs(lx - x1) > 2 or abs(ly - (y1 - 4)) > 2:
            cv2.line(out, (lx, ly), (x1, y1), colour, 1)

    people = sum(1 for t in tracks if t.confirmed)
    status = f"fps={fps:.1f} people={people} sightings={sightings_count}"
    cv2.putText(out, status, (8, h - 10), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
