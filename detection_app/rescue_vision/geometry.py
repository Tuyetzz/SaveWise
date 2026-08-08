"""Pinhole geometry: bbox -> bearing and coarse distance. PRD 6.7.

Pure module. No I/O, no clock, no cv2.
"""

from __future__ import annotations

import math

from rescue_vision.config import Config
from rescue_vision.types import BBox, Measurement


def focal_px(frame_width: int, hfov_deg: float) -> float:
    """Focal length in pixels implied by the frame width and horizontal FOV."""
    return frame_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def bearing_deg(cx: float, frame_width: int, hfov_deg: float) -> float:
    """Signed horizontal angle from the optical axis to a bbox centre.

    Negative = left of centre, positive = right (PRD Appendix B).

    Uses the tan form, not the linear approximation. It costs nothing and is
    correct at the frame edges, which is exactly where a person first appears
    during a sweep.
    """
    f_px = focal_px(frame_width, hfov_deg)
    return math.degrees(math.atan((cx - frame_width / 2.0) / f_px))


def distance_m(bbox_height_px: float, f_px: float, assumed_height_m: float) -> float:
    """Coarse pinhole distance from bbox height. +/-25-30% at best when valid."""
    if bbox_height_px <= 0:
        return math.inf
    return (assumed_height_m * f_px) / bbox_height_px


def distance_validity(
    bbox: BBox, frame_w: int, frame_h: int, dist_m: float, cfg: Config
) -> tuple[bool, str | None]:
    """Decide whether a distance estimate is trustworthy. PRD 6.7 table.

    Returns a reason string rather than a bare bool so the annotated frame and
    the log can say *why* an estimate was rejected. Debugging "the distance is
    wrong" is far quicker when the frame already says `not_upright`.
    """
    if bbox.y1 <= cfg.edge_margin_px:
        return False, "clipped_top"
    if bbox.y2 >= frame_h - cfg.edge_margin_px:
        return False, "clipped_bottom"
    if bbox.aspect_ratio < cfg.min_aspect_ratio:
        # Likely lying down or crouching -- the 1.7 m assumption is invalid.
        return False, "not_upright"
    if bbox.height < cfg.min_bbox_height_px:
        return False, "bbox_too_small"
    if not (cfg.distance_min_m <= dist_m <= cfg.distance_max_m):
        return False, "implausible_distance"
    return True, None


def measure(bbox: BBox, frame_w: int, frame_h: int, cfg: Config) -> Measurement:
    """Full geometry for one bbox."""
    f_px = focal_px(frame_w, cfg.hfov_deg)
    bearing = bearing_deg(bbox.cx, frame_w, cfg.hfov_deg)
    dist = distance_m(bbox.height, f_px, cfg.assumed_human_height_m)
    valid, reason = distance_validity(bbox, frame_w, frame_h, dist, cfg)
    return Measurement(
        bearing_deg=bearing,
        distance_m=dist,
        distance_valid=valid,
        invalid_reason=reason,
    )


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. Used to match confirm-pass boxes to tracks."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0
