"""Two-tier detection cascade. PRD 6.5.

Tier 1 (scan): nano model, small input, every frame, feeds the tracker.
Tier 2 (confirm): small model, larger input, only when the scan pass found a
candidate and the rate limit allows.

The pipeline depends on the Detector Protocol, not on CascadeDetector, so the
whole loop can be tested with no model and no camera.

NMS-free caveat: YOLO26 is end-to-end, so the usual iou= NMS threshold has no
effect. Tune recall and precision with conf= only. Don't waste demo-day time
turning a knob that isn't wired to anything.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from rescue_vision.config import Config
from rescue_vision.types import BBox, RawDetection

log = logging.getLogger(__name__)


class Detector(Protocol):
    """What the pipeline needs from a detector."""

    def scan(self, frame: np.ndarray) -> list[RawDetection]: ...

    def confirm(self, frame: np.ndarray) -> list[BBox]: ...

    def should_confirm(self, has_candidates: bool, now: float) -> bool: ...


class _RateLimitedConfirm:
    """Shared escalation policy: confirm only on candidates, and not too often.

    The clock only advances when a confirm is actually granted, so a stream of
    declined requests can never starve the next one.
    """

    def __init__(self, confirm_min_interval: float) -> None:
        self._interval = confirm_min_interval
        self._last_confirm: float | None = None

    def should_confirm(self, has_candidates: bool, now: float) -> bool:
        if not has_candidates:
            return False
        if (
            self._last_confirm is not None
            and (now - self._last_confirm) < self._interval
        ):
            return False
        self._last_confirm = now
        return True


class CascadeDetector(_RateLimitedConfirm):
    """Ultralytics-backed cascade. Loads both models up front."""

    def __init__(self, cfg: Config) -> None:
        from ultralytics import YOLO  # heavy import, kept out of module scope

        super().__init__(cfg.confirm_min_interval)
        self._cfg = cfg
        log.info("loading scan model %s", cfg.scan_model)
        self._scan_model = YOLO(cfg.scan_model)
        log.info("loading confirm model %s", cfg.confirm_model)
        self._confirm_model = YOLO(cfg.confirm_model)

    def scan(self, frame: np.ndarray) -> list[RawDetection]:
        """Every-frame pass. Runs through the tracker so IDs stay stable."""
        results = self._scan_model.track(
            frame,
            persist=True,
            tracker=self._cfg.tracker_cfg,
            imgsz=self._cfg.scan_imgsz,
            conf=self._cfg.scan_conf,
            classes=[self._cfg.person_class_id],
            verbose=False,
        )
        return _to_detections(results, with_ids=True)

    def confirm(self, frame: np.ndarray) -> list[BBox]:
        """Candidate-only pass. These boxes carry no track IDs; the caller
        associates them with tracks by IoU."""
        results = self._confirm_model.predict(
            frame,
            imgsz=self._cfg.confirm_imgsz,
            conf=self._cfg.confirm_conf,
            classes=[self._cfg.person_class_id],
            verbose=False,
        )
        return [d.bbox for d in _to_detections(results, with_ids=False)]


def _to_detections(results, with_ids: bool) -> list[RawDetection]:
    """Convert an Ultralytics Results list into our own types."""
    out: list[RawDetection] = []
    if not results:
        return out
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    if with_ids and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int).tolist()
    else:
        ids = [None] * len(xyxy)

    for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
        out.append(
            RawDetection(
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                track_id=tid,
            )
        )
    return out


class ScriptedDetector(_RateLimitedConfirm):
    """Test double: replays a fixed list of per-frame detections.

    Lets the full pipeline be exercised with no model, no camera and no
    network -- the only way the smoke test can run anywhere.
    """

    def __init__(
        self,
        script: list[list[RawDetection]],
        confirm_all: bool = True,
        confirm_min_interval: float = 0.0,
    ) -> None:
        super().__init__(confirm_min_interval)
        self._script = script
        self._index = 0
        self._confirm_all = confirm_all
        self._last_scan: list[RawDetection] = []

    def scan(self, frame: np.ndarray) -> list[RawDetection]:
        if self._index >= len(self._script):
            self._last_scan = []
            return []
        self._last_scan = self._script[self._index]
        self._index += 1
        return self._last_scan

    def confirm(self, frame: np.ndarray) -> list[BBox]:
        if not self._confirm_all:
            return []
        return [d.bbox for d in self._last_scan]
