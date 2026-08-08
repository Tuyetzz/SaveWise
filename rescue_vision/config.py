"""All tunables in one place. Values copied verbatim from PRD Appendix A.

These are starting values, not tuned ones. Tune on the real chassis -- carpet
and hard floor behave differently enough to matter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Camera (OV5647 / Camera Module v1) ---
    # CHANGE hfov_deg if using Camera Module v2/v3, or every bearing and
    # distance below silently goes wrong.
    hfov_deg: float = 53.5
    vfov_deg: float = 41.4
    frame_width: int = 640
    frame_height: int = 480
    exposure_time_us: int = 4000  # tune in the actual demo lighting
    analogue_gain: float = 4.0

    # --- Detection ---
    scan_model: str = "yolo26n.pt"  # "yolo26n.onnx" on the Pi
    confirm_model: str = "yolo26s.pt"  # "yolo26s.onnx" on the Pi
    scan_imgsz: int = 480
    confirm_imgsz: int = 640
    scan_conf: float = 0.25
    confirm_conf: float = 0.45
    person_class_id: int = 0
    confirm_min_interval: float = 0.15
    n_confirm: int = 2
    confirm_iou_match: float = 0.5  # IoU to associate a confirm box to a track

    # --- Tracking ---
    tracker_cfg: str = "bytetrack.yaml"
    ema_alpha: float = 0.4
    track_max_age_frames: int = 30  # drop tracks unseen this long

    # --- Distance ---
    assumed_human_height_m: float = 1.7
    min_bbox_height_px: float = 40.0
    min_aspect_ratio: float = 1.2
    distance_min_m: float = 0.3
    distance_max_m: float = 30.0
    edge_margin_px: float = 2.0

    # --- Output ---
    # The rover drives itself (straight lines, left turn on blockage from its
    # own front sensor), so this subsystem observes and logs only. There are no
    # control, motor, or safety constants: nothing here can move anything.
    save_frames: bool = True  # one best frame per sighting
    jpeg_quality: int = 90
    confidence_sample_interval: float = 1.0  # display only; logs stay per-frame
