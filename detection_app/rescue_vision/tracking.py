"""Per-track accumulation: EMA smoothing and confirm-pass promotion. PRD 6.5, 6.7.

Pure module: no I/O, and the clock is passed in rather than read.
"""

from __future__ import annotations

from rescue_vision.config import Config
from rescue_vision.geometry import iou, measure
from rescue_vision.types import BBox, RawDetection, TrackState


def ema(previous: float | None, new: float, alpha: float) -> float:
    """Exponential moving average. The first sample passes through unchanged."""
    if previous is None:
        return new
    return alpha * new + (1.0 - alpha) * previous


class TrackStore:
    """Holds smoothed state per track ID and decides when a track is human.

    Track IDs come from ByteTrack via the scan pass. Detections without a track
    ID are ignored: identity is the tracker's job, and there is nothing to
    accumulate state onto without it.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._tracks: dict[int, TrackState] = {}

    def update(
        self,
        detections: list[RawDetection],
        frame_w: int,
        frame_h: int,
        frame_index: int,
        now: float,
    ) -> None:
        """Fold this frame's scan detections into per-track smoothed state."""
        for d in detections:
            if d.track_id is None:
                continue
            m = measure(d.bbox, frame_w, frame_h, self._cfg)
            existing = self._tracks.get(d.track_id)

            if existing is None:
                self._tracks[d.track_id] = TrackState(
                    track_id=d.track_id,
                    bbox=d.bbox,
                    confidence=d.confidence,
                    bearing_deg=m.bearing_deg,
                    distance_m=m.distance_m,
                    distance_valid=m.distance_valid,
                    invalid_reason=m.invalid_reason,
                    last_seen_frame=frame_index,
                    display_confidence=d.confidence,
                    confidence_sampled_at=now,
                )
                continue

            a = self._cfg.ema_alpha
            existing.bearing_deg = ema(existing.bearing_deg, m.bearing_deg, a)
            # Only smooth distance across frames where it is trustworthy;
            # blending in a rejected estimate would poison the average.
            if m.distance_valid:
                base = existing.distance_m if existing.distance_valid else None
                existing.distance_m = ema(base, m.distance_m, a)
            else:
                existing.distance_m = m.distance_m
            existing.distance_valid = m.distance_valid
            existing.invalid_reason = m.invalid_reason
            existing.bbox = d.bbox
            existing.confidence = d.confidence
            existing.last_seen_frame = frame_index
            self._sample_display_confidence(existing, now)

    def _sample_display_confidence(self, track: TrackState, now: float) -> None:
        """Refresh the on-screen confidence at most once per interval.

        The live value keeps updating every frame for the log. Only what a
        human reads off the annotated frame is held steady.
        """
        last = track.confidence_sampled_at
        if last is None or (now - last) >= self._cfg.confidence_sample_interval:
            track.display_confidence = track.confidence
            track.confidence_sampled_at = now

    def apply_confirmations(self, confirm_boxes: list[BBox]) -> set[int]:
        """Match confirm-pass boxes to tracks by IoU and promote on N_CONFIRM.

        The confirm model returns its own boxes with no track IDs, so
        association is by overlap. Each track is credited at most once per pass.
        """
        matched: set[int] = set()
        for track in self._tracks.values():
            best = max((iou(track.bbox, cb) for cb in confirm_boxes), default=0.0)
            if best >= self._cfg.confirm_iou_match:
                track.confirm_count += 1
                if track.confirm_count >= self._cfg.n_confirm:
                    track.confirmed = True
                matched.add(track.track_id)
        return matched

    def prune(self, frame_index: int) -> None:
        """Drop tracks the tracker has stopped reporting."""
        max_age = self._cfg.track_max_age_frames
        self._tracks = {
            tid: t
            for tid, t in self._tracks.items()
            if frame_index - t.last_seen_frame <= max_age
        }

    def tracks(self) -> list[TrackState]:
        """Every retained track, including ones not seen this frame.

        Prefer `visible_tracks()` for anything that steers the rover or claims
        a person is present.
        """
        return list(self._tracks.values())

    def visible_tracks(self, frame_index: int) -> list[TrackState]:
        """Only tracks the detector reported on this very frame.

        Retention and visibility are different questions. A track is retained
        past its last sighting so ByteTrack can re-associate the same ID and
        keep its smoothed history -- but a retained track is not evidence that
        a person is still there. Steering or logging off a stale track means
        commanding a turn toward someone who has already left the frame, and
        writing detection rows for a bbox that holds nobody.
        """
        return [t for t in self._tracks.values() if t.last_seen_frame == frame_index]

    def confirmed_tracks(self) -> list[TrackState]:
        return [t for t in self._tracks.values() if t.confirmed]
