"""Pick exactly one target per frame. PRD 6.8.

Pure module: the clock is passed in, never read.
"""

from __future__ import annotations

from rescue_vision.config import Config
from rescue_vision.types import TrackState


class TargetSelector:
    """Applies the PRD 6.8 priority rule with TARGET_HOLD stickiness."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._current_id: int | None = None
        self._selected_at: float = 0.0

    def select(self, tracks: list[TrackState], now: float) -> TrackState | None:
        candidates = [t for t in tracks if t.confirmed]
        if not candidates:
            self._current_id = None
            return None

        by_id = {t.track_id: t for t in candidates}

        # Stickiness: hold the previous target briefly even if another now
        # scores higher, so the rover does not flip-flop between two people.
        if self._current_id in by_id:
            if now - self._selected_at < self._cfg.target_hold:
                return by_id[self._current_id]

        best = min(candidates, key=self._priority)
        if best.track_id != self._current_id:
            self._current_id = best.track_id
            self._selected_at = now
        return best

    @staticmethod
    def _priority(t: TrackState) -> tuple[int, float, int]:
        """Sort key, lower is better.

        Tracks with a trustworthy distance form tier 0 and rank by distance;
        the rest form tier 1 and rank by descending bbox area, a reasonable
        proxy for nearest. Track ID breaks ties so the choice is deterministic.
        """
        if t.distance_valid:
            return (0, t.distance_m, t.track_id)
        return (1, -t.bbox.area, t.track_id)
