from rescue_vision.config import Config
from rescue_vision.selection import TargetSelector
from rescue_vision.types import BBox, TrackState

CFG = Config()


def track(track_id, distance_m=5.0, distance_valid=True, confirmed=True, bbox=None):
    return TrackState(
        track_id=track_id,
        bbox=bbox or BBox(300.0, 100.0, 360.0, 400.0),
        confidence=0.9,
        bearing_deg=0.0,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=confirmed,
    )


def test_no_tracks_selects_nothing():
    assert TargetSelector(CFG).select([], now=0.0) is None


def test_unconfirmed_tracks_are_never_selected():
    sel = TargetSelector(CFG)
    assert sel.select([track(1, confirmed=False)], now=0.0) is None


def test_nearest_valid_distance_wins():
    sel = TargetSelector(CFG)
    chosen = sel.select([track(1, distance_m=8.0), track(2, distance_m=3.0)], now=0.0)
    assert chosen.track_id == 2


def test_largest_bbox_wins_when_no_distance_is_valid():
    small = BBox(300.0, 100.0, 320.0, 200.0)
    large = BBox(100.0, 50.0, 300.0, 450.0)
    sel = TargetSelector(CFG)
    chosen = sel.select(
        [
            track(1, distance_valid=False, bbox=small),
            track(2, distance_valid=False, bbox=large),
        ],
        now=0.0,
    )
    assert chosen.track_id == 2


def test_a_track_with_valid_distance_beats_one_without():
    sel = TargetSelector(CFG)
    chosen = sel.select(
        [track(1, distance_valid=False), track(2, distance_m=20.0)], now=0.0
    )
    assert chosen.track_id == 2


def test_lowest_track_id_breaks_ties_for_determinism():
    sel = TargetSelector(CFG)
    chosen = sel.select([track(5, distance_m=4.0), track(2, distance_m=4.0)], now=0.0)
    assert chosen.track_id == 2


def test_target_is_sticky_within_target_hold():
    """Stops the rover flip-flopping between two people."""
    sel = TargetSelector(CFG)
    assert sel.select([track(1, distance_m=5.0)], now=0.0).track_id == 1
    chosen = sel.select([track(1, distance_m=5.0), track(2, distance_m=1.0)], now=0.5)
    assert chosen.track_id == 1


def test_target_switches_after_target_hold_expires():
    sel = TargetSelector(CFG)
    sel.select([track(1, distance_m=5.0)], now=0.0)
    chosen = sel.select([track(1, distance_m=5.0), track(2, distance_m=1.0)], now=1.5)
    assert chosen.track_id == 2


def test_stickiness_does_not_resurrect_a_vanished_target():
    sel = TargetSelector(CFG)
    sel.select([track(1)], now=0.0)
    chosen = sel.select([track(2)], now=0.1)
    assert chosen.track_id == 2
