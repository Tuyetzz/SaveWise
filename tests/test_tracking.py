import pytest

from rescue_vision.config import Config
from rescue_vision.geometry import bearing_deg
from rescue_vision.tracking import TrackStore, ema
from rescue_vision.types import BBox, RawDetection

CFG = Config()


def det(x1, y1, x2, y2, track_id, conf=0.9):
    return RawDetection(BBox(x1, y1, x2, y2), conf, track_id)


def test_ema_with_no_previous_value_returns_the_new_value():
    assert ema(None, 10.0, 0.4) == 10.0


def test_ema_blends_toward_the_new_value():
    assert ema(0.0, 10.0, 0.4) == pytest.approx(4.0)


def test_ema_repeated_converges_on_the_new_value():
    v = 0.0
    for _ in range(50):
        v = ema(v, 10.0, 0.4)
    assert v == pytest.approx(10.0, abs=1e-6)


def test_update_creates_a_track_for_a_new_id():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=7)], 640, 480, 0, now=0.0)
    tracks = store.tracks()
    assert len(tracks) == 1
    assert tracks[0].track_id == 7


def test_detections_without_a_track_id_are_ignored():
    """The tracker owns identity. An untracked box has nothing to accumulate onto."""
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=None)], 640, 480, 0, now=0.0)
    assert store.tracks() == []


def test_bearing_is_smoothed_rather_than_jumping():
    """Raw per-frame bearing is noisy enough to make the rover visibly twitch."""
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0, now=0.0)
    first = store.tracks()[0].bearing_deg
    store.update([det(600.0, 100.0, 660.0, 400.0, track_id=1)], 640, 480, 1, now=0.1)
    second = store.tracks()[0].bearing_deg
    raw_second = bearing_deg(630.0, 640, CFG.hfov_deg)
    assert first < second < raw_second


def test_a_track_starts_unconfirmed():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0, now=0.0)
    assert store.tracks()[0].confirmed is False
    assert store.confirmed_tracks() == []


def test_track_is_promoted_after_n_confirm_matching_confirm_passes():
    store = TrackStore(CFG)
    box = BBox(300.0, 100.0, 360.0, 400.0)
    store.update([RawDetection(box, 0.9, 1)], 640, 480, 0, now=0.0)

    matched = store.apply_confirmations([box])
    assert matched == {1}
    assert store.tracks()[0].confirm_count == 1
    assert store.tracks()[0].confirmed is False  # n_confirm is 2

    store.apply_confirmations([box])
    assert store.tracks()[0].confirmed is True
    assert [t.track_id for t in store.confirmed_tracks()] == [1]


def test_confirmation_does_not_match_a_distant_box():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0, now=0.0)
    matched = store.apply_confirmations([BBox(0.0, 0.0, 20.0, 20.0)])
    assert matched == set()
    assert store.tracks()[0].confirm_count == 0


def test_once_confirmed_a_track_stays_confirmed():
    """PRD 6.5: promoted tracks stay confirmed until the tracker drops them."""
    store = TrackStore(CFG)
    box = BBox(300.0, 100.0, 360.0, 400.0)
    store.update([RawDetection(box, 0.9, 1)], 640, 480, 0, now=0.0)
    store.apply_confirmations([box])
    store.apply_confirmations([box])
    assert store.tracks()[0].confirmed is True

    for i in range(1, 5):
        store.update([RawDetection(box, 0.9, 1)], 640, 480, i, now=float(i) * 0.1)
    assert store.tracks()[0].confirmed is True


def test_prune_drops_tracks_unseen_for_longer_than_max_age():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 0, now=0.0)
    store.prune(frame_index=CFG.track_max_age_frames + 1)
    assert store.tracks() == []


def test_prune_keeps_a_recently_seen_track():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, track_id=1)], 640, 480, 10, now=0.0)
    store.prune(frame_index=11)
    assert len(store.tracks()) == 1


# --- Amendment A: displayed confidence is sampled at 1 Hz ---


def test_display_confidence_is_set_on_the_first_frame():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.80)], 640, 480, 0, now=0.0)
    assert store.tracks()[0].display_confidence == pytest.approx(0.80)


def test_display_confidence_is_held_between_samples():
    """The whole point: a value redrawn at 10 Hz is unreadable."""
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.80)], 640, 480, 0, now=0.0)
    for i, t in enumerate([0.1, 0.3, 0.6, 0.9], start=1):
        store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.40)], 640, 480, i, now=t)
    assert store.tracks()[0].display_confidence == pytest.approx(0.80)


def test_display_confidence_refreshes_after_the_sample_interval():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.80)], 640, 480, 0, now=0.0)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.40)], 640, 480, 1, now=1.2)
    assert store.tracks()[0].display_confidence == pytest.approx(0.40)


def test_live_confidence_still_updates_every_frame():
    """Logs must stay precise even while the display is held."""
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.80)], 640, 480, 0, now=0.0)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.41)], 640, 480, 1, now=0.1)
    track = store.tracks()[0]
    assert track.confidence == pytest.approx(0.41)
    assert track.display_confidence == pytest.approx(0.80)


def test_each_track_samples_its_confidence_independently():
    store = TrackStore(CFG)
    store.update([det(300.0, 100.0, 360.0, 400.0, 1, conf=0.80)], 640, 480, 0, now=0.0)
    store.update(
        [
            det(300.0, 100.0, 360.0, 400.0, 1, conf=0.50),
            det(400.0, 100.0, 460.0, 400.0, 2, conf=0.70),
        ],
        640,
        480,
        1,
        now=0.2,
    )
    by_id = {t.track_id: t for t in store.tracks()}
    assert by_id[1].display_confidence == pytest.approx(0.80)  # held
    assert by_id[2].display_confidence == pytest.approx(0.70)  # first sample
