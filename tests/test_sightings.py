import json

import numpy as np
import pytest

from rescue_vision.config import Config
from rescue_vision.sightings import SightingRecorder
from rescue_vision.types import BBox, TrackState

CFG = Config()
FRAME = np.zeros((48, 64, 3), np.uint8)


def track(track_id=1, conf=0.9, bearing=0.0, distance_m=4.0, distance_valid=True):
    return TrackState(
        track_id=track_id,
        bbox=BBox(10.0, 5.0, 30.0, 45.0),
        confidence=conf,
        bearing_deg=bearing,
        distance_m=distance_m,
        distance_valid=distance_valid,
        confirmed=True,
    )


def rec(tmp_path):
    return SightingRecorder(CFG, tmp_path)


def rows(tmp_path):
    path = tmp_path / "sightings.jsonl"
    if not path.exists():
        return []
    text = path.read_text().strip()
    return [json.loads(line) for line in text.splitlines()] if text else []


def test_one_person_over_many_frames_is_a_single_sighting(tmp_path):
    """The whole point of the pivot: 50 frames of one person is 1 row."""
    r = rec(tmp_path)
    for i in range(50):
        r.observe([track()], FRAME, now=float(i) * 0.1)
        r.finalise_absent({1}, now=float(i) * 0.1)
    r.close()
    assert len(rows(tmp_path)) == 1
    assert rows(tmp_path)[0]["frames_seen"] == 50


def test_nothing_is_written_while_the_person_is_still_visible(tmp_path):
    r = rec(tmp_path)
    for i in range(5):
        r.observe([track()], FRAME, now=float(i))
        r.finalise_absent({1}, now=float(i))
    assert rows(tmp_path) == []


def test_sighting_is_written_once_the_track_has_been_gone_for_the_grace_period(
    tmp_path,
):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.finalise_absent({1}, now=0.0)
    r.finalise_absent(set(), now=1.0)  # inside the 1.5s grace -- still open
    assert rows(tmp_path) == []
    r.finalise_absent(set(), now=2.0)  # past it -- now closed
    assert len(rows(tmp_path)) == 1


def test_close_finalises_a_sighting_still_open_at_journey_end(tmp_path):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.close()
    assert len(rows(tmp_path)) == 1


def test_peak_confidence_is_the_maximum_not_the_last_value(tmp_path):
    r = rec(tmp_path)
    for i, c in enumerate([0.5, 0.95, 0.6]):
        r.observe([track(conf=c, bearing=float(i))], FRAME, now=float(i))
    r.close()
    row = rows(tmp_path)[0]
    assert row["peak_confidence"] == pytest.approx(0.95)
    assert row["peak_confidence_at_s"] == pytest.approx(1.0)
    assert row["bearing_at_peak_deg"] == pytest.approx(1.0)


def test_mean_confidence_is_averaged_over_frames_seen(tmp_path):
    r = rec(tmp_path)
    for c in (0.6, 0.8):
        r.observe([track(conf=c)], FRAME, now=0.0)
    r.close()
    assert rows(tmp_path)[0]["mean_confidence"] == pytest.approx(0.7)


def test_journey_time_starts_at_the_first_observed_frame(tmp_path):
    """Default clock is time.monotonic (seconds since boot), so an unset start
    would report a person 'seen at 19610s'. Latch on the first frame instead --
    which also excludes the seconds spent loading models."""
    r = SightingRecorder(CFG, tmp_path)
    r.observe([track()], FRAME, now=19610.4)
    r.observe([track()], FRAME, now=19612.4)
    r.close()
    row = rows(tmp_path)[0]
    assert row["first_seen_s"] == pytest.approx(0.0)
    assert row["last_seen_s"] == pytest.approx(2.0)


def test_journey_time_latches_even_on_a_frame_with_nobody_in_it(tmp_path):
    """The journey starts when the rover starts, not when it first sees someone."""
    r = SightingRecorder(CFG, tmp_path)
    r.observe([], FRAME, now=1000.0)
    r.observe([track()], FRAME, now=1003.0)
    r.close()
    assert rows(tmp_path)[0]["first_seen_s"] == pytest.approx(3.0)


def test_timings_are_relative_to_journey_start(tmp_path):
    """Wall-clock timestamps are useless in a journey report."""
    r = SightingRecorder(CFG, tmp_path, journey_start=1000.0)
    r.observe([track()], FRAME, now=1012.4)
    r.observe([track()], FRAME, now=1017.9)
    r.close()
    row = rows(tmp_path)[0]
    assert row["first_seen_s"] == pytest.approx(12.4)
    assert row["last_seen_s"] == pytest.approx(17.9)
    assert row["duration_s"] == pytest.approx(5.5)


def test_bearing_range_spans_the_whole_sighting(tmp_path):
    r = rec(tmp_path)
    for b in (-24.1, 0.0, 11.5):
        r.observe([track(bearing=b)], FRAME, now=0.0)
    r.close()
    assert rows(tmp_path)[0]["bearing_range_deg"] == [
        pytest.approx(-24.1),
        pytest.approx(11.5),
    ]


def test_closest_distance_uses_only_trustworthy_estimates(tmp_path):
    r = rec(tmp_path)
    r.observe([track(distance_m=5.0, distance_valid=True)], FRAME, now=0.0)
    r.observe([track(distance_m=0.4, distance_valid=False)], FRAME, now=1.0)
    r.observe([track(distance_m=3.5, distance_valid=True)], FRAME, now=2.0)
    r.close()
    row = rows(tmp_path)[0]
    assert row["closest_distance_m"] == pytest.approx(3.5)
    assert row["distance_valid_frames"] == 2


def test_closest_distance_is_null_when_no_frame_had_a_valid_estimate(tmp_path):
    """A prone person fails the aspect-ratio check -- and is the rescue target."""
    r = rec(tmp_path)
    r.observe([track(distance_valid=False)], FRAME, now=0.0)
    r.close()
    row = rows(tmp_path)[0]
    assert row["closest_distance_m"] is None
    assert row["distance_valid_frames"] == 0


def test_two_people_produce_two_sightings(tmp_path):
    r = rec(tmp_path)
    r.observe([track(1), track(2)], FRAME, now=0.0)
    r.close()
    assert len(rows(tmp_path)) == 2


def test_reacquisition_under_a_new_id_is_a_second_sighting(tmp_path):
    """Honest over-counting beats a merge that silently fuses two people."""
    r = rec(tmp_path)
    r.observe([track(1)], FRAME, now=0.0)
    r.finalise_absent(set(), now=1.0)
    r.observe([track(7)], FRAME, now=2.0)
    r.close()
    assert len(rows(tmp_path)) == 2
    assert [x["sighting_id"] for x in rows(tmp_path)] == [1, 2]


def test_unconfirmed_tracks_are_never_recorded(tmp_path):
    r = rec(tmp_path)
    t = track()
    t.confirmed = False
    r.observe([t], FRAME, now=0.0)
    r.close()
    assert rows(tmp_path) == []


def test_exactly_one_image_is_written_per_sighting(tmp_path):
    """Fixes the PRD disk risk: three people means three JPEGs, not hundreds."""
    r = rec(tmp_path)
    for i in range(40):
        r.observe([track(1), track(2)], FRAME, now=float(i))
    r.close()
    images = list((tmp_path / "sightings").glob("*.jpg"))
    assert len(images) == 2
    assert len(rows(tmp_path)) == 2


def test_the_saved_image_is_the_peak_confidence_frame(tmp_path):
    import cv2

    dim = np.full((48, 64, 3), 10, np.uint8)
    bright = np.full((48, 64, 3), 240, np.uint8)
    r = rec(tmp_path)
    r.observe([track(conf=0.4)], dim, now=0.0)
    r.observe([track(conf=0.95)], bright, now=1.0)
    r.observe([track(conf=0.5)], dim, now=2.0)
    r.close()
    saved = cv2.imread(str(tmp_path / rows(tmp_path)[0]["best_frame"]))
    assert saved.mean() > 200


def test_a_dropped_frame_does_not_split_one_person_into_two_sightings(tmp_path):
    """The measured defect: at camera-module noise levels the detector misses
    ~23% of frames, and closing on the first miss logged one person ~9 times."""
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.finalise_absent({1}, now=0.0)
    r.finalise_absent(set(), now=0.2)  # missed frame, inside the grace window
    r.observe([track()], FRAME, now=0.4)  # same ByteTrack id returns
    r.finalise_absent({1}, now=0.4)
    r.close()
    assert len(rows(tmp_path)) == 1


def test_a_long_absence_still_closes_the_sighting(tmp_path):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.finalise_absent(set(), now=5.0)
    assert len(rows(tmp_path)) == 1
    r.observe([track()], FRAME, now=6.0)
    r.close()
    assert len(rows(tmp_path)) == 2


def test_duration_reports_time_actually_seen_not_the_grace_period(tmp_path):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.observe([track()], FRAME, now=2.0)
    r.finalise_absent(set(), now=9.0)
    assert rows(tmp_path)[0]["duration_s"] == pytest.approx(2.0)


@pytest.mark.parametrize("drop_rate", [0.0, 0.1, 0.23, 0.4])
def test_one_person_is_one_sighting_at_every_realistic_dropout_rate(
    tmp_path, drop_rate
):
    """The demo's central claim, as a test."""
    import random

    rng = random.Random(0)
    r = rec(tmp_path)
    for i in range(60):
        now = i * 0.1
        seen = rng.random() >= drop_rate
        tracks = [track()] if seen else []
        r.observe(tracks, FRAME, now)
        r.finalise_absent({t.track_id for t in tracks}, now)
    r.close()
    assert len(rows(tmp_path)) == 1


def test_journey_duration_covers_the_whole_run(tmp_path):
    r = rec(tmp_path)
    r.observe([], FRAME, now=100.0)
    r.observe([track()], FRAME, now=104.5)
    r.close()
    assert r.journey_duration_s == pytest.approx(4.5)


def test_every_row_declares_the_schema(tmp_path):
    r = rec(tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.close()
    assert rows(tmp_path)[0]["schema"] == "rescue.sighting.v1"


def test_summary_returns_the_finalised_records(tmp_path):
    r = rec(tmp_path)
    r.observe([track(1)], FRAME, now=0.0)
    r.observe([track(2)], FRAME, now=0.0)
    r.close()
    assert len(r.summary()) == 2


def test_frame_saving_can_be_disabled(tmp_path):
    r = SightingRecorder(Config(save_frames=False), tmp_path)
    r.observe([track()], FRAME, now=0.0)
    r.close()
    assert rows(tmp_path)[0]["best_frame"] is None
    assert not (tmp_path / "sightings").exists()
