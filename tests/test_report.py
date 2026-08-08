from rescue_vision.report import build_report
from rescue_vision.types import Sighting


def sighting(sighting_id=1, distance=3.2, frame=None):
    return Sighting(
        sighting_id=sighting_id,
        track_id=sighting_id,
        first_seen_s=12.4,
        last_seen_s=17.9,
        frames_seen=48,
        peak_confidence=0.93,
        confidence_sum=48 * 0.81,
        bearing_at_peak_deg=-8.3,
        closest_distance_m=distance,
        best_frame_path=frame,
    )


def test_report_is_written_and_names_the_people_found(tmp_path):
    path = build_report([sighting(1), sighting(2)], tmp_path, duration_s=134.0)
    html = path.read_text(encoding="utf-8")
    assert path.name == "report.html"
    assert "P1" in html and "P2" in html


def test_an_empty_journey_still_produces_a_report(tmp_path):
    html = build_report([], tmp_path, duration_s=60.0).read_text(encoding="utf-8")
    assert "No humans" in html


def test_each_card_carries_its_person_colour(tmp_path):
    from rescue_vision.palette import colour_for

    html = build_report([sighting(1), sighting(2)], tmp_path, 10.0).read_text("utf-8")
    assert colour_for(1) in html
    assert colour_for(2) in html


def test_images_are_inlined_so_the_file_stands_alone(tmp_path):
    import cv2
    import numpy as np

    frames = tmp_path / "sightings"
    frames.mkdir()
    cv2.imwrite(str(frames / "s1.jpg"), np.zeros((20, 20, 3), np.uint8))
    html = build_report(
        [sighting(1, frame="sightings/s1.jpg")], tmp_path, 10.0
    ).read_text("utf-8")
    assert "data:image/jpeg;base64," in html


def test_a_missing_image_degrades_instead_of_crashing(tmp_path):
    html = build_report(
        [sighting(1, frame="sightings/gone.jpg")], tmp_path, 10.0
    ).read_text("utf-8")
    assert "P1" in html


def test_unknown_distance_is_not_reported_as_a_number(tmp_path):
    html = build_report([sighting(1, distance=None)], tmp_path, 10.0).read_text("utf-8")
    assert "not measurable" in html


def test_report_states_the_coverage_limitation(tmp_path):
    """Judges reward a team that knows what it did not search."""
    html = build_report([sighting(1)], tmp_path, 10.0).read_text("utf-8")
    assert "53.5" in html


def test_report_does_not_claim_to_know_where_anyone_was(tmp_path):
    """Without odometry, position would be fiction."""
    html = build_report([sighting(1)], tmp_path, 10.0).read_text("utf-8")
    assert "never <em>where</em>" in html


def test_bearing_is_described_as_a_direction_not_a_coordinate(tmp_path):
    html = build_report([sighting(1)], tmp_path, 10.0).read_text("utf-8")
    assert "to the left" in html
