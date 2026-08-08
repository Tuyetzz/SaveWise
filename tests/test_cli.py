import pytest

from rescue_vision.cli import build_parser, config_from_args, format_summary, should_quit
from rescue_vision.types import Sighting


def test_source_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_there_is_no_flag_that_can_move_the_rover():
    """The rover drives itself; this program only observes."""
    help_text = build_parser().format_help()
    for gone in ("--rover", "--kp", "--deadband-deg", "--min-turn", "--stby-pin"):
        assert gone not in help_text


def test_raw_log_is_off_by_default():
    args = build_parser().parse_args(["--source", "0"])
    assert args.raw_log is False


def test_raw_log_can_be_enabled():
    args = build_parser().parse_args(["--source", "0", "--raw-log"])
    assert args.raw_log is True


def test_model_paths_are_overridable_for_the_pi():
    args = build_parser().parse_args(
        [
            "--source",
            "picamera",
            "--scan-model",
            "yolo26n.onnx",
            "--confirm-model",
            "yolo26s.onnx",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.scan_model == "yolo26n.onnx"
    assert cfg.confirm_model == "yolo26s.onnx"


def test_confirm_interval_is_tunable_as_the_main_fps_lever():
    args = build_parser().parse_args(["--source", "0", "--confirm-min-interval", "0.5"])
    assert config_from_args(args).confirm_min_interval == 0.5


def test_no_confirm_disables_the_confirm_tier():
    import math

    args = build_parser().parse_args(["--source", "0", "--no-confirm"])
    assert math.isinf(config_from_args(args).confirm_min_interval)


def test_confidence_floors_are_tunable_for_a_noisy_camera():
    args = build_parser().parse_args(
        ["--source", "0", "--scan-conf", "0.15", "--confirm-conf", "0.35"]
    )
    cfg = config_from_args(args)
    assert cfg.scan_conf == 0.15
    assert cfg.confirm_conf == 0.35


def test_scan_imgsz_is_tunable():
    args = build_parser().parse_args(["--source", "0", "--scan-imgsz", "320"])
    assert config_from_args(args).scan_imgsz == 320


def test_no_save_frames_flag_disables_frame_output():
    args = build_parser().parse_args(["--source", "0", "--no-save-frames"])
    assert config_from_args(args).save_frames is False


def test_q_and_escape_quit_the_live_display():
    """A webcam stream never ends, so the operator needs a way out."""
    assert should_quit(ord("q")) is True
    assert should_quit(ord("Q")) is True
    assert should_quit(27) is True


def test_other_keys_do_not_quit():
    assert should_quit(255) is False
    assert should_quit(ord("a")) is False


def test_defaults_are_left_untouched_when_no_overrides_are_given():
    from rescue_vision.config import Config

    args = build_parser().parse_args(["--source", "0"])
    assert config_from_args(args) == Config()


def _sighting(sighting_id=1, distance=3.2):
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
    )


def test_summary_reports_an_empty_journey_plainly():
    assert "no humans detected" in format_summary([])


def test_summary_lists_one_row_per_sighting():
    text = format_summary([_sighting(1), _sighting(2)])
    assert "2 human sighting(s)" in text
    assert "0.93" in text


def test_summary_handles_an_unknown_distance():
    """A prone person often has no trustworthy distance -- must not crash."""
    text = format_summary([_sighting(distance=None)])
    assert "--" in text
