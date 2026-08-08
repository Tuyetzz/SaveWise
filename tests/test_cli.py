import pytest

from rescue_vision.cli import build_parser, config_from_args


def test_source_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_select_the_console_rover():
    """Motors must never engage by accident on a dev machine."""
    args = build_parser().parse_args(["--source", "clip.mp4"])
    assert args.rover == "console"


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


def test_kp_and_deadband_are_tunable_from_the_command_line():
    """These get tuned on the chassis, so they must not need a code edit."""
    args = build_parser().parse_args(
        ["--source", "0", "--kp", "0.05", "--deadband-deg", "8"]
    )
    cfg = config_from_args(args)
    assert cfg.kp == 0.05
    assert cfg.deadband_deg == 8.0


def test_no_save_frames_flag_disables_frame_output():
    args = build_parser().parse_args(["--source", "0", "--no-save-frames"])
    assert config_from_args(args).save_frames is False


def test_confirm_interval_is_tunable_as_the_main_fps_lever():
    args = build_parser().parse_args(["--source", "0", "--confirm-min-interval", "0.5"])
    assert config_from_args(args).confirm_min_interval == 0.5


def test_no_confirm_disables_the_confirm_tier():
    import math

    args = build_parser().parse_args(["--source", "0", "--no-confirm"])
    assert math.isinf(config_from_args(args).confirm_min_interval)


def test_scan_imgsz_is_tunable():
    args = build_parser().parse_args(["--source", "0", "--scan-imgsz", "320"])
    assert config_from_args(args).scan_imgsz == 320


def test_q_and_escape_quit_the_live_display():
    """A webcam stream never ends, so the operator needs a way out."""
    from rescue_vision.cli import should_quit

    assert should_quit(ord("q")) is True
    assert should_quit(ord("Q")) is True
    assert should_quit(27) is True


def test_other_keys_do_not_quit():
    from rescue_vision.cli import should_quit

    assert should_quit(255) is False  # waitKey's "nothing pressed", masked
    assert should_quit(ord("a")) is False


def test_defaults_are_left_untouched_when_no_overrides_are_given():
    from rescue_vision.config import Config

    args = build_parser().parse_args(["--source", "0"])
    assert config_from_args(args) == Config()
