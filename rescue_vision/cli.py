"""Command-line entry point. Wires the two swappable ends of the pipe."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from rescue_vision.config import Config
from rescue_vision.events import EventWriter
from rescue_vision.frame_source import create_frame_source
from rescue_vision.pipeline import Pipeline
from rescue_vision.rover import ConsoleRover, GpioZeroRover

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rescue_vision",
        description="Person detection, bearing, and turn command for a rescue rover.",
    )
    p.add_argument(
        "--source",
        required=True,
        help='Frame source: a video path, a webcam index ("0"), or "picamera".',
    )
    p.add_argument(
        "--rover",
        choices=["console", "gpiozero"],
        default="console",
        help="Motor backend. Defaults to console so motors never move by accident.",
    )
    p.add_argument("--scan-model", default=None, help="Override the scan model path.")
    p.add_argument(
        "--confirm-model", default=None, help="Override the confirm model path."
    )
    p.add_argument("--output-dir", default="output", help="Where JSONL and frames go.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--display",
        action="store_true",
        help="Show a preview window. NEVER use this on the headless Pi.",
    )
    p.add_argument(
        "--mjpeg-port",
        type=int,
        default=None,
        help="Serve an annotated MJPEG preview on this port. Use this instead "
        "of --display on the headless Pi.",
    )
    p.add_argument("--save-video", default=None, help="Write an annotated MP4 here.")
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--kp", type=float, default=None)
    p.add_argument("--deadband-deg", type=float, default=None)
    p.add_argument("--min-turn", type=float, default=None)
    p.add_argument(
        "--stby-pin",
        type=int,
        default=None,
        help="TB6612FNG standby pin. Omit for an L298N.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    if args.scan_model:
        overrides["scan_model"] = args.scan_model
    if args.confirm_model:
        overrides["confirm_model"] = args.confirm_model
    if args.kp is not None:
        overrides["kp"] = args.kp
    if args.deadband_deg is not None:
        overrides["deadband_deg"] = args.deadband_deg
    if args.min_turn is not None:
        overrides["min_turn"] = args.min_turn
    if args.no_save_frames:
        overrides["save_frames"] = False
    return dataclasses.replace(Config(), **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = config_from_args(args)

    out = Path(args.output_dir)
    writer = EventWriter(out / "events.jsonl", out / "detections", cfg)

    if args.rover == "gpiozero":
        rover = GpioZeroRover(cfg, stby_pin=args.stby_pin)
    else:
        rover = ConsoleRover(cfg)

    source = create_frame_source(args.source, cfg)

    # Imported here so --help stays fast and doesn't pay the ultralytics cost.
    from rescue_vision.detector import CascadeDetector

    detector = CascadeDetector(cfg)
    pipeline = Pipeline(detector, rover, writer, cfg)

    preview = None
    if args.mjpeg_port is not None:
        from rescue_vision.preview import MjpegServer

        preview = MjpegServer(args.mjpeg_port)
        preview.start()
        log.info("watch the annotated feed at %s", preview.url)

    video_writer = None
    if args.save_video:
        import cv2

        video_writer = cv2.VideoWriter(
            args.save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            15.0,
            (source.width, source.height),
        )

    def on_frame(result) -> None:
        if result.annotated is None:
            return
        if video_writer is not None:
            video_writer.write(result.annotated)
        if preview is not None:
            preview.publish(result.annotated)
        if args.display:
            import cv2

            cv2.imshow("rescue_vision", result.annotated)
            cv2.waitKey(1)

    try:
        count = pipeline.run(source, max_frames=args.max_frames, on_frame=on_frame)
        log.info("processed %d frames", count)
        return 0
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130
    finally:
        # Order matters: motors first, always.
        rover.close()
        source.close()
        writer.close()
        if preview is not None:
            preview.stop()
        if video_writer is not None:
            video_writer.release()
        if args.display:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
