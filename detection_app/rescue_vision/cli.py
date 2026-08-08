"""Command-line entry point.

The rover drives itself. This program only watches and records: there is no
flag here that can move anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from rescue_vision.config import Config
from rescue_vision.frame_source import create_frame_source
from rescue_vision.pipeline import Pipeline
from rescue_vision.sightings import SightingRecorder
from rescue_vision.types import Sighting

log = logging.getLogger(__name__)


class QuitRequested(Exception):
    """Raised from the display callback when the operator presses q or Esc."""


def should_quit(key: int) -> bool:
    """True for 'q', 'Q', or Esc."""
    return key in (ord("q"), ord("Q"), 27)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rescue_vision",
        description="Logs humans detected during a rover journey, with confidence.",
    )
    p.add_argument(
        "--source",
        required=True,
        help='Frame source: a video path, a webcam index ("0"), or "picamera".',
    )
    p.add_argument("--scan-model", default=None, help="Override the scan model path.")
    p.add_argument(
        "--confirm-model", default=None, help="Override the confirm model path."
    )
    p.add_argument("--output-dir", default="output", help="Where the journey log goes.")
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
    p.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Skip the one best frame saved per sighting.",
    )
    p.add_argument(
        "--raw-log",
        action="store_true",
        help="Also write per-frame detections to events.jsonl. Off by default: "
        "the journey log is the deliverable, this is for debugging a miss.",
    )
    p.add_argument(
        "--confirm-min-interval",
        type=float,
        default=None,
        help="Seconds between confirm passes. THE main FPS lever: the confirm "
        "tier costs ~60%% of the frame budget, so raising this to 0.5 roughly "
        "doubles throughput. Raise this before shrinking --scan-imgsz.",
    )
    p.add_argument(
        "--scan-imgsz",
        type=int,
        default=None,
        help="Scan-pass input size. Shrinking this costs real detections of "
        "distant and prone people -- try --confirm-min-interval first.",
    )
    p.add_argument(
        "--scan-conf",
        type=float,
        default=None,
        help="Scan-pass confidence floor (default 0.25). LOWER this to ~0.15 "
        "on a noisy camera module: measured recall drops ~23%% under Pi-like "
        "noise, and the confirm tier plus N_CONFIRM still guard precision.",
    )
    p.add_argument(
        "--confirm-conf",
        type=float,
        default=None,
        help="Confirm-pass confidence floor (default 0.45).",
    )
    p.add_argument(
        "--no-confirm",
        action="store_true",
        help="Disable the confirm tier entirely, leaning on N_CONFIRM over the "
        "scan model. Last-resort FPS measure.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    if args.scan_model:
        overrides["scan_model"] = args.scan_model
    if args.confirm_model:
        overrides["confirm_model"] = args.confirm_model
    if args.no_save_frames:
        overrides["save_frames"] = False
    if args.confirm_min_interval is not None:
        overrides["confirm_min_interval"] = args.confirm_min_interval
    if args.scan_imgsz is not None:
        overrides["scan_imgsz"] = args.scan_imgsz
    if args.scan_conf is not None:
        overrides["scan_conf"] = args.scan_conf
    if args.confirm_conf is not None:
        overrides["confirm_conf"] = args.confirm_conf
    if args.no_confirm:
        # An interval no run will ever reach, so the escalation check always
        # declines. Keeps one code path instead of a special case.
        overrides["confirm_min_interval"] = float("inf")
    return dataclasses.replace(Config(), **overrides)


def format_summary(sightings: list[Sighting]) -> str:
    """The end-of-journey report -- what you would actually show a judge."""
    if not sightings:
        return "\nJOURNEY COMPLETE - no humans detected.\n"

    lines = [
        "",
        f"JOURNEY COMPLETE - {len(sightings)} human sighting(s)",
        "",
        f"  {'#':>3}  {'seen at':>9}  {'for':>7}  {'peak conf':>9}  "
        f"{'bearing':>8}  {'nearest':>8}",
        f"  {'-' * 3}  {'-' * 9}  {'-' * 7}  {'-' * 9}  {'-' * 8}  {'-' * 8}",
    ]
    for s in sightings:
        dist = (
            f"{s.closest_distance_m:.1f}m" if s.closest_distance_m is not None else "  --"
        )
        lines.append(
            f"  {s.sighting_id:>3}  {s.first_seen_s:>8.1f}s  {s.duration_s:>6.1f}s  "
            f"{s.peak_confidence:>9.2f}  {s.bearing_at_peak_deg:>+7.1f}d  {dist:>8}"
        )
    lines += [
        "",
        "  Times are seconds from journey start; bearing is degrees off the",
        "  rover's heading at that moment (negative = left). Without odometry",
        "  the log records when a person was seen, not where they were.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = config_from_args(args)
    out = Path(args.output_dir)

    source = create_frame_source(args.source, cfg)

    # Imported here so --help stays fast and doesn't pay the ultralytics cost.
    from rescue_vision.detector import CascadeDetector

    detector = CascadeDetector(cfg)
    recorder = SightingRecorder(cfg, out)

    raw_writer = None
    if args.raw_log:
        from rescue_vision.events import RawEventWriter

        raw_writer = RawEventWriter(out / "events.jsonl", cfg)

    pipeline = Pipeline(detector, recorder, cfg, raw_writer)

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

            cv2.imshow("rescue_vision  [q or Esc to quit]", result.annotated)
            if should_quit(cv2.waitKey(1) & 0xFF):
                raise QuitRequested

    try:
        count = pipeline.run(source, max_frames=args.max_frames, on_frame=on_frame)
        log.info("processed %d frames", count)
        return 0
    except QuitRequested:
        log.info("quit requested")
        return 0
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130
    finally:
        # pipeline.run already closed the recorder in its own finally, so the
        # summary below reflects every sighting including any still open when
        # the journey ended.
        print(format_summary(recorder.summary()))

        from rescue_vision.report import build_report

        report = build_report(recorder.summary(), out, recorder.journey_duration_s)
        print(f"  Report: {report.resolve()}\n")

        source.close()
        if raw_writer is not None:
            raw_writer.close()
        if preview is not None:
            preview.stop()
        if video_writer is not None:
            video_writer.release()
        if args.display:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
