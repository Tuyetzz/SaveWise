"""Frame capture behind one interface. PRD FR12, 6.10.

Everything downstream of this module is identical on Windows and the Pi.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from rescue_vision.config import Config

log = logging.getLogger(__name__)


class FrameSource(ABC):
    """A stream of BGR frames."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Next frame, or None when the stream is exhausted."""

    @property
    @abstractmethod
    def width(self) -> int: ...

    @property
    @abstractmethod
    def height(self) -> int: ...

    def close(self) -> None:
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VideoFileSource(FrameSource):
    """Recorded clip. The Windows dev path."""

    def __init__(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"video file not found: {path}")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video file: {path}")
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def close(self) -> None:
        self._cap.release()


class WebcamSource(FrameSource):
    """Laptop webcam. Dev convenience when no clip is available."""

    def __init__(self, index: int, cfg: Config) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def close(self) -> None:
        self._cap.release()


class PiCameraSource(FrameSource):
    """picamera2 capture on the Pi 5.

    UNTESTED: written from the spec, never executed against hardware.
    picamera2 cannot be imported on Windows, so the import is lazy.

    Auto-exposure indoors picks 20-30 ms and smears a person into the
    background while the rover turns. Fixed short exposure plus raised gain is
    a requirement, not polish (PRD 6.6).
    """

    def __init__(self, cfg: Config) -> None:
        from picamera2 import Picamera2  # lazy: Pi only

        self._cam = Picamera2()
        video_cfg = self._cam.create_video_configuration(
            main={"size": (cfg.frame_width, cfg.frame_height), "format": "RGB888"}
        )
        self._cam.configure(video_cfg)
        self._cam.start()
        self._cam.set_controls(
            {
                "AeEnable": False,
                "ExposureTime": cfg.exposure_time_us,
                "AnalogueGain": cfg.analogue_gain,
            }
        )
        self._w = cfg.frame_width
        self._h = cfg.frame_height

    def read(self) -> np.ndarray | None:
        return self._cam.capture_array()

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


def create_frame_source(spec: str, cfg: Config) -> FrameSource:
    """Build a source from a CLI spec.

    "picamera"    -> PiCameraSource (Pi only)
    "0", "1", ... -> WebcamSource at that index
    anything else -> VideoFileSource for that path
    """
    if spec == "picamera":
        return PiCameraSource(cfg)
    if spec.isdigit():
        return WebcamSource(int(spec), cfg)
    return VideoFileSource(spec)
