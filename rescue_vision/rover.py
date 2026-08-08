"""Motor abstraction. PRD 6.9.

Safety contract for every backend:
  - motors default to stopped
  - stop() on KeyboardInterrupt, on any uncaught exception, and in a finally
  - a watchdog stops the drive if drive() goes quiet for WATCHDOG_TIMEOUT

The watchdog lives in the base class so both backends get it and neither can
forget it.

BENCH-TEST WITH THE WHEELS OFF THE GROUND FIRST. EVERY TIME.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable

from rescue_vision.config import Config
from rescue_vision.control import mix

log = logging.getLogger(__name__)

# Placeholder wiring -- EDIT TO MATCH YOUR CHASSIS (PRD 6.9, still an open
# question). If the rover turns away from people, swap the forward/backward
# pins here rather than negating KP.
MOTOR_PINS: dict[str, dict[str, int]] = {
    "left": {"forward": 17, "backward": 18, "enable": 12},
    "right": {"forward": 22, "backward": 23, "enable": 13},
}


class RoverController(ABC):
    """Base controller: mixing, watchdog, and a guaranteed stop."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._lock = threading.RLock()
        self._closed = False
        self._watchdog_fired = True  # start stopped
        self._timer: threading.Timer | None = None
        self._restart_timer()

    def drive(self, turn: float, forward: float) -> None:
        """Apply a normalized command and pet the watchdog."""
        left, right = mix(turn, forward)
        with self._lock:
            if self._closed:
                return
            self._watchdog_fired = False
            self._restart_timer()
            self._apply(left, right)

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._stop()

    def __enter__(self) -> "RoverController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _restart_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._cfg.watchdog_timeout, self._on_watchdog)
        self._timer.daemon = True
        self._timer.start()

    def _on_watchdog(self) -> None:
        with self._lock:
            if self._closed or self._watchdog_fired:
                return
            self._watchdog_fired = True
            log.warning(
                "watchdog: no drive() within %.2fs, stopping motors",
                self._cfg.watchdog_timeout,
            )
            self._stop()

    @abstractmethod
    def _apply(self, left: float, right: float) -> None:
        """Send per-side speeds in [-1, +1] to the hardware."""

    @abstractmethod
    def _stop(self) -> None:
        """Bring both sides to a halt. Must be safe to call repeatedly."""


class ConsoleRover(RoverController):
    """Logs commands instead of moving anything. Default, and the Windows path."""

    def __init__(self, cfg: Config, sink: Callable[[str], None] = print) -> None:
        self.commands: list[tuple[float, float]] = []
        self._sink = sink
        super().__init__(cfg)

    def _apply(self, left: float, right: float) -> None:
        self.commands.append((left, right))
        self._sink(f"[rover] left={left:+.2f} right={right:+.2f}")

    def _stop(self) -> None:
        self.commands.append((0.0, 0.0))
        self._sink("[rover] stop   left=+0.00 right=+0.00")


class GpioZeroRover(RoverController):
    """Real L298N / TB6612FNG via gpiozero. Pi 5 only.

    UNTESTED: written from the spec, never executed against hardware. Expect
    the gpiochip number and the pin mapping to need adjustment on first Pi boot
    (PRD 6.9). RPi.GPIO and pigpio do NOT work on Pi 5 -- gpiozero backed by
    lgpio is the supported path.
    """

    def __init__(
        self,
        cfg: Config,
        pins: dict[str, dict[str, int]] | None = None,
        stby_pin: int | None = None,
    ) -> None:
        from gpiozero import DigitalOutputDevice, Motor  # lazy: Pi only

        pins = pins or MOTOR_PINS
        self._left = Motor(
            forward=pins["left"]["forward"],
            backward=pins["left"]["backward"],
            enable=pins["left"]["enable"],
            pwm=True,
        )
        self._right = Motor(
            forward=pins["right"]["forward"],
            backward=pins["right"]["backward"],
            enable=pins["right"]["enable"],
            pwm=True,
        )
        # The TB6612FNG ignores every input until STBY is driven high. The
        # L298N has no such pin; leave stby_pin as None for it.
        self._stby = DigitalOutputDevice(stby_pin) if stby_pin is not None else None
        if self._stby is not None:
            self._stby.on()
        super().__init__(cfg)

    @staticmethod
    def _drive_one(motor, speed: float) -> None:
        if speed > 0:
            motor.forward(min(1.0, speed))
        elif speed < 0:
            motor.backward(min(1.0, -speed))
        else:
            motor.stop()

    def _apply(self, left: float, right: float) -> None:
        self._drive_one(self._left, left)
        self._drive_one(self._right, right)

    def _stop(self) -> None:
        self._left.stop()
        self._right.stop()

    def close(self) -> None:
        super().close()
        self._left.close()
        self._right.close()
        if self._stby is not None:
            self._stby.off()
            self._stby.close()
