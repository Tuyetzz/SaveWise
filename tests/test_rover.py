import time

from rescue_vision.config import Config
from rescue_vision.rover import ConsoleRover

CFG = Config()


def test_drive_mixes_to_left_and_right():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.drive(turn=0.0, forward=0.5)
    assert rover.commands[-1] == (0.5, 0.5)
    rover.close()


def test_stop_sends_zero_to_both_sides():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.drive(turn=0.5, forward=0.5)
    rover.stop()
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_watchdog_stops_the_motors_when_drive_goes_quiet():
    """A vision pipeline that hangs must not leave a rover driving into a wall."""
    cfg = Config(watchdog_timeout=0.1)
    rover = ConsoleRover(cfg, sink=lambda _: None)
    rover.drive(turn=0.0, forward=0.5)
    assert rover.commands[-1] == (0.5, 0.5)
    time.sleep(0.35)
    assert rover.commands[-1] == (0.0, 0.0)
    rover.close()


def test_watchdog_does_not_fire_while_drive_is_called_regularly():
    cfg = Config(watchdog_timeout=0.3)
    rover = ConsoleRover(cfg, sink=lambda _: None)
    for _ in range(6):
        rover.drive(turn=0.0, forward=0.4)
        time.sleep(0.05)
    assert rover.commands[-1] == (0.4, 0.4)
    rover.close()


def test_close_stops_the_motors():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.drive(turn=0.0, forward=0.5)
    rover.close()
    assert rover.commands[-1] == (0.0, 0.0)


def test_drive_after_close_is_ignored():
    """Nothing may re-engage the motors once the controller is shut down."""
    rover = ConsoleRover(CFG, sink=lambda _: None)
    rover.close()
    rover.drive(turn=0.0, forward=1.0)
    assert rover.commands[-1] == (0.0, 0.0)


def test_context_manager_stops_on_exception():
    rover = ConsoleRover(CFG, sink=lambda _: None)
    try:
        with rover:
            rover.drive(turn=0.0, forward=0.9)
            raise RuntimeError("pipeline blew up")
    except RuntimeError:
        pass
    assert rover.commands[-1] == (0.0, 0.0)


def test_console_rover_writes_to_its_sink():
    lines = []
    rover = ConsoleRover(CFG, sink=lines.append)
    rover.drive(turn=0.25, forward=0.0)
    rover.close()
    assert any("left" in line for line in lines)


def test_gpiozero_rover_import_does_not_require_gpiozero():
    """The module must import cleanly on Windows -- gpiozero is imported lazily."""
    from rescue_vision.rover import GpioZeroRover

    assert GpioZeroRover is not None
