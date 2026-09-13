import time

import numpy as np

from neurofly16px.config import NightConfig
from neurofly16px.device.schedule import ScheduledDisplay, in_window
from neurofly16px.types import new_frame


class FakeDisplay:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.brightness: list[int] = []
        self.closed = False

    def show(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def set_brightness(self, level: int) -> None:
        self.brightness.append(level)

    def close(self) -> None:
        self.closed = True


def make(hour: float, cfg: NightConfig | None = None) -> tuple[ScheduledDisplay, FakeDisplay, list]:
    now = [hour]
    inner = FakeDisplay()
    sched = ScheduledDisplay(
        inner,
        cfg or NightConfig(),
        clock=lambda: now[0],
        localtime=lambda t: time.struct_time((2026, 9, 13, int(t), int(t % 1 * 60), 0, 6, 256, 0)),
    )
    return sched, inner, now


def test_window_wraps_past_midnight() -> None:
    assert in_window(23.5, 23.0, 7.0) and in_window(3.0, 23.0, 7.0)
    assert not in_window(12.0, 23.0, 7.0)
    assert in_window(12.0, 8.0, 20.0) and not in_window(21.0, 8.0, 20.0)
    assert not in_window(5.0, 7.0, 7.0), "an empty window never sleeps"


def test_daytime_passes_frames_through() -> None:
    sched, inner, _ = make(12.0)
    frame = new_frame()
    frame[0, 0] = (9, 9, 9)
    sched.show(frame)
    assert len(inner.frames) == 1 and tuple(inner.frames[0][0, 0]) == (9, 9, 9)
    assert inner.brightness == [60] and not sched.asleep


def test_night_blanks_once_then_stops_pushing() -> None:
    sched, inner, _ = make(23.5)
    frame = new_frame()
    frame[5, 5] = (200, 0, 0)
    for _ in range(5):
        sched.show(frame)
    assert sched.asleep
    assert len(inner.frames) == 1 and not inner.frames[0].any(), "one black frame, then silence"
    assert inner.brightness == [0]


def test_morning_wakes_the_panel_up() -> None:
    sched, inner, now = make(23.5)
    sched.show(new_frame())
    now[0] = 8.0
    frame = new_frame()
    frame[1, 1] = (5, 5, 5)
    sched.show(frame)
    assert not sched.asleep and len(inner.frames) == 2
    assert inner.brightness == [0, 60]


def test_disabled_schedule_never_sleeps() -> None:
    sched, inner, _ = make(2.0, NightConfig(enabled=False))
    sched.show(new_frame())
    assert not sched.asleep and len(inner.frames) == 1


def test_wrapper_forwards_close_and_counters() -> None:
    sched, inner, _ = make(12.0)
    inner.frames_sent = 7
    assert sched.frames_sent == 7
    sched.close()
    assert inner.closed
