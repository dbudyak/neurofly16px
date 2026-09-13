"""Night mode: the panel goes dark between two local hours.

A desk toy that glows at 3 a.m. is a bug. `ScheduledDisplay` wraps any `Display`:
outside the night window it passes frames through at the configured brightness,
inside it blanks the panel once and then stops pushing frames until morning.
The clock is injected, so the tests do not wait for midnight.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from neurofly16px.config import NightConfig
from neurofly16px.device.base import Display
from neurofly16px.types import Frame, new_frame

log = logging.getLogger(__name__)


def in_window(hour: float, start: float, end: float) -> bool:
    """Is `hour` inside [start, end)? The window may wrap past midnight."""
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


class ScheduledDisplay:
    """Display wrapper that sleeps at night and sets the daytime brightness."""

    def __init__(
        self,
        display: Display,
        cfg: NightConfig,
        clock: Callable[[], float] = time.time,
        localtime: Callable[[float], time.struct_time] = time.localtime,
    ) -> None:
        self._display = display
        self._cfg = cfg
        self._clock = clock
        self._localtime = localtime
        self._asleep: bool | None = None

    @property
    def asleep(self) -> bool:
        return bool(self._asleep)

    def local_hour(self) -> float:
        t = self._localtime(self._clock())
        return t.tm_hour + t.tm_min / 60.0

    def show(self, frame: Frame) -> None:
        asleep = self._cfg.enabled and in_window(self.local_hour(), self._cfg.start, self._cfg.end)
        if asleep != self._asleep:
            self._asleep = asleep
            log.info("night mode %s", "on: panel dark" if asleep else "off: panel awake")
            self._set_brightness(self._cfg.brightness if asleep else None)
            if asleep:
                self._display.show(new_frame())  # one black frame, then silence
        if not asleep:
            self._display.show(frame)

    def close(self) -> None:
        self._display.close()

    def _set_brightness(self, level: int | None) -> None:
        setter = getattr(self._display, "set_brightness", None)
        if setter is None:
            return
        try:
            setter(level if level is not None else self._cfg.day_brightness)
        except OSError as exc:  # the link may be down; the driver will reconnect
            log.warning("could not set brightness: %s", exc)

    def __getattr__(self, name: str):
        """Counters and the like come from the wrapped display."""
        return getattr(self._display, name)
