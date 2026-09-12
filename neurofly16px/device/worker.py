"""Runs a Display on its own thread so a slow or stalled device never blocks the sim."""

from __future__ import annotations

import logging
import threading

from neurofly16px.device.base import Display
from neurofly16px.types import Frame

log = logging.getLogger(__name__)


class DisplayWorker:
    """One-slot mailbox: the latest frame wins, older pending frames are counted and dropped."""

    def __init__(self, display: Display) -> None:
        self._display = display
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="display", daemon=True)
        self.frames_shown = 0
        self.frames_dropped = 0

    @property
    def display(self) -> Display:
        """The wrapped display, so callers can read its own counters."""
        return self._display

    def start(self) -> None:
        self._thread.start()

    def show(self, frame: Frame) -> None:
        with self._cond:
            if self._pending is not None:
                self.frames_dropped += 1
            self._pending = frame
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=5.0)
        self._display.close()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._pending is None and self._stop:
                    return
                frame, self._pending = self._pending, None
            try:
                self._display.show(frame)  # type: ignore[arg-type]
                self.frames_shown += 1
            except Exception:
                log.exception("display.show failed; frame dropped")
