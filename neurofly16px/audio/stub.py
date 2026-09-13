"""Scripted audio features for running without a microphone."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence

from neurofly16px.types import SILENCE, AudioFeatures

TALK = AudioFeatures(t=0.0, rms=0.45, onset=False, direction=None)
CLAP_RIGHT = AudioFeatures(t=0.0, rms=0.95, onset=True, direction=0.6)
CLAP_LEFT = AudioFeatures(t=0.0, rms=0.95, onset=True, direction=-0.6)

DEMO_SCRIPT: tuple[tuple[float, AudioFeatures], ...] = (
    (6.0, SILENCE),
    (0.2, CLAP_RIGHT),
    (5.0, SILENCE),
    (8.0, TALK),
    (4.0, SILENCE),
    (0.2, CLAP_LEFT),
    (6.0, SILENCE),
)
"""A room with someone in it: quiet, a clap, quiet, talking, quiet, another clap.

Lets the behaviour layers be exercised end to end without a microphone.
"""


class StubAudio:
    """Loops through (duration_s, features) segments; None means permanent silence."""

    def __init__(
        self,
        script: Sequence[tuple[float, AudioFeatures]] | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._script = list(script or [])
        self._period = sum(d for d, _ in self._script)
        self._clock = clock
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = self._clock()

    def stop(self) -> None:
        pass

    def latest(self) -> AudioFeatures:
        now = self._clock()
        if not self._script:
            # a timestamp, not the frozen SILENCE constant: this is a working
            # device reporting a quiet room, not a missing one
            return dataclasses.replace(SILENCE, t=now)
        elapsed = (now - self._t0) % self._period
        for duration, features in self._script:
            if elapsed < duration:
                return dataclasses.replace(features, t=now)
            elapsed -= duration
        return dataclasses.replace(self._script[-1][1], t=now)
