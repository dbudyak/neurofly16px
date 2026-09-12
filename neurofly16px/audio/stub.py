"""Scripted audio features for running without a microphone."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence

from neurofly16px.types import SILENCE, AudioFeatures


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
            return SILENCE
        elapsed = (now - self._t0) % self._period
        for duration, features in self._script:
            if elapsed < duration:
                return dataclasses.replace(features, t=now)
            elapsed -= duration
        return dataclasses.replace(self._script[-1][1], t=now)
