"""Fixed command sequence, looped. Exercises every mode without audio."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from neurofly16px.types import IDLE, AudioFeatures, FlyState, SteeringCommand

DEMO_SCRIPT: tuple[tuple[float, SteeringCommand], ...] = (
    (2.0, IDLE),
    (4.0, SteeringCommand(0.8, 0.0, "walk")),
    (2.0, SteeringCommand(0.5, 1.0, "walk")),
    (2.0, SteeringCommand(0.5, -1.0, "walk")),
    (0.3, SteeringCommand(0.0, 0.0, "fly")),
    (3.0, SteeringCommand(1.0, 0.0, "walk")),
)


class ScriptedBehavior:
    def __init__(
        self,
        script: Sequence[tuple[float, SteeringCommand]] = DEMO_SCRIPT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not script:
            raise ValueError("script must have at least one segment")
        self._script = list(script)
        self._period = sum(d for d, _ in self._script)
        self._clock = clock
        self._t0 = clock()

    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand:
        del audio, fly
        elapsed = (self._clock() - self._t0) % self._period
        for duration, cmd in self._script:
            if elapsed < duration:
                return cmd
            elapsed -= duration
        return self._script[-1][1]
