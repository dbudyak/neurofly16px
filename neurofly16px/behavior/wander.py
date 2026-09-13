"""What the fly does with no ears: a slow random walk.

Used when no audio is arriving at all — no microphone configured, the device
gone, or the capture process dead. Standing perfectly still in that case looks
like a crash; wandering looks like a fly.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from neurofly16px.config import WanderConfig
from neurofly16px.types import AudioFeatures, FlyState, SteeringCommand


class WanderBehavior:
    """Walks, turns a little, pauses now and then. No input."""

    def __init__(
        self,
        cfg: WanderConfig,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._rng = rng or random.Random()
        self._until = clock()
        self._cmd = SteeringCommand(forward=cfg.forward, turn=0.0, mode="walk")

    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand:
        del audio, fly
        now = self._clock()
        if now >= self._until:
            self._cmd = self._next_leg()
            self._until = now + self._rng.uniform(self._cfg.leg_min_s, self._cfg.leg_max_s)
        return self._cmd

    def _next_leg(self) -> SteeringCommand:
        if self._rng.random() < self._cfg.pause_chance:
            return SteeringCommand(forward=0.0, turn=0.0, mode="idle")
        turn = self._rng.uniform(-1.0, 1.0) * self._cfg.turn_amount
        return SteeringCommand(forward=self._cfg.forward, turn=turn, mode="walk")
