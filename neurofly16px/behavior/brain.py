"""Behaviour driven by the connectome model instead of the state machine."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from neurofly16px.brain.runner import BrainProcess
from neurofly16px.config import BrainConfig
from neurofly16px.types import AudioFeatures, FlyState, SteeringCommand

log = logging.getLogger(__name__)


class BrainBehavior:
    """Posts audio to the brain process and returns whatever it last decided."""

    def __init__(
        self,
        cfg: BrainConfig,
        process: BrainProcess | None = None,
        clock: Callable[[], float] = time.monotonic,
        log_every_s: float = 10.0,
    ) -> None:
        self._process = process or BrainProcess(cfg)
        self._clock = clock
        self._log_every = log_every_s
        self._next_log = clock()
        self._process.start()

    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand:
        del fly
        self._process.send_audio(audio)
        activity = self._process.poll()
        now = self._clock()
        if now >= self._next_log:
            self._next_log = now + self._log_every
            log.info(
                "brain at %.1fx real time, %.1f simulated s, dn_walking %.2f Hz, mode %s",
                activity.speed_ratio,
                activity.sim_time_s,
                activity.rates_hz.get("dn_walking", 0.0),
                activity.command.mode,
            )
        return activity.command

    def close(self) -> None:
        self._process.stop()
