"""Hand-written behaviour: idle / walk / startle, driven by audio features.

Sign conventions (neurofly16px.types): `turn` is positive counter-clockwise,
i.e. to the fly's left; `direction` is positive when the sound is on the right.
So turning *toward* a sound means a negative turn, and away from it a positive
one.

`update()` is pure apart from the injected clock and RNG, so the whole state
machine is testable as feature sequence -> command sequence.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable
from typing import Literal

from neurofly16px.config import FsmConfig
from neurofly16px.types import AudioFeatures, FlyState, SteeringCommand

log = logging.getLogger(__name__)

State = Literal["idle", "walk", "startle"]


class FsmBehavior:
    def __init__(
        self,
        cfg: FsmConfig,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._rng = rng or random.Random()
        now = clock()
        self._state: State = "idle"
        self._loud_since: float | None = None
        self._quiet_since: float | None = now
        self._startle_at = -math.inf
        self._startle_side = 1.0
        self._next_twitch = now + self._twitch_gap()
        self._twitch_until = now
        self._twitch_turn = 0.0
        self._drift_turn = 0.0
        self._next_drift = now

    @property
    def state(self) -> State:
        return self._state

    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand:
        del fly  # the FSM reacts to sound only; the body state is unused for now
        now = self._clock()
        cfg = self._cfg
        self._track_loudness(audio, now)

        if self._should_startle(audio, now):
            self._state = "startle"
            self._startle_at = now
            self._startle_side = self._side_away_from(audio.direction)
            log.info("startle (rms %.2f, direction %s)", audio.rms, audio.direction)

        if self._state == "startle":
            elapsed = now - self._startle_at
            if elapsed < cfg.startle_s:
                return SteeringCommand(forward=0.0, turn=self._startle_side, mode="fly")
            if elapsed < cfg.startle_s + cfg.charge_s:
                return SteeringCommand(forward=cfg.charge_forward, turn=0.0, mode="walk")
            self._state = "walk"

        if self._state == "idle" and self._held_loud(now) >= cfg.walk_dwell_s:
            self._state = "walk"
        elif self._state == "walk" and self._held_quiet(now) >= cfg.idle_dwell_s:
            self._state = "idle"

        if self._state == "walk":
            return SteeringCommand(
                forward=cfg.walk_forward, turn=self._walk_turn(audio, now), mode="walk"
            )
        return SteeringCommand(forward=0.0, turn=self._idle_turn(now), mode="idle")

    # --- internals ----------------------------------------------------------

    def _track_loudness(self, audio: AudioFeatures, now: float) -> None:
        if audio.rms > self._cfg.t_walk:
            self._loud_since = self._loud_since if self._loud_since is not None else now
        else:
            self._loud_since = None
        if audio.rms < self._cfg.t_idle:
            self._quiet_since = self._quiet_since if self._quiet_since is not None else now
        else:
            self._quiet_since = None

    def _held_loud(self, now: float) -> float:
        return 0.0 if self._loud_since is None else now - self._loud_since

    def _held_quiet(self, now: float) -> float:
        return 0.0 if self._quiet_since is None else now - self._quiet_since

    def _should_startle(self, audio: AudioFeatures, now: float) -> bool:
        if not (audio.onset and audio.rms > self._cfg.t_startle):
            return False
        return now - self._startle_at >= self._cfg.startle_cooldown_s

    def _side_away_from(self, direction: float | None) -> float:
        if direction is None or direction == 0.0:
            return self._rng.choice((-1.0, 1.0))
        return 1.0 if direction > 0 else -1.0  # sound on the right -> turn left

    def _walk_turn(self, audio: AudioFeatures, now: float) -> float:
        if audio.direction is not None:
            return float(-self._cfg.k_dir * math.sin(audio.direction))  # toward the sound
        if now >= self._next_drift:
            self._drift_turn = self._rng.uniform(-1.0, 1.0) * self._cfg.drift_turn_amount
            self._next_drift = now + self._twitch_gap()
        return self._drift_turn

    def _idle_turn(self, now: float) -> float:
        if now >= self._next_twitch:
            self._twitch_turn = self._rng.choice((-1.0, 1.0)) * self._cfg.idle_turn_amount
            self._twitch_until = now + self._cfg.idle_turn_s
            self._next_twitch = self._twitch_until + self._twitch_gap()
        return self._twitch_turn if now < self._twitch_until else 0.0

    def _twitch_gap(self) -> float:
        return self._rng.uniform(self._cfg.idle_turn_min_s, self._cfg.idle_turn_max_s)
