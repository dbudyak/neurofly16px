"""Descending-neuron activity -> SteeringCommand.

The fly's brain talks to its body through ~1,300 descending neurons, and the
clusters in `data/populations.json` say what each group is for. The rules here
are deliberately explicit rather than learned: a giant-fibre burst is an escape,
the walking cluster's rate is forward speed, its left-right difference is a turn.
PLAN.md 6.3 keeps a PCA fallback in reserve if these rules turn out dead or
chaotic.
"""

from __future__ import annotations

import logging
import math

from neurofly16px.config import BrainConfig
from neurofly16px.types import SteeringCommand

log = logging.getLogger(__name__)

WALK_LEFT = ("dn_walking",)
GIANT_FIBRE = ("dn_DNp01_left", "dn_DNp01_right")


class DescendingReadout:
    """Windowed spike rates in, one steering command out."""

    def __init__(self, cfg: BrainConfig) -> None:
        self._cfg = cfg
        self._smoothed: dict[str, float] = {}
        self._fly_until = -math.inf

    def smooth(self, rates_hz: dict[str, float]) -> dict[str, float]:
        """Exponential moving average over every population seen so far.

        A population missing from this window counts as zero, not as its previous
        value: leaving it frozen would let one burst hold the body forever.
        """
        alpha = self._cfg.ema_alpha
        for name in set(self._smoothed) | set(rates_hz):
            previous = self._smoothed.get(name, 0.0)
            self._smoothed[name] = previous + alpha * (rates_hz.get(name, 0.0) - previous)
        return dict(self._smoothed)

    def update(self, rates_hz: dict[str, float], now: float) -> SteeringCommand:
        """`rates_hz` is per-neuron Hz per population, over the last window."""
        rates = self.smooth(rates_hz)
        cfg = self._cfg

        escape = max(rates.get(name, 0.0) for name in GIANT_FIBRE)
        if escape >= cfg.giant_fibre_hz:
            if now >= self._fly_until:
                log.info("giant fibre burst at %.1f Hz: escape", escape)
            self._fly_until = now + cfg.fly_s
        if now < self._fly_until:
            return SteeringCommand(forward=0.0, turn=self._turn(rates), mode="fly")

        walking = rates.get("dn_walking", 0.0)
        turn = self._turn(rates)
        if walking < cfg.idle_hz:
            return SteeringCommand(forward=0.0, turn=turn * 0.5, mode="idle")
        forward = min(1.0, walking / cfg.walk_hz)
        return SteeringCommand(forward=forward, turn=turn, mode="walk")

    def _turn(self, rates: dict[str, float]) -> float:
        """Left minus right, over the steering-related clusters, normalised."""
        left = rates.get("dn_walking_left", 0.0) + rates.get("dn_head_orienting_left", 0.0)
        right = rates.get("dn_walking_right", 0.0) + rates.get("dn_head_orienting_right", 0.0)
        difference = left - right
        return max(-1.0, min(1.0, difference / self._cfg.turn_gain))
