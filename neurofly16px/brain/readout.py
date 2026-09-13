"""Descending-neuron activity -> SteeringCommand.

The fly's brain talks to its body through ~1,300 descending neurons. Which of
them sound actually reaches is a measurement, not a guess, and on this connectome
(`docs/banc.md`) it is unambiguous: Johnston's-organ drive lights up the giant
fibre DNp01 (50 Hz at speech level, 184 Hz at clap level), then the
threat-response and takeoff clusters, and leaves the walking cluster silent. So
the readout is built from what responds:

* the giant fibre, with habituation, decides escape;
* the mean descending rate is how much the brain is asking the body to do;
* the left-right contrast of descending activity is which way.

PLAN.md 6.3 held a PCA over all descending rates in reserve for exactly this
case. This is its simplest form: two aggregates of the same population.
"""

from __future__ import annotations

import logging
import math

from neurofly16px.config import BrainConfig
from neurofly16px.types import SteeringCommand

log = logging.getLogger(__name__)

GIANT_FIBRE = ("dn_DNp01_left", "dn_DNp01_right")
_EPS = 1e-9


class DescendingReadout:
    """Windowed spike rates in, one steering command out."""

    def __init__(self, cfg: BrainConfig) -> None:
        self._cfg = cfg
        self._smoothed: dict[str, float] = {}
        self._gf_baseline = 0.0
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
        """`rates_hz` is per-neuron Hz per population over the last window.

        `now` is wall-clock seconds: the rates are biology, but the hold times are
        what the fly on the desk does, and the brain does not run at real time.
        """
        rates = self.smooth(rates_hz)
        cfg = self._cfg
        turn = self._turn(rates)

        if self._escaping(rates, now):
            return SteeringCommand(forward=0.0, turn=turn, mode="fly")

        drive = rates.get("dn_all", 0.0)
        if drive < cfg.idle_hz:
            return SteeringCommand(forward=0.0, turn=turn * 0.5, mode="idle")
        forward = min(1.0, drive / cfg.drive_hz)
        return SteeringCommand(forward=forward, turn=turn, mode="walk")

    # --- internals ----------------------------------------------------------

    def _escaping(self, rates: dict[str, float], now: float) -> bool:
        cfg = self._cfg
        giant_fibre = max(rates.get(name, 0.0) for name in GIANT_FIBRE)
        baseline = self._gf_baseline
        self._gf_baseline += cfg.gf_baseline_alpha * (giant_fibre - baseline)
        novel = giant_fibre >= cfg.gf_novelty_ratio * baseline
        if giant_fibre >= cfg.giant_fibre_hz and novel:
            if now >= self._fly_until:
                log.info(
                    "giant fibre at %.0f Hz over a %.0f Hz baseline: escape",
                    giant_fibre,
                    baseline,
                )
            self._fly_until = now + cfg.fly_s
        return now < self._fly_until

    def _turn(self, rates: dict[str, float]) -> float:
        """Left-right contrast of descending activity, in [-1, 1].

        A contrast rather than a difference, so it does not scale with how loud
        the room is.
        """
        left = rates.get("dn_all_left", 0.0)
        right = rates.get("dn_all_right", 0.0)
        contrast = (left - right) / (left + right + _EPS)
        return max(-1.0, min(1.0, contrast * self._cfg.turn_gain))
