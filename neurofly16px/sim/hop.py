"""A visual jump in place. The walking body cannot fly (docs/plan-assessment.md #6)."""

from __future__ import annotations

import math


class HopOverlay:
    """Shared by the stub sim and the flybody sim so a startle looks the same in both."""

    def __init__(self, duration_s: float, height_cm: float, wingbeat_hz: float) -> None:
        self._duration = duration_s
        self._height = height_cm
        self._wingbeat = wingbeat_hz
        self._start: float | None = None

    def trigger(self, t: float) -> None:
        if not self.active(t):
            self._start = t

    def active(self, t: float) -> bool:
        return self._start is not None and 0.0 <= t - self._start < self._duration

    def sample(self, t: float) -> tuple[float, bool, float]:
        """(height cm, airborne, wing phase 0..1) at time t."""
        if not self.active(t):
            self._start = None
            return 0.0, False, 0.0
        elapsed = t - self._start  # type: ignore[operator]
        z = self._height * math.sin(math.pi * elapsed / self._duration)
        wing = (elapsed * self._wingbeat) % 1.0
        return z, True, wing
