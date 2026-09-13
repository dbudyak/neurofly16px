"""Audio features -> Johnston's-organ drive.

The fly hears with its antennae: the Johnston's organ neurons sit at the base of
each and fire to antennal vibration. JO-A and JO-B carry sound, JO-C and JO-E
wind and gravity (`docs/banc.md`), so loudness drives the sound populations and
the wind ones stay quiet unless asked for.

Loudness is compressed logarithmically — the range from a quiet room to a clap is
orders of magnitude, and firing rates are not — and split left/right by the
bearing when there is one.
"""

from __future__ import annotations

import math

from neurofly16px.config import BrainConfig
from neurofly16px.types import AudioFeatures


def compress(rms: float, r_max: float, k: float) -> float:
    """0..1 loudness -> 0..r_max Hz, log-compressed."""
    clipped = min(1.0, max(0.0, rms))
    return r_max * math.log1p(k * clipped) / math.log1p(k)


def balance(direction: float | None) -> tuple[float, float]:
    """(left, right) gains summing to 2.0; equal when there is no bearing."""
    if direction is None:
        return 1.0, 1.0
    d = max(-1.0, min(1.0, math.sin(direction)))
    return 1.0 - d, 1.0 + d


class JohnstonStimulus:
    """Turns the latest AudioFeatures into per-population rates in Hz."""

    def __init__(self, cfg: BrainConfig) -> None:
        self._cfg = cfg
        self._onset_until = -math.inf

    def rates(self, audio: AudioFeatures, now: float) -> dict[str, float]:
        cfg = self._cfg
        if audio.onset:
            self._onset_until = now + cfg.onset_ms / 1000.0
        base = compress(audio.rms, cfg.r_max_hz, cfg.stim_k)
        if now < self._onset_until:
            base = max(base, cfg.onset_hz)
        left, right = balance(audio.direction)
        drive = {
            "jo_sound_left": base * left,
            "jo_sound_right": base * right,
        }
        if cfg.wind_fraction > 0.0:
            drive["jo_wind_left"] = base * left * cfg.wind_fraction
            drive["jo_wind_right"] = base * right * cfg.wind_fraction
        return drive
