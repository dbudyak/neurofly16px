"""PCM blocks -> AudioFeatures. Pure numpy, runs in the capture thread.

Loudness is measured *relative to the room*, not normalised to it: a slow
noise-floor tracker follows the quiet level (fast down, slow up), and anything
above it is scaled by a fixed reference RMS, so silence reads 0.0 and speech
reads high. A pure AGC would instead pull any steady level to the middle of the
range, which makes a quiet room indistinguishable from a conversation — that is
what the first version did and why it was replaced (`docs/setup.md`, "Audio").

Onsets are blocks whose RMS stands out against the running median of the last
second: steady noise (a fan) never fires, a clap does. Direction is the
inter-channel level difference, only meaningful with two channels.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from neurofly16px.config import AudioConfig
from neurofly16px.types import AudioFeatures

_EPS = 1e-12
_NOISE_FALL = 0.1  # per block, so the floor follows a quieting room in ~0.2 s


class FeatureExtractor:
    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        self._block_s = cfg.block_ms / 1000.0
        self._noise = cfg.rms_floor
        self._rise = 1.0 - math.exp(-self._block_s / cfg.noise_tau_s)
        window = max(1, round(cfg.median_window_s / self._block_s))
        self._history: deque[float] = deque(maxlen=window)

    @property
    def noise(self) -> float:
        """Current background-noise estimate in raw RMS; useful when tuning."""
        return self._noise

    def push(self, block: np.ndarray, t: float) -> AudioFeatures:
        """block: (n_samples, channels) float32 in [-1, 1]."""
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        rms = float(np.sqrt(np.mean(np.square(samples))))
        onset = self._is_onset(rms)
        self._history.append(rms)
        self._track_noise(rms)
        above = rms - self._cfg.noise_margin * self._noise
        loudness = min(1.0, max(0.0, above / self._cfg.loud_rms))
        return AudioFeatures(t=t, rms=loudness, onset=onset, direction=self._direction(samples))

    # --- internals ----------------------------------------------------------

    def _is_onset(self, rms: float) -> bool:
        if not self._history:
            return False
        median = float(np.median(self._history))
        return rms > self._cfg.onset_k * max(median, self._cfg.rms_floor)

    def _track_noise(self, rms: float) -> None:
        rate = _NOISE_FALL if rms < self._noise else self._rise
        self._noise = max(self._cfg.rms_floor, self._noise + rate * (rms - self._noise))

    def _direction(self, samples: np.ndarray) -> float | None:
        if samples.shape[1] < 2:
            return None
        left = float(np.sqrt(np.mean(np.square(samples[:, 0]))))
        right = float(np.sqrt(np.mean(np.square(samples[:, 1]))))
        balance = (right - left) / (right + left + _EPS)
        if abs(balance) < self._cfg.direction_min_balance:
            return None  # coincident capsules or a centred source: no usable bearing
        return float(math.asin(max(-1.0, min(1.0, balance))))
