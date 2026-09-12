"""Numpy port of the flybody DMPO walking policy.

Architecture (docs/flybody.md, "Pretrained policies"): observations flattened and
concatenated in a fixed key order -> Linear -> LayerNorm -> tanh -> Linear -> ELU
-> Linear -> ELU -> Linear (mean of the Gaussian head). Only the mean is needed at
test time; CanonicalSpecWrapper clips it to [-1, 1] downstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LN_EPS = 1e-5
_WEIGHT_NAMES = (
    "w0",
    "b0",
    "ln_scale",
    "ln_offset",
    "w1",
    "b1",
    "w2",
    "b2",
    "w_mean",
    "b_mean",
)


def flatten_observation(obs: Mapping[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate the flattened observables in the given key order as float32."""
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in keys])


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


@dataclass(frozen=True)
class NumpyPolicy:
    """Deterministic (mean) forward pass of the exported walking policy."""

    obs_keys: tuple[str, ...]
    w0: np.ndarray
    b0: np.ndarray
    ln_scale: np.ndarray
    ln_offset: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w_mean: np.ndarray
    b_mean: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> NumpyPolicy:
        with np.load(path, allow_pickle=False) as f:
            keys = tuple(str(k) for k in f["obs_keys"])
            weights = {name: f[name].astype(np.float32) for name in _WEIGHT_NAMES}
        return cls(obs_keys=keys, **weights)

    @property
    def action_dim(self) -> int:
        return int(self.w_mean.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.w0.shape[0])

    def __call__(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.forward(flatten_observation(obs, self.obs_keys))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (obs_dim,) or (batch, obs_dim) -> canonical action(s), float32."""
        h = x @ self.w0 + self.b0
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mean) / np.sqrt(var + _LN_EPS) * self.ln_scale + self.ln_offset
        h = np.tanh(h)
        h = _elu(h @ self.w1 + self.b1)
        h = _elu(h @ self.w2 + self.b2)
        return (h @ self.w_mean + self.b_mean).astype(np.float32)
