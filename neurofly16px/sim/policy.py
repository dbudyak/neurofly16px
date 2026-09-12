"""Numpy port of the flybody DMPO walking policy.

Architecture (docs/flybody.md, "Pretrained policies"): observations are flattened
and concatenated in a fixed key order, then

    Linear -> LayerNorm(eps 1e-5) -> tanh -> (Linear -> ELU) * n -> Linear(mean)

Only the mean of the Gaussian head is needed at test time; `CanonicalSpecWrapper`
clips it to [-1, 1] downstream. The shipped walking policy has n = 3 hidden layers
of 512 units (measured, see docs/flybody.md "Host results"), so the number of
hidden layers is read from the npz rather than hard-coded.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LN_EPS = 1e-5


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
    hidden: tuple[tuple[np.ndarray, np.ndarray], ...]
    w_mean: np.ndarray
    b_mean: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> NumpyPolicy:
        with np.load(path, allow_pickle=False) as f:
            hidden = []
            for i in itertools.count(1):
                if f"w{i}" not in f:
                    break
                hidden.append((f[f"w{i}"].astype(np.float32), f[f"b{i}"].astype(np.float32)))
            return cls(
                obs_keys=tuple(str(k) for k in f["obs_keys"]),
                w0=f["w0"].astype(np.float32),
                b0=f["b0"].astype(np.float32),
                ln_scale=f["ln_scale"].astype(np.float32),
                ln_offset=f["ln_offset"].astype(np.float32),
                hidden=tuple(hidden),
                w_mean=f["w_mean"].astype(np.float32),
                b_mean=f["b_mean"].astype(np.float32),
            )

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "obs_keys": np.array(self.obs_keys),
            "w0": self.w0,
            "b0": self.b0,
            "ln_scale": self.ln_scale,
            "ln_offset": self.ln_offset,
            "w_mean": self.w_mean,
            "b_mean": self.b_mean,
        }
        for i, (w, b) in enumerate(self.hidden, start=1):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

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
        h = np.asarray(x, dtype=np.float32) @ self.w0 + self.b0
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mean) / np.sqrt(var + _LN_EPS) * self.ln_scale + self.ln_offset
        h = np.tanh(h)
        for w, b in self.hidden:
            h = _elu(h @ w + b)
        return (h @ self.w_mean + self.b_mean).astype(np.float32)


def make_policy(
    obs_keys: Sequence[str],
    w0: np.ndarray,
    b0: np.ndarray,
    ln_scale: np.ndarray,
    ln_offset: np.ndarray,
    hidden: Sequence[tuple[np.ndarray, np.ndarray]],
    w_mean: np.ndarray,
    b_mean: np.ndarray,
) -> NumpyPolicy:
    """Build a policy from plain sequences (used by scripts/export_policy.py)."""
    return NumpyPolicy(
        obs_keys=tuple(obs_keys),
        w0=np.asarray(w0, np.float32),
        b0=np.asarray(b0, np.float32),
        ln_scale=np.asarray(ln_scale, np.float32),
        ln_offset=np.asarray(ln_offset, np.float32),
        hidden=tuple((np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in hidden),
        w_mean=np.asarray(w_mean, np.float32),
        b_mean=np.asarray(b_mean, np.float32),
    )
