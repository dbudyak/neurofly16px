"""Who the neurons are: identity, taxonomy, transmitter sign and soma position.

This is the half of the connectome that needs no login — the BANC metadata and
annotation tables published in `htem/BANC-project`. The wiring (who connects to
whom) is gated behind a FlyWire/Codex sign-in and arrives in Phase 6.0b; until
then this module gives the populations to stimulate and read out, and the map the
activity is drawn on.

Sources and the sign convention are in `docs/banc.md`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EXCITATORY = ("acetylcholine", "dopamine", "octopamine", "serotonin", "tyramine")
INHIBITORY = ("gaba", "glutamate", "histamine")
"""Shiu et al.'s convention plus BANC's two extra transmitters (docs/banc.md)."""

CATEGORICAL = (
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "cell_function",
    "side",
    "region",
    "neurotransmitter",
)


def transmitter_sign(transmitter: str | None) -> int:
    """+1 excitatory, -1 inhibitory; unknown transmitters are assumed excitatory."""
    if transmitter is None:
        return 1
    name = transmitter.strip().lower()
    if name in INHIBITORY:
        return -1
    return 1


@dataclass(frozen=True)
class Anatomy:
    """Per-neuron arrays, all the same length and in the same order."""

    ids: np.ndarray
    sign: np.ndarray
    soma_xyz: np.ndarray
    labels: dict[str, np.ndarray]
    populations: dict[str, list[int]]

    @classmethod
    def load(cls, npz_path: str | Path, populations_path: str | Path | None = None) -> Anatomy:
        with np.load(npz_path, allow_pickle=False) as f:
            labels = {name: f[name] for name in CATEGORICAL if name in f}
            ids, sign, soma = f["ids"], f["sign"], f["soma_xyz"]
        populations: dict[str, list[int]] = {}
        if populations_path is not None:
            with open(populations_path) as handle:
                loaded = json.load(handle)
            populations = {name: entry["indices"] for name, entry in loaded.items()}
        return cls(ids=ids, sign=sign, soma_xyz=soma, labels=labels, populations=populations)

    def __len__(self) -> int:
        return len(self.ids)

    def index_of(self, population: str) -> np.ndarray:
        return np.asarray(self.populations.get(population, ()), dtype=np.int64)

    def soma_image(
        self,
        width: int,
        height: int,
        axes: tuple[int, int] = (0, 1),
        flip: bool = False,
    ) -> np.ndarray:
        """Neuron count per pixel, for the viewer's heat map.

        The default projection is x (left-right) against y (the body axis), which
        is what separates the structures: measured soma means are y = 155 um for
        the brain and 778 um for the ventral nerve cord, against a 19 um
        difference in x. Unflipped, small y sits at the top, so the fly is drawn
        head-up. Neurons without a soma position are dropped, never binned at the
        origin.
        """
        known = np.isfinite(self.soma_xyz).all(axis=1)
        if not known.any():
            return np.zeros((height, width), np.int32)
        points = self.soma_xyz[known][:, list(axes)]
        return self.bin_points(points, width, height, flip)

    def soma_bins(
        self,
        width: int,
        height: int,
        axes: tuple[int, int] = (0, 1),
        flip: bool = False,
    ) -> np.ndarray:
        """Pixel index per neuron (-1 where the soma position is unknown).

        Computed once so that per-update the viewer only has to `bincount` the
        spikes of the neurons that fired.
        """
        bins = np.full(len(self.ids), -1, np.int32)
        known = np.isfinite(self.soma_xyz).all(axis=1)
        if not known.any():
            return bins
        points = self.soma_xyz[known][:, list(axes)]
        cols, rows = self._pixel_coords(points, width, height, flip)
        bins[known] = rows * width + cols
        return bins

    @staticmethod
    def _pixel_coords(
        points: np.ndarray, width: int, height: int, flip: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        low = points.min(axis=0)
        high = points.max(axis=0)
        span = np.where(high > low, high - low, 1.0)
        scaled = (points - low) / span
        vertical = 1.0 - scaled[:, 1] if flip else scaled[:, 1]
        cols = np.clip((scaled[:, 0] * (width - 1)).round(), 0, width - 1).astype(np.int32)
        rows = np.clip((vertical * (height - 1)).round(), 0, height - 1).astype(np.int32)
        return cols, rows

    @classmethod
    def bin_points(
        cls, points: np.ndarray, width: int, height: int, flip: bool = False
    ) -> np.ndarray:
        cols, rows = cls._pixel_coords(points, width, height, flip)
        flat = np.bincount(rows * width + cols, minlength=width * height)
        return flat.reshape(height, width).astype(np.int32)
