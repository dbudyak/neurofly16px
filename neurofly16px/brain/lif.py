"""Leaky integrate-and-fire propagation over the BANC wiring.

The model is Shiu et al. 2024 (`docs/banc.md`): every neuron is the same leaky
integrator, every synapse is worth `w_syn` millivolts times its synapse count,
and the sign comes from the presynaptic neuron's transmitter. Wiring is real;
the dynamics are a deliberately coarse assumption.

Propagation is one sparse matrix-vector product per step: `arriving = Wt @ spikes`
with the wiring transposed so each row is a postsynaptic neuron. The first
version gathered only the rows of the neurons that spiked, which sounds cheaper
and measured 8x slower: `nonzero()` and reading the gather size force a
GPU-to-CPU sync on every step, while an SpMV over 1.4 M edges is ~11 MB of
traffic and needs no synchronisation at all.

Spikes arrive `t_dly` later, so they are accumulated into a ring of
`t_dly / dt` + 1 slots and folded into the conductance when their slot comes up.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

from neurofly16px.config import BrainConfig

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; running the brain on the CPU")
        return torch.device("cpu")
    return torch.device(requested)


class LifBrain:
    """One fly's central nervous system, stepped at `dt_ms`."""

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        cfg: BrainConfig,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or resolve_device(cfg.device)
        self.n = len(indptr) - 1
        self.indptr = torch.as_tensor(indptr, dtype=torch.int64, device=self.device)
        self.indices = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        self._incoming = self._transpose(indptr, indices, weights)

        self.delay_steps = max(1, round(cfg.t_dly_ms / cfg.dt_ms))
        self.refractory_steps = max(1, round(cfg.t_rfc_ms / cfg.dt_ms))
        self._slots = self.delay_steps + 1
        self._decay = math.exp(-cfg.dt_ms / cfg.tau_ms)
        self._membrane = cfg.dt_ms / cfg.t_mbr_ms
        self._dt_s = cfg.dt_ms / 1000.0

        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self._drive_prob: torch.Tensor | None = None
        self.weight_scale = cfg.weight_scale
        self.runaway_events = 0
        self.steps = 0
        self._peak_spikes = torch.zeros((), dtype=torch.int64, device=self.device)
        self.reset()

    def _transpose(
        self, indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray
    ) -> torch.Tensor:
        """The wiring as CSR by *postsynaptic* neuron, ready for `Wt @ spikes`."""
        pre = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(indptr))
        order = np.lexsort((pre, indices))
        post_sorted = indices[order]
        row_counts = np.bincount(post_sorted, minlength=self.n)
        row_ptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(row_counts, out=row_ptr[1:])
        return torch.sparse_csr_tensor(
            torch.as_tensor(row_ptr, dtype=torch.int64, device=self.device),
            torch.as_tensor(pre[order], dtype=torch.int64, device=self.device),
            torch.as_tensor(weights[order], dtype=torch.float32, device=self.device),
            size=(self.n, self.n),
            device=self.device,
        )

    @classmethod
    def load(cls, cfg: BrainConfig, device: torch.device | None = None) -> LifBrain:
        with np.load(cfg.connectome_path, allow_pickle=False) as f:
            brain = cls(f["indptr"], f["indices"], f["weights"], cfg, device)
        log.info(
            "brain: %d neurons, %d edges on %s", brain.n, len(brain.indices), brain.device
        )
        return brain

    def reset(self) -> None:
        zeros = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.v = zeros + self.cfg.v_0
        self.g = zeros.clone()
        self.refractory_left = torch.zeros(self.n, dtype=torch.int16, device=self.device)
        self.arriving = torch.zeros(
            (self._slots, self.n), dtype=torch.float32, device=self.device
        )
        self.steps = 0

    def set_drive(self, rates_hz: torch.Tensor | None) -> None:
        """Set the Poisson drive once, as a per-neuron Hz vector (mostly zeros).

        Kept dense on purpose: selecting the stimulated neurons would give the step
        a data-dependent shape, and every such shape costs a GPU-to-CPU sync, which
        measured slower than drawing one random number per neuron.
        """
        if rates_hz is None:
            self._drive_prob = None
            return
        self._drive_prob = rates_hz.to(self.device, torch.float32) * self._dt_s

    def step(self, rates_hz: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one `dt`; returns the boolean spike vector for this step.

        Passing `rates_hz` is shorthand for `set_drive` followed by `step()`; the
        loop sets the drive when the audio changes and then steps repeatedly.
        """
        if rates_hz is not None:
            self.set_drive(rates_hz)
        ready = self.refractory_left == 0
        spiked = (self.v > self.cfg.v_th) & ready
        if self._drive_prob is not None:
            # Stimulation ignores the refractory period, as in Shiu et al., where the
            # PoissonInput weight is far above threshold and the target's refractory
            # period is set to zero so it fires at the requested rate (docs/banc.md).
            draw = torch.rand(self.n, generator=self.generator, device=self.device)
            spiked = spiked | (draw < self._drive_prob)

        self._guard_runaway(spiked)
        self._propagate(spiked)

        slot = self.steps % self._slots
        self.g = self.g + self.arriving[slot]
        self.arriving[slot].zero_()

        active = self.refractory_left == 0
        self.g = torch.where(active, self.g * self._decay, self.g)
        self.v = torch.where(
            active, self.v + self._membrane * (self.cfg.v_0 - self.v + self.g), self.v
        )

        self.v = torch.where(spiked, torch.full_like(self.v, self.cfg.v_rst), self.v)
        self.g = torch.where(spiked, torch.zeros_like(self.g), self.g)
        self.refractory_left = torch.where(
            spiked,
            torch.full_like(self.refractory_left, self.refractory_steps),
            torch.clamp(self.refractory_left - 1, min=0),
        )
        self.steps += 1
        return spiked

    # --- internals ----------------------------------------------------------

    def _propagate(self, spiked: torch.Tensor) -> None:
        """Add every spiking neuron's outgoing weights to the slot `t_dly` ahead."""
        contribution = torch.mv(self._incoming, spiked.to(self.weights.dtype))
        slot = (self.steps + self.delay_steps) % self._slots
        self.arriving[slot].add_(contribution, alpha=self.cfg.w_syn * self.weight_scale)

    def _guard_runaway(self, spiked: torch.Tensor) -> None:
        """Halve the global weight if the network lights up; the expected failure mode.

        Checked every `guard_every_steps` steps: reading the spike count off the
        GPU is a synchronisation, and doing it every step costs more than the
        propagation.
        """
        self._peak_spikes = torch.maximum(self._peak_spikes, spiked.sum())
        if self.steps % self.cfg.guard_every_steps:
            return
        count = int(self._peak_spikes)
        self._peak_spikes.zero_()
        fraction = count / self.n
        if fraction <= self.cfg.max_spike_fraction or count < self.cfg.min_runaway_spikes:
            return
        self.weight_scale *= 0.5
        self.runaway_events += 1
        log.warning(
            "%d neurons (%.1f%%) spiking at step %d; weight_scale halved to %.4f",
            count,
            100 * fraction,
            self.steps,
            self.weight_scale,
        )


def rates_from_populations(
    n: int,
    drive: dict[str, float],
    populations: dict[str, list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Per-neuron Hz vector from {population name: rate}."""
    rates = torch.zeros(n, dtype=torch.float32, device=device)
    for name, hz in drive.items():
        index = populations.get(name)
        if not index:
            log.warning("population %r is empty or unknown; no drive applied", name)
            continue
        rates[torch.as_tensor(index, dtype=torch.int64, device=device)] = hz
    return rates
