"""The brain in its own process.

The LIF model is launch-bound on the CPU side (docs/banc.md) and so is MuJoCo, so
sharing one process would slow both. The brain therefore runs separately and the
two talk through one-slot mailboxes: the body posts the latest audio features, the
brain posts the latest steering command plus the activity the viewer draws.
Neither side ever waits for the other; if the brain falls behind, the body keeps
its own real-time clock and the lag is reported.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field

import numpy as np

from neurofly16px.config import BrainConfig
from neurofly16px.types import IDLE, SILENCE, AudioFeatures, SteeringCommand

log = logging.getLogger(__name__)

HEAT_WIDTH = 160
HEAT_HEIGHT = 120


@dataclass
class BrainActivity:
    """One window of brain state, for the viewer and the body."""

    t: float
    command: SteeringCommand
    rates_hz: dict[str, float] = field(default_factory=dict)
    heat: np.ndarray | None = None
    """Spike counts per heat-map pixel over the window, shape (HEAT_HEIGHT, HEAT_WIDTH)."""
    steps: int = 0
    sim_time_s: float = 0.0
    wall_time_s: float = 0.0
    weight_scale: float = 1.0
    runaway_events: int = 0

    @property
    def speed_ratio(self) -> float:
        """Simulated seconds per wall second; 1.0 is real time."""
        return self.sim_time_s / self.wall_time_s if self.wall_time_s else 0.0


def _latest(q: mp.Queue):
    """Drain a queue and return the last item, or None."""
    item = None
    while True:
        try:
            item = q.get_nowait()
        except (queue.Empty, OSError):
            return item


def _publish(q: mp.Queue, item) -> None:
    """Replace whatever is waiting: the newest value is the only one that matters."""
    _latest(q)
    try:
        q.put_nowait(item)
    except (queue.Full, OSError):
        pass


def brain_loop(cfg: BrainConfig, audio_q: mp.Queue, out_q: mp.Queue, stop: mp.Event) -> None:
    """Run the LIF model until `stop`, publishing one BrainActivity per window."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s brain: %(message)s")
    import torch

    from neurofly16px.brain.anatomy import Anatomy
    from neurofly16px.brain.lif import LifBrain, rates_from_populations
    from neurofly16px.brain.readout import DescendingReadout
    from neurofly16px.brain.stimulus import JohnstonStimulus

    brain = LifBrain.load(cfg)
    anatomy = Anatomy.load(cfg.anatomy_path, cfg.populations_path)
    with open(cfg.populations_path) as f:
        populations = {name: entry["indices"] for name, entry in json.load(f).items()}
    watched = {
        name: torch.as_tensor(index, dtype=torch.int64, device=brain.device)
        for name, index in populations.items()
        if index
    }
    bins = torch.as_tensor(
        anatomy.soma_bins(HEAT_WIDTH, HEAT_HEIGHT), dtype=torch.int64, device=brain.device
    )
    drawn = bins >= 0
    bin_index = bins.clamp(min=0)

    viewer = None
    if cfg.viewer:
        from neurofly16px.viewer.server import ViewerServer

        viewer = ViewerServer(cfg.viewer_host, cfg.viewer_port)
        viewer.start()
    viewer_every = max(1, round(1000 / cfg.window_ms / max(cfg.viewer_hz, 0.1)))
    window_count = 0
    ceiling = 1

    stimulus = JohnstonStimulus(cfg)
    readout = DescendingReadout(cfg)
    steps_per_window = max(1, round(cfg.window_ms / cfg.dt_ms))
    window_s = cfg.window_ms / 1000.0
    audio = SILENCE
    sim_time = 0.0
    started = time.monotonic()
    log.info("brain ready: %d neurons on %s", brain.n, brain.device)

    while not stop.is_set():
        newest = _latest(audio_q)
        if newest is not None:
            audio = newest
        drive = stimulus.rates(audio, sim_time)
        brain.set_drive(rates_from_populations(brain.n, drive, populations, brain.device))

        # One accumulator vector for the whole window: summing each population, or
        # binning the heat map, on every step costs more than the model itself.
        fired = torch.zeros(brain.n, dtype=torch.int32, device=brain.device)
        for _ in range(steps_per_window):
            fired += brain.step()
        sim_time += window_s

        counts = {name: fired[index].sum() for name, index in watched.items()}
        heat = torch.zeros(HEAT_HEIGHT * HEAT_WIDTH, dtype=torch.int32, device=brain.device)
        heat.index_add_(0, bin_index, torch.where(drawn, fired, torch.zeros_like(fired)))
        rates = {
            name: int(value) / len(populations[name]) / window_s for name, value in counts.items()
        }
        # Hold times (escape, idle) are wall-clock: they shape what happens on the
        # desk, and the brain's own clock runs at a fraction of real time.
        command = readout.update(rates, time.monotonic())
        heat_image = heat.reshape(HEAT_HEIGHT, HEAT_WIDTH).cpu().numpy()

        window_count += 1
        if viewer is not None and window_count % viewer_every == 0:
            flat = heat_image.ravel()
            # A slowly falling ceiling keeps the colours steady between bursts.
            ceiling = max(int(flat.max()), 1, int(ceiling * 0.9))
            viewer.publish(
                {
                    "heat": flat.astype(int).tolist(),
                    "ceiling": ceiling,
                    "rates": {k: round(v, 3) for k, v in rates.items()},
                    "mode": command.mode,
                    "forward": round(command.forward, 3),
                    "turn": round(command.turn, 3),
                    "sim_time": round(sim_time, 2),
                    "ratio": round(sim_time / max(time.monotonic() - started, 1e-9), 3),
                    "spiking": int(fired.sum()),
                    "weight_scale": round(brain.weight_scale, 4),
                }
            )
        _publish(
            out_q,
            BrainActivity(
                t=sim_time,
                command=command,
                rates_hz=rates,
                heat=heat_image,
                steps=brain.steps,
                sim_time_s=sim_time,
                wall_time_s=time.monotonic() - started,
                weight_scale=brain.weight_scale,
                runaway_events=brain.runaway_events,
            ),
        )
    log.info("brain stopped after %.1f simulated seconds", sim_time)


class BrainProcess:
    """Owns the child process and the two mailboxes."""

    def __init__(self, cfg: BrainConfig, context: str = "spawn") -> None:
        self._cfg = cfg
        self._ctx = mp.get_context(context)
        self._audio_q: mp.Queue = self._ctx.Queue(maxsize=4)
        self._out_q: mp.Queue = self._ctx.Queue(maxsize=4)
        self._stop = self._ctx.Event()
        self._process: mp.Process | None = None
        self.latest = BrainActivity(t=0.0, command=IDLE)

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = self._ctx.Process(
            target=brain_loop,
            args=(self._cfg, self._audio_q, self._out_q, self._stop),
            name="brain",
            daemon=True,
        )
        self._process.start()
        log.info("brain process %d started", self._process.pid)

    def send_audio(self, audio: AudioFeatures) -> None:
        _publish(self._audio_q, audio)

    def poll(self) -> BrainActivity:
        """Latest activity, or the previous one; never blocks."""
        newest = _latest(self._out_q)
        if newest is not None:
            self.latest = newest
        return self.latest

    def command(self) -> SteeringCommand:
        return self.poll().command

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                log.warning("brain process did not stop; terminating")
                self._process.terminate()
            self._process = None
