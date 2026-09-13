"""Benchmark the LIF brain and run the sugar-GRN sanity check.

    uv run python scripts/bench_brain.py [--seconds 1.0]

Real time is 10,000 steps/s at dt = 0.1 ms; anything less means the brain runs in
slow motion, which the main loop tolerates and logs (PLAN.md 6.5).
"""

import argparse
import json
import logging
import time

import torch

from neurofly16px.brain.lif import LifBrain, rates_from_populations
from neurofly16px.config import BrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("bench")

WATCH = ("mn_proboscis", "dn_walking", "dn_all", "jo_sound_left")


def count_spikes(brain: LifBrain, watch: dict[str, torch.Tensor], steps: int) -> dict[str, int]:
    totals = {name: torch.zeros((), dtype=torch.int64, device=brain.device) for name in watch}
    for _ in range(steps):
        spiked = brain.step()
        for name, index in watch.items():
            totals[name] += spiked[index].sum()
    return {name: int(value) for name, value in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1.0, help="simulated seconds per condition")
    ap.add_argument("--rate", type=float, default=100.0, help="stimulation rate, Hz")
    args = ap.parse_args()

    cfg = BrainConfig()
    brain = LifBrain.load(cfg)
    with open(cfg.populations_path) as f:
        populations = {name: entry["indices"] for name, entry in json.load(f).items()}
    watch = {
        name: torch.as_tensor(populations[name], dtype=torch.int64, device=brain.device)
        for name in WATCH
        if populations.get(name)
    }
    steps = round(args.seconds * 1000 / cfg.dt_ms)

    drive = rates_from_populations(brain.n, {"sugar_grn": args.rate}, populations, brain.device)
    brain.set_drive(drive)
    for _ in range(300):
        brain.step()
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        brain.step()
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    rate = steps / wall
    log.info(
        "%.0f steps/s = %.2fx real time (%.0f us/step) on %s",
        rate,
        rate / (1000 / cfg.dt_ms),
        1e6 / rate,
        brain.device,
    )

    brain.reset()
    brain.set_drive(drive)
    stimulated = count_spikes(brain, watch, steps)
    brain.reset()
    brain.set_drive(None)
    control = count_spikes(brain, watch, steps)

    log.info("spikes in %.1f simulated second(s), sugar GRNs at %.0f Hz:", args.seconds, args.rate)
    for name in watch:
        size = len(populations[name])
        log.info(
            "  %-16s %7d  (%5.1f Hz per neuron)   control: %d",
            name,
            stimulated[name],
            stimulated[name] / size / args.seconds,
            control[name],
        )
    log.info("runaway events: %d, weight_scale %.4f", brain.runaway_events, brain.weight_scale)


if __name__ == "__main__":
    main()
