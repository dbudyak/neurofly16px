"""Fixed-step main loop: the sim tracks the wall clock, behaviour and display are rate-limited."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from neurofly16px.audio.base import AudioSource
from neurofly16px.behavior.base import Behavior
from neurofly16px.config import LoopConfig
from neurofly16px.device.base import Display
from neurofly16px.record import Recorder
from neurofly16px.render.base import Renderer
from neurofly16px.sim.base import FlySim
from neurofly16px.types import IDLE

log = logging.getLogger(__name__)


@dataclass
class LoopStats:
    sim_steps: int = 0
    frames: int = 0
    behavior_updates: int = 0
    dropped_steps: int = 0
    wall_s: float = 0.0


def run(
    cfg: LoopConfig,
    *,
    audio: AudioSource,
    behavior: Behavior,
    sim: FlySim,
    renderer: Renderer,
    display: Display,
    duration_s: float | None = None,
    recorder: Recorder | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LoopStats:
    """Run until `duration_s` of wall time has passed (or forever when None)."""
    stats = LoopStats()
    dt = sim.control_dt
    behavior_period = 1.0 / cfg.behavior_hz
    frame_period = 1.0 / cfg.fps
    audio.start()
    fly = sim.reset()
    cmd = IDLE
    t0 = clock()
    sim_time = 0.0
    next_behavior = 0.0
    next_frame = 0.0
    last_lag_log = 0.0
    try:
        while True:
            now = clock() - t0
            if duration_s is not None and now >= duration_s:
                break
            if now >= next_behavior:
                cmd = behavior.update(audio.latest(), fly)
                stats.behavior_updates += 1
                while next_behavior <= now:
                    next_behavior += behavior_period
            steps = 0
            while sim_time + dt <= now and steps < cfg.max_catchup_steps:
                fly = sim.step(cmd)
                sim_time += dt
                steps += 1
            if sim_time + dt <= (clock() - t0):
                behind = int(((clock() - t0) - sim_time) / dt)
                stats.dropped_steps += behind
                sim_time += behind * dt
                if now - last_lag_log > 5.0:
                    log.warning("sim cannot keep up: dropped %d steps so far", stats.dropped_steps)
                    last_lag_log = now
            if now >= next_frame:
                frame = renderer.render(fly)
                display.show(frame)
                if recorder is not None:
                    recorder.add(fly, cmd, frame)
                stats.frames += 1
                next_frame = now + frame_period
            wake = min(next_behavior, sim_time + dt, next_frame)
            delay = wake - (clock() - t0)
            if delay > 0:
                sleep(delay)
    finally:
        audio.stop()
        display.close()
        stats.wall_s = clock() - t0
        stats.sim_steps = round(sim_time / dt) - stats.dropped_steps
    return stats
