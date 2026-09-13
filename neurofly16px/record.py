"""Recording a run: the frames that were displayed, and the state behind each one."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from neurofly16px.types import FlyState, Frame, SteeringCommand

log = logging.getLogger(__name__)

STATE_DTYPE = np.dtype(
    [
        ("t", "f4"),
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("heading", "f4"),
        ("speed", "f4"),
        ("airborne", "?"),
        ("wing_phase", "f4"),
        ("legs_down", "?", (6,)),
    ]
)
COMMAND_DTYPE = np.dtype([("forward", "f4"), ("turn", "f4"), ("mode", "U5")])


class Recorder:
    """Collects one row per displayed frame; `save` writes a compressed npz."""

    def __init__(self, max_frames: int = 200_000) -> None:
        self._max = max_frames
        self._frames: list[Frame] = []
        self._states: list[tuple] = []
        self._commands: list[tuple] = []
        self.overflowed = False

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, fly: FlyState, cmd: SteeringCommand, frame: Frame) -> None:
        if len(self._frames) >= self._max:
            if not self.overflowed:
                log.warning("recording capped at %d frames; the rest is dropped", self._max)
                self.overflowed = True
            return
        self._frames.append(frame.copy())
        self._states.append(
            (
                fly.t,
                fly.x,
                fly.y,
                fly.z,
                fly.heading,
                fly.speed,
                fly.airborne,
                fly.wing_phase,
                fly.legs_down,
            )
        )
        self._commands.append((cmd.forward, cmd.turn, cmd.mode))

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = (
            np.stack(self._frames)
            if self._frames
            else np.zeros((0, 16, 16, 3), np.uint8)
        )
        np.savez_compressed(
            out,
            frames=frames,
            states=np.array(self._states, dtype=STATE_DTYPE),
            commands=np.array(self._commands, dtype=COMMAND_DTYPE),
        )
        log.info("recorded %d frames to %s", len(self._frames), out)
        return out
