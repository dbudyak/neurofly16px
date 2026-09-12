"""Body simulation stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import FlyState, SteeringCommand


class FlySim(Protocol):
    control_dt: float
    """Seconds of simulated time advanced by one step()."""

    def reset(self) -> FlyState: ...

    def step(self, cmd: SteeringCommand) -> FlyState: ...
