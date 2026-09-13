"""Messages exchanged between pipeline stages. Nothing else crosses stage boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["idle", "walk", "fly"]

Surface = Literal["floor", "right", "ceiling", "left", "air"]
"""Which face of the box the fly is on; "air" means in flight.

Flies walk on walls and ceilings, so the panel is a room seen from the side
rather than a floor with sky above it (docs/plans/2026-09-13-phase7-box-world.md).
"""

SURFACES: tuple[Surface, ...] = ("floor", "right", "ceiling", "left")
"""The four walkable faces, counter-clockwise from the bottom."""

LEG_NAMES: tuple[str, ...] = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")


@dataclass(frozen=True)
class AudioFeatures:
    t: float
    """Monotonic time of the audio block, s."""
    rms: float
    """Loudness after slow automatic gain control, 0..1."""
    onset: bool
    """A transient was detected in this block."""
    direction: float | None
    """Azimuth estimate in rad, -pi/2 = left .. +pi/2 = right; None when mono."""


@dataclass(frozen=True)
class SteeringCommand:
    forward: float
    """-1..1, fraction of the maximum forward speed."""
    turn: float
    """-1..1, fraction of the maximum yaw rate; positive = counter-clockwise."""
    mode: Mode


@dataclass(frozen=True)
class FlyState:
    t: float
    """Simulation time, s."""
    x: float
    y: float
    """World position, cm."""
    z: float
    """Height above the floor, cm; 0 while walking."""
    heading: float
    """rad, counter-clockwise from +x."""
    speed: float
    """Ground speed, cm/s."""
    airborne: bool
    legs_down: tuple[bool, bool, bool, bool, bool, bool]
    """Foot contact per leg, ordered as LEG_NAMES."""
    wing_phase: float
    """0..1, meaningful only while airborne."""
    surface: Surface = "floor"
    """Which face she is attached to; "air" while flying.

    Defaults to the floor, which is what a flat-ground sim produces on its own.
    """


Frame = np.ndarray
"""uint8 (16, 16, 3): [row, col, rgb], row 0 is the top of the panel."""

SIDE = 16

SILENCE = AudioFeatures(t=0.0, rms=0.0, onset=False, direction=None)
IDLE = SteeringCommand(forward=0.0, turn=0.0, mode="idle")


def new_frame() -> Frame:
    return np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
