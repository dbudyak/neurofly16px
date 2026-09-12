"""Messages exchanged between pipeline stages. Nothing else crosses stage boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["idle", "walk", "fly"]

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


Frame = np.ndarray
"""uint8 (16, 16, 3): [row, col, rgb], row 0 is the top of the panel."""

SIDE = 16

SILENCE = AudioFeatures(t=0.0, rms=0.0, onset=False, direction=None)
IDLE = SteeringCommand(forward=0.0, turn=0.0, mode="idle")


def new_frame() -> Frame:
    return np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
