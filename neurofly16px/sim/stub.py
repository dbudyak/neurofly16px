"""Kinematic stand-in for the physics: integrates speed and yaw, fakes a tripod gait."""

from __future__ import annotations

import math

from neurofly16px.config import HopConfig, StubSimConfig
from neurofly16px.sim.hop import HopOverlay
from neurofly16px.types import FlyState, SteeringCommand

CONTROL_DT = 0.002


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return a - 2 * math.pi * math.floor((a + math.pi) / (2 * math.pi))


def tripod_legs(phase: float, moving: bool) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Foot contact per leg (T1L, T1R, T2L, T2R, T3L, T3R). Tripod A = T1L, T2R, T3L."""
    if not moving:
        return (True, True, True, True, True, True)
    a = phase < 0.5
    return (a, not a, not a, a, a, not a)


class StubSim:
    """Same interface and control rate as the flybody sim, without the physics."""

    control_dt = CONTROL_DT

    def __init__(self, cfg: StubSimConfig, hop: HopConfig) -> None:
        self._cfg = cfg
        self._hop = HopOverlay(hop.duration_s, hop.height_cm, hop.wingbeat_hz)
        self.reset()

    def reset(self) -> FlyState:
        self._t = 0.0
        self._x = self._y = 0.0
        self._heading = 0.0
        self._phase = 0.0
        self._speed = 0.0
        return self._state()

    def step(self, cmd: SteeringCommand) -> FlyState:
        dt = self.control_dt
        if cmd.mode == "fly":
            self._hop.trigger(self._t)
        forward = 0.0 if cmd.mode == "idle" else max(-1.0, min(1.0, cmd.forward))
        turn = max(-1.0, min(1.0, cmd.turn))
        v = forward * self._cfg.v_max_cm_s
        w = turn * self._cfg.w_max_rad_s
        self._heading = wrap_angle(self._heading + w * dt)
        self._x += v * math.cos(self._heading) * dt
        self._y += v * math.sin(self._heading) * dt
        self._phase = (self._phase + abs(v) * dt / self._cfg.stride_cm) % 1.0
        self._speed = abs(v)
        self._t += dt
        return self._state()

    def _state(self) -> FlyState:
        z, airborne, wing = self._hop.sample(self._t)
        return FlyState(
            t=self._t,
            x=self._x,
            y=self._y,
            z=z,
            heading=self._heading,
            speed=self._speed,
            airborne=airborne,
            legs_down=tripod_legs(self._phase, self._speed > 0.0),
            wing_phase=wing,
        )
