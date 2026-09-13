import dataclasses

import numpy as np
import pytest

from neurofly16px import types as t


def test_frame_is_black_uint8_16x16() -> None:
    f = t.new_frame()
    assert f.shape == (16, 16, 3) and f.dtype == np.uint8 and not f.any()


def test_messages_are_frozen() -> None:
    cmd = t.SteeringCommand(forward=0.5, turn=0.0, mode="walk")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.forward = 1.0  # type: ignore[misc]


def test_fly_state_leg_order() -> None:
    assert t.LEG_NAMES == ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
    fly = t.FlyState(
        t=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        airborne=False,
        legs_down=(True,) * 6,
        wing_phase=0.0,
    )
    assert len(fly.legs_down) == 6


def test_constants() -> None:
    assert t.SILENCE.rms == 0.0 and t.SILENCE.onset is False and t.SILENCE.direction is None
    assert t.IDLE.mode == "idle"


def test_surface_defaults_to_the_floor() -> None:
    fly = t.FlyState(
        t=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        airborne=False,
        legs_down=(True,) * 6,
        wing_phase=0.0,
    )
    assert fly.surface == "floor", "a flat-ground sim needs to say nothing"
    assert t.SURFACES == ("floor", "right", "ceiling", "left")
    assert "air" not in t.SURFACES, "air is a state, not a walkable face"
