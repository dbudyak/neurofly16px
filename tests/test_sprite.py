import math

import numpy as np

from neurofly16px.config import RenderConfig
from neurofly16px.render import sprite as s
from neurofly16px.types import FlyState

CORNER = RenderConfig(arena_cm=8.0, origin_at_center=False)


def fly(**kw) -> FlyState:
    base = dict(
        t=0.0,
        x=4.0,
        y=4.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        airborne=False,
        legs_down=(True,) * 6,
        wing_phase=0.0,
    )
    base.update(kw)
    return FlyState(**base)


def px(frame: np.ndarray, col: int, y: int) -> tuple[int, ...]:
    return tuple(int(v) for v in frame[15 - y, col])


def test_heading_zero_layout() -> None:
    frame = s.SpriteRenderer(CORNER).render(fly())
    # centre pixel: 4 cm / 8 cm * 16 = 8
    assert px(frame, 8, 8) == s.BODY and px(frame, 7, 8) == s.BODY and px(frame, 9, 8) == s.BODY
    assert px(frame, 10, 8) == s.HEAD
    # front-left leg (T1L, down): c + h + 2n = (9, 10); front-right: (9, 6)
    assert px(frame, 9, 10) == s.LEG_DOWN and px(frame, 9, 6) == s.LEG_DOWN
    assert px(frame, 8, 10) == s.LEG_DOWN and px(frame, 7, 10) == s.LEG_DOWN
    assert px(frame, 0, 0) == s.BG
    assert frame.dtype == np.uint8 and frame.shape == (16, 16, 3)


def test_raised_leg_hugs_the_body() -> None:
    frame = s.SpriteRenderer(CORNER).render(
        fly(legs_down=(False, True, True, True, True, True))
    )
    assert px(frame, 9, 9) == s.LEG_UP and px(frame, 9, 10) == s.BG


def test_heading_rotates_and_wraps() -> None:
    frame = s.SpriteRenderer(CORNER).render(fly(x=7.9, y=4.0, heading=math.pi / 2))
    # centre col = 15 (7.9/8*16 = 15.8 -> 15); head 2 px up: (15, 10)
    assert px(frame, 15, 10) == s.HEAD
    # left normal for heading +y is -x: front-left leg at c + h + 2n = (13, 9)
    assert px(frame, 13, 9) == s.LEG_DOWN


def test_airborne_shows_wings_and_changes_body_colour() -> None:
    r = s.SpriteRenderer(CORNER)
    up = r.render(fly(airborne=True, z=0.5, wing_phase=0.25))
    assert px(up, 8, 8) == s.BODY_AIRBORNE and px(up, 7, 10) == s.WING and px(up, 7, 6) == s.WING
    down = r.render(fly(airborne=True, z=0.5, wing_phase=0.75))
    assert px(down, 7, 10) != s.WING


def test_deterministic() -> None:
    r = s.SpriteRenderer(CORNER)
    a, b = r.render(fly(heading=0.3)), r.render(fly(heading=0.3))
    assert np.array_equal(a, b)


def test_origin_at_center_puts_the_fly_mid_panel() -> None:
    frame = s.SpriteRenderer(RenderConfig(arena_cm=8.0)).render(fly(x=0.0, y=0.0))
    assert px(frame, 8, 8) == s.BODY and px(frame, 10, 8) == s.HEAD
