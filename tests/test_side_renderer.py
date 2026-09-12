import math

import numpy as np

from neurofly16px.config import RenderConfig
from neurofly16px.render import side as s
from neurofly16px.types import FlyState

CFG = RenderConfig(arena_cm=8.0, height_cm=2.0, origin_at_center=True)


def fly(**kw) -> FlyState:
    base = dict(
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
    base.update(kw)
    return FlyState(**base)


def px(frame: np.ndarray, row: int, col: int) -> tuple[int, ...]:
    return tuple(int(v) for v in frame[row, col])


def render(**kw) -> np.ndarray:
    return s.SideRenderer(CFG).render(fly(**kw))


def test_standing_fly_sits_on_the_floor_facing_right() -> None:
    frame = render()
    assert px(frame, 15, 0) == s.GROUND and px(frame, 15, 15) == s.GROUND  # floor line
    # centre column for x = 0 with origin_at_center is 8
    assert px(frame, 13, 7) == s.BODY and px(frame, 13, 8) == s.BODY and px(frame, 13, 9) == s.BODY
    assert px(frame, 13, 10) == s.HEAD  # head in front, to the right
    assert px(frame, 14, 7) == s.LEG_DOWN
    assert px(frame, 14, 8) == s.LEG_DOWN
    assert px(frame, 14, 9) == s.LEG_DOWN
    assert px(frame, 12, 7) == s.WING_FOLDED and px(frame, 12, 8) == s.WING_FOLDED
    assert px(frame, 0, 0) == s.BG
    assert frame.dtype == np.uint8 and frame.shape == (16, 16, 3)


def test_heading_backwards_mirrors_the_fly() -> None:
    frame = render(heading=math.pi)
    assert px(frame, 13, 6) == s.HEAD  # head now on the left
    assert px(frame, 13, 9) == s.BODY


def test_heading_across_the_view_still_picks_a_side() -> None:
    assert px(render(heading=math.pi / 4), 13, 10) == s.HEAD
    assert px(render(heading=3 * math.pi / 4), 13, 6) == s.HEAD


def test_lifted_leg_swings_forward_and_dims() -> None:
    frame = render(legs_down=(False, True, True, True, True, True))  # T1 left = front leg up
    assert px(frame, 14, 9) == s.BG  # front foot left the floor
    assert px(frame, 14, 10) == s.LEG_UP  # swinging forward, under the head


def test_hop_lifts_the_whole_fly_off_the_floor() -> None:
    frame = render(z=0.5, airborne=True, wing_phase=0.25)
    lift = round(0.5 * 16 / 2.0)  # 4 rows
    assert px(frame, 13 - lift, 8) == s.BODY_AIRBORNE
    assert px(frame, 14 - lift, 8) == s.LEG_UP  # legs tucked
    assert px(frame, 13, 8) == s.BG  # nothing left on the ground
    assert px(frame, 15, 8) == s.GROUND  # the floor stays


def test_wings_beat_between_two_rows_while_airborne() -> None:
    up = render(z=0.3, airborne=True, wing_phase=0.2)
    down = render(z=0.3, airborne=True, wing_phase=0.7)
    lift = round(0.3 * 16 / 2.0)
    assert px(up, 11 - lift, 8) == s.WING  # up-stroke, clear of the back
    assert px(down, 12 - lift, 8) == s.WING  # down-stroke, just above the thorax
    assert px(down, 13 - lift, 8) == s.BODY_AIRBORNE  # never painted over the body


def test_position_wraps_around_the_panel() -> None:
    frame = render(x=3.9)  # 3.9 + 4 = 7.9 cm of an 8 cm arena -> column 15
    assert px(frame, 13, 14) == s.BODY and px(frame, 13, 15) == s.BODY
    assert px(frame, 13, 0) == s.BODY  # body wraps round the edge
    assert px(frame, 13, 1) == s.HEAD


def test_ground_can_be_switched_off() -> None:
    frame = s.SideRenderer(RenderConfig(show_ground=False)).render(fly())
    assert px(frame, 15, 0) == s.BG


def test_deterministic() -> None:
    a, b = render(heading=0.3, x=1.1), render(heading=0.3, x=1.1)
    assert np.array_equal(a, b)
