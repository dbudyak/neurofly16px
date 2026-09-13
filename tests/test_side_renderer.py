"""The side view: a fly standing on each wall of the room, and flying between them."""

import math

import numpy as np

from neurofly16px.config import RenderConfig
from neurofly16px.render import side as s
from neurofly16px.types import FlyState

BOX = 4.0
CFG = RenderConfig(arena_cm=BOX)


def fly(**kw) -> FlyState:
    base = dict(
        t=0.0,
        x=2.0,
        y=0.0,
        z=0.0,
        heading=0.0,
        speed=1.0,
        airborne=False,
        legs_down=(True,) * 6,
        wing_phase=0.0,
        surface="floor",
    )
    base.update(kw)
    return FlyState(**base)


def render(**kw) -> np.ndarray:
    return s.SideRenderer(CFG, box_cm=BOX).render(fly(**kw))


def px(frame: np.ndarray, row: int, col: int) -> tuple[int, ...]:
    return tuple(int(v) for v in frame[row, col])


def find(frame: np.ndarray, colour: tuple[int, int, int]) -> list[tuple[int, int]]:
    rows, cols = np.nonzero((frame == np.array(colour, np.uint8)).all(-1))
    return sorted(zip(rows.tolist(), cols.tolist(), strict=True))


def test_the_room_is_drawn_as_an_outline() -> None:
    frame = render()
    assert px(frame, 15, 0) == s.GROUND and px(frame, 0, 15) == s.GROUND
    assert px(frame, 7, 0) == s.GROUND and px(frame, 7, 15) == s.GROUND, "both walls"
    assert px(frame, 7, 7) == s.BG, "the room is empty in the middle"


def test_standing_on_the_floor_faces_right_with_legs_down() -> None:
    frame = render(x=2.0, y=0.0, heading=0.0)
    # x = 2 of 4 cm -> column 8 (rounded on a 15-pixel span); feet on the bottom row
    head = find(frame, s.HEAD)[0]
    body = find(frame, s.BODY)
    legs = find(frame, s.LEG_DOWN)
    assert all(row == 14 for row, _ in body), "the body sits one pixel above her feet"
    assert all(row == 15 for row, _ in legs), "and her feet are on the floor"
    assert head[1] > max(col for _, col in body), "head in front, to the right"


def test_walking_the_other_way_mirrors_her() -> None:
    frame = render(heading=math.pi)
    head = find(frame, s.HEAD)[0]
    body = find(frame, s.BODY)
    assert head[1] < min(col for _, col in body), "head now on the left"


def test_on_the_ceiling_she_hangs_upside_down() -> None:
    frame = render(surface="ceiling", x=2.0, y=BOX, heading=math.pi)
    body = find(frame, s.BODY)
    legs = find(frame, s.LEG_DOWN)
    assert all(row == 1 for row, _ in body), "body below the ceiling line"
    assert all(row == 0 for row, _ in legs), "feet against the ceiling"


def test_on_the_right_wall_she_stands_sideways() -> None:
    frame = render(surface="right", x=BOX, y=2.0, heading=math.pi / 2)
    body = find(frame, s.BODY)
    legs = find(frame, s.LEG_DOWN)
    assert all(col == 14 for _, col in body), "body one pixel in from the wall"
    assert all(col == 15 for _, col in legs), "feet against the wall"


def test_on_the_left_wall_too() -> None:
    frame = render(surface="left", x=0.0, y=2.0, heading=-math.pi / 2)
    body = find(frame, s.BODY)
    legs = find(frame, s.LEG_DOWN)
    assert all(col == 1 for _, col in body)
    assert all(col == 0 for _, col in legs)


def test_a_lifted_leg_swings_forward_and_dims() -> None:
    frame = render(legs_down=(False, True, True, True, True, True))  # front leg up
    assert find(frame, s.LEG_UP), "the lifted leg is drawn, dimmer"
    assert len(find(frame, s.LEG_DOWN)) == 2, "only two feet left on the floor"


def test_the_whole_fly_stays_on_the_panel_at_a_corner() -> None:
    """Her head used to fall off the edge every time she rounded one."""
    for x in (0.0, 0.2, 3.8, BOX):
        frame = render(x=x, y=0.0, heading=0.0)
        assert len(find(frame, s.BODY)) == 3, f"body clipped at x={x}"
        assert len(find(frame, s.HEAD)) == 1, f"head clipped at x={x}"


def test_flying_draws_her_in_the_middle_of_the_room_with_both_wings() -> None:
    frame = render(surface="air", x=2.0, y=2.0, airborne=True, heading=math.pi / 2, wing_phase=0.2)
    body = find(frame, s.BODY_AIRBORNE)
    wings = find(frame, s.WING)
    assert body, "she is drawn"
    assert all(2 < row < 13 for row, _ in body), "away from the walls"
    assert len(wings) == 4, "a wing either side, beating"
    assert not find(frame, s.LEG_DOWN), "nothing planted while airborne"


def test_flight_direction_rotates_the_body() -> None:
    right = render(surface="air", x=2.0, y=2.0, airborne=True, heading=0.0)
    up = render(surface="air", x=2.0, y=2.0, airborne=True, heading=math.pi / 2)
    assert find(right, s.HEAD)[0] != find(up, s.HEAD)[0]
    head_right, head_up = find(right, s.HEAD)[0], find(up, s.HEAD)[0]
    assert head_right[1] > 8, "flying right, head to the right"
    assert head_up[0] < 8, "flying up, head above"


def test_deterministic() -> None:
    a, b = render(heading=0.3, x=1.1), render(heading=0.3, x=1.1)
    assert np.array_equal(a, b)


def test_a_fly_against_the_wall_is_still_drawn_whole() -> None:
    for x, y in ((0.0, 2.0), (BOX, 2.0), (2.0, 0.0), (2.0, BOX)):
        frame = render(surface="air", x=x, y=y, airborne=True, heading=0.0)
        assert len(find(frame, s.BODY_AIRBORNE)) == 3, f"body clipped at ({x}, {y})"
        assert len(find(frame, s.HEAD)) == 1, f"head clipped at ({x}, {y})"
