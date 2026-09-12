"""FlyState -> 16x16 sprite, seen from the side.

The panel is a vertical slice of the world: columns are world x, rows are height
above the floor. The fly walks along the bottom on a dim floor line and lifts off
it when she hops, which is what you see on a desk toy from across the room. World
y (depth into the screen) is not drawn; it only shows up through the heading,
which decides whether she faces left or right.

Layout, facing right, with the floor on the bottom row:

        row 11   . . w w . .        up-stroke (airborne only)
        row 12   . . w w . .        wings: folded when walking, down-stroke when airborne
        row 13   . a a t h .        abdomen, thorax, head
        row 14   . l . l . l        legs: bright planted, dim while swinging
        row 15   # # # # # #        floor

The whole fly shifts up by `z` while airborne.
"""

from __future__ import annotations

import math

from neurofly16px.config import RenderConfig
from neurofly16px.types import SIDE, FlyState, Frame, new_frame

BG = (0, 0, 0)
BODY = (255, 120, 0)
BODY_AIRBORNE = (255, 60, 30)
HEAD = (255, 235, 170)
LEG_DOWN = (200, 200, 200)
LEG_UP = (60, 60, 60)
WING_FOLDED = (40, 70, 110)
WING = (140, 200, 255)
GROUND = (35, 25, 15)

_BODY_ALONG = (-1, 0, 1)
"""Abdomen, thorax and the segment in front of it, in body lengths from the centre."""
_HEAD_ALONG = 2
_LEG_ALONG = (-1, 0, 1)
"""Hind, middle and front leg, under the matching body segment."""
_NEAR_LEGS = (4, 2, 0)
"""FlyState.legs_down indices for the near side (T3, T2, T1 = hind, middle, front)."""


class SideRenderer:
    """The renderer the Ditoo actually shows; `SpriteRenderer` is the top-down one."""

    def __init__(self, cfg: RenderConfig) -> None:
        self._arena = cfg.arena_cm
        self._offset = cfg.arena_cm / 2 if cfg.origin_at_center else 0.0
        self._height_cm = cfg.height_cm
        self._show_ground = cfg.show_ground
        self.ground_row = SIDE - 1
        self.feet_row = self.ground_row - 1
        self.body_row = self.ground_row - 2
        self.wing_row = self.ground_row - 3

    def render(self, fly: FlyState) -> Frame:
        frame = new_frame()
        if self._show_ground:
            frame[self.ground_row, :] = GROUND

        column = int(((fly.x + self._offset) % self._arena) * (SIDE / self._arena))
        facing = 1 if math.cos(fly.heading) >= 0.0 else -1
        lift = self._lift(fly.z)
        body_row = self.body_row - lift

        def put(col_offset: int, row: int, colour: tuple[int, int, int]) -> None:
            if 0 <= row < SIDE:
                frame[row, (column + facing * col_offset) % SIDE] = colour

        self._draw_legs(fly, put, body_row + 1, facing)
        self._draw_wings(fly, put, body_row - 1)
        body = BODY_AIRBORNE if fly.airborne else BODY
        for along in _BODY_ALONG:
            put(along, body_row, body)
        put(_HEAD_ALONG, body_row, HEAD)
        return frame

    # --- parts --------------------------------------------------------------

    def _draw_legs(self, fly: FlyState, put, row: int, facing: int) -> None:
        for along, leg in zip(_LEG_ALONG, _NEAR_LEGS, strict=True):
            if fly.airborne:
                put(along, row, LEG_UP)  # tucked under the body
            elif fly.legs_down[leg]:
                put(along, row, LEG_DOWN)
            else:
                put(along + 1, row, LEG_UP)  # swinging forward

    def _draw_wings(self, fly: FlyState, put, row: int) -> None:
        if not fly.airborne:
            put(-1, row, WING_FOLDED)
            put(0, row, WING_FOLDED)
            return
        # Both strokes stay above the back, or the body would paint over them:
        # the pair blinks between two rows at wingbeat_hz, which reads as flapping.
        stroke = row - 1 if fly.wing_phase < 0.5 else row
        put(-1, stroke, WING)
        put(0, stroke, WING)

    def _lift(self, z_cm: float) -> int:
        return int(round(max(0.0, z_cm) * SIDE / self._height_cm))
