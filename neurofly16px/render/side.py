"""FlyState -> 16x16 sprite of a fly in a box, seen from the side.

The panel is a room: floor, two walls, a ceiling. The fly is drawn in a local
frame -- `u` along the surface she is standing on, `v` pointing out of it into the
room -- and that frame is rotated per surface, so she hangs upside down from the
ceiling and stands sideways on a wall with her legs against it. In flight the
same body is drawn along her velocity with both wings beating.

Layout on the floor, facing right (`u` right, `v` up):

        v=3   . . w w . .        up-stroke (airborne only)
        v=2   . . w w . .        wings: folded walking, down-stroke flying
        v=1   . a a t h .        abdomen, thorax, head
        v=0   . l . l . l        legs: bright planted, dim while swinging
        wall  # # # # # #        the room's outline
"""

from __future__ import annotations

import math

from neurofly16px.config import RenderConfig
from neurofly16px.types import SIDE, FlyState, Frame, Surface, new_frame

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
"""Abdomen, thorax and the segment in front, in body lengths from the centre."""
_HEAD_ALONG = 2
_LEG_ALONG = (-1, 0, 1)
"""Hind, middle and front leg, under the matching body segment."""
_NEAR_LEGS = (4, 2, 0)
"""FlyState.legs_down indices for the near side (T3, T2, T1)."""

# Local (u, v) -> panel (column, row) per surface. u runs along the surface in the
# direction of increasing arc length, v points into the room.
_FRAMES: dict[Surface, tuple[tuple[int, int], tuple[int, int]]] = {
    "floor": ((1, 0), (0, -1)),
    "right": ((0, -1), (-1, 0)),
    "ceiling": ((-1, 0), (0, 1)),
    "left": ((0, 1), (1, 0)),
}
"""Each entry is (u as (dcol, drow), v as (dcol, drow)); row 0 is the top."""


class SideRenderer:
    """The renderer the Ditoo shows: a fly walking the walls of a 16x16 room."""

    def __init__(self, cfg: RenderConfig, box_cm: float | None = None) -> None:
        self._box = box_cm if box_cm is not None else cfg.arena_cm
        self._show_ground = cfg.show_ground
        self.ground_row = SIDE - 1

    def render(self, fly: FlyState) -> Frame:
        frame = new_frame()
        if self._show_ground:
            self._draw_room(frame)
        if fly.surface == "air":
            self._draw_flying(frame, fly)
        else:
            self._draw_standing(frame, fly)
        return frame

    # --- the room -----------------------------------------------------------

    def _draw_room(self, frame: Frame) -> None:
        frame[SIDE - 1, :] = GROUND
        frame[0, :] = GROUND
        frame[:, 0] = GROUND
        frame[:, SIDE - 1] = GROUND

    # --- the fly ------------------------------------------------------------

    def _draw_standing(self, frame: Frame, fly: FlyState) -> None:
        u_axis, v_axis = _FRAMES[fly.surface]
        facing = self._facing(fly)
        col, row = self._anchor(fly, facing)

        def put(along: int, out: int, colour: tuple[int, int, int]) -> None:
            c = col + u_axis[0] * along * facing + v_axis[0] * out
            r = row + u_axis[1] * along * facing + v_axis[1] * out
            if 0 <= r < SIDE and 0 <= c < SIDE:
                frame[r, c] = colour

        for along, leg in zip(_LEG_ALONG, _NEAR_LEGS, strict=True):
            if fly.legs_down[leg]:
                put(along, 0, LEG_DOWN)
            else:
                put(along + 1, 0, LEG_UP)  # swinging forward
        put(-1, 2, WING_FOLDED)
        put(0, 2, WING_FOLDED)
        for along in _BODY_ALONG:
            put(along, 1, BODY)
        put(_HEAD_ALONG, 1, HEAD)

    def _draw_flying(self, frame: Frame, fly: FlyState) -> None:
        col, row = self._to_pixel(fly.x, fly.y)
        # Keep the whole fly on the panel: at the walls her head would otherwise
        # stick out past the edge and vanish.
        margin = _HEAD_ALONG
        col = max(margin, min(SIDE - 1 - margin, col))
        row = max(margin, min(SIDE - 1 - margin, row))
        hx, hy = math.cos(fly.heading), math.sin(fly.heading)

        def put(along: float, out: float, colour: tuple[int, int, int]) -> None:
            c = col + _round(along * hx - out * hy)
            r = row - _round(along * hy + out * hx)
            if 0 <= r < SIDE and 0 <= c < SIDE:
                frame[r, c] = colour

        stroke = 1 if fly.wing_phase < 0.5 else 2
        put(-1, stroke, WING)
        put(0, stroke, WING)
        put(-1, -stroke, WING)
        put(0, -stroke, WING)
        for along in _BODY_ALONG:
            put(along, 0, BODY_AIRBORNE)
        put(_HEAD_ALONG, 0, HEAD)

    # --- geometry -----------------------------------------------------------

    def _to_pixel(self, x: float, y: float) -> tuple[int, int]:
        scale = (SIDE - 1) / self._box
        col = int(round(min(self._box, max(0.0, x)) * scale))
        row = SIDE - 1 - int(round(min(self._box, max(0.0, y)) * scale))
        return col, row

    def _anchor(self, fly: FlyState, facing: int) -> tuple[int, int]:
        """Where her feet touch the surface, kept far enough from the corner to fit.

        She is four pixels long; without this her head falls off the panel every
        time she rounds a corner, which reads as a glitch rather than as a fly.
        """
        col, row = self._to_pixel(fly.x, fly.y)
        u_axis, _ = _FRAMES[fly.surface]
        ahead = _HEAD_ALONG * facing
        behind = _BODY_ALONG[0] * facing
        if u_axis[0]:  # she runs along the columns
            col = _clamp_span(col, u_axis[0] * ahead, u_axis[0] * behind)
        else:  # along the rows
            row = _clamp_span(row, u_axis[1] * ahead, u_axis[1] * behind)
        return col, row

    @staticmethod
    def _facing(fly: FlyState) -> int:
        """+1 when she walks along increasing arc length, -1 when she walks back."""
        tangent = {
            "floor": (1.0, 0.0),
            "right": (0.0, 1.0),
            "ceiling": (-1.0, 0.0),
            "left": (0.0, -1.0),
        }[fly.surface]
        along = math.cos(fly.heading) * tangent[0] + math.sin(fly.heading) * tangent[1]
        return 1 if along >= 0 else -1


def _clamp_span(value: int, ahead: int, behind: int) -> int:
    """Shift `value` so that both ends of the body stay on the panel."""
    low = -min(0, ahead, behind)
    high = SIDE - 1 - max(0, ahead, behind)
    return max(low, min(high, value))


def _round(v: float) -> int:
    """Round half away from zero, so offsets either side stay symmetric."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))
