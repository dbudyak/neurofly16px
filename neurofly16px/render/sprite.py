"""FlyState -> 16x16 sprite: body, head, six leg ticks, wings when airborne.

Not to scale: the real fly is 2.5 mm long; the sprite is 5 px so posture, heading
and gait stay visible. World -> pixel: a fixed arena of `arena_cm` mapped to 16 px
with toroidal wrap.
"""

from __future__ import annotations

import math

from neurofly16px.config import RenderConfig
from neurofly16px.types import SIDE, FlyState, Frame, new_frame

BG = (0, 0, 0)
BODY = (255, 120, 0)
BODY_AIRBORNE = (255, 40, 40)
HEAD = (255, 255, 200)
LEG_DOWN = (210, 210, 210)
LEG_UP = (70, 70, 70)
WING = (120, 190, 255)

_LEG_ALONG = (1.0, 0.0, -1.0)  # T1, T2, T3 offsets along the heading


def _round(v: float) -> int:
    """Round half away from zero, so left/right offsets stay symmetric."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


class SpriteRenderer:
    def __init__(self, cfg: RenderConfig) -> None:
        self._arena = cfg.arena_cm

    def render(self, fly: FlyState) -> Frame:
        frame = new_frame()
        scale = SIDE / self._arena
        cx = int((fly.x % self._arena) * scale)
        cy = int((fly.y % self._arena) * scale)
        hx, hy = math.cos(fly.heading), math.sin(fly.heading)
        nx, ny = -hy, hx

        def put(dx: float, dy: float, colour: tuple[int, int, int]) -> None:
            col = (cx + _round(dx)) % SIDE
            row = (SIDE - 1 - (cy + _round(dy))) % SIDE
            frame[row, col] = colour

        for i, along in enumerate(_LEG_ALONG):
            for side, sign in ((0, 1.0), (1, -1.0)):
                down = fly.legs_down[2 * i + side]
                reach = 1.5 if down else 1.0
                put(
                    along * hx + sign * reach * nx,
                    along * hy + sign * reach * ny,
                    LEG_DOWN if down else LEG_UP,
                )
        if fly.airborne and fly.wing_phase < 0.5:
            put(-hx + 2 * nx, -hy + 2 * ny, WING)
            put(-hx - 2 * nx, -hy - 2 * ny, WING)
        body = BODY_AIRBORNE if fly.airborne else BODY
        put(-hx, -hy, body)
        put(0.0, 0.0, body)
        put(hx, hy, body)
        put(2 * hx, 2 * hy, HEAD)
        return frame
