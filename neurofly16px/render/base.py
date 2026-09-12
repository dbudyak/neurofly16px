"""Rendering stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import FlyState, Frame


class Renderer(Protocol):
    def render(self, fly: FlyState) -> Frame: ...
