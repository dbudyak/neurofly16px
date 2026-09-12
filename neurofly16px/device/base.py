"""Display stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import Frame


class Display(Protocol):
    def show(self, frame: Frame) -> None: ...

    def close(self) -> None: ...
