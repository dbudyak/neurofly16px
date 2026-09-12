"""ANSI 24-bit colour rendering of a frame; each pixel is two spaces with a background colour."""

from __future__ import annotations

import sys
from typing import TextIO

from neurofly16px.types import Frame


class TerminalDisplay:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self._out = stream
        self._out.write("\x1b[?25l\x1b[2J")

    def show(self, frame: Frame) -> None:
        lines = []
        for row in frame:
            cells = "".join(f"\x1b[48;2;{r};{g};{b}m  " for r, g, b in row.tolist())
            lines.append(cells + "\x1b[0m\n")
        self._out.write("\x1b[H" + "".join(lines))
        self._out.flush()

    def close(self) -> None:
        self._out.write("\x1b[0m\x1b[?25h\n")
        self._out.flush()
