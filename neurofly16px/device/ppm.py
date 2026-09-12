"""Writes every frame as a numbered binary PPM (no image library needed)."""

from __future__ import annotations

from pathlib import Path

from neurofly16px.types import SIDE, Frame


class PpmDisplay:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def show(self, frame: Frame) -> None:
        path = self._dir / f"frame_{self._n:06d}.ppm"
        path.write_bytes(f"P6\n{SIDE} {SIDE}\n255\n".encode() + frame.tobytes())
        self._n += 1

    def close(self) -> None:
        pass
