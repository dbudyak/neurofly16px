"""Audio feature stage."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import AudioFeatures


class AudioSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest(self) -> AudioFeatures:
        """Most recent features; never blocks."""
        ...
