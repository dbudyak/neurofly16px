"""Behaviour stage: audio features + body state -> steering."""

from __future__ import annotations

from typing import Protocol

from neurofly16px.types import AudioFeatures, FlyState, SteeringCommand


class Behavior(Protocol):
    def update(self, audio: AudioFeatures, fly: FlyState) -> SteeringCommand: ...
